"""
GAMMA SWEEP on the prune-MDP (actor-critic) — tests whether discounting (γ<1)
changes the RL result. ALL prior RL runs used γ=1.0, which (with the telescoping
reward) makes the return path-independent ⇒ degenerate MDP. γ<1 breaks that
path-independence, BUT weights the final mask only by γ^(T-1) and rewards the
intermediate accuracy trajectory (≈ anytime pruning). This run measures both.

Config (flagged): AC (best RL), γ∈{0.9,0.95,0.99,1.0}, 3 seeds, 300 episodes,
80% prune target, chunk 16 — identical to multi_seed_ac.py except γ.

Measures:
  (1) final-mask full-test drop vs γ (headline; predict ≥ γ=1, i.e. worse/equal).
  (2) anytime curve: greedy-rollout accuracy vs %pruned (predict γ<1 = smoother
      descent but lower final acc).

Output: experiments/latest/rl/gamma_sweep/{plot.png, summary.txt, run.log}
Run: venv/bin/python scripts/rl/gamma_sweep.py
"""

import os
import sys
import random
import datetime

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(".")
from src.model import MLP
from src.dataset import get_mnist_loaders
from src.pruners.rl_policy import PolicyNet
from src.pruners.rl_value import ValueNet
from src.rl.env import PruneEnv
from src.rl.ac_train import run_episode_ac, actor_critic_update
from src.interpretability import analyze_pruner, evaluate_with_gates

CKPT_PATH   = "experiments/checkpoints/mnist_model.pt"
CONFIG_PATH = "configs/config.yaml"
OUT_DIR     = "experiments/latest/rl/gamma_sweep"

GAMMAS        = [0.9, 0.95, 0.99, 1.0]
SEEDS         = [0, 1, 2]
MAX_PRUNE     = 0.80
PRUNE_CHUNK   = 16
N_EPISODES    = 300
LR            = 1e-3
ENTROPY_COEF  = 0.01
VALUE_COEF    = 0.5
CALIB_BATCH   = 256
EVAL_BATCH    = 256
RECALIB_EVERY = 5
GREEDY_EVAL_EVERY = 10
BILSTM_DROP   = 3.68          # reference (BiLSTM sw=0.5, 5-seed)


def set_seed(s):
    torch.manual_seed(s); np.random.seed(s); random.seed(s)


def load_model(device):
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    m = MLP(**ckpt["config"]).to(device); m.load_state_dict(ckpt["state_dict"]); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def build_fixed_batches(loader, n_total, device):
    xs, ys = [], []; it = iter(loader); grabbed = 0
    while grabbed < n_total:
        x, y = next(it); xs.append(x); ys.append(y); grabbed += x.size(0)
    return torch.cat(xs)[:n_total].to(device), torch.cat(ys)[:n_total].to(device)


def train_ac_gamma(model, train_loader, test_loader, device, gamma, n_hidden):
    """Train AC at the given gamma; return final-mask drop + best greedy anytime curve."""
    calib_x, _     = build_fixed_batches(train_loader, CALIB_BATCH, device)
    eval_x, eval_y = build_fixed_batches(test_loader,  EVAL_BATCH,  device)
    env = PruneEnv(model, calib_x, eval_x, eval_y, device,
                   max_prune_fraction=MAX_PRUNE, prune_chunk=PRUNE_CHUNK,
                   recalibrate_every=RECALIB_EVERY)
    policy    = PolicyNet(env.feat_dim, env.global_dim, hidden=64).to(device)
    value_net = ValueNet (env.feat_dim, env.global_dim, hidden=64).to(device)
    opt = torch.optim.Adam(list(policy.parameters()) + list(value_net.parameters()), lr=LR)

    best_greedy_acc, best_gates, best_traj = -1.0, None, None
    for ep in range(1, N_EPISODES + 1):
        lp, ent, rew, val, info = run_episode_ac(env, policy, value_net, PRUNE_CHUNK)
        actor_critic_update(opt, policy, value_net, lp, ent, rew, val,
                            entropy_coef=ENTROPY_COEF, value_coef=VALUE_COEF, gamma=gamma)
        if ep % GREEDY_EVAL_EVERY == 0:
            with torch.no_grad():
                _, _, g_rew, _, g_info = run_episode_ac(env, policy, value_net,
                                                        PRUNE_CHUNK, greedy=True)
            if g_info["final_acc"] > best_greedy_acc:
                best_greedy_acc = g_info["final_acc"]
                best_gates = [m.clone().detach().cpu() for m in env.masks]
                # anytime curve: acc_i = orig + cumsum(rewards); frac_i = (i+1)*chunk/N
                accs = g_info["orig_acc"] + np.cumsum(g_rew)
                fracs = [min((i + 1) * PRUNE_CHUNK / n_hidden, MAX_PRUNE)
                         for i in range(len(g_rew))]
                best_traj = (fracs, list(accs))

    if best_gates is None:   # safety
        with torch.no_grad():
            run_episode_ac(env, policy, value_net, PRUNE_CHUNK, greedy=True)
        best_gates = [m.clone().detach().cpu() for m in env.masks]
    test_acc = evaluate_with_gates(model, best_gates, test_loader, device)
    res = analyze_pruner(model, gates=best_gates, calib_loader=train_loader,
                         device=device, n_calib_batches=5)
    return {"test_acc": test_acc, "frac_pruned": res["frac_pruned_total"], "traj": best_traj}


def stats(v):
    a = np.array(v, float)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else 0.0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_mnist_loaders(**cfg["data"])
    base = load_model(device)
    sizes = [L.out_features for L in
             [m for m in base.modules() if isinstance(m, torch.nn.Linear)][:-1]]
    n_hidden = sum(sizes)
    full = [torch.ones(s, dtype=torch.bool, device=device) for s in sizes]
    baseline = evaluate_with_gates(base, full, test_loader, device)
    print(f"Device {device} | baseline {baseline*100:.2f}% | n_hidden {n_hidden}")
    print(f"GAMMAS={GAMMAS} SEEDS={SEEDS} episodes={N_EPISODES} target={MAX_PRUNE}\n")

    per_gamma = {}          # gamma -> {"drops":[...], "trajs":[...]}
    for gamma in GAMMAS:
        drops, trajs = [], []
        for seed in SEEDS:
            model = load_model(device); set_seed(seed)
            t0 = datetime.datetime.now()
            r = train_ac_gamma(model, train_loader, test_loader, device, gamma, n_hidden)
            dt = (datetime.datetime.now() - t0).total_seconds()
            drop = (baseline - r["test_acc"]) * 100
            drops.append(drop); trajs.append(r["traj"])
            print(f"  gamma={gamma:<5} seed={seed}: pruned={r['frac_pruned']*100:5.1f}% "
                  f"drop={drop:6.2f}pp [{dt:.0f}s]", flush=True)
        per_gamma[gamma] = {"drops": drops, "trajs": trajs}
        dm, ds = stats(drops)
        print(f"  -> gamma={gamma}: drop {dm:.2f} ± {ds:.2f}pp\n", flush=True)

    # ── plot ───────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    gs = [str(g) for g in GAMMAS]
    means = [stats(per_gamma[g]["drops"])[0] for g in GAMMAS]
    sds   = [stats(per_gamma[g]["drops"])[1] for g in GAMMAS]
    ax1.bar(gs, means, yerr=sds, color="#16a085", alpha=0.85, capsize=8)
    for i, g in enumerate(GAMMAS):
        ax1.scatter([i] * len(per_gamma[g]["drops"]), per_gamma[g]["drops"],
                    color="k", s=25, zorder=3)
    ax1.axhline(BILSTM_DROP, color="#e67e22", ls="--", label=f"BiLSTM {BILSTM_DROP}pp")
    ax1.set_xlabel("γ (discount)"); ax1.set_ylabel("final-mask drop (pp) @80% prune")
    ax1.set_title("Final-mask drop vs γ (mean ± std, 3 seeds)", fontweight="bold")
    ax1.legend(); ax1.grid(axis="y", alpha=0.3)

    cmap = plt.cm.viridis(np.linspace(0, 1, len(GAMMAS)))
    for g, c in zip(GAMMAS, cmap):
        # average the seed trajectories (same length: deterministic step count)
        trajs = [t for t in per_gamma[g]["trajs"] if t is not None]
        if not trajs:
            continue
        L = min(len(t[0]) for t in trajs)
        fracs = np.array(trajs[0][0][:L]) * 100
        accs = np.mean([np.array(t[1][:L]) for t in trajs], axis=0) * 100
        ax2.plot(fracs, accs, color=c, lw=2, label=f"γ={g}")
    ax2.set_xlabel("% pruned (greedy trajectory)")
    ax2.set_ylabel("eval-batch accuracy (%)")
    ax2.set_title("Anytime curve — accuracy vs sparsity (best greedy rollout)", fontweight="bold")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/plot.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ── summary ─────────────────────────────────────────────────────────────────
    lines = [
        "=" * 64,
        "GAMMA SWEEP — actor-critic, 80% prune target, 3 seeds, 300 ep",
        f"baseline {baseline*100:.2f}%   reference: BiLSTM {BILSTM_DROP}pp, γ=1 prior AC 4.71±1.55",
        "=" * 64,
        f"{'gamma':>6} | {'final-mask drop (pp)':>22} | per-seed",
        "-" * 64,
    ]
    for g in GAMMAS:
        dm, ds = stats(per_gamma[g]["drops"])
        ps = ", ".join(f"{d:.2f}" for d in per_gamma[g]["drops"])
        lines.append(f"{g:>6} | {dm:>10.2f} ± {ds:<8.2f} | [{ps}]")
    lines += ["-" * 64,
              "γ^(T-1) final-mask weight (T≈100): γ=0.9→3e-5, 0.95→6e-3, 0.99→0.37, 1.0→1.0",
              "=" * 64]
    with open(f"{OUT_DIR}/summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nSaved {OUT_DIR}/plot.png")


if __name__ == "__main__":
    main()
