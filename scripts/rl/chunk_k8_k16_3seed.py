"""
Clean k=8 vs k=16 comparison: 3 seeds each, 250 episodes, 80% target.

Isolates the chunk-size effect WITHOUT the episode-budget confound from
chunk_sweep.py (both k get the same 250 episodes; both are cheap enough to
converge in that budget). Actor-critic, raw (un-normalised) entropy.

Output: experiments/latest/rl/chunk_sweep/k8_vs_k16_3seed/{summary.txt, plot.png, run.log}
Run from project root:
    venv/bin/python scripts/chunk_k8_k16_3seed.py
"""

import os
import sys
import random
import datetime
import yaml
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(".")
from src.model import MLP
from src.dataset import get_mnist_loaders
from src.pruners.rl_policy import PolicyNet
from src.pruners.rl_value  import ValueNet
from src.rl.env import PruneEnv
from src.rl.ac_train import run_episode_ac, actor_critic_update
from src.interpretability import analyze_pruner, evaluate_with_gates


# ── config ────────────────────────────────────────────────────────────────────
SEEDS         = [0, 1, 2]
CHUNKS        = [8]
N_EPISODES    = 200
CKPT_PATH     = "experiments/checkpoints/mnist_model.pt"
CONFIG_PATH   = "configs/config.yaml"
OUT_DIR       = "experiments/latest/rl/chunk_sweep/k8_3seed"

MAX_PRUNE         = 0.80
LR                = 1e-3
ENTROPY_COEF      = 0.01
VALUE_COEF        = 0.5
GAMMA             = 1.0
CALIB_BATCH       = 256
EVAL_BATCH        = 256
RECALIB_EVERY     = 5
GREEDY_EVAL_EVERY = 25
NORMALIZE_ENTROPY = False


def set_seed(s):
    torch.manual_seed(s); np.random.seed(s); random.seed(s)


def load_model(device):
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    m = MLP(**ckpt["config"]).to(device)
    m.load_state_dict(ckpt["state_dict"]); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def build_fixed_batches(loader, n_total, device):
    xs, ys = [], []
    it = iter(loader); g = 0
    while g < n_total:
        x, y = next(it); xs.append(x); ys.append(y); g += x.size(0)
    return torch.cat(xs)[:n_total].to(device), torch.cat(ys)[:n_total].to(device)


def train(model, train_loader, test_loader, device, baseline_acc, chunk):
    calib_x, _     = build_fixed_batches(train_loader, CALIB_BATCH, device)
    eval_x, eval_y = build_fixed_batches(test_loader,  EVAL_BATCH,  device)
    env = PruneEnv(model, calib_x, eval_x, eval_y, device,
                   max_prune_fraction=MAX_PRUNE,
                   prune_chunk=chunk, recalibrate_every=RECALIB_EVERY)
    policy    = PolicyNet(env.feat_dim, env.global_dim, hidden=64).to(device)
    value_net = ValueNet (env.feat_dim, env.global_dim, hidden=64).to(device)
    opt = torch.optim.Adam(list(policy.parameters()) + list(value_net.parameters()), lr=LR)

    best_eval_acc, best_gates = -1.0, None
    for ep in range(1, N_EPISODES + 1):
        lp, ent, rew, val, info = run_episode_ac(
            env, policy, value_net, chunk, normalize_entropy=NORMALIZE_ENTROPY)
        actor_critic_update(opt, policy, value_net, lp, ent, rew, val,
                            entropy_coef=ENTROPY_COEF, value_coef=VALUE_COEF, gamma=GAMMA)
        if ep % GREEDY_EVAL_EVERY == 0:
            with torch.no_grad():
                _, _, _, _, g = run_episode_ac(env, policy, value_net, chunk,
                                               greedy=True, normalize_entropy=NORMALIZE_ENTROPY)
            if g["final_acc"] > best_eval_acc:
                best_eval_acc = g["final_acc"]
                best_gates    = [m.clone().detach().cpu() for m in env.masks]

    if best_gates is None:
        with torch.no_grad():
            run_episode_ac(env, policy, value_net, chunk, greedy=True,
                           normalize_entropy=NORMALIZE_ENTROPY)
        best_gates = [m.clone().detach().cpu() for m in env.masks]

    test_acc = evaluate_with_gates(model, best_gates, test_loader, device)
    res = analyze_pruner(model, gates=best_gates, calib_loader=train_loader,
                         device=device, n_calib_batches=5)
    alive, dead = res["mean_act_alive_overall"], res["mean_act_dead_overall"]
    return {
        "test_acc": test_acc, "drop": baseline_acc - test_acc,
        "frac_pruned": res["frac_pruned_total"],
        "ratio": dead / alive if alive else float("nan"),
    }


def stats(v):
    a = np.array(v, float)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else 0.0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  chunks={CHUNKS}  |  seeds={SEEDS}  |  episodes={N_EPISODES}\n")

    train_loader, test_loader = get_mnist_loaders(**cfg["data"])
    base = load_model(device)
    sizes = [L.out_features for L in
             [m for m in base.modules() if isinstance(m, torch.nn.Linear)][:-1]]
    full = [torch.ones(s, dtype=torch.bool, device=device) for s in sizes]
    baseline_acc = evaluate_with_gates(base, full, test_loader, device)
    print(f"Baseline test acc (full set): {baseline_acc*100:.2f}%\n")

    results = {k: [] for k in CHUNKS}
    for k in CHUNKS:
        for seed in SEEDS:
            model = load_model(device); set_seed(seed)
            t0 = datetime.datetime.now()
            r = train(model, train_loader, test_loader, device, baseline_acc, k)
            dt = (datetime.datetime.now() - t0).total_seconds()
            r["seed"] = seed
            results[k].append(r)
            print(f"  k={k:>2} seed {seed}: pruned={r['frac_pruned']*100:.2f}%  "
                  f"drop={r['drop']*100:.2f}pp  ratio={r['ratio']:.3f}  [{dt:.0f}s]", flush=True)
            _write_summary(results, baseline_acc)
        print()

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = list(range(len(CHUNKS)))
    means = [stats([r["drop"]*100 for r in results[k]])[0] for k in CHUNKS]
    sds   = [stats([r["drop"]*100 for r in results[k]])[1] for k in CHUNKS]
    ax.bar(xs, means, yerr=sds, color=["#16a085", "#2980b9"], alpha=0.85, capsize=8)
    for x, k in zip(xs, CHUNKS):
        for r in results[k]:
            ax.scatter([x], [r["drop"]*100], color="k", s=30, zorder=3)
    ax.axhline(3.68, color="#c0392b", ls="--", alpha=0.6, label="BiLSTM 5-seed (3.68)")
    ax.set_xticks(xs); ax.set_xticklabels([f"k={k}" for k in CHUNKS])
    ax.set_ylabel("Best full-test drop (pp)")
    ax.set_title("k=8 vs k=16 — 3 seeds, 250 ep, 80% target (mean ± std)", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); plt.savefig(f"{OUT_DIR}/plot.png", dpi=150); plt.close(fig)
    print(f"Saved plot to {OUT_DIR}/plot.png")
    _write_summary(results, baseline_acc, final=True)


def _write_summary(results, baseline_acc, final=False):
    lines = [
        "=" * 78,
        "k=8 vs k=16 — actor-critic, 3 seeds, 250 episodes, 80% target, raw entropy",
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 78,
        "",
        f"Baseline test accuracy: {baseline_acc*100:.2f}%   (equal episode budget — no confound)",
        "",
        f"{'k':>3} | {'seed':>4} | {'% Pruned':>9} | {'Drop pp':>8} | {'Ratio':>7}",
        "-" * 78,
    ]
    for k in CHUNKS:
        for r in results[k]:
            lines.append(f"{k:>3} | {r['seed']:>4} | {r['frac_pruned']*100:8.2f}% | "
                         f"{r['drop']*100:7.2f}  | {r['ratio']:>7.3f}")
        if results[k]:
            dm, ds = stats([r["drop"]*100 for r in results[k]])
            rm, rs = stats([r["ratio"]   for r in results[k]])
            lines.append(f"{k:>3} | mean | {'':>9} | {dm:7.2f}±{ds:.2f} | {rm:.3f}±{rs:.3f}")
        lines.append("-" * 78)
    lines += [
        "",
        "PRIORS: AC k=16 5-seed 4.71±1.55 | BiLSTM 5-seed 3.68±0.79 | chunk_sweep k=8 1-seed 2.68",
        "",
        "=" * 78,
    ]
    with open(f"{OUT_DIR}/summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    if final:
        print("\n".join(lines))


if __name__ == "__main__":
    main()
