"""
Prune-chunk sweep on the actor-critic pruner (1 seed, 80% target).

Hypothesis: shrinking the chunk k (neurons pruned per macro-step) should
(a) reduce variance — k=1 removes the multi-categorical action entirely,
    leaving a single exact categorical choice per step (no without-replacement
    joint-logprob proxy), and
(b) possibly improve performance — finer control over prune order, longer and
    more customisable MDP.

Counterforce: smaller k → longer horizon → episodes cost ~1/k more compute, and
each episode is still only ONE gradient update. So per-k episode budgets are
CAPPED to keep wall-time bounded while giving each k enough updates to show its
trend. Budgets are NOT equal across k — noted in the summary.

Raw (un-normalised) entropy, since normalising regressed the result earlier.

Output: experiments/latest/rl/chunk_sweep/{k8,k4,k1}/ + summary.txt + plot.png
Run from project root:
    venv/bin/python scripts/chunk_sweep.py
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
SEED          = 0
CKPT_PATH     = "experiments/checkpoints/mnist_model.pt"
CONFIG_PATH   = "configs/config.yaml"
OUT_DIR       = "experiments/latest/rl/chunk_sweep"

# (chunk, episodes, greedy_eval_every) — episodes capped inversely to cost
SWEEP = [
    (8, 250, 20),
    (4, 200, 20),
    (1, 100, 20),
]

MAX_PRUNE         = 0.80
LR                = 1e-3
ENTROPY_COEF      = 0.01      # raw (un-normalised) entropy
VALUE_COEF        = 0.5
GAMMA             = 1.0
CALIB_BATCH       = 256
EVAL_BATCH        = 256
RECALIB_EVERY     = 5
NORMALIZE_ENTROPY = False


def set_seed(s: int) -> None:
    torch.manual_seed(s); np.random.seed(s); random.seed(s)


def load_model(device) -> MLP:
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    model = MLP(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def build_fixed_batches(loader, n_total: int, device):
    xs, ys = [], []
    it = iter(loader); g = 0
    while g < n_total:
        x, y = next(it); xs.append(x); ys.append(y); g += x.size(0)
    return torch.cat(xs)[:n_total].to(device), torch.cat(ys)[:n_total].to(device)


def train_one_chunk(model, train_loader, test_loader, device, baseline_acc,
                    chunk, episodes, greedy_every, out_sub):
    os.makedirs(out_sub, exist_ok=True)
    calib_x, _     = build_fixed_batches(train_loader, CALIB_BATCH, device)
    eval_x, eval_y = build_fixed_batches(test_loader,  EVAL_BATCH,  device)
    env = PruneEnv(model, calib_x, eval_x, eval_y, device,
                   max_prune_fraction=MAX_PRUNE,
                   prune_chunk=chunk, recalibrate_every=RECALIB_EVERY)
    policy    = PolicyNet(env.feat_dim, env.global_dim, hidden=64).to(device)
    value_net = ValueNet (env.feat_dim, env.global_dim, hidden=64).to(device)
    opt = torch.optim.Adam(
        list(policy.parameters()) + list(value_net.parameters()), lr=LR)

    returns, value_losses = [], []
    best_eval_acc   = -1.0
    best_gates      = None

    for ep in range(1, episodes + 1):
        log_probs, entropies, rewards, values, info = run_episode_ac(
            env, policy, value_net, chunk, normalize_entropy=NORMALIZE_ENTROPY)
        stats = actor_critic_update(
            opt, policy, value_net, log_probs, entropies, rewards, values,
            entropy_coef=ENTROPY_COEF, value_coef=VALUE_COEF, gamma=GAMMA)
        returns.append(info["return"]); value_losses.append(stats["value_loss"])

        if ep % greedy_every == 0:
            with torch.no_grad():
                _, _, _, _, g_info = run_episode_ac(
                    env, policy, value_net, chunk, greedy=True,
                    normalize_entropy=NORMALIZE_ENTROPY)
            if g_info["final_acc"] > best_eval_acc:
                best_eval_acc = g_info["final_acc"]
                best_gates    = [m.clone().detach().cpu() for m in env.masks]
            print(f"    [k={chunk}] ep {ep:>3}/{episodes}  "
                  f"return={info['return']:+.3f}  "
                  f"greedy_eval_acc={g_info['final_acc']*100:.2f}%  "
                  f"steps/ep={info['n_steps']}", flush=True)

    if best_gates is None:
        with torch.no_grad():
            run_episode_ac(env, policy, value_net, chunk, greedy=True,
                           normalize_entropy=NORMALIZE_ENTROPY)
        best_gates = [m.clone().detach().cpu() for m in env.masks]

    # full-test eval of best greedy mask
    test_acc = evaluate_with_gates(model, best_gates, test_loader, device)
    result   = analyze_pruner(model, gates=best_gates, calib_loader=train_loader,
                              device=device, n_calib_batches=5)
    alive = result["mean_act_alive_overall"]; dead = result["mean_act_dead_overall"]

    # per-chunk training plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(returns, color="#2980b9", lw=1)
    axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Return")
    axes[0].set_title(f"k={chunk} return"); axes[0].grid(alpha=0.3)
    axes[1].plot(value_losses, color="#d35400", lw=1)
    axes[1].set_xlabel("Episode"); axes[1].set_ylabel("Value loss")
    axes[1].set_title(f"k={chunk} critic MSE"); axes[1].grid(alpha=0.3)
    fig.tight_layout(); plt.savefig(f"{out_sub}/training.png", dpi=150); plt.close(fig)

    return {
        "chunk"      : chunk,
        "episodes"   : episodes,
        "steps_per_ep": info["n_steps"],
        "test_acc"   : test_acc,
        "drop"       : baseline_acc - test_acc,
        "frac_pruned": result["frac_pruned_total"],
        "ratio"      : dead / alive if alive else float("nan"),
        "return_final": float(np.mean(returns[-10:])),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  seed={SEED}  |  sweep={[s[0] for s in SWEEP]}\n")

    train_loader, test_loader = get_mnist_loaders(**cfg["data"])
    base = load_model(device)
    sizes = [L.out_features for L in
             [m for m in base.modules() if isinstance(m, torch.nn.Linear)][:-1]]
    full = [torch.ones(s, dtype=torch.bool, device=device) for s in sizes]
    baseline_acc = evaluate_with_gates(base, full, test_loader, device)
    print(f"Baseline test acc (full set): {baseline_acc*100:.2f}%\n")

    results = []
    for chunk, episodes, greedy_every in SWEEP:
        print(f"━━━━━ k={chunk}  ({episodes} episodes) ━━━━━")
        model = load_model(device)
        set_seed(SEED)
        t0 = datetime.datetime.now()
        r = train_one_chunk(model, train_loader, test_loader, device, baseline_acc,
                            chunk, episodes, greedy_every, f"{OUT_DIR}/k{chunk}")
        r["secs"] = (datetime.datetime.now() - t0).total_seconds()
        results.append(r)
        print(f"  → k={chunk}: pruned={r['frac_pruned']*100:.2f}%  "
              f"test_acc={r['test_acc']*100:.2f}%  drop={r['drop']*100:.2f}pp  "
              f"ratio={r['ratio']:.3f}  [{r['secs']:.0f}s]\n")
        _write_summary(results, baseline_acc)   # incremental save

    # ── comparison plot ──────────────────────────────────────────────────────
    ks    = [r["chunk"] for r in results]
    drops = [r["drop"]*100 for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, drops, "o-", color="#16a085", lw=2, ms=9)
    for k, d in zip(ks, drops):
        ax.annotate(f"{d:.2f}", (k, d), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9)
    ax.axhline(3.68, color="#2980b9", ls="--", alpha=0.6, label="BiLSTM 5-seed mean (3.68)")
    ax.axhline(4.71, color="#c0392b", ls=":",  alpha=0.6, label="AC k=16 5-seed mean (4.71)")
    ax.set_xlabel("Prune chunk k"); ax.set_ylabel("Best full-test drop (pp)")
    ax.set_xticks(ks); ax.invert_xaxis()
    ax.set_title("Chunk-size sweep — accuracy drop vs k (1 seed, 80% target)",
                 fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); plt.savefig(f"{OUT_DIR}/plot.png", dpi=150); plt.close(fig)
    print(f"Saved comparison plot to {OUT_DIR}/plot.png")

    _write_summary(results, baseline_acc, final=True)


def _write_summary(results, baseline_acc, final=False):
    lines = [
        "=" * 78,
        "PRUNE-CHUNK SWEEP — actor-critic, 1 seed, 80% target, raw entropy",
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 78,
        "",
        f"Base model            : MLP 784 -> 1024 -> 1024 -> 10",
        f"Baseline test accuracy: {baseline_acc*100:.2f}%",
        f"Seed                  : {SEED}",
        "NOTE: episode budgets are CAPPED per-k (smaller k costs ~1/k more/episode).",
        "      Comparison is across k at each k's plateau, not equal episodes.",
        "",
        f"{'k':>3} | {'episodes':>8} | {'steps/ep':>8} | {'% Pruned':>9} | "
        f"{'Test Acc':>9} | {'Drop pp':>8} | {'Ratio':>7} | {'secs':>6}",
        "-" * 78,
    ]
    for r in results:
        lines.append(
            f"{r['chunk']:>3} | {r['episodes']:>8} | {r['steps_per_ep']:>8} | "
            f"{r['frac_pruned']*100:8.2f}% | {r['test_acc']*100:8.2f}% | "
            f"{r['drop']*100:7.2f}  | {r['ratio']:>7.3f} | {r['secs']:>6.0f}")
    lines += [
        "-" * 78,
        "",
        "PRIORS (k=16): AC 5-seed 4.71±1.55 | BiLSTM 5-seed 3.68±0.79 | REINFORCE 6.01±4.17",
        "",
        "=" * 78,
    ]
    with open(f"{OUT_DIR}/summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    if final:
        print("\n".join(lines))


if __name__ == "__main__":
    main()
