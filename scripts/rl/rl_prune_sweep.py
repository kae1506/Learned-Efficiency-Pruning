"""
Sweep RL pruner across max-prune fractions in [0.65, 0.85].

For each fraction, train a fresh policy with REINFORCE for N_EPISODES and
record the best greedy and best sampled accuracies achieved during training.
Plot accuracy drop vs max prune fraction with a 3% threshold marker.

Run from project root:
    venv/bin/python scripts/rl_prune_sweep.py
"""

import os
import sys
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
from src.rl.env import PruneEnv
from src.rl.train import run_episode, reinforce_update


# ── config ────────────────────────────────────────────────────────────────────
CONFIG_PATH   = "configs/config.yaml"
CKPT_PATH     = "experiments/checkpoints/mnist_model.pt"
OUT_DIR       = "experiments/latest/rl/reinforce/sweep_65_85"

PRUNE_FRACTIONS = np.round(np.linspace(0.65, 0.85, 10), 4).tolist()

N_EPISODES         = 250
PRUNE_CHUNK        = 16
LR                 = 1e-3
ENTROPY_COEF       = 0.01
BASELINE_DECAY     = 0.95
GAMMA              = 1.0
CALIB_BATCH        = 256
EVAL_BATCH         = 256
RECALIB_EVERY      = 5
GREEDY_EVAL_EVERY  = 10  # more frequent greedy eval for accurate "best greedy"


def load_model(path: str, device) -> MLP:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = MLP(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def build_fixed_batches(loader, n_total: int, device):
    xs, ys = [], []
    it = iter(loader)
    grabbed = 0
    while grabbed < n_total:
        x, y = next(it)
        xs.append(x); ys.append(y)
        grabbed += x.size(0)
    return torch.cat(xs)[:n_total].to(device), torch.cat(ys)[:n_total].to(device)


def train_one_fraction(model, calib_x, eval_x, eval_y, device, max_prune):
    env = PruneEnv(
        model, calib_x, eval_x, eval_y, device,
        max_prune_fraction=max_prune,
        prune_chunk=PRUNE_CHUNK,
        recalibrate_every=RECALIB_EVERY,
    )
    policy = PolicyNet(env.feat_dim, env.global_dim, hidden=64).to(device)
    opt    = torch.optim.Adam(policy.parameters(), lr=LR)

    baseline = 0.0
    best_sampled_acc = -1.0
    best_greedy_acc  = -1.0
    best_sampled_ep  = 0
    best_greedy_ep   = 0
    final_frac       = 0.0

    for ep in range(1, N_EPISODES + 1):
        log_probs, entropies, rewards, info = run_episode(env, policy, PRUNE_CHUNK)
        ep_return = sum(rewards)
        _ = reinforce_update(
            opt, policy, log_probs, entropies, rewards,
            baseline=baseline,
            entropy_coef=ENTROPY_COEF,
            gamma=GAMMA,
        )
        baseline = BASELINE_DECAY * baseline + (1 - BASELINE_DECAY) * ep_return

        if info["final_acc"] > best_sampled_acc:
            best_sampled_acc = info["final_acc"]
            best_sampled_ep  = ep
        final_frac = info["frac_pruned"]

        if ep % GREEDY_EVAL_EVERY == 0:
            with torch.no_grad():
                _, _, _, g_info = run_episode(env, policy, PRUNE_CHUNK, greedy=True)
            if g_info["final_acc"] > best_greedy_acc:
                best_greedy_acc = g_info["final_acc"]
                best_greedy_ep  = ep

    return {
        "max_prune"        : max_prune,
        "orig_acc"         : env.orig_acc,
        "best_sampled_acc" : best_sampled_acc,
        "best_sampled_ep"  : best_sampled_ep,
        "best_greedy_acc"  : best_greedy_acc,
        "best_greedy_ep"   : best_greedy_ep,
        "achieved_frac"    : final_frac,
    }


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Fractions to sweep: {PRUNE_FRACTIONS}")
    print(f"Episodes per fraction: {N_EPISODES}")
    print()

    model = load_model(CKPT_PATH, device)
    train_loader, test_loader = get_mnist_loaders(**cfg["data"])
    calib_x, _      = build_fixed_batches(train_loader, CALIB_BATCH, device)
    eval_x, eval_y  = build_fixed_batches(test_loader,  EVAL_BATCH,  device)

    results = []
    for i, frac in enumerate(PRUNE_FRACTIONS, 1):
        print(f"[{i}/{len(PRUNE_FRACTIONS)}] max_prune = {frac:.4f}  ", end="", flush=True)
        t0 = datetime.datetime.now()
        r = train_one_fraction(model, calib_x, eval_x, eval_y, device, frac)
        dt = (datetime.datetime.now() - t0).total_seconds()
        sampled_drop = (r["orig_acc"] - r["best_sampled_acc"]) * 100
        greedy_drop  = (r["orig_acc"] - r["best_greedy_acc"])  * 100
        print(f"|  best_sampled={r['best_sampled_acc']*100:6.2f}% (drop {sampled_drop:5.2f}%)"
              f"  best_greedy={r['best_greedy_acc']*100:6.2f}% (drop {greedy_drop:5.2f}%)"
              f"  [{dt:5.1f}s]")
        results.append(r)

    # ── aggregate plot: accuracy drop vs max prune fraction ───────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    fracs            = np.array([r["max_prune"]              for r in results]) * 100
    drops_sampled    = np.array([(r["orig_acc"] - r["best_sampled_acc"]) * 100 for r in results])
    drops_greedy     = np.array([(r["orig_acc"] - r["best_greedy_acc"])  * 100 for r in results])
    achieved         = np.array([r["achieved_frac"] for r in results]) * 100

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(fracs, drops_greedy,  "o-", color="#2980b9", label="best greedy",  lw=2, ms=7)
    ax.plot(fracs, drops_sampled, "s--", color="#27ae60", label="best sampled", lw=1.5, ms=6, alpha=0.8)
    ax.axhline(3.0, color="#c0392b", ls=":", lw=1.5, label="3% drop threshold")
    ax.set_xlabel("Target prune fraction (%)")
    ax.set_ylabel("Best accuracy drop from original (pp)")
    ax.set_title("RL Pruner — accuracy drop vs target sparsity", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")

    # Annotate each greedy point
    for f, d in zip(fracs, drops_greedy):
        ax.annotate(f"{d:.1f}", (f, d), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color="#2980b9")

    fig.tight_layout()
    plot_path = f"{OUT_DIR}/plot.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot to {plot_path}")

    # ── summary ───────────────────────────────────────────────────────────────
    # find largest fraction whose best-greedy drop is still <= 3%
    under_3pct = [r for r in results if (r["orig_acc"] - r["best_greedy_acc"]) * 100 <= 3.0]
    optimal = max(under_3pct, key=lambda r: r["max_prune"]) if under_3pct else None

    lines = [
        "=" * 78,
        "RL PRUNER SWEEP — accuracy drop vs target prune fraction",
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 78,
        "",
        f"Base model        : MLP 784 -> 1024 -> 1024 -> 10  (2048 hidden neurons)",
        f"Orig eval acc     : {results[0]['orig_acc']*100:.2f}%",
        f"Episodes per run  : {N_EPISODES}",
        f"Greedy eval every : {GREEDY_EVAL_EVERY} episodes",
        f"Fractions swept   : {len(PRUNE_FRACTIONS)} in [0.65, 0.85]",
        "",
        f"{'Target':>8} | {'Achieved':>9} | {'BestSampled':>11} | {'SampDrop':>9} | "
        f"{'BestGreedy':>10} | {'GreedyDrop':>10}",
        "-" * 78,
    ]
    for r in results:
        sd = (r["orig_acc"] - r["best_sampled_acc"]) * 100
        gd = (r["orig_acc"] - r["best_greedy_acc"])  * 100
        lines.append(
            f"{r['max_prune']*100:7.2f}% | {r['achieved_frac']*100:8.2f}% | "
            f"{r['best_sampled_acc']*100:10.2f}% | {sd:8.2f}% | "
            f"{r['best_greedy_acc']*100:9.2f}% | {gd:9.2f}%"
        )
    lines += ["", "=" * 78]
    if optimal is not None:
        gd = (optimal["orig_acc"] - optimal["best_greedy_acc"]) * 100
        lines += [
            f"OPTIMAL @ 3% drop tolerance",
            f"  Target prune fraction : {optimal['max_prune']*100:.2f}%",
            f"  Best greedy accuracy  : {optimal['best_greedy_acc']*100:.2f}%",
            f"  Best greedy drop      : {gd:.2f}%",
        ]
    else:
        lines += ["No swept fraction kept best-greedy drop below 3%."]
    lines += ["=" * 78]

    summary_path = f"{OUT_DIR}/summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
