"""
Multi-seed comparison of BiLSTM hypernetwork pruning (sw=0.5) vs
REINFORCE pruning (80% target). Trains both methods from scratch on
each seed, evaluates on the full MNIST test set, runs interpretability,
and aggregates mean ± std.

Output: experiments/latest/rl/variance_study/reinforce_vs_bilstm/{plot.png, summary.txt, run.log}
Run from project root:
    venv/bin/python scripts/multi_seed_compare.py
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
from src.pruners.bilstm import Pruner as BiLSTMPruner
from src.pruners.rl_policy import PolicyNet
from src.prune_train import pruner_step, get_hidden_weights
from src.rl.env import PruneEnv
from src.rl.train import run_episode, reinforce_update
from src.interpretability import analyze_pruner, evaluate_with_gates


# ── config ────────────────────────────────────────────────────────────────────
SEEDS         = [0, 1, 2, 3, 4]
CKPT_PATH     = "experiments/checkpoints/mnist_model.pt"
CONFIG_PATH   = "configs/config.yaml"
OUT_DIR       = "experiments/latest/rl/variance_study/reinforce_vs_bilstm"

# BiLSTM
BILSTM_SW      = 0.5
BILSTM_STEPS   = 1000
BILSTM_LR      = 1e-3
BILSTM_SAMPLES = 64

# RL
RL_MAX_PRUNE         = 0.80
RL_PRUNE_CHUNK       = 16
RL_EPISODES          = 300
RL_LR                = 1e-3
RL_ENTROPY_COEF      = 0.01
RL_BASELINE_DECAY    = 0.95
RL_GAMMA             = 1.0
RL_CALIB_BATCH       = 256
RL_EVAL_BATCH        = 256
RL_RECALIB_EVERY     = 5
RL_GREEDY_EVAL_EVERY = 10


def set_seed(s: int) -> None:
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)


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
    it = iter(loader)
    grabbed = 0
    while grabbed < n_total:
        x, y = next(it)
        xs.append(x); ys.append(y)
        grabbed += x.size(0)
    return torch.cat(xs)[:n_total].to(device), torch.cat(ys)[:n_total].to(device)


def train_bilstm(model, train_loader, test_loader, device) -> dict:
    hidden_weights = get_hidden_weights(model)
    layer_shapes = [(w.shape[0], w.shape[1]) for w in hidden_weights]
    pruner = BiLSTMPruner(layer_shapes).to(device)
    opt    = torch.optim.Adam(pruner.parameters(), lr=BILSTM_LR)

    it = iter(train_loader)
    for _ in range(BILSTM_STEPS):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader); x, y = next(it)
        x, y = x[:BILSTM_SAMPLES].to(device), y[:BILSTM_SAMPLES].to(device)
        pruner_step(pruner, model, opt, x, y, BILSTM_SW)

    pruner.eval()
    with torch.no_grad():
        gates = pruner(get_hidden_weights(model))

    test_acc = evaluate_with_gates(model, gates, test_loader, device)
    result   = analyze_pruner(model, gates=gates, calib_loader=train_loader,
                              device=device, n_calib_batches=5)
    alive = result["mean_act_alive_overall"]
    dead  = result["mean_act_dead_overall"]
    return {
        "test_acc"   : test_acc,
        "frac_pruned": result["frac_pruned_total"],
        "alive_act"  : alive,
        "dead_act"   : dead,
        "ratio"      : dead / alive if alive else float("nan"),
        "per_layer"  : [p["frac_pruned"] for p in result["per_layer"]],
    }


def train_rl(model, train_loader, test_loader, device) -> dict:
    calib_x, _     = build_fixed_batches(train_loader, RL_CALIB_BATCH, device)
    eval_x, eval_y = build_fixed_batches(test_loader,  RL_EVAL_BATCH,  device)
    env = PruneEnv(model, calib_x, eval_x, eval_y, device,
                   max_prune_fraction=RL_MAX_PRUNE,
                   prune_chunk=RL_PRUNE_CHUNK,
                   recalibrate_every=RL_RECALIB_EVERY)
    policy = PolicyNet(env.feat_dim, env.global_dim, hidden=64).to(device)
    opt    = torch.optim.Adam(policy.parameters(), lr=RL_LR)

    baseline          = 0.0
    best_greedy_acc   = -1.0
    best_greedy_gates = None

    for ep in range(1, RL_EPISODES + 1):
        log_probs, entropies, rewards, info = run_episode(env, policy, RL_PRUNE_CHUNK)
        ep_return = sum(rewards)
        _ = reinforce_update(opt, policy, log_probs, entropies, rewards,
                             baseline=baseline,
                             entropy_coef=RL_ENTROPY_COEF, gamma=RL_GAMMA)
        baseline = RL_BASELINE_DECAY * baseline + (1 - RL_BASELINE_DECAY) * ep_return

        if ep % RL_GREEDY_EVAL_EVERY == 0:
            with torch.no_grad():
                _, _, _, g_info = run_episode(env, policy, RL_PRUNE_CHUNK, greedy=True)
            if g_info["final_acc"] > best_greedy_acc:
                best_greedy_acc   = g_info["final_acc"]
                best_greedy_gates = [m.clone().detach().cpu() for m in env.masks]

    if best_greedy_gates is None:
        with torch.no_grad():
            _, _, _, _ = run_episode(env, policy, RL_PRUNE_CHUNK, greedy=True)
        best_greedy_gates = [m.clone().detach().cpu() for m in env.masks]

    test_acc = evaluate_with_gates(model, best_greedy_gates, test_loader, device)
    result   = analyze_pruner(model, gates=best_greedy_gates,
                              calib_loader=train_loader,
                              device=device, n_calib_batches=5)
    alive = result["mean_act_alive_overall"]
    dead  = result["mean_act_dead_overall"]
    return {
        "test_acc"       : test_acc,
        "best_greedy_acc": best_greedy_acc,
        "frac_pruned"    : result["frac_pruned_total"],
        "alive_act"      : alive,
        "dead_act"       : dead,
        "ratio"          : dead / alive if alive else float("nan"),
        "per_layer"      : [p["frac_pruned"] for p in result["per_layer"]],
    }


def stats(values):
    a = np.array(values, dtype=float)
    if len(a) <= 1:
        return float(a.mean()), 0.0
    return float(a.mean()), float(a.std(ddof=1))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  seeds={SEEDS}\n")

    train_loader, test_loader = get_mnist_loaders(**cfg["data"])
    base_model = load_model(device)
    layer_sizes = [L.out_features for L in
                   [m for m in base_model.modules() if isinstance(m, torch.nn.Linear)][:-1]]
    baseline_gates = [torch.ones(s, dtype=torch.bool, device=device) for s in layer_sizes]
    baseline_acc = evaluate_with_gates(base_model, baseline_gates, test_loader, device)
    print(f"Baseline test acc (full set): {baseline_acc*100:.2f}%\n")

    bilstm_results, rl_results = [], []

    for i, seed in enumerate(SEEDS):
        print(f"━━━━━ seed {seed} ({i+1}/{len(SEEDS)}) ━━━━━")

        # BiLSTM ────────────────────────────────────────────────────────────
        model = load_model(device)
        set_seed(seed)
        print("  [bilstm]", end=" ", flush=True)
        t0 = datetime.datetime.now()
        br = train_bilstm(model, train_loader, test_loader, device)
        dt = (datetime.datetime.now() - t0).total_seconds()
        br["seed"] = seed
        bilstm_results.append(br)
        print(f"pruned={br['frac_pruned']*100:5.2f}%  "
              f"test_acc={br['test_acc']*100:5.2f}%  "
              f"drop={(baseline_acc-br['test_acc'])*100:5.2f}pp  "
              f"ratio={br['ratio']:.3f}  [{dt:.1f}s]")

        # RL ────────────────────────────────────────────────────────────────
        model = load_model(device)
        set_seed(seed)
        print("  [rl    ]", end=" ", flush=True)
        t0 = datetime.datetime.now()
        rr = train_rl(model, train_loader, test_loader, device)
        dt = (datetime.datetime.now() - t0).total_seconds()
        rr["seed"] = seed
        rl_results.append(rr)
        print(f"pruned={rr['frac_pruned']*100:5.2f}%  "
              f"test_acc={rr['test_acc']*100:5.2f}%  "
              f"drop={(baseline_acc-rr['test_acc'])*100:5.2f}pp  "
              f"ratio={rr['ratio']:.3f}  [{dt:.1f}s]")
        print()

    # ── aggregate ────────────────────────────────────────────────────────────
    b_pruned = [r["frac_pruned"] for r in bilstm_results]
    b_drops  = [baseline_acc - r["test_acc"] for r in bilstm_results]
    b_ratios = [r["ratio"]      for r in bilstm_results]
    r_pruned = [r["frac_pruned"] for r in rl_results]
    r_drops  = [baseline_acc - r["test_acc"] for r in rl_results]
    r_ratios = [r["ratio"]      for r in rl_results]

    bpm, bps = stats(b_pruned); bdm, bds = stats(b_drops); brm, brs = stats(b_ratios)
    rpm, rps = stats(r_pruned); rdm, rds = stats(r_drops); rrm, rrs = stats(r_ratios)

    # ── plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"BiLSTM (sw={BILSTM_SW}) vs REINFORCE ({int(RL_MAX_PRUNE*100)}% target) — "
                 f"{len(SEEDS)} seeds", fontsize=13, fontweight="bold")
    labels  = [f"BiLSTM\nsw={BILSTM_SW}", f"REINFORCE\n{int(RL_MAX_PRUNE*100)}%"]
    cols    = ["#2980b9", "#c0392b"]

    panels = [
        ("Neurons pruned (%)", b_pruned, r_pruned, [bpm*100, rpm*100], [bps*100, rps*100], 100),
        ("Accuracy drop (pp)", b_drops,  r_drops,  [bdm*100, rdm*100], [bds*100, rds*100], 1),
        ("Dead/alive ratio",   b_ratios, r_ratios, [brm,     rrm],     [brs,     rrs],     1),
    ]
    for ax, (ylab, bvals, rvals, means, sds, scale) in zip(axes, panels):
        ax.bar(labels, means, yerr=sds, color=cols, alpha=0.85, capsize=8)
        for i, vals in enumerate([bvals, rvals]):
            scaled = [v * scale for v in vals]
            ax.scatter([i]*len(scaled), scaled, color="k", s=25, zorder=3)
        ax.set_ylabel(ylab)
        ax.set_title(ylab + "  (mean ± std)")
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    plt.savefig(f"{OUT_DIR}/plot.png", dpi=150)
    plt.close(fig)
    print(f"Saved plot to {OUT_DIR}/plot.png")

    # ── summary ──────────────────────────────────────────────────────────────
    lines = [
        "=" * 84,
        f"MULTI-SEED COMPARISON — BiLSTM (sw={BILSTM_SW}) vs REINFORCE ({int(RL_MAX_PRUNE*100)}% target)",
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Seeds: {SEEDS}",
        "=" * 84,
        "",
        f"Base model               : MLP 784 -> 1024 -> 1024 -> 10",
        f"Baseline test accuracy   : {baseline_acc*100:.2f}%  (full MNIST test set)",
        "",
        f"BILSTM   steps={BILSTM_STEPS}   sw={BILSTM_SW}   lr={BILSTM_LR}",
        f"RL       episodes={RL_EPISODES}   max_prune={RL_MAX_PRUNE}   "
            f"chunk={RL_PRUNE_CHUNK}   lr={RL_LR}",
        "",
        "PER-SEED RESULTS",
        "-" * 84,
        f"{'Seed':>4} | {'Method':>10} | {'% Pruned':>9} | {'Test Acc':>9} | "
            f"{'Drop pp':>8} | {'Ratio':>7}",
        "-" * 84,
    ]
    for br, rr in zip(bilstm_results, rl_results):
        lines.append(
            f"{br['seed']:>4} | {'BiLSTM':>10} | {br['frac_pruned']*100:8.2f}% | "
            f"{br['test_acc']*100:8.2f}% | {(baseline_acc-br['test_acc'])*100:7.2f}  | "
            f"{br['ratio']:>7.3f}"
        )
        lines.append(
            f"{rr['seed']:>4} | {'REINFORCE':>10} | {rr['frac_pruned']*100:8.2f}% | "
            f"{rr['test_acc']*100:8.2f}% | {(baseline_acc-rr['test_acc'])*100:7.2f}  | "
            f"{rr['ratio']:>7.3f}"
        )

    lines += [
        "",
        "AGGREGATE (mean ± std across seeds)",
        "-" * 84,
        f"{'Method':>10} | {'% Pruned':>16} | {'Drop pp':>16} | {'Dead/Alive':>16}",
        "-" * 84,
        f"{'BiLSTM':>10} | {bpm*100:7.2f} ± {bps*100:5.2f}  | "
            f"{bdm*100:7.2f} ± {bds*100:5.2f}  | "
            f"{brm:7.3f} ± {brs:5.3f}",
        f"{'REINFORCE':>10} | {rpm*100:7.2f} ± {rps*100:5.2f}  | "
            f"{rdm*100:7.2f} ± {rds*100:5.2f}  | "
            f"{rrm:7.3f} ± {rrs:5.3f}",
        "",
        "=" * 84,
    ]

    with open(f"{OUT_DIR}/summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved summary to {OUT_DIR}/summary.txt")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
