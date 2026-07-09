"""
Multi-seed actor-critic comparison on the prune-MDP at 80% target.

Same seeds, hyperparameters, episode budget, and env config as the prior
multi_seed_compare.py REINFORCE run, so the only delta is the substitution
of EMA scalar baseline with learned V(s) and reward-to-go advantages.

Output: experiments/latest/rl/variance_study/actor_critic/{summary.txt, summary.png, training.png, run.log}
Run from project root:
    venv/bin/python scripts/multi_seed_ac.py
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
SEEDS         = [0, 1, 2, 3, 4]
CKPT_PATH     = "experiments/checkpoints/mnist_model.pt"
CONFIG_PATH   = "configs/config.yaml"
OUT_DIR       = "experiments/latest/rl/variance_study/actor_critic"

MAX_PRUNE         = 0.80
PRUNE_CHUNK       = 16
N_EPISODES        = 300
LR                = 1e-3
ENTROPY_COEF      = 0.01     # same as REINFORCE run for isolation of baseline effect
VALUE_COEF        = 0.5
GAMMA             = 1.0
CALIB_BATCH       = 256
EVAL_BATCH        = 256
RECALIB_EVERY     = 5
GREEDY_EVAL_EVERY = 10


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
    it = iter(loader); grabbed = 0
    while grabbed < n_total:
        x, y = next(it); xs.append(x); ys.append(y); grabbed += x.size(0)
    return torch.cat(xs)[:n_total].to(device), torch.cat(ys)[:n_total].to(device)


def train_ac(model, train_loader, test_loader, device) -> dict:
    calib_x, _     = build_fixed_batches(train_loader, CALIB_BATCH, device)
    eval_x, eval_y = build_fixed_batches(test_loader,  EVAL_BATCH,  device)
    env = PruneEnv(model, calib_x, eval_x, eval_y, device,
                   max_prune_fraction=MAX_PRUNE,
                   prune_chunk=PRUNE_CHUNK,
                   recalibrate_every=RECALIB_EVERY)
    policy    = PolicyNet(env.feat_dim, env.global_dim, hidden=64).to(device)
    value_net = ValueNet (env.feat_dim, env.global_dim, hidden=64).to(device)
    opt = torch.optim.Adam(
        list(policy.parameters()) + list(value_net.parameters()),
        lr=LR,
    )

    best_greedy_acc   = -1.0
    best_greedy_gates = None
    returns_per_ep    = []
    value_losses      = []

    for ep in range(1, N_EPISODES + 1):
        log_probs, entropies, rewards, values, info = run_episode_ac(
            env, policy, value_net, PRUNE_CHUNK
        )
        stats = actor_critic_update(
            opt, policy, value_net, log_probs, entropies, rewards, values,
            entropy_coef=ENTROPY_COEF, value_coef=VALUE_COEF, gamma=GAMMA,
        )
        returns_per_ep.append(info["return"])
        value_losses.append(stats["value_loss"])

        if ep % GREEDY_EVAL_EVERY == 0:
            with torch.no_grad():
                _, _, _, _, g_info = run_episode_ac(
                    env, policy, value_net, PRUNE_CHUNK, greedy=True
                )
            if g_info["final_acc"] > best_greedy_acc:
                best_greedy_acc   = g_info["final_acc"]
                best_greedy_gates = [m.clone().detach().cpu() for m in env.masks]

    if best_greedy_gates is None:
        with torch.no_grad():
            _, _, _, _, _ = run_episode_ac(env, policy, value_net, PRUNE_CHUNK, greedy=True)
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
        "returns"        : returns_per_ep,
        "value_losses"   : value_losses,
    }


def stats_fn(values):
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
    base = load_model(device)
    sizes = [L.out_features for L in
             [m for m in base.modules() if isinstance(m, torch.nn.Linear)][:-1]]
    full_gates = [torch.ones(s, dtype=torch.bool, device=device) for s in sizes]
    baseline_acc = evaluate_with_gates(base, full_gates, test_loader, device)
    print(f"Baseline test acc (full set): {baseline_acc*100:.2f}%\n")

    ac_results = []
    for i, seed in enumerate(SEEDS):
        print(f"━━━━━ seed {seed} ({i+1}/{len(SEEDS)}) ━━━━━")
        model = load_model(device)
        set_seed(seed)
        print("  [actor-critic]", end=" ", flush=True)
        t0 = datetime.datetime.now()
        r = train_ac(model, train_loader, test_loader, device)
        dt = (datetime.datetime.now() - t0).total_seconds()
        r["seed"] = seed
        ac_results.append(r)
        print(f"pruned={r['frac_pruned']*100:5.2f}%  "
              f"test_acc={r['test_acc']*100:5.2f}%  "
              f"drop={(baseline_acc-r['test_acc'])*100:5.2f}pp  "
              f"ratio={r['ratio']:.3f}  [{dt:.1f}s]\n")

    pruned = [r["frac_pruned"] for r in ac_results]
    drops  = [baseline_acc - r["test_acc"] for r in ac_results]
    ratios = [r["ratio"] for r in ac_results]
    pm, ps = stats_fn(pruned)
    dm, ds = stats_fn(drops)
    rm, rs = stats_fn(ratios)

    # ── plot per-seed training curves ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for r in ac_results:
        axes[0].plot(r["returns"],      lw=1, alpha=0.6, label=f"seed {r['seed']}")
        axes[1].plot(r["value_losses"], lw=1, alpha=0.6, label=f"seed {r['seed']}")
    axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Episode return")
    axes[0].set_title("Return per seed"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Episode"); axes[1].set_ylabel("Value loss")
    axes[1].set_title("Critic MSE per seed"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.suptitle(f"Actor-critic — 5 seeds @ {int(MAX_PRUNE*100)}% prune",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.savefig(f"{OUT_DIR}/training.png", dpi=150)
    plt.close(fig)

    # ── plot aggregate bars + dots ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Actor-critic 5-seed summary (80% prune target)",
                 fontsize=13, fontweight="bold")
    panels = [
        ("Neurons pruned (%)", [p*100 for p in pruned], pm*100, ps*100),
        ("Accuracy drop (pp)", [d*100 for d in drops],  dm*100, ds*100),
        ("Dead/Alive ratio",   ratios,                   rm,     rs),
    ]
    for ax, (title, vals, mean, sd) in zip(axes, panels):
        ax.bar(["Actor-Critic"], [mean], yerr=[sd], color="#16a085", alpha=0.85, capsize=8)
        ax.scatter([0]*len(vals), vals, color="k", s=25, zorder=3)
        ax.set_ylabel(title); ax.set_title(title + "  (mean ± std)")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plt.savefig(f"{OUT_DIR}/summary.png", dpi=150)
    plt.close(fig)

    # ── text summary ─────────────────────────────────────────────────────────
    lines = [
        "=" * 84,
        f"MULTI-SEED ACTOR-CRITIC — {int(MAX_PRUNE*100)}% prune target",
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Seeds: {SEEDS}",
        "=" * 84,
        "",
        "Base model               : MLP 784 -> 1024 -> 1024 -> 10",
        f"Baseline test accuracy   : {baseline_acc*100:.2f}%  (full MNIST test set)",
        "",
        "Algorithm: REINFORCE with learned V(s) baseline + reward-to-go advantages",
        f"  episodes={N_EPISODES}   max_prune={MAX_PRUNE}   chunk={PRUNE_CHUNK}",
        f"  lr={LR}   entropy_coef={ENTROPY_COEF}   value_coef={VALUE_COEF}",
        "",
        "PER-SEED RESULTS",
        "-" * 84,
        f"{'Seed':>4} | {'% Pruned':>9} | {'Test Acc':>9} | {'Drop pp':>8} | {'Ratio':>7}",
        "-" * 84,
    ]
    for r in ac_results:
        lines.append(
            f"{r['seed']:>4} | {r['frac_pruned']*100:8.2f}% | {r['test_acc']*100:8.2f}% | "
            f"{(baseline_acc-r['test_acc'])*100:7.2f}  | {r['ratio']:>7.3f}"
        )
    lines += [
        "",
        "AGGREGATE (mean ± std across seeds)",
        "-" * 84,
        f"{'% Pruned':>20} | {'Drop pp':>20} | {'Dead/Alive':>20}",
        f"{pm*100:9.2f} ± {ps*100:6.2f}   | {dm*100:9.2f} ± {ds*100:6.2f}   | "
        f"{rm:11.3f} ± {rs:6.3f}",
        "",
        "CROSS-METHOD COMPARISON (priors from multi_seed/summary.txt)",
        "-" * 84,
        f"{'Method':>17} | {'Drop pp':>18} | {'Dead/Alive':>18}",
        f"{'BiLSTM sw=0.5':>17} |    3.68 ± 0.79     |    0.308 ± 0.007",
        f"{'REINFORCE':>17} |    6.01 ± 4.17     |    0.486 ± 0.144",
        f"{'Actor-Critic':>17} |    {dm*100:5.2f} ± {ds*100:5.2f}     |    {rm:5.3f} ± {rs:5.3f}",
        "",
        "=" * 84,
    ]
    with open(f"{OUT_DIR}/summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
