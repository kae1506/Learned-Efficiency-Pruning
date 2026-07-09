"""
Actor-critic + NORMALISED ENTROPY convergence study.

Same env / policy / value setup as multi_seed_ac.py, with two changes:
  - entropy is normalised by log(N_alive) each step (entropy ∈ [0,1])
  - 3 seeds, 500 episodes

At every greedy checkpoint (every 10 episodes) the current greedy mask is
evaluated on the FULL MNIST test set, so we can trace the accuracy-drop
trajectory and answer: does the drop meaningfully change after episode 300,
and if so where does it plateau?

Output: experiments/latest/rl/variance_study/actor_critic_norment/{summary.txt, convergence.png, run.log}
Run from project root:
    venv/bin/python scripts/multi_seed_ac_norment.py
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
from src.interpretability import evaluate_with_gates


# ── config ────────────────────────────────────────────────────────────────────
SEEDS         = [0, 1, 2]
CKPT_PATH     = "experiments/checkpoints/mnist_model.pt"
CONFIG_PATH   = "configs/config.yaml"
OUT_DIR       = "experiments/latest/rl/variance_study/actor_critic_norment"

MAX_PRUNE         = 0.80
PRUNE_CHUNK       = 16
N_EPISODES        = 500
LR                = 1e-3
ENTROPY_COEF      = 0.01     # now applied to NORMALISED entropy ∈ [0,1]
VALUE_COEF        = 0.5
GAMMA             = 1.0
CALIB_BATCH       = 256
EVAL_BATCH        = 256
RECALIB_EVERY     = 5
GREEDY_EVAL_EVERY = 10
NORMALIZE_ENTROPY = True

# convergence detection threshold (pp of full-test drop)
PLATEAU_THRESHOLD = 0.30


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


def train_ac_traced(model, train_loader, test_loader, device, baseline_acc) -> dict:
    calib_x, _     = build_fixed_batches(train_loader, CALIB_BATCH, device)
    eval_x, eval_y = build_fixed_batches(test_loader,  EVAL_BATCH,  device)
    env = PruneEnv(model, calib_x, eval_x, eval_y, device,
                   max_prune_fraction=MAX_PRUNE,
                   prune_chunk=PRUNE_CHUNK,
                   recalibrate_every=RECALIB_EVERY)
    policy    = PolicyNet(env.feat_dim, env.global_dim, hidden=64).to(device)
    value_net = ValueNet (env.feat_dim, env.global_dim, hidden=64).to(device)
    opt = torch.optim.Adam(
        list(policy.parameters()) + list(value_net.parameters()), lr=LR,
    )

    ckpt_eps, ckpt_drops = [], []   # full-test drop at each greedy checkpoint
    best_full_acc   = -1.0
    best_full_gates = None

    for ep in range(1, N_EPISODES + 1):
        log_probs, entropies, rewards, values, info = run_episode_ac(
            env, policy, value_net, PRUNE_CHUNK, normalize_entropy=NORMALIZE_ENTROPY
        )
        actor_critic_update(
            opt, policy, value_net, log_probs, entropies, rewards, values,
            entropy_coef=ENTROPY_COEF, value_coef=VALUE_COEF, gamma=GAMMA,
        )

        if ep % GREEDY_EVAL_EVERY == 0:
            with torch.no_grad():
                run_episode_ac(env, policy, value_net, PRUNE_CHUNK,
                               greedy=True, normalize_entropy=NORMALIZE_ENTROPY)
            greedy_mask = [m.clone().detach().cpu() for m in env.masks]
            full_acc  = evaluate_with_gates(model, greedy_mask, test_loader, device)
            full_drop = baseline_acc - full_acc
            ckpt_eps.append(ep)
            ckpt_drops.append(full_drop)
            if full_acc > best_full_acc:
                best_full_acc   = full_acc
                best_full_gates = greedy_mask

    return {
        "ckpt_eps"   : ckpt_eps,
        "ckpt_drops" : ckpt_drops,                       # per-checkpoint drop
        "best_drop"  : baseline_acc - best_full_acc,     # best over all checkpoints
        "frac_pruned": env.fraction_pruned,
    }


def running_min(xs):
    out, m = [], float("inf")
    for x in xs:
        m = min(m, x)
        out.append(m)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  seeds={SEEDS}  |  episodes={N_EPISODES}  |  norm_entropy={NORMALIZE_ENTROPY}\n")

    train_loader, test_loader = get_mnist_loaders(**cfg["data"])
    base = load_model(device)
    sizes = [L.out_features for L in
             [m for m in base.modules() if isinstance(m, torch.nn.Linear)][:-1]]
    full_gates = [torch.ones(s, dtype=torch.bool, device=device) for s in sizes]
    baseline_acc = evaluate_with_gates(base, full_gates, test_loader, device)
    print(f"Baseline test acc (full set): {baseline_acc*100:.2f}%\n")

    results = []
    for i, seed in enumerate(SEEDS):
        print(f"━━━━━ seed {seed} ({i+1}/{len(SEEDS)}) ━━━━━")
        model = load_model(device)
        set_seed(seed)
        t0 = datetime.datetime.now()
        r = train_ac_traced(model, train_loader, test_loader, device, baseline_acc)
        dt = (datetime.datetime.now() - t0).total_seconds()
        r["seed"] = seed
        results.append(r)
        # print the trajectory compactly
        print(f"  best_drop={r['best_drop']*100:.2f}pp  pruned={r['frac_pruned']*100:.2f}%  [{dt:.1f}s]")
        traj = "  ".join(f"e{e}:{d*100:.1f}" for e, d in
                         zip(r["ckpt_eps"][::5], r["ckpt_drops"][::5]))  # every 50 eps
        print(f"  drop@[50,100,...]: {traj}\n")

    # ── aggregate running-best curve across seeds ────────────────────────────
    eps = results[0]["ckpt_eps"]                          # same grid for all seeds
    rb_per_seed = [running_min(r["ckpt_drops"]) for r in results]
    rb_arr = np.array(rb_per_seed) * 100                  # [n_seeds, n_ckpts] in pp
    mean_rb = rb_arr.mean(axis=0)
    std_rb  = rb_arr.std(axis=0, ddof=1) if len(SEEDS) > 1 else np.zeros_like(mean_rb)

    # raw (non-running-best) mean drop too, for context
    raw_arr = np.array([r["ckpt_drops"] for r in results]) * 100
    mean_raw = raw_arr.mean(axis=0)

    final_rb = mean_rb[-1]

    # convergence: first episode where remaining improvement < threshold
    convergence_ep = eps[-1]
    for e, v in zip(eps, mean_rb):
        if (v - final_rb) < PLATEAU_THRESHOLD:
            convergence_ep = e
            break

    # specifically: change between ep 300 and ep 500
    def rb_at(target_ep):
        idx = min(range(len(eps)), key=lambda j: abs(eps[j] - target_ep))
        return mean_rb[idx], eps[idx]
    rb300, e300 = rb_at(300)
    rb500, e500 = rb_at(500)
    change_300_500 = rb300 - rb500

    # ── plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for r, rb in zip(results, rb_per_seed):
        ax.plot(eps, [d*100 for d in r["ckpt_drops"]], lw=0.8, alpha=0.3,
                label=f"seed {r['seed']} (raw)")
    ax.plot(eps, mean_rb, color="#16a085", lw=2.5, label="mean running-best drop")
    ax.fill_between(eps, mean_rb - std_rb, mean_rb + std_rb, color="#16a085", alpha=0.15)
    ax.axvline(300, color="gray", ls="--", alpha=0.6, label="episode 300")
    ax.axvline(convergence_ep, color="#c0392b", ls=":", lw=2,
               label=f"plateau @ ep {convergence_ep} (<{PLATEAU_THRESHOLD}pp left)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Full-test accuracy drop (pp)")
    ax.set_title("Actor-critic + normalised entropy — convergence of best mask\n"
                 f"(3 seeds, 500 episodes, {int(MAX_PRUNE*100)}% prune target)",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.savefig(f"{OUT_DIR}/convergence.png", dpi=150)
    plt.close(fig)
    print(f"Saved convergence plot to {OUT_DIR}/convergence.png")

    # ── summary ──────────────────────────────────────────────────────────────
    best_drops = [r["best_drop"]*100 for r in results]
    bd_mean = float(np.mean(best_drops))
    bd_std  = float(np.std(best_drops, ddof=1)) if len(SEEDS) > 1 else 0.0

    lines = [
        "=" * 84,
        "ACTOR-CRITIC + NORMALISED ENTROPY — convergence study",
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Seeds: {SEEDS}   Episodes: {N_EPISODES}   normalize_entropy={NORMALIZE_ENTROPY}",
        "=" * 84,
        "",
        "Base model               : MLP 784 -> 1024 -> 1024 -> 10",
        f"Baseline test accuracy   : {baseline_acc*100:.2f}%",
        f"Entropy coef             : {ENTROPY_COEF}  (on normalised entropy ∈ [0,1])",
        f"Prune target             : {int(MAX_PRUNE*100)}%",
        "",
        "BEST-MASK FULL-TEST DROP (over all checkpoints)",
        "-" * 84,
        f"{'Seed':>4} | {'% Pruned':>9} | {'Best drop (pp)':>15}",
        "-" * 84,
    ]
    for r in results:
        lines.append(f"{r['seed']:>4} | {r['frac_pruned']*100:8.2f}% | {r['best_drop']*100:14.2f}")
    lines += [
        "-" * 84,
        f"{'mean':>4} |           | {bd_mean:8.2f} ± {bd_std:.2f}",
        "",
        "CONVERGENCE ANALYSIS (mean running-best drop across seeds)",
        "-" * 84,
        f"  Final (ep {e500}) drop          : {rb500:.2f} pp",
        f"  Drop at ep {e300}               : {rb300:.2f} pp",
        f"  Improvement ep {e300}->{e500}     : {change_300_500:.2f} pp",
        f"  Plateau episode (<{PLATEAU_THRESHOLD}pp left): {convergence_ep}",
        "",
    ]
    if change_300_500 < PLATEAU_THRESHOLD:
        lines.append(f"  VERDICT: accuracy drop does NOT meaningfully change after ep 300 "
                     f"(only {change_300_500:.2f}pp gained from 300->500).")
    else:
        lines.append(f"  VERDICT: drop DOES keep improving after ep 300 "
                     f"({change_300_500:.2f}pp gained 300->500); "
                     f"meaningful change stops around ep {convergence_ep}.")
    lines += [
        "",
        "Per-checkpoint mean running-best drop (pp):",
        "  " + "  ".join(f"e{e}:{v:.2f}" for e, v in zip(eps, mean_rb)),
        "",
        "PRIORS (5-seed):  BiLSTM 3.68±0.79   REINFORCE 6.01±4.17   AC(raw-ent) 4.71±1.55",
        "",
        "=" * 84,
    ]
    with open(f"{OUT_DIR}/summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
