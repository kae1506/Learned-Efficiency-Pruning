"""
Per-neuron Bernoulli action — actor-critic, 3 seeds, 250 episodes, 80% target.

Replaces the multinomial "pick-k" action with independent per-neuron Bernoulli
prune decisions (exact factorised log-prob, clean per-neuron credit). Init bias
set so expected pruned/step ≈ 16 at start → episode length comparable to the
k=16 chunk regime. Entropy = SUM of per-neuron Bernoulli entropies (scales with
N_alive, the analog of the raw categorical entropy that worked best); coef small
(0.001) so the bonus magnitude (~0.1 at start) matches the categorical 0.076.

Reuses src.rl.ac_train.actor_critic_update (action-agnostic).

Output: experiments/latest/rl/bernoulli_3seed/{summary.txt, plot.png, run.log}
Run from project root:
    venv/bin/python scripts/bernoulli_3seed.py
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
from src.pruners.rl_bernoulli_policy import BernoulliPolicyNet
from src.pruners.rl_value import ValueNet
from src.rl.env import PruneEnv
from src.rl.bernoulli_train import run_episode_bernoulli
from src.rl.ac_train import actor_critic_update
from src.interpretability import analyze_pruner, evaluate_with_gates


# ── config ────────────────────────────────────────────────────────────────────
SEEDS         = [0, 1, 2]
N_EPISODES    = 250
CKPT_PATH     = "experiments/checkpoints/mnist_model.pt"
CONFIG_PATH   = "configs/config.yaml"
OUT_DIR       = "experiments/latest/rl/bernoulli_3seed"

MAX_PRUNE         = 0.80
INIT_BIAS         = -4.8     # sigmoid(-4.8)*2048 ≈ 16 expected pruned/step at start
LR                = 1e-3
ENTROPY_COEF      = 0.001    # on SUM-of-Bernoulli entropy (~0.1 bonus at start)
VALUE_COEF        = 0.5
GAMMA             = 1.0
CALIB_BATCH       = 256
EVAL_BATCH        = 256
RECALIB_EVERY     = 5
GREEDY_EVAL_EVERY = 25


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


def train(model, train_loader, test_loader, device, baseline_acc):
    calib_x, _     = build_fixed_batches(train_loader, CALIB_BATCH, device)
    eval_x, eval_y = build_fixed_batches(test_loader,  EVAL_BATCH,  device)
    env = PruneEnv(model, calib_x, eval_x, eval_y, device,
                   max_prune_fraction=MAX_PRUNE,
                   prune_chunk=1, recalibrate_every=RECALIB_EVERY)  # chunk unused by Bernoulli
    policy    = BernoulliPolicyNet(env.feat_dim, env.global_dim, hidden=64, init_bias=INIT_BIAS).to(device)
    value_net = ValueNet(env.feat_dim, env.global_dim, hidden=64).to(device)
    opt = torch.optim.Adam(list(policy.parameters()) + list(value_net.parameters()), lr=LR)

    best_eval_acc, best_gates = -1.0, None
    returns = []
    for ep in range(1, N_EPISODES + 1):
        lp, ent, rew, val, info = run_episode_bernoulli(env, policy, value_net)
        actor_critic_update(opt, policy, value_net, lp, ent, rew, val,
                            entropy_coef=ENTROPY_COEF, value_coef=VALUE_COEF, gamma=GAMMA)
        returns.append(info["return"])
        if ep % GREEDY_EVAL_EVERY == 0:
            with torch.no_grad():
                _, _, _, _, g = run_episode_bernoulli(env, policy, value_net, greedy=True)
            if g["final_acc"] > best_eval_acc:
                best_eval_acc = g["final_acc"]
                best_gates    = [m.clone().detach().cpu() for m in env.masks]
            print(f"    ep {ep:>3}/{N_EPISODES}  return={info['return']:+.3f}  "
                  f"greedy_acc={g['final_acc']*100:.2f}%  greedy_pruned={g['frac_pruned']*100:.1f}%  "
                  f"steps/ep={info['n_steps']}", flush=True)

    if best_gates is None:
        with torch.no_grad():
            run_episode_bernoulli(env, policy, value_net, greedy=True)
        best_gates = [m.clone().detach().cpu() for m in env.masks]

    test_acc = evaluate_with_gates(model, best_gates, test_loader, device)
    res = analyze_pruner(model, gates=best_gates, calib_loader=train_loader,
                         device=device, n_calib_batches=5)
    alive, dead = res["mean_act_alive_overall"], res["mean_act_dead_overall"]
    return {
        "test_acc": test_acc, "drop": baseline_acc - test_acc,
        "frac_pruned": res["frac_pruned_total"],
        "ratio": dead / alive if alive else float("nan"),
        "return_final": float(np.mean(returns[-10:])),
    }


def stats(v):
    a = np.array(v, float)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else 0.0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Bernoulli action  |  seeds={SEEDS}  |  episodes={N_EPISODES}\n")

    train_loader, test_loader = get_mnist_loaders(**cfg["data"])
    base = load_model(device)
    sizes = [L.out_features for L in
             [m for m in base.modules() if isinstance(m, torch.nn.Linear)][:-1]]
    full = [torch.ones(s, dtype=torch.bool, device=device) for s in sizes]
    baseline_acc = evaluate_with_gates(base, full, test_loader, device)
    print(f"Baseline test acc (full set): {baseline_acc*100:.2f}%\n")

    results = []
    for seed in SEEDS:
        print(f"━━━━━ seed {seed} ━━━━━")
        model = load_model(device); set_seed(seed)
        t0 = datetime.datetime.now()
        r = train(model, train_loader, test_loader, device, baseline_acc)
        r["seed"] = seed
        r["secs"] = (datetime.datetime.now() - t0).total_seconds()
        results.append(r)
        print(f"  → seed {seed}: pruned={r['frac_pruned']*100:.2f}%  "
              f"test_acc={r['test_acc']*100:.2f}%  drop={r['drop']*100:.2f}pp  "
              f"ratio={r['ratio']:.3f}  [{r['secs']:.0f}s]\n", flush=True)
        _write_summary(results, baseline_acc)

    # ── plot ──────────────────────────────────────────────────────────────────
    dm, ds = stats([r["drop"]*100 for r in results])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["Bernoulli\nAC"], [dm], yerr=[ds], color="#8e44ad", alpha=0.85, capsize=8)
    for r in results:
        ax.scatter([0], [r["drop"]*100], color="k", s=30, zorder=3)
    ax.axhline(3.68, color="#2980b9", ls="--", alpha=0.6, label="BiLSTM 5-seed (3.68)")
    ax.axhline(4.71, color="#c0392b", ls=":",  alpha=0.6, label="AC k=16 5-seed (4.71)")
    ax.set_ylabel("Best full-test drop (pp)")
    ax.set_title("Per-neuron Bernoulli action — 3 seeds, 250 ep, 80% (mean ± std)",
                 fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); plt.savefig(f"{OUT_DIR}/plot.png", dpi=150); plt.close(fig)
    print(f"Saved plot to {OUT_DIR}/plot.png")
    _write_summary(results, baseline_acc, final=True)


def _write_summary(results, baseline_acc, final=False):
    dm, ds = stats([r["drop"]*100 for r in results]) if results else (0, 0)
    rm, rs = stats([r["ratio"] for r in results]) if results else (0, 0)
    lines = [
        "=" * 78,
        "PER-NEURON BERNOULLI ACTION — actor-critic, 3 seeds, 250 ep, 80% target",
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 78,
        "",
        f"Baseline test accuracy: {baseline_acc*100:.2f}%",
        f"Action     : independent Bernoulli per alive neuron (exact factorised log-prob)",
        f"Init bias  : {INIT_BIAS} (≈16 expected pruned/step at start)",
        f"Entropy    : SUM per-neuron Bernoulli H, coef={ENTROPY_COEF}",
        "",
        f"{'seed':>4} | {'% Pruned':>9} | {'Test Acc':>9} | {'Drop pp':>8} | {'Ratio':>7} | {'secs':>6}",
        "-" * 78,
    ]
    for r in results:
        lines.append(f"{r['seed']:>4} | {r['frac_pruned']*100:8.2f}% | {r['test_acc']*100:8.2f}% | "
                     f"{r['drop']*100:7.2f}  | {r['ratio']:>7.3f} | {r['secs']:>6.0f}")
    lines += [
        "-" * 78,
        f"mean | {'':>9} | {'':>9} | {dm:7.2f}±{ds:.2f} | {rm:.3f}±{rs:.3f}",
        "",
        "PRIORS: AC k=16 5-seed 4.71±1.55 | AC k=8 3-seed 5.54±3.44 | BiLSTM 5-seed 3.68±0.79",
        "",
        "=" * 78,
    ]
    with open(f"{OUT_DIR}/summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    if final:
        print("\n".join(lines))


if __name__ == "__main__":
    main()
