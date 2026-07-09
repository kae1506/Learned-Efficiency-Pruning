"""
Rigorous iso-accuracy architecture comparison (Q9 / "which shape prunes best
at a fixed accuracy budget").

Unlike iso_accuracy_sweep.py (which re-thresholds ONE pruner's fixed ranking and
is ~1-2pp pessimistic), this RETRAINS a fresh BiLSTM pruner at each sparsity
weight (sw). The pruner is genuinely optimised for each operating point, so the
survivor count at the 2pp budget is the real achievable number, not a pessimistic
proxy. Run for 2 seeds per (model, sw) to get error bars — BiLSTM is near-
deterministic so 2 seeds is enough.

Procedure per architecture:
  1. Load the cached frozen base model, measure its full-test baseline accuracy.
  2. For each sw in a per-model grid (grids are centred where each shape is
     expected to cross 2pp — narrow needs tiny sw, deep needs large sw) and each
     seed: train a pruner, apply its binary mask, measure full-test drop +
     surviving-neuron count.
  3. For each seed, pick the most aggressive pruner (fewest survivors) whose drop
     is still <= 2pp. Average that survivor count across seeds.

Compare survivors@2pp across the four shapes: do they converge (absolute task
floor) or differ (architecture-dependent)?

Output: experiments/latest/hypernetwork/iso_accuracy_retrain/{summary.txt, plot.png, run.log}
Run from project root:
    venv/bin/python scripts/hypernetwork/iso_accuracy_retrain.py
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
from src.prune_train import pruner_step, get_hidden_weights
from src.interpretability import evaluate_with_gates


# ── config ────────────────────────────────────────────────────────────────────
# Per-model sparsity-weight grids. Each grid is chosen to BRACKET the 2pp drop
# for that architecture, based on prior runs:
#   - narrow [205,205]: extremely sensitive (sw=0.5 already gave 16pp) → tiny sw.
#   - medium [1024,1024]: 2pp lands near sw≈0.3 (from earlier sw sweep).
#   - deep [512x4]: very prune-tolerant (sw=0.5 gave only 1.05pp) → large sw.
#   - wide [2048]: sw=0.5 gave 7.3pp → small-ish sw.
# (name, checkpoint, [sw grid])
MODELS = [
    ("narrow [205,205]",  "experiments/checkpoints/mnist_narrow205x2.pt", [0.01, 0.03, 0.06, 0.10, 0.20]),
    ("medium [1024,1024]","experiments/checkpoints/mnist_model.pt",        [0.15, 0.25, 0.30, 0.40, 0.55]),
    ("deep [512x4]",      "experiments/checkpoints/mnist_deep4x512.pt",     [0.50, 0.80, 1.20, 1.80, 2.50]),
    ("wide [2048]",       "experiments/checkpoints/mnist_wide2048.pt",      [0.05, 0.10, 0.20, 0.30, 0.45]),
]
SEEDS         = [0, 1]
TARGET_DROP   = 2.0       # pp — the iso-accuracy budget
PRUNER_STEPS  = 1000
PRUNER_LR     = 1e-3
SAMPLES       = 64

CONFIG_PATH   = "configs/config.yaml"
OUT_DIR       = "experiments/latest/hypernetwork/iso_accuracy_retrain"


def set_seed(s: int) -> None:
    """Seed all RNGs so a (model, sw, seed) run is reproducible."""
    torch.manual_seed(s); np.random.seed(s); random.seed(s)


def load_frozen(ckpt_path: str, device) -> MLP:
    """Load a trained MLP checkpoint and freeze it (pruner never updates base)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    m = MLP(**ckpt["config"]).to(device)
    m.load_state_dict(ckpt["state_dict"]); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def train_pruner_and_eval(model, train_loader, test_loader, device, sw, baseline_acc):
    """
    Train one BiLSTM pruner on `model` at sparsity weight `sw`, then apply its
    binary mask and measure full-test accuracy drop + surviving neuron count.
    Returns dict with sw, drop (pp), survivors, frac_pruned.
    """
    # Build a pruner sized to this model's hidden-layer shapes.
    layer_shapes = [(w.shape[0], w.shape[1]) for w in get_hidden_weights(model)]
    pruner = BiLSTMPruner(layer_shapes).to(device)
    opt = torch.optim.Adam(pruner.parameters(), lr=PRUNER_LR)

    # Standard pruner training loop: each step draws a minibatch and nudges the
    # gates to minimise (CE_pruned - CE_orig) + sw * mean(gate).
    it = iter(train_loader)
    for _ in range(PRUNER_STEPS):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader); x, y = next(it)
        x, y = x[:SAMPLES].to(device), y[:SAMPLES].to(device)
        pruner_step(pruner, model, opt, x, y, sw)

    # Read off the final binary mask and evaluate it on the FULL test set.
    pruner.eval()
    with torch.no_grad():
        gates = pruner(get_hidden_weights(model))           # list of {0,1} per layer
    test_acc = evaluate_with_gates(model, gates, test_loader, device)
    n_total = sum(int(g.numel()) for g in gates)
    survivors = int(sum(int(g.sum().item()) for g in gates))
    return {
        "sw": sw,
        "drop": (baseline_acc - test_acc) * 100,            # in pp
        "survivors": survivors,
        "n_total": n_total,
        "frac_pruned": 1 - survivors / n_total,
    }


def best_within_budget(points):
    """
    Among a seed's (sw -> result) points, return the one that prunes the MOST
    (fewest survivors) while staying within TARGET_DROP. If none qualify, return
    the lowest-drop point (and the caller flags that 2pp was unreachable).
    """
    ok = [p for p in points if p["drop"] <= TARGET_DROP]
    if ok:
        return min(ok, key=lambda p: p["survivors"]), True   # most-pruned ≤2pp
    return min(points, key=lambda p: p["drop"]), False        # closest we got


def summarize(values):
    """Mean and half-range (used as the error bar with only 2 seeds)."""
    a = np.array(values, float)
    return float(a.mean()), float((a.max() - a.min()) / 2)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  target ≤{TARGET_DROP}pp  |  seeds={SEEDS}\n")
    train_loader, test_loader = get_mnist_loaders(**cfg["data"])

    # results[name] = {
    #   "n_total", "baseline", "best": [per-seed best-within-budget dicts],
    #   "all": {seed: [all sw points]}, "reached": bool
    # }
    results = {}
    for name, ckpt, sw_grid in MODELS:
        print(f"=== {name}  (sw grid {sw_grid}) ===")
        model = load_frozen(ckpt, device)
        full = [torch.ones(w.shape[0], dtype=torch.bool, device=device)
                for w in get_hidden_weights(model)]
        baseline_acc = evaluate_with_gates(model, full, test_loader, device)

        per_seed_best, all_points, reached_all = [], {}, True
        for seed in SEEDS:
            points = []
            for sw in sw_grid:
                set_seed(seed)                              # reseed per pruner train
                r = train_pruner_and_eval(model, train_loader, test_loader,
                                          device, sw, baseline_acc)
                points.append(r)
                print(f"  seed {seed} sw={sw:<5}: drop={r['drop']:6.2f}pp  "
                      f"survivors={r['survivors']:>4}/{r['n_total']}  "
                      f"pruned={r['frac_pruned']*100:5.1f}%", flush=True)
            best, reached = best_within_budget(points)
            reached_all = reached_all and reached
            per_seed_best.append(best)
            all_points[seed] = points
            tag = "" if reached else "  (!! never reached 2pp)"
            print(f"  → seed {seed} best ≤{TARGET_DROP}pp: survivors={best['survivors']}  "
                  f"sw={best['sw']}  drop={best['drop']:.2f}pp{tag}\n", flush=True)

        results[name] = {
            "n_total": per_seed_best[0]["n_total"],
            "baseline": baseline_acc,
            "best": per_seed_best,
            "all": all_points,
            "reached": reached_all,
        }
        _write_summary(results)

    # ── plot: survivors@2pp per architecture, 2-seed mean ± half-range ────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    names = list(results.keys())
    colors = ["#e74c3c", "#2980b9", "#e67e22", "#9b59b6"]
    means, errs, totals = [], [], []
    for name in names:
        sv = [b["survivors"] for b in results[name]["best"]]
        m, e = summarize(sv)
        means.append(m); errs.append(e); totals.append(results[name]["n_total"])
    xs = np.arange(len(names))
    ax.bar(xs, totals, color="#dddddd", label="total hidden neurons")        # grey = full size
    ax.bar(xs, means, yerr=errs, color=colors, alpha=0.9, capsize=8,
           label="survivors @ ≤2pp")                                         # coloured = survivors
    for x, m, t in zip(xs, means, totals):
        ax.text(x, m + 25, f"{m:.0f}", ha="center", fontsize=10, fontweight="bold")
        ax.text(x, t + 25, f"/{t}", ha="center", fontsize=8, color="gray")
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Neurons")
    ax.set_title(f"Iso-accuracy survivors (≤{TARGET_DROP}pp) — retrained per sw, 2 seeds",
                 fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); plt.savefig(f"{OUT_DIR}/plot.png", dpi=150); plt.close(fig)
    print(f"Saved plot to {OUT_DIR}/plot.png")
    _write_summary(results, final=True)


def _write_summary(results, final=False):
    lines = [
        "=" * 80,
        f"ISO-ACCURACY ARCHITECTURE COMPARISON (retrained per sw) — survivors @ ≤{TARGET_DROP}pp",
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Seeds: {SEEDS}   (fresh pruner retrained at every sw, not threshold-swept)",
        "=" * 80,
        "",
        f"{'Model':>20} | {'Total':>6} | {'Baseline':>8} | {'Survivors@2pp':>16} | {'% Pruned':>9} | {'sw':>10}",
        "-" * 80,
    ]
    for name, r in results.items():
        sv = [b["survivors"] for b in r["best"]]
        m, e = summarize(sv)
        fr = [b["frac_pruned"] * 100 for b in r["best"]]
        fm, fe = summarize(fr)
        sws = "/".join(str(b["sw"]) for b in r["best"])
        flag = "" if r["reached"] else "  (!! >2pp)"
        lines.append(f"{name:>20} | {r['n_total']:>6} | {r['baseline']*100:7.2f}% | "
                     f"{m:6.0f} ± {e:<5.0f}  | {fm:6.1f}±{fe:<4.1f}% | {sws:>10}{flag}")
    lines += [
        "-" * 80,
        "",
        "vs threshold-sweep proxy (iso_accuracy_sweep): narrow 360 / medium 1179 / deep 744 / wide 806",
        "",
        "READ: survivors@2pp similar across all → absolute task floor.",
        "      differ by architecture → prunability is shape-dependent (redundancy distribution).",
        "=" * 80,
    ]
    with open(f"{OUT_DIR}/summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    if final:
        print("\n".join(lines))


if __name__ == "__main__":
    main()
