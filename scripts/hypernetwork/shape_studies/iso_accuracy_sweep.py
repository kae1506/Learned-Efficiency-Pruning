"""
Iso-accuracy survivor sweep (Q9): is the minimal subnetwork an ABSOLUTE task
floor or network-relative?

For each cached model shape, train ONE BiLSTM pruner, then sweep a global
threshold on its continuous per-neuron scores (no retraining) to trace the full
accuracy-vs-survivors curve. Read off the ABSOLUTE surviving-neuron count at a
fixed accuracy budget (2pp drop) and compare across shapes:
  - survivors converge to ~same count  → ABSOLUTE task floor.
  - survivors scale with starting width → network-relative.

Cheap: 1 pruner train + ~35 forward-pass evals per model (no sw sweep, no base
retraining — all four base models are cached).

Output: experiments/latest/hypernetwork/iso_accuracy_sweep/{summary.txt, plot.png, run.log}
Run from project root:
    venv/bin/python scripts/hypernetwork/iso_accuracy_sweep.py
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
MODELS = [
    ("narrow [205,205]", "experiments/checkpoints/mnist_narrow205x2.pt"),
    ("medium [1024,1024]", "experiments/checkpoints/mnist_model.pt"),
    ("deep [512x4]",    "experiments/checkpoints/mnist_deep4x512.pt"),
    ("wide [2048]",     "experiments/checkpoints/mnist_wide2048.pt"),
]
SW            = 0.5
PRUNER_STEPS  = 1000
PRUNER_LR     = 1e-3
SAMPLES       = 64
SEED          = 0
TARGET_DROP   = 2.0       # pp — the iso-accuracy budget
KEEP_FRACS    = np.linspace(1.0, 0.03, 33)

CONFIG_PATH   = "configs/config.yaml"
OUT_DIR       = "experiments/latest/hypernetwork/iso_accuracy_sweep"


def set_seed(s):
    torch.manual_seed(s); np.random.seed(s); random.seed(s)


def load_frozen(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    m = MLP(**ckpt["config"]).to(device)
    m.load_state_dict(ckpt["state_dict"]); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def train_pruner(model, train_loader, device):
    set_seed(SEED)
    shapes = [(w.shape[0], w.shape[1]) for w in get_hidden_weights(model)]
    pruner = BiLSTMPruner(shapes).to(device)
    opt = torch.optim.Adam(pruner.parameters(), lr=PRUNER_LR)
    it = iter(train_loader)
    for _ in range(PRUNER_STEPS):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader); x, y = next(it)
        x, y = x[:SAMPLES].to(device), y[:SAMPLES].to(device)
        pruner_step(pruner, model, opt, x, y, SW)
    pruner.eval()
    return pruner


def sweep_curve(model, pruner, test_loader, device, baseline_acc):
    scores = pruner.scores(get_hidden_weights(model))          # list per layer
    n_total = sum(s.numel() for s in scores)
    flat = torch.cat([s.flatten() for s in scores])            # [n_total]
    curve = []
    for frac in KEEP_FRACS:
        k = max(1, int(round(frac * n_total)))
        thresh = torch.topk(flat, k).values.min()              # k-th largest score
        gates = [(s >= thresh).float() for s in scores]        # 1 = keep
        kept = int(sum(int(g.sum().item()) for g in gates))
        acc = evaluate_with_gates(model, gates, test_loader, device)
        curve.append({"kept": kept, "frac_pruned": 1 - kept / n_total,
                      "acc": acc, "drop": baseline_acc - acc})
    return n_total, curve


def survivors_at_target(curve):
    """Min surviving neurons among points within TARGET_DROP of baseline."""
    ok = [c for c in curve if c["drop"] * 100 <= TARGET_DROP]
    if not ok:
        return None
    return min(ok, key=lambda c: c["kept"])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  target drop = {TARGET_DROP}pp\n")
    train_loader, test_loader = get_mnist_loaders(**cfg["data"])

    rows, curves = [], {}
    for name, ckpt in MODELS:
        print(f"=== {name} ===")
        model = load_frozen(ckpt, device)
        full = [torch.ones(w.shape[0], dtype=torch.bool, device=device)
                for w in get_hidden_weights(model)]
        baseline_acc = evaluate_with_gates(model, full, test_loader, device)
        pruner = train_pruner(model, train_loader, device)
        n_total, curve = sweep_curve(model, pruner, test_loader, device, baseline_acc)
        curves[name] = (n_total, baseline_acc, curve)
        tgt = survivors_at_target(curve)
        if tgt:
            rows.append({"name": name, "n_total": n_total, "baseline": baseline_acc,
                         "kept": tgt["kept"], "frac_pruned": tgt["frac_pruned"],
                         "drop": tgt["drop"]})
            print(f"  n_total={n_total}  baseline={baseline_acc*100:.2f}%  "
                  f"@≤{TARGET_DROP}pp: kept={tgt['kept']}  pruned={tgt['frac_pruned']*100:.1f}%\n", flush=True)
        else:
            rows.append({"name": name, "n_total": n_total, "baseline": baseline_acc,
                         "kept": n_total, "frac_pruned": 0.0, "drop": 0.0})
            print(f"  n_total={n_total}  (never within {TARGET_DROP}pp even at full)\n", flush=True)
        _write_summary(rows)

    # ── plot: accuracy-drop vs survivors, all models ─────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"narrow [205,205]": "#e74c3c", "medium [1024,1024]": "#2980b9",
              "deep [512x4]": "#e67e22", "wide [2048]": "#9b59b6"}
    for name, (n_total, base, curve) in curves.items():
        xs = [c["kept"] for c in curve]
        ys = [c["drop"] * 100 for c in curve]
        ax.plot(xs, ys, "o-", ms=3, lw=1.3, color=colors.get(name), label=name)
    ax.axhline(TARGET_DROP, color="k", ls="--", alpha=0.6, label=f"{TARGET_DROP}pp budget")
    ax.set_xlabel("Surviving neurons (absolute count)")
    ax.set_ylabel("Full-test accuracy drop (pp)")
    ax.set_ylim(-0.5, 12)
    ax.set_title("Iso-accuracy survivor curves — do shapes converge to one floor?",
                 fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); plt.savefig(f"{OUT_DIR}/plot.png", dpi=150); plt.close(fig)
    print(f"Saved plot to {OUT_DIR}/plot.png")
    _write_summary(rows, final=True)


def _write_summary(rows, final=False):
    lines = [
        "=" * 78,
        f"ISO-ACCURACY SURVIVOR SWEEP — absolute survivors at ≤{TARGET_DROP}pp drop",
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 78,
        "",
        "One BiLSTM pruner per model; global threshold sweep on continuous scores.",
        "Question: do absolute survivor counts CONVERGE (absolute task floor) or",
        "SCALE with starting width (network-relative)?",
        "",
        f"{'Model':>20} | {'Total':>6} | {'Baseline':>8} | {'Kept@2pp':>9} | {'% Pruned':>9}",
        "-" * 78,
    ]
    for r in rows:
        lines.append(f"{r['name']:>20} | {r['n_total']:>6} | {r['baseline']*100:7.2f}% | "
                     f"{r['kept']:>9} | {r['frac_pruned']*100:8.1f}%")
    lines += [
        "-" * 78,
        "",
        "READ: kept@2pp similar across all 4 → ABSOLUTE task floor (minimal useful",
        "      subnet is task-determined; prune% is just starting distance above it).",
        "      kept@2pp grows with total → network-relative.",
        "=" * 78,
    ]
    with open(f"{OUT_DIR}/summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    if final:
        print("\n".join(lines))


if __name__ == "__main__":
    main()
