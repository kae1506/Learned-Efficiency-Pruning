"""
Neuron-pruning vs WEIGHT/compute savings, plus the iso-accuracy survivor curves.

Two outputs:
  (1) curves.png — accuracy-drop vs surviving-neurons line plot (one line per
      architecture), parsed from the iso_accuracy_retrain run.log. Same style as
      iso_accuracy_sweep's plot but for the rigorous retrained data.
  (2) weights.png + summary — the real question: a hidden neuron carries a
      different number of WEIGHTS depending on its layer's fan-in/fan-out, so
      neuron-pruning % ≠ weight-pruning %. We retrain each model's best-≤2pp
      pruner, read the per-layer surviving counts, and compute the ACTUAL pruned
      parameter count (and hence FLOP/compute savings).

Why weights ≠ neurons:
  - input matrix (784 × h1) and output matrix (h_last × 10): only ONE side is
    pruned → weight savings LINEAR in neuron %.
  - hidden→hidden matrices (only deep/medium have them): BOTH sides pruned →
    weight savings QUADRATIC (≈ 1 - f_keep^2). Deep gets this discount; wide
    (single hidden layer, no hidden→hidden matrix) does not.

Pruned param count = sum over consecutive layer pairs of (survivors_in × survivors_out),
with input dim = 784 (fixed) and output dim = 10 (fixed). For an MLP, inference
MACs ≈ this same sum, so weight-savings % == compute-savings %.

Output: experiments/latest/hypernetwork/iso_accuracy_retrain/{curves.png, weights.png, weight_summary.txt}
Run from project root:
    venv/bin/python scripts/hypernetwork/weight_analysis.py
"""

import os
import re
import sys
import random
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


# Each model: (display name, checkpoint, best sw found at ≤2pp in the retrain run)
MODELS = [
    ("narrow [205,205]",  "experiments/checkpoints/mnist_narrow205x2.pt", 0.08),
    ("medium [1024,1024]","experiments/checkpoints/mnist_model.pt",        0.30),
    ("deep [512x4]",      "experiments/checkpoints/mnist_deep4x512.pt",     0.80),
    ("wide [2048]",       "experiments/checkpoints/mnist_wide2048.pt",      0.10),
]
COLORS = {"narrow [205,205]": "#e74c3c", "medium [1024,1024]": "#2980b9",
          "deep [512x4]": "#e67e22", "wide [2048]": "#9b59b6"}

RUN_LOG       = "experiments/latest/hypernetwork/iso_accuracy_retrain/run.log"
OUT_DIR       = "experiments/latest/hypernetwork/iso_accuracy_retrain"
INPUT_DIM     = 784
OUTPUT_DIM    = 10
PRUNER_STEPS  = 1000
PRUNER_LR     = 1e-3
SAMPLES       = 64
SEED          = 0


def set_seed(s):
    torch.manual_seed(s); np.random.seed(s); random.seed(s)


def load_frozen(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    m = MLP(**ckpt["config"]).to(device)
    m.load_state_dict(ckpt["state_dict"]); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


# ── (1) parse run.log → survivor/drop curves per model ────────────────────────
def parse_curves():
    """
    Returns {model_name: [(survivors, drop_pp), ...]} by reading the per-sw lines
    of the retrain run.log. Model boundaries are the '=== name (sw grid ...' lines.
    """
    curves = {}
    cur = None
    line_re = re.compile(r"survivors=\s*(\d+)/\d+\s+pruned.*")  # marker
    drop_re = re.compile(r"drop=\s*(-?\d+\.\d+)pp\s+survivors=\s*(\d+)/")
    with open(RUN_LOG) as f:
        for line in f:
            if line.startswith("=== "):
                # e.g. "=== narrow [205,205]  (sw grid [...]) ==="
                name = line.split("===")[1].split("  (sw grid")[0].strip()
                cur = name
                curves[cur] = []
            elif cur and "sw=" in line and "drop=" in line and "best" not in line:
                m = drop_re.search(line)
                if m:
                    drop = float(m.group(1)); surv = int(m.group(2))
                    curves[cur].append((surv, drop))
    return curves


def plot_curves(curves):
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, pts in curves.items():
        pts = sorted(pts)                       # by survivors ascending
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, "o-", ms=5, lw=1.5, color=COLORS.get(name), label=name)
    ax.axhline(2.0, color="k", ls="--", alpha=0.6, label="2.0pp budget")
    ax.set_xlabel("Surviving neurons (absolute count)")
    ax.set_ylabel("Full-test accuracy drop (pp)")
    ax.set_title("Iso-accuracy survivor curves (retrained per sw, 2 seeds)",
                 fontweight="bold")
    ax.set_ylim(-1, 8)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); plt.savefig(f"{OUT_DIR}/curves.png", dpi=150); plt.close(fig)
    print(f"Saved curves to {OUT_DIR}/curves.png")


# ── (2) per-layer survivors → actual weight / compute savings ─────────────────
def layer_dims(model):
    """Full hidden-layer widths (out_features of every Linear except the last)."""
    lins = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    return [L.out_features for L in lins[:-1]]


def param_count(widths):
    """
    Weight count of an MLP with given hidden `widths`, fixed INPUT_DIM/OUTPUT_DIM.
    = sum of consecutive (in × out) over [input, *hidden, output].
    Equals inference MACs for an MLP, so this is also the compute proxy.
    """
    dims = [INPUT_DIM] + list(widths) + [OUTPUT_DIM]
    return sum(dims[i] * dims[i + 1] for i in range(len(dims) - 1))


def analyze_weights(device):
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    train_loader, test_loader = get_mnist_loaders(**cfg["data"])

    rows = []
    for name, ckpt, sw in MODELS:
        model = load_frozen(ckpt, device)
        full_widths = layer_dims(model)
        orig_params = param_count(full_widths)

        # retrain the best-≤2pp pruner and read the per-layer surviving counts
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
            pruner_step(pruner, model, opt, x, y, sw)
        pruner.eval()
        with torch.no_grad():
            gates = pruner(get_hidden_weights(model))

        surv_widths = [int(g.sum().item()) for g in gates]   # survivors per layer
        pruned_params = param_count(surv_widths)
        full_mask = [torch.ones(w.shape[0], dtype=torch.bool, device=device)
                     for w in get_hidden_weights(model)]
        baseline_acc = evaluate_with_gates(model, full_mask, test_loader, device)
        acc = evaluate_with_gates(model, gates, test_loader, device)

        n_neurons = sum(full_widths)
        n_surv = sum(surv_widths)
        rows.append({
            "name": name, "sw": sw,
            "drop": (baseline_acc - acc) * 100,
            "neuron_pruned": 1 - n_surv / n_neurons,
            "weight_pruned": 1 - pruned_params / orig_params,
            "orig_params": orig_params, "pruned_params": pruned_params,
            "surv_widths": surv_widths, "full_widths": full_widths,
        })
        print(f"  {name:>20}: neurons -{(1-n_surv/n_neurons)*100:5.1f}%  "
              f"weights -{(1-pruned_params/orig_params)*100:5.1f}%  "
              f"final={pruned_params/1e3:6.1f}K weights  (drop {rows[-1]['drop']:.2f}pp, "
              f"layers {surv_widths})", flush=True)
    return rows


def plot_weights(rows):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    names = [r["name"] for r in rows]
    xs = np.arange(len(names))
    cols = [COLORS[r["name"]] for r in rows]

    # left: neuron-pruned % vs weight-pruned % (grouped bars)
    w = 0.38
    axes[0].bar(xs - w/2, [r["neuron_pruned"]*100 for r in rows], w,
                color="#95a5a6", label="neurons pruned %")
    axes[0].bar(xs + w/2, [r["weight_pruned"]*100 for r in rows], w,
                color=cols, label="weights pruned %")
    axes[0].set_xticks(xs); axes[0].set_xticklabels(names, fontsize=8, rotation=10)
    axes[0].set_ylabel("% removed")
    axes[0].set_title("Neuron-pruning % vs WEIGHT-pruning %", fontweight="bold")
    axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)

    # right: absolute weight count remaining (the real compute floor)
    axes[1].bar(xs, [r["orig_params"]/1e3 for r in rows], color="#dddddd",
                label="original weights")
    axes[1].bar(xs, [r["pruned_params"]/1e3 for r in rows], color=cols,
                label="weights @ ≤2pp")
    for x, r in zip(xs, rows):
        axes[1].text(x, r["pruned_params"]/1e3 + 15, f"{r['pruned_params']/1e3:.0f}K",
                     ha="center", fontsize=9, fontweight="bold")
    axes[1].set_xticks(xs); axes[1].set_xticklabels(names, fontsize=8, rotation=10)
    axes[1].set_ylabel("Weights (thousands)")
    axes[1].set_title("Absolute weights remaining (= compute floor)", fontweight="bold")
    axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout(); plt.savefig(f"{OUT_DIR}/weights.png", dpi=150); plt.close(fig)
    print(f"Saved weights plot to {OUT_DIR}/weights.png")


def write_weight_summary(rows):
    lines = [
        "=" * 86,
        "NEURON-PRUNING vs WEIGHT/COMPUTE SAVINGS — at the ≤2pp operating point",
        "=" * 86,
        "",
        "Pruned weights = inference MACs for an MLP, so weight% == compute%.",
        "Input(784×h1) & output(h_last×10) matrices: 1-sided prune → LINEAR savings.",
        "Hidden→hidden matrices (deep/medium only): 2-sided → QUADRATIC savings.",
        "",
        f"{'Model':>20} | {'sw':>5} | {'Neurons -%':>10} | {'Weights -%':>10} | "
        f"{'Orig W':>9} | {'Final W':>9} | survivors/layer",
        "-" * 86,
    ]
    for r in rows:
        lines.append(
            f"{r['name']:>20} | {r['sw']:>5} | {r['neuron_pruned']*100:9.1f}% | "
            f"{r['weight_pruned']*100:9.1f}% | {r['orig_params']/1e3:7.0f}K | "
            f"{r['pruned_params']/1e3:7.0f}K | {r['surv_widths']}")
    lines += [
        "-" * 86,
        "",
        "KEY: compare 'Final W' across the three 2048-neuron nets (wide/medium/deep).",
        "If they converge → a WEIGHT/COMPUTE floor exists even though the NEURON",
        "floor does not (neuron survivors were 314/490/814). Weights are the better",
        "lens for 'actual compute saved'.",
        "=" * 86,
    ]
    with open(f"{OUT_DIR}/weight_summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    print("=== (1) survivor/drop curves from run.log ===")
    curves = parse_curves()
    plot_curves(curves)
    print("\n=== (2) weight / compute savings (retrain best-sw pruner per model) ===")
    rows = analyze_weights(device)
    plot_weights(rows)
    write_weight_summary(rows)


if __name__ == "__main__":
    main()
