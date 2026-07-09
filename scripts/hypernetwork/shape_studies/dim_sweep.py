"""
Hyperparameter sweep: hidden dimension vs % nodes pruned by BiLSTM pruner.

For each hidden dimension d in DIMS:
  1. Train a fresh MLP  784 -> [d, d] -> 10  for TRAIN_EPOCHS epochs.
  2. Run the BiLSTM pruner for PRUNER_STEPS steps.
  3. Record final avg_gate  ->  % pruned = (1 - avg_gate) * 100.

Output: experiments/latest/hypernetwork/dim_sweep/plot.png  +  summary.txt
Run from project root: venv/bin/python scripts/dim_sweep.py
"""

import os, sys, copy
sys.path.append(".")

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.model import MLP
from src.dataset import get_mnist_loaders
from src.train import train_epoch, evaluate
from src.pruners.bilstm import Pruner
from src.prune_train import pruner_step, get_hidden_weights

# ── Config ────────────────────────────────────────────────────────────────────
DIMS          = [32, 64, 128, 256, 512, 1024, 2048]
TRAIN_EPOCHS  = 5        # fast enough; accuracy stabilises by epoch 4
PRUNER_STEPS  = 600
PRUNER_LR     = 0.001
SPARSITY_W    = 0.05
SAMPLES       = 64
BATCH_SIZE    = 128
DATA_DIR      = "./data"
OUT_DIR       = "experiments/latest/hypernetwork/dim_sweep"

os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

train_loader, test_loader = get_mnist_loaders(
    data_dir=DATA_DIR, batch_size=BATCH_SIZE, num_workers=0
)

# ── Sweep ─────────────────────────────────────────────────────────────────────
results = []   # (dim, test_acc, pct_pruned, final_gate)

for dim in DIMS:
    print(f"{'='*55}")
    print(f"  dim = {dim}  (784 -> [{dim}, {dim}] -> 10)")
    print(f"{'='*55}")

    # ── 1. Train MNIST model ─────────────────────────────────────────────────
    model = MLP(input_dim=784, hidden_dims=[dim, dim],
                output_dim=10, dropout=0.1).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, TRAIN_EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, opt, device)
        te_loss, te_acc = evaluate(model, test_loader, device)
        print(f"  epoch {epoch}/{TRAIN_EPOCHS}  "
              f"train_acc={tr_acc:.4f}  test_acc={te_acc:.4f}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ── 2. Run BiLSTM pruner ─────────────────────────────────────────────────
    hidden_weights = get_hidden_weights(model)
    layer_shapes   = [(w.shape[0], w.shape[1]) for w in hidden_weights]
    pruner   = Pruner(layer_shapes).to(device)
    p_opt    = torch.optim.Adam(pruner.parameters(), lr=PRUNER_LR)
    data_it  = iter(train_loader)

    gate_history = []
    for step in range(1, PRUNER_STEPS + 1):
        try:
            x, y = next(data_it)
        except StopIteration:
            data_it = iter(train_loader)
            x, y = next(data_it)

        x, y = x[:SAMPLES].to(device), y[:SAMPLES].to(device)
        metrics = pruner_step(pruner, model, p_opt, x, y, SPARSITY_W)
        gate_history.append(metrics["avg_gate"])

        if step % 100 == 0:
            print(f"  [pruner] step {step:>4}  "
                  f"avg_gate={metrics['avg_gate']:.4f}  "
                  f"pruned_acc={metrics['pruned_acc']:.4f}")

    final_gate = gate_history[-1]
    pct_pruned = (1 - final_gate) * 100
    print(f"  -> final: {pct_pruned:.1f}% pruned  "
          f"(avg_gate={final_gate:.4f})  test_acc={te_acc:.4f}\n")

    results.append((dim, te_acc * 100, pct_pruned, final_gate))

# ── Plot ──────────────────────────────────────────────────────────────────────
dims       = [r[0] for r in results]
test_accs  = [r[1] for r in results]
pct_pruned = [r[2] for r in results]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Hidden Dimension Sweep — MNIST BiLSTM Pruner", fontsize=13, fontweight="bold")

# Left: % pruned vs dim
ax1.plot(dims, pct_pruned, "o-", color="#e67e22", linewidth=2, markersize=7, markerfacecolor="white", markeredgewidth=2)
for d, p in zip(dims, pct_pruned):
    ax1.annotate(f"{p:.1f}%", (d, p), textcoords="offset points",
                 xytext=(0, 9), ha="center", fontsize=8.5, color="#884400")
ax1.set_xlabel("Hidden dimension (both layers)", fontsize=11)
ax1.set_ylabel("Nodes pruned (%)", fontsize=11)
ax1.set_title("% Nodes Pruned vs Dimension", fontsize=11)
ax1.set_xscale("log", base=2)
ax1.set_xticks(dims)
ax1.set_xticklabels([str(d) for d in dims])
ax1.set_ylim(0, 100)
ax1.grid(True, alpha=0.3)

# Right: test accuracy vs dim
ax2.plot(dims, test_accs, "s-", color="#2980b9", linewidth=2, markersize=7, markerfacecolor="white", markeredgewidth=2)
for d, a in zip(dims, test_accs):
    ax2.annotate(f"{a:.1f}%", (d, a), textcoords="offset points",
                 xytext=(0, 9), ha="center", fontsize=8.5, color="#1a5276")
ax2.set_xlabel("Hidden dimension (both layers)", fontsize=11)
ax2.set_ylabel("Test accuracy (%)", fontsize=11)
ax2.set_title("Pre-Pruning Test Accuracy vs Dimension", fontsize=11)
ax2.set_xscale("log", base=2)
ax2.set_xticks(dims)
ax2.set_xticklabels([str(d) for d in dims])
ax2.set_ylim(85, 100)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
plot_path = f"{OUT_DIR}/plot.png"
plt.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"Saved plot to {plot_path}")

# ── Text summary ──────────────────────────────────────────────────────────────
summary_path = f"{OUT_DIR}/summary.txt"
lines = [
    "=" * 55,
    "DIMENSION SWEEP — BiLSTM PRUNER",
    f"Pruner steps: {PRUNER_STEPS}  |  sparsity_weight: {SPARSITY_W}",
    f"Train epochs: {TRAIN_EPOCHS}",
    "=" * 55,
    f"{'Dim':>6}  {'Test Acc':>10}  {'% Pruned':>10}  {'Avg Gate':>10}",
    "-" * 55,
]
for dim, acc, pct, gate in results:
    lines.append(f"{dim:>6}  {acc:>9.2f}%  {pct:>9.1f}%  {gate:>10.4f}")
lines += ["=" * 55]

with open(summary_path, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Saved summary to {summary_path}")
