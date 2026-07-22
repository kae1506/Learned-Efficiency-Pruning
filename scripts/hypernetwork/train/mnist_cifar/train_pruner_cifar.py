"""
CIFAR-big MLP pruner: λ sweep.

Goal
----
Train a BiLSTM weight-conditioned pruner on the FROZEN CIFAR_big base model
(experiments/checkpoints/cifar_big.pt, 87.39% test acc, ~10.4M params).
Prune ONLY the MLP head (fc1/fc2/fc3) — fc4 (output) is left untouched.

The fc head holds 9.04M of the 10.4M params (~87%), so MLP-only pruning
already targets most of the weight memory. Convs stay frozen and intact.

Why soft λ (vs hard top-K): F12 in crisp-findings — under hard global top-K,
layers with very different counts (1024 / 512 / 256) get starved.
Soft λ gives global subset-selection gradient and lets each layer settle
on its own ratio.

Design knobs (confirmed with user, 2026-06):
- Layers pruned : fc1 (1024×8192), fc2 (512×1024), fc3 (256×512)
- Pruner       : BiLSTM, embed_dim=64, lstm_hidden=128
- λ sweep      : {0.01, 0.03, 0.1, 0.3}
- Train length : 5 epochs over CIFAR train (batch 256)
- Mask applies : output-side of each fc layer (multiplicative on rows)
- Device       : MPS

Run from project root:
  venv/bin/python scripts/hypernetwork/train_pruner_cifar.py
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import time
import argparse

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import v2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(".")
from src.pruners.bilstm import Pruner
from scripts.base.train_cifar import CIFARNetBig, CIFAR_MEAN, CIFAR_STD


CKPT_PATH = "experiments/checkpoints/cifar_big.pt"
OUT_ROOT  = "experiments/latest/hypernetwork/cifar_lambda_sweep"


# ─────────────────────────────────────────────────────────────────────────────
# Base-model loading + frozen forward pass with FC gates applied
# ─────────────────────────────────────────────────────────────────────────────

def load_cifar_big(device) -> CIFARNetBig:
    """Load the frozen CIFAR_big base model. BN runs in eval mode (running stats)."""
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model = CIFARNetBig(output_dim=ckpt["config"]["output_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def get_fc_weights(model: CIFARNetBig) -> list[torch.Tensor]:
    """Detached weight matrices for the pruner's three FC targets (output layer excluded)."""
    return [model.fc1.weight.detach(),
            model.fc2.weight.detach(),
            model.fc3.weight.detach()]


def masked_forward(model: CIFARNetBig, gates: list[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass through CIFAR_big with per-neuron gates applied to fc1/fc2/fc3 ROWS.

    Conv blocks (frozen) run unchanged. For each gated fc layer we scale its
    weight rows + bias by `gate` (binary STE in forward, real-valued in backward),
    so gradients flow to the pruner. fc4 is unmasked.
    """
    # ── Convs (frozen) ────────────────────────────────────────────────────────
    h = model.pool(F.relu(model.bn1(model.conv1(x))))
    h = model.pool(F.relu(model.bn2(model.conv2(h))))
    h = model.pool(F.relu(model.bn3(model.conv3(h))))
    h = h.view(h.size(0), -1)                          # [B, 8192]

    # ── fc1/fc2/fc3 with row-wise gating ──────────────────────────────────────
    for linear, gate in [(model.fc1, gates[0]),
                         (model.fc2, gates[1]),
                         (model.fc3, gates[2])]:
        w = linear.weight.detach() * gate.unsqueeze(1)  # [out, in]
        b = linear.bias.detach()   * gate               # [out]
        h = F.relu(F.linear(h, w, b))

    # ── fc4 (output) — unmasked ───────────────────────────────────────────────
    return F.linear(h, model.fc4.weight.detach(), model.fc4.bias.detach())


# ─────────────────────────────────────────────────────────────────────────────
# Pruner training step (CE_pruned − CE_orig) + λ · mean(gates)
# ─────────────────────────────────────────────────────────────────────────────

def pruner_step(pruner, base_model, optimizer, x, y, sparsity_weight):
    optimizer.zero_grad()

    fc_weights = get_fc_weights(base_model)
    gates = pruner(fc_weights)

    with torch.no_grad():
        orig_logits = base_model(x)
        ce_orig  = F.cross_entropy(orig_logits, y)
        orig_acc = (orig_logits.argmax(1) == y).float().mean().item()

    pruned_logits = masked_forward(base_model, gates, x)
    ce_pruned = F.cross_entropy(pruned_logits, y)

    with torch.no_grad():
        pruned_acc = (pruned_logits.argmax(1) == y).float().mean().item()

    # Sparsity term: per-layer keep fraction, averaged across layers.
    # Lower → more pruned. Soft penalty → global subset-selection gradient.
    sparsity_loss = sum(g.mean() for g in gates) / len(gates)

    loss = (ce_pruned - ce_orig) + sparsity_weight * sparsity_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(pruner.parameters(), max_norm=1.0)
    optimizer.step()

    avg_gate = sum(g.mean().item() for g in gates) / len(gates)
    return {"loss": loss.item(), "ce_orig": ce_orig.item(), "ce_pruned": ce_pruned.item(),
            "orig_acc": orig_acc, "pruned_acc": pruned_acc,
            "acc_drop": orig_acc - pruned_acc, "avg_gate": avg_gate}


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def get_loaders(batch_size: int, num_workers: int = 2):
    """Plain CIFAR-10 train/test loaders (no augmentation — pruner sees clean data)."""
    tf = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    train = torchvision.datasets.CIFAR10(root="./data", train=True,  download=True, transform=tf)
    test  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=tf)
    tl = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    vl = torch.utils.data.DataLoader(test,  batch_size=256,        shuffle=False, num_workers=num_workers)
    return tl, vl


# ─────────────────────────────────────────────────────────────────────────────
# Full-test-set evaluation with final gates applied
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_with_gates(base_model, gates, test_loader, device) -> tuple[float, float]:
    """Returns (original_test_acc, pruned_test_acc) over the full test set."""
    correct_orig = correct_pruned = total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        orig_logits   = base_model(x)
        pruned_logits = masked_forward(base_model, gates, x)
        correct_orig   += (orig_logits.argmax(1)   == y).sum().item()
        correct_pruned += (pruned_logits.argmax(1) == y).sum().item()
        total          += y.size(0)
    return correct_orig / total, correct_pruned / total


# ─────────────────────────────────────────────────────────────────────────────
# Plotting + summary
# ─────────────────────────────────────────────────────────────────────────────

def _smooth(values, window=50):
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo:i + 1]) / (i - lo + 1))
    return out


def plot_one_run(history, save_path, title):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    steps = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    axes[0].plot(steps, history["loss"], alpha=0.2, color="steelblue")
    axes[0].plot(steps, _smooth(history["loss"]), color="steelblue", lw=2)
    axes[0].axhline(0, color="gray", ls="--", lw=0.8)
    axes[0].set_title("Pruner loss  (CE_pruned − CE_orig + λ·keep)")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("loss"); axes[0].grid(alpha=0.3)

    op = [a*100 for a in history["orig_acc"]]
    pp = [a*100 for a in history["pruned_acc"]]
    axes[1].plot(steps, op, alpha=0.15, color="steelblue")
    axes[1].plot(steps, pp, alpha=0.15, color="tomato")
    axes[1].plot(steps, _smooth(op), color="steelblue", lw=2, label="orig")
    axes[1].plot(steps, _smooth(pp), color="tomato",    lw=2, label="pruned")
    axes[1].set_title("mini-batch accuracy")
    axes[1].set_xlabel("step"); axes[1].set_ylabel("acc (%)"); axes[1].grid(alpha=0.3); axes[1].legend()

    pct = [(1-g)*100 for g in history["avg_gate"]]
    axes[2].plot(steps, pct, alpha=0.2, color="darkorange")
    axes[2].plot(steps, _smooth(pct), color="darkorange", lw=2)
    axes[2].set_title("% neurons pruned (avg across fc1/fc2/fc3)")
    axes[2].set_xlabel("step"); axes[2].set_ylabel("% pruned"); axes[2].set_ylim(0, 100); axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(results, save_path):
    """Final-pruning vs final-test-acc trade-off across λ values."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    pct_pruned = [r["pct_pruned"] for r in results]
    pruned_acc = [r["pruned_test_acc"]*100 for r in results]
    orig_acc   = results[0]["orig_test_acc"]*100
    labels     = [f"λ={r['lambda']}" for r in results]

    ax.scatter(pct_pruned, pruned_acc, s=120, c="tomato", zorder=3)
    for x, y, lab in zip(pct_pruned, pruned_acc, labels):
        ax.annotate(lab, (x, y), xytext=(8, 4), textcoords="offset points", fontsize=10)
    ax.axhline(orig_acc, color="steelblue", ls="--", lw=1.2,
               label=f"unpruned test acc = {orig_acc:.2f}%")
    ax.set_xlabel("% MLP neurons pruned (fc1+fc2+fc3 avg)")
    ax.set_ylabel("pruned test accuracy (%)")
    ax.set_title("CIFAR_big MLP pruning: λ-sweep trade-off", fontweight="bold")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_run_summary(path, lam, layer_shapes, history, per_layer_kept, orig_test, pruned_test, total_time):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    final_gate = history["avg_gate"][-1]
    pct_pruned = (1 - final_gate) * 100
    lines = [
        f"CIFAR_big MLP pruner — λ = {lam}",
        f"layers pruned : {layer_shapes}",
        f"steps         : {len(history['loss'])}",
        f"wall time     : {total_time:.1f}s",
        "-" * 56,
        f"final avg keep gate           : {final_gate:.4f}",
        f"final % MLP neurons pruned    : {pct_pruned:.2f}%",
        f"per-layer kept (fc1/fc2/fc3)  : {per_layer_kept}",
        "-" * 56,
        f"FULL test set (10000 imgs):",
        f"  original (unpruned)  acc    : {orig_test*100:.2f}%",
        f"  pruned               acc    : {pruned_test*100:.2f}%",
        f"  drop                        : {(orig_test - pruned_test)*100:+.2f}pp",
    ]
    with open(path, "w") as f: f.write("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Per-λ training run
# ─────────────────────────────────────────────────────────────────────────────

def train_one(lam, base_model, train_loader, test_loader, args, device):
    layer_shapes = [(w.shape[0], w.shape[1]) for w in get_fc_weights(base_model)]
    pruner = Pruner(layer_shapes, embed_dim=args.embed_dim, lstm_hidden=args.lstm_hidden).to(device)
    opt = torch.optim.Adam(pruner.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in pruner.parameters())
    print(f"\n── λ = {lam} ── pruner params: {n_params:,} "
          f"(embed_dim={args.embed_dim}, lstm_hidden={args.lstm_hidden})", flush=True)

    history = {k: [] for k in ["loss", "orig_acc", "pruned_acc", "acc_drop", "avg_gate"]}
    t0 = time.time()
    step = 0
    for ep in range(1, args.epochs + 1):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            m = pruner_step(pruner, base_model, opt, x, y, lam)
            for k in history: history[k].append(m[k])
            step += 1
            if step % 50 == 0:
                print(f"  step {step:>4} ep{ep} | loss {m['loss']:+.3f} | "
                      f"orig {m['orig_acc']:.3f} | pruned {m['pruned_acc']:.3f} | "
                      f"keep {m['avg_gate']:.3f}", flush=True)
    total_time = time.time() - t0

    # Final gates → eval on full test set
    pruner.eval()
    with torch.no_grad():
        final_gates = pruner(get_fc_weights(base_model))
    per_layer_kept = [int(g.sum().item()) for g in final_gates]
    layer_sizes    = [g.numel() for g in final_gates]
    orig_test, pruned_test = evaluate_with_gates(base_model, final_gates, test_loader, device)

    final_gate = history["avg_gate"][-1]
    pct_pruned = (1 - final_gate) * 100
    print(f"  → final keep {final_gate:.3f}  pruned {pct_pruned:.2f}%  | "
          f"per-layer kept {per_layer_kept}/{layer_sizes}  | "
          f"orig→pruned test acc {orig_test*100:.2f}→{pruned_test*100:.2f}%", flush=True)

    return {
        "lambda": lam, "history": history,
        "per_layer_kept": list(zip(per_layer_kept, layer_sizes)),
        "pct_pruned": pct_pruned, "orig_test_acc": orig_test,
        "pruned_test_acc": pruned_test, "total_time": total_time,
        "layer_shapes": layer_shapes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas",     type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3])
    ap.add_argument("--epochs",      type=int,   default=5)
    ap.add_argument("--batch_size",  type=int,   default=256)
    ap.add_argument("--embed_dim",   type=int,   default=64)
    ap.add_argument("--lstm_hidden", type=int,   default=128)
    ap.add_argument("--lr",          type=float, default=0.001)
    ap.add_argument("--device",      type=str,   default="mps")
    args = ap.parse_args()

    # ── Device ─────────────────────────────────────────────────────────────────
    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    base_model = load_cifar_big(device)
    n_base = sum(p.numel() for p in base_model.parameters())
    print(f"Loaded CIFAR_big — {n_base:,} params, frozen.")

    train_loader, test_loader = get_loaders(args.batch_size)

    os.makedirs(OUT_ROOT, exist_ok=True)
    results = []
    for lam in args.lambdas:
        res = train_one(lam, base_model, train_loader, test_loader, args, device)
        results.append(res)

        run_dir = os.path.join(OUT_ROOT, f"lambda_{lam}")
        plot_one_run(res["history"], os.path.join(run_dir, "plot.png"),
                     title=f"CIFAR_big MLP pruner — λ={lam} — "
                           f"{res['pct_pruned']:.1f}% pruned, "
                           f"test {res['pruned_test_acc']*100:.2f}%")
        write_run_summary(os.path.join(run_dir, "summary.txt"), lam,
                          res["layer_shapes"], res["history"],
                          res["per_layer_kept"], res["orig_test_acc"],
                          res["pruned_test_acc"], res["total_time"])

    # ── Comparison plot + sweep summary ────────────────────────────────────────
    plot_comparison(results, os.path.join(OUT_ROOT, "comparison.png"))

    lines = ["CIFAR_big MLP pruner — λ sweep",
             f"layers : fc1(1024×8192) fc2(512×1024) fc3(256×512); fc4 untouched",
             f"pruner : BiLSTM embed_dim={args.embed_dim} lstm_hidden={args.lstm_hidden}",
             f"train  : {args.epochs} epochs × batch {args.batch_size} on {device}",
             "-" * 72,
             f"{'lambda':>8} | {'% pruned':>10} | {'orig acc':>9} | {'pruned acc':>11} | "
             f"{'drop':>7} | {'fc1':>10} {'fc2':>10} {'fc3':>10}",
             "-" * 72]
    for r in results:
        kept = r["per_layer_kept"]
        lines.append(
            f"{r['lambda']:>8} | {r['pct_pruned']:>9.2f}% | "
            f"{r['orig_test_acc']*100:>8.2f}% | {r['pruned_test_acc']*100:>10.2f}% | "
            f"{(r['orig_test_acc']-r['pruned_test_acc'])*100:>+6.2f}pp | "
            f"{kept[0][0]:>4}/{kept[0][1]:<5} {kept[1][0]:>4}/{kept[1][1]:<5} {kept[2][0]:>4}/{kept[2][1]:<5}")
    with open(os.path.join(OUT_ROOT, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nResults → {OUT_ROOT}/")


if __name__ == "__main__":
    main()
