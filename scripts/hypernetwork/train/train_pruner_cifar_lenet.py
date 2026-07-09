"""
CIFAR-LeNet (small, 63K-param base model) MLP pruner: single-seed λ sweep.

Goal
----
Pendant experiment to train_pruner_cifar.py — same protocol but on the small
LeNet-style CIFAR base (`experiments/checkpoints/cifar_cnn.pt`, ~64.8% test acc,
63,106 params). Lets us compare prunability across a 10.4M-param big model and
a 63K-param small model trained on the same dataset.

Base model structure (from scripts/base/train_cifar.py, class CIFARNet):
  conv1  Conv2d(3, 6, 5)   → pool         32x32x3  → 14x14x6
  conv2  Conv2d(6, 16, 5)  → pool         14x14x6  → 5x5x16  (flatten = 400)
  fc1    Linear(400, 128)     ← prunable  (51,328 params, 81% of total)
  fc2    Linear(128, 64)      ← prunable  (8,256 params, 13% of total)
  fc3    Linear(64, 10)       output      (650 params)

So MLP-only pruning (fc1 + fc2 = 94% of weights) attacks essentially the whole
model. Conv blocks together hold only ~2.9K params and are left untouched.

Design choices (mirror the big-model run for direct comparability):
- λ sweep      : {0.01, 0.03, 0.1, 0.3}   (initial wide sweep range)
- Pruner       : BiLSTM, embed_dim=64, lstm_hidden=128
- Train length : 5 epochs over CIFAR train, batch 256
- Device       : CPU — LeNet is too small for MPS; the base trained on CPU originally
- Seed         : single (this is the wide first-pass; can multi-seed later if interesting)

Run from project root:
  venv/bin/python scripts/hypernetwork/train_pruner_cifar_lenet.py
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import time
import argparse

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
from scripts.base.train_cifar import CIFARNet, CIFAR_MEAN, CIFAR_STD


CKPT_PATH = "experiments/checkpoints/cifar_cnn.pt"
DEFAULT_OUT_ROOT = "experiments/latest/hypernetwork/cifar_lenet_lambda_sweep"


# ─────────────────────────────────────────────────────────────────────────────
# Base-model loading + frozen forward pass with FC gates applied
# ─────────────────────────────────────────────────────────────────────────────

def load_cifar_lenet(device) -> CIFARNet:
    """Load the frozen LeNet CIFAR base. No BatchNorm here — pure Conv2d/Linear."""
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model = CIFARNet(output_dim=ckpt["config"]["output_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def get_fc_weights(model: CIFARNet) -> list[torch.Tensor]:
    """Detached fc1/fc2 weight matrices for the pruner (fc3 = output, excluded)."""
    return [model.fc1.weight.detach(),
            model.fc2.weight.detach()]


def masked_forward(model: CIFARNet, gates: list[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """
    LeNet forward with per-neuron gates on fc1, fc2 ROWS. fc3 unmasked.
    No BN in this architecture — simpler than the big model.
    """
    # ── Convs (frozen) ────────────────────────────────────────────────────────
    h = model.pool(F.relu(model.conv1(x)))     # 32x32x3 → 14x14x6
    h = model.pool(F.relu(model.conv2(h)))     # 14x14x6 → 5x5x16
    h = h.view(h.size(0), -1)                  # flatten → 400

    # ── fc1, fc2 with row-wise gating ─────────────────────────────────────────
    for linear, gate in [(model.fc1, gates[0]),
                         (model.fc2, gates[1])]:
        w = linear.weight.detach() * gate.unsqueeze(1)
        b = linear.bias.detach()   * gate
        h = F.relu(F.linear(h, w, b))

    # ── fc3 (output) — unmasked ───────────────────────────────────────────────
    return F.linear(h, model.fc3.weight.detach(), model.fc3.bias.detach())


# ─────────────────────────────────────────────────────────────────────────────
# Pruner training step
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

    sparsity_loss = sum(g.mean() for g in gates) / len(gates)
    loss = (ce_pruned - ce_orig) + sparsity_weight * sparsity_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(pruner.parameters(), max_norm=1.0)
    optimizer.step()

    per_layer_keep = [g.mean().item() for g in gates]
    avg_gate = sum(per_layer_keep) / len(per_layer_keep)
    return {"loss": loss.item(), "orig_acc": orig_acc, "pruned_acc": pruned_acc,
            "acc_drop": orig_acc - pruned_acc, "avg_gate": avg_gate,
            "per_layer_keep": per_layer_keep}


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def get_loaders(batch_size: int, num_workers: int = 2):
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
    """3 panels: loss, mini-batch acc, per-layer % pruned (fc1 + fc2)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    steps = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    axes[0].plot(steps, history["loss"], alpha=0.2, color="steelblue")
    axes[0].plot(steps, _smooth(history["loss"]), color="steelblue", lw=2)
    axes[0].axhline(0, color="gray", ls="--", lw=0.8)
    axes[0].set_title("Pruner loss"); axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss"); axes[0].grid(alpha=0.3)

    op = [a*100 for a in history["orig_acc"]]
    pp = [a*100 for a in history["pruned_acc"]]
    axes[1].plot(steps, op, alpha=0.15, color="steelblue")
    axes[1].plot(steps, pp, alpha=0.15, color="tomato")
    axes[1].plot(steps, _smooth(op), color="steelblue", lw=2, label="orig")
    axes[1].plot(steps, _smooth(pp), color="tomato",    lw=2, label="pruned")
    axes[1].set_title("mini-batch acc"); axes[1].set_xlabel("step")
    axes[1].set_ylabel("acc (%)"); axes[1].grid(alpha=0.3); axes[1].legend()

    colors = {"fc1": "#c0392b", "fc2": "#2980b9"}
    for i, name in enumerate(["fc1", "fc2"]):
        per = [(1 - k) * 100 for k in history["per_layer_keep"][i]]
        axes[2].plot(steps, per, alpha=0.2, color=colors[name])
        axes[2].plot(steps, _smooth(per), color=colors[name], lw=2, label=name)
    axes[2].set_title("per-layer % pruned"); axes[2].set_xlabel("step")
    axes[2].set_ylabel("% pruned"); axes[2].set_ylim(0, 100)
    axes[2].grid(alpha=0.3); axes[2].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(results, save_path):
    """Pareto: % pruned vs pruned test acc across λ values (single seed)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    pct_pruned = [r["pct_pruned"] for r in results]
    pruned_acc = [r["pruned_test_acc"]*100 for r in results]
    orig_acc   = results[0]["orig_test_acc"]*100

    ax.scatter(pct_pruned, pruned_acc, s=120, c="tomato", zorder=3)
    for x, y, r in zip(pct_pruned, pruned_acc, results):
        ax.annotate(f"λ={r['lambda']}", (x, y), xytext=(8, 4),
                    textcoords="offset points", fontsize=10)
    ax.axhline(orig_acc, color="steelblue", ls="--", lw=1.2,
               label=f"unpruned test acc = {orig_acc:.2f}%")
    ax.set_xlabel("% MLP neurons pruned (fc1+fc2 avg)")
    ax.set_ylabel("pruned test accuracy (%)")
    ax.set_title("CIFAR LeNet MLP pruning: λ-sweep trade-off",
                 fontweight="bold")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_run_summary(path, lam, layer_shapes, history, per_layer_kept,
                      orig_test, pruned_test, total_time):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    final_gate = history["avg_gate"][-1]
    pct_pruned = (1 - final_gate) * 100
    lines = [
        f"CIFAR LeNet MLP pruner — λ = {lam}",
        f"layers : {layer_shapes}",
        f"steps  : {len(history['loss'])}",
        f"time   : {total_time:.1f}s",
        "-" * 56,
        f"final avg keep gate          : {final_gate:.4f}",
        f"final % MLP neurons pruned   : {pct_pruned:.2f}%",
        f"per-layer kept (fc1/fc2)     : {per_layer_kept}",
        "-" * 56,
        f"FULL test set (10000 imgs):",
        f"  original (unpruned) acc    : {orig_test*100:.2f}%",
        f"  pruned              acc    : {pruned_test*100:.2f}%",
        f"  drop                       : {(orig_test - pruned_test)*100:+.2f}pp",
    ]
    with open(path, "w") as f: f.write("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Per-λ training run
# ─────────────────────────────────────────────────────────────────────────────

def train_one(lam, seed, base_model, train_loader, test_loader, args, device):
    """One (λ, seed) pruner training run. Re-seeds torch/np so seeds across the
    sweep give genuinely independent inits (the base model is shared and frozen
    so seeding only matters for the pruner's init + dataloader shuffle)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    layer_shapes = [(w.shape[0], w.shape[1]) for w in get_fc_weights(base_model)]
    pruner = Pruner(layer_shapes, embed_dim=args.embed_dim, lstm_hidden=args.lstm_hidden).to(device)
    opt = torch.optim.Adam(pruner.parameters(), lr=args.lr)

    n_pruner = sum(p.numel() for p in pruner.parameters())
    tag = f"λ={lam} seed={seed}"
    print(f"\n── {tag} ── pruner params: {n_pruner:,} "
          f"(embed_dim={args.embed_dim}, lstm_hidden={args.lstm_hidden})", flush=True)

    history = {"loss": [], "orig_acc": [], "pruned_acc": [], "acc_drop": [],
               "avg_gate": [], "per_layer_keep": [[], []]}
    t0 = time.time()
    step = 0
    for ep in range(1, args.epochs + 1):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            m = pruner_step(pruner, base_model, opt, x, y, lam)
            history["loss"].append(m["loss"])
            history["orig_acc"].append(m["orig_acc"])
            history["pruned_acc"].append(m["pruned_acc"])
            history["acc_drop"].append(m["acc_drop"])
            history["avg_gate"].append(m["avg_gate"])
            for i, k in enumerate(m["per_layer_keep"]):
                history["per_layer_keep"][i].append(k)
            step += 1
            if step % args.log_every == 0:
                fc1_p, fc2_p = [(1 - k) * 100 for k in m["per_layer_keep"]]
                print(f"  [{tag}] step {step:>4} ep{ep} | loss {m['loss']:+.3f} | "
                      f"orig {m['orig_acc']:.3f} pruned {m['pruned_acc']:.3f} | "
                      f"pruned% fc1={fc1_p:5.1f} fc2={fc2_p:5.1f} "
                      f"(avg={(fc1_p+fc2_p)/2:5.1f})", flush=True)
    total_time = time.time() - t0

    pruner.eval()
    with torch.no_grad():
        final_gates = pruner(get_fc_weights(base_model))
    per_layer_kept = [int(g.sum().item()) for g in final_gates]
    layer_sizes    = [g.numel() for g in final_gates]
    orig_test, pruned_test = evaluate_with_gates(base_model, final_gates, test_loader, device)

    final_gate = history["avg_gate"][-1]
    pct_pruned = (1 - final_gate) * 100
    print(f"  → [{tag}] final keep {final_gate:.3f}  pruned {pct_pruned:.2f}%  | "
          f"per-layer kept {per_layer_kept}/{layer_sizes}  | "
          f"orig→pruned test acc {orig_test*100:.2f}→{pruned_test*100:.2f}%  | "
          f"{total_time:.0f}s", flush=True)

    return {"lambda": lam, "seed": seed, "history": history,
            "per_layer_kept": list(zip(per_layer_kept, layer_sizes)),
            "pct_pruned": pct_pruned, "orig_test_acc": orig_test,
            "pruned_test_acc": pruned_test, "total_time": total_time,
            "layer_shapes": layer_shapes}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def plot_efficiency(per_lambda_stats, save_path):
    """Pruning efficiency curve: (mean % pruned) / (mean acc-drop pp) vs λ.

    The local maximum identifies the λ where the pruner gets the most pruning
    per unit of accuracy loss — a sharper, more interpretable target than the
    raw Pareto frontier. Hypothesised to coincide with λ_sim (the lowest λ at
    which all layers commit in the simultaneous regime).

    Caveat: when mean drop is very small or negative (pruning helps), the
    ratio explodes / flips sign. We guard with a floor of 0.5pp to keep the
    plot readable.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    lambdas = [s["lambda"] for s in per_lambda_stats]
    eff = []
    for s in per_lambda_stats:
        drop_pp = (s["orig_test_acc"] - s["pruned_test_acc_mean"]) * 100
        denom = max(drop_pp, 0.5)          # floor to keep plot finite
        eff.append(s["pct_pruned_mean"] / denom)
    ax.plot(lambdas, eff, "o-", color="darkorange", markersize=10, lw=2)
    for lam, e in zip(lambdas, eff):
        ax.annotate(f"{e:.1f}", (lam, e), xytext=(6, 4),
                    textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("λ (log scale)")
    ax.set_ylabel("efficiency  =  (% pruned)  /  max(drop pp, 0.5)")
    ax.set_title("Pruning efficiency vs λ  (local-max = λ_sim candidate)",
                 fontweight="bold")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_multiseed_comparison(per_lambda_stats, save_path):
    """% pruned vs pruned test acc, error bars from across seeds."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.5))
    lambdas    = [s["lambda"] for s in per_lambda_stats]
    pp_mean    = [s["pct_pruned_mean"]          for s in per_lambda_stats]
    pp_std     = [s["pct_pruned_std"]           for s in per_lambda_stats]
    acc_mean   = [s["pruned_test_acc_mean"]*100 for s in per_lambda_stats]
    acc_std    = [s["pruned_test_acc_std"]*100  for s in per_lambda_stats]
    orig_acc   = per_lambda_stats[0]["orig_test_acc"]*100

    ax.errorbar(pp_mean, acc_mean, xerr=pp_std, yerr=acc_std,
                fmt="o", color="tomato", markersize=10, capsize=4, lw=1.5, zorder=3)
    for lam, x, y in zip(lambdas, pp_mean, acc_mean):
        ax.annotate(f"λ={lam}", (x, y), xytext=(8, 4),
                    textcoords="offset points", fontsize=10)
    ax.axhline(orig_acc, color="steelblue", ls="--", lw=1.2,
               label=f"unpruned test acc = {orig_acc:.2f}%")
    ax.set_xlabel("% MLP neurons pruned (fc1+fc2 avg)")
    ax.set_ylabel("pruned test accuracy (%)")
    ax.set_title("CIFAR LeNet MLP — multi-seed Pareto (mean ± stdev)",
                 fontweight="bold")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas",     type=float, nargs="+",
                    default=[0.01, 0.03, 0.1, 0.3])
    ap.add_argument("--seeds",       type=int,   nargs="+", default=[0],
                    help="Seeds to run per λ. Multi-seed enables error bars + efficiency.")
    ap.add_argument("--epochs",      type=int,   default=5)
    ap.add_argument("--batch_size",  type=int,   default=256)
    ap.add_argument("--embed_dim",   type=int,   default=64)
    ap.add_argument("--lstm_hidden", type=int,   default=128)
    ap.add_argument("--lr",          type=float, default=0.001)
    ap.add_argument("--log_every",   type=int,   default=50)
    ap.add_argument("--device",      type=str,   default="cpu")
    ap.add_argument("--out_dir",     type=str,   default=DEFAULT_OUT_ROOT,
                    help="Output dir. Override per-sweep so wide/fine don't overwrite each other.")
    args = ap.parse_args()
    out_root = args.out_dir

    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}  λs={args.lambdas}  seeds={args.seeds}  epochs={args.epochs}")

    base_model = load_cifar_lenet(device)
    n_base = sum(p.numel() for p in base_model.parameters())
    print(f"Loaded CIFAR LeNet — {n_base:,} params, frozen.")

    train_loader, test_loader = get_loaders(args.batch_size)
    os.makedirs(out_root, exist_ok=True)

    # ── λ × seed nested loop ──────────────────────────────────────────────────
    all_results = []
    for lam in args.lambdas:
        for seed in args.seeds:
            res = train_one(lam, seed, base_model, train_loader, test_loader, args, device)
            all_results.append(res)

            # Per-(λ, seed) artefacts. With single seed the dir is the legacy
            # `lambda_<λ>/` (back-compat); with multiple seeds we nest.
            if len(args.seeds) == 1:
                run_dir = os.path.join(out_root, f"lambda_{lam}")
            else:
                run_dir = os.path.join(out_root, f"lambda_{lam}", f"seed_{seed}")
            plot_one_run(res["history"], os.path.join(run_dir, "plot.png"),
                         title=f"CIFAR LeNet MLP pruner — λ={lam} seed={seed} — "
                               f"{res['pct_pruned']:.1f}% pruned, "
                               f"test {res['pruned_test_acc']*100:.2f}%")
            write_run_summary(os.path.join(run_dir, "summary.txt"), lam,
                              res["layer_shapes"], res["history"],
                              res["per_layer_kept"], res["orig_test_acc"],
                              res["pruned_test_acc"], res["total_time"])

    # ── Aggregate per-λ across seeds ──────────────────────────────────────────
    per_lambda_stats = []
    for lam in args.lambdas:
        runs = [r for r in all_results if r["lambda"] == lam]
        pcts = [r["pct_pruned"]      for r in runs]
        accs = [r["pruned_test_acc"] for r in runs]
        per_lambda_stats.append({
            "lambda": lam,
            "pct_pruned_mean":      float(np.mean(pcts)),
            "pct_pruned_std":       float(np.std(pcts)),
            "pruned_test_acc_mean": float(np.mean(accs)),
            "pruned_test_acc_std":  float(np.std(accs)),
            "orig_test_acc":        runs[0]["orig_test_acc"],
            "runs": runs,
        })

    # Comparison + efficiency plots
    if len(args.seeds) == 1:
        plot_comparison([r for r in all_results], os.path.join(out_root, "comparison.png"))
    else:
        plot_multiseed_comparison(per_lambda_stats, os.path.join(out_root, "comparison.png"))
    plot_efficiency(per_lambda_stats, os.path.join(out_root, "efficiency.png"))

    # ── Summary table ─────────────────────────────────────────────────────────
    lines = [f"CIFAR LeNet MLP pruner — λ sweep, seeds={args.seeds}, epochs={args.epochs}",
             f"base   : 63K-param LeNet, fc1(128×400) fc2(64×128); fc3 untouched",
             f"pruner : BiLSTM embed_dim={args.embed_dim} lstm_hidden={args.lstm_hidden}",
             f"train  : batch {args.batch_size} on {device}",
             "-" * 96,
             f"{'lambda':>7} {'seed':>5} | {'% pruned':>10} | {'orig acc':>9} | {'pruned acc':>11} | "
             f"{'drop':>7} | {'efficiency':>11} | {'fc1':>10} {'fc2':>10}",
             "-" * 96]
    for s in per_lambda_stats:
        for r in s["runs"]:
            kept = r["per_layer_kept"]
            drop = (r["orig_test_acc"] - r["pruned_test_acc"]) * 100
            eff  = r["pct_pruned"] / max(drop, 0.5)
            lines.append(
                f"{r['lambda']:>7} {r['seed']:>5} | "
                f"{r['pct_pruned']:>9.2f}% | "
                f"{r['orig_test_acc']*100:>8.2f}% | {r['pruned_test_acc']*100:>10.2f}% | "
                f"{drop:>+6.2f}pp | "
                f"{eff:>10.2f}  | "
                f"{kept[0][0]:>4}/{kept[0][1]:<5} {kept[1][0]:>4}/{kept[1][1]:<5}")
        if len(args.seeds) > 1:
            drop_m = (s["orig_test_acc"] - s["pruned_test_acc_mean"]) * 100
            eff_m  = s["pct_pruned_mean"] / max(drop_m, 0.5)
            lines.append(
                f"{s['lambda']:>7} {'MEAN':>5} | "
                f"{s['pct_pruned_mean']:>7.2f}±{s['pct_pruned_std']:>4.2f}% | "
                f"{s['orig_test_acc']*100:>8.2f}% | "
                f"{s['pruned_test_acc_mean']*100:>7.2f}±{s['pruned_test_acc_std']*100:>3.2f}% | "
                f"{drop_m:>+6.2f}pp | "
                f"{eff_m:>10.2f}  | -")
            lines.append("-" * 96)
    with open(os.path.join(out_root, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nResults → {out_root}/")


if __name__ == "__main__":
    main()
