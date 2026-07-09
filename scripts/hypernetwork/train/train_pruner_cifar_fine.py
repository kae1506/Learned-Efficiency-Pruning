"""
CIFAR_big MLP pruner: FINE λ sweep around the elbow (λ≈0.03), multi-seed.

Follows up on cifar_lambda_sweep, which found λ=0.03 as a Pareto winner
(70.5% pruned, −1.53pp on test) but only one seed per λ. This script:

  • Sweeps λ ∈ {0.02, 0.03, 0.05, 0.07} (4 points bracketing the elbow).
  • Runs 3 seeds per λ → 12 pruner trainings total.
  • Live per-step log shows PER-LAYER % pruned per seed during training,
    not just an averaged "keep" number, so layer-specific dynamics are
    visible while it runs.

Output:
  experiments/latest/hypernetwork/cifar_lambda_fine/
    lambda_<λ>/seed_<s>/{plot.png, summary.txt}
    comparison.png                 (seed-mean ± stdev across λ)
    summary.txt                    (table: λ × seed and seed-averaged row)

Run from project root:
  venv/bin/python scripts/hypernetwork/train_pruner_cifar_fine.py
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import time
import argparse

import numpy as np
import torch
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
OUT_ROOT  = "experiments/latest/hypernetwork/cifar_lambda_fine"


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
    """Detached fc1/fc2/fc3 weight matrices fed to the pruner (fc4 excluded)."""
    return [model.fc1.weight.detach(),
            model.fc2.weight.detach(),
            model.fc3.weight.detach()]


def masked_forward(model: CIFARNetBig, gates: list[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """
    CIFAR_big forward with per-neuron gates applied to fc1/fc2/fc3 ROWS.
    Conv blocks (frozen, BN in eval mode) run unchanged. fc4 unmasked.
    """
    h = model.pool(F.relu(model.bn1(model.conv1(x))))
    h = model.pool(F.relu(model.bn2(model.conv2(h))))
    h = model.pool(F.relu(model.bn3(model.conv3(h))))
    h = h.view(h.size(0), -1)                          # [B, 8192]

    for linear, gate in [(model.fc1, gates[0]),
                         (model.fc2, gates[1]),
                         (model.fc3, gates[2])]:
        w = linear.weight.detach() * gate.unsqueeze(1)
        b = linear.bias.detach()   * gate
        h = F.relu(F.linear(h, w, b))

    return F.linear(h, model.fc4.weight.detach(), model.fc4.bias.detach())


# ─────────────────────────────────────────────────────────────────────────────
# Pruner training step. Returns avg + PER-LAYER kept fraction so the live
# log can show fc1/fc2/fc3 dynamics independently.
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

    # Per-layer keep fractions (gates are binary STE → fraction == #kept / #total)
    per_layer_keep = [g.mean().item() for g in gates]
    avg_gate = sum(per_layer_keep) / len(per_layer_keep)

    return {"loss": loss.item(), "orig_acc": orig_acc, "pruned_acc": pruned_acc,
            "acc_drop": orig_acc - pruned_acc, "avg_gate": avg_gate,
            "per_layer_keep": per_layer_keep}


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def get_loaders(batch_size: int, num_workers: int = 2):
    """CIFAR-10 loaders with clean (no-aug) transform — pruner sees same dist as test."""
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
    """Per-(λ, seed) plot. Three panels: loss, acc, per-layer % pruned."""
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

    # Per-layer % pruned over time — the key new view
    colors = {"fc1": "#c0392b", "fc2": "#2980b9", "fc3": "#27ae60"}
    for i, name in enumerate(["fc1", "fc2", "fc3"]):
        per = [(1 - k) * 100 for k in history["per_layer_keep"][i]]
        axes[2].plot(steps, per, alpha=0.2, color=colors[name])
        axes[2].plot(steps, _smooth(per), color=colors[name], lw=2, label=name)
    axes[2].set_title("per-layer % pruned"); axes[2].set_xlabel("step")
    axes[2].set_ylabel("% pruned"); axes[2].set_ylim(0, 100)
    axes[2].grid(alpha=0.3); axes[2].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(per_lambda_stats, save_path):
    """Final % pruned vs test acc, with seed mean ± stdev error bars across λ."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.5))
    lambdas    = [s["lambda"] for s in per_lambda_stats]
    pp_mean    = [s["pct_pruned_mean"]   for s in per_lambda_stats]
    pp_std     = [s["pct_pruned_std"]    for s in per_lambda_stats]
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
    ax.set_xlabel("% MLP neurons pruned (fc1+fc2+fc3 avg)")
    ax.set_ylabel("pruned test accuracy (%)")
    ax.set_title("CIFAR_big MLP fine λ sweep — seed mean ± stdev",
                 fontweight="bold")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_run_summary(path, lam, seed, layer_shapes, history,
                      per_layer_kept, orig_test, pruned_test, total_time):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    final_gate = history["avg_gate"][-1]
    pct_pruned = (1 - final_gate) * 100
    lines = [
        f"CIFAR_big MLP pruner — λ = {lam}, seed = {seed}",
        f"layers : {layer_shapes}",
        f"steps  : {len(history['loss'])}",
        f"time   : {total_time:.1f}s",
        "-" * 56,
        f"final avg keep gate           : {final_gate:.4f}",
        f"final % MLP neurons pruned    : {pct_pruned:.2f}%",
        f"per-layer kept (fc1/fc2/fc3)  : {per_layer_kept}",
        "-" * 56,
        f"FULL test set (10000 imgs):",
        f"  original (unpruned) acc     : {orig_test*100:.2f}%",
        f"  pruned              acc     : {pruned_test*100:.2f}%",
        f"  drop                        : {(orig_test - pruned_test)*100:+.2f}pp",
    ]
    with open(path, "w") as f: f.write("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Per-(λ, seed) training run
# ─────────────────────────────────────────────────────────────────────────────

def train_one(lam, seed, base_model, train_loader, test_loader, args, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    layer_shapes = [(w.shape[0], w.shape[1]) for w in get_fc_weights(base_model)]
    pruner = Pruner(layer_shapes, embed_dim=args.embed_dim, lstm_hidden=args.lstm_hidden).to(device)
    opt = torch.optim.Adam(pruner.parameters(), lr=args.lr)

    tag = f"λ={lam} seed={seed}"
    print(f"\n── {tag} ── pruner params: "
          f"{sum(p.numel() for p in pruner.parameters()):,}", flush=True)

    history = {"loss": [], "orig_acc": [], "pruned_acc": [], "acc_drop": [],
               "avg_gate": [], "per_layer_keep": [[], [], []]}
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
                # PER-LAYER % pruned in the live log — the requested view.
                fc1_p, fc2_p, fc3_p = [(1 - k) * 100 for k in m["per_layer_keep"]]
                print(f"  [{tag}] step {step:>4} ep{ep} | "
                      f"loss {m['loss']:+.3f} | "
                      f"orig {m['orig_acc']:.3f} pruned {m['pruned_acc']:.3f} | "
                      f"pruned% fc1={fc1_p:5.1f} fc2={fc2_p:5.1f} fc3={fc3_p:5.1f} "
                      f"(avg={(fc1_p+fc2_p+fc3_p)/3:5.1f})", flush=True)
    total_time = time.time() - t0

    # Final eval on full test set
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

    return {
        "lambda": lam, "seed": seed, "history": history,
        "per_layer_kept": list(zip(per_layer_kept, layer_sizes)),
        "pct_pruned": pct_pruned, "orig_test_acc": orig_test,
        "pruned_test_acc": pruned_test, "total_time": total_time,
        "layer_shapes": layer_shapes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main: λ × seed nested loop, then per-λ aggregation + comparison plot
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas",     type=float, nargs="+",
                    default=[0.02, 0.03, 0.05, 0.07])
    ap.add_argument("--seeds",       type=int,   nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs",      type=int,   default=5)
    ap.add_argument("--batch_size",  type=int,   default=256)
    ap.add_argument("--embed_dim",   type=int,   default=64)
    ap.add_argument("--lstm_hidden", type=int,   default=128)
    ap.add_argument("--lr",          type=float, default=0.001)
    ap.add_argument("--log_every",   type=int,   default=50)
    ap.add_argument("--device",      type=str,   default="mps")
    args = ap.parse_args()

    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    base_model = load_cifar_big(device)
    print(f"Loaded CIFAR_big — {sum(p.numel() for p in base_model.parameters()):,} params, frozen.")

    train_loader, test_loader = get_loaders(args.batch_size)
    os.makedirs(OUT_ROOT, exist_ok=True)

    # ── λ × seed nested loop ──────────────────────────────────────────────────
    all_results = []   # flat list of per-(λ, seed) result dicts
    for lam in args.lambdas:
        for seed in args.seeds:
            res = train_one(lam, seed, base_model, train_loader, test_loader, args, device)
            all_results.append(res)
            run_dir = os.path.join(OUT_ROOT, f"lambda_{lam}", f"seed_{seed}")
            plot_one_run(res["history"], os.path.join(run_dir, "plot.png"),
                         title=f"CIFAR_big MLP pruner — λ={lam} seed={seed} — "
                               f"{res['pct_pruned']:.1f}% pruned, "
                               f"test {res['pruned_test_acc']*100:.2f}%")
            write_run_summary(os.path.join(run_dir, "summary.txt"), lam, seed,
                              res["layer_shapes"], res["history"],
                              res["per_layer_kept"], res["orig_test_acc"],
                              res["pruned_test_acc"], res["total_time"])

    # ── Aggregate per-λ across seeds (mean ± stdev) ───────────────────────────
    per_lambda_stats = []
    for lam in args.lambdas:
        runs = [r for r in all_results if r["lambda"] == lam]
        pcts   = [r["pct_pruned"]      for r in runs]
        accs   = [r["pruned_test_acc"] for r in runs]
        per_lambda_stats.append({
            "lambda": lam,
            "pct_pruned_mean": float(np.mean(pcts)),
            "pct_pruned_std":  float(np.std(pcts)),
            "pruned_test_acc_mean": float(np.mean(accs)),
            "pruned_test_acc_std":  float(np.std(accs)),
            "orig_test_acc":  runs[0]["orig_test_acc"],
            "runs": runs,
        })

    plot_comparison(per_lambda_stats, os.path.join(OUT_ROOT, "comparison.png"))

    # ── Top-level table: every (λ, seed) row + per-λ aggregated row ───────────
    lines = ["CIFAR_big MLP pruner — FINE λ sweep × seeds",
             f"layers : fc1(1024×8192) fc2(512×1024) fc3(256×512); fc4 untouched",
             f"pruner : BiLSTM embed_dim={args.embed_dim} lstm_hidden={args.lstm_hidden}",
             f"train  : {args.epochs} epochs × batch {args.batch_size} on {device}",
             f"seeds  : {args.seeds}",
             "-" * 90,
             f"{'lambda':>7} {'seed':>5} | {'% pruned':>10} | {'orig acc':>9} | "
             f"{'pruned acc':>11} | {'drop':>7} | "
             f"{'fc1':>10} {'fc2':>10} {'fc3':>10}",
             "-" * 90]
    for s in per_lambda_stats:
        for r in s["runs"]:
            kept = r["per_layer_kept"]
            lines.append(
                f"{r['lambda']:>7} {r['seed']:>5} | "
                f"{r['pct_pruned']:>9.2f}% | "
                f"{r['orig_test_acc']*100:>8.2f}% | "
                f"{r['pruned_test_acc']*100:>10.2f}% | "
                f"{(r['orig_test_acc']-r['pruned_test_acc'])*100:>+6.2f}pp | "
                f"{kept[0][0]:>4}/{kept[0][1]:<5} "
                f"{kept[1][0]:>4}/{kept[1][1]:<5} "
                f"{kept[2][0]:>4}/{kept[2][1]:<5}")
        lines.append(
            f"{s['lambda']:>7} {'MEAN':>5} | "
            f"{s['pct_pruned_mean']:>8.2f}±{s['pct_pruned_std']:.2f}% | "
            f"{s['orig_test_acc']*100:>8.2f}% | "
            f"{s['pruned_test_acc_mean']*100:>7.2f}±{s['pruned_test_acc_std']*100:.2f}% | "
            f"{(s['orig_test_acc']-s['pruned_test_acc_mean'])*100:>+6.2f}pp | "
            f"-")
        lines.append("-" * 90)

    with open(os.path.join(OUT_ROOT, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nResults → {OUT_ROOT}/")


if __name__ == "__main__":
    main()
