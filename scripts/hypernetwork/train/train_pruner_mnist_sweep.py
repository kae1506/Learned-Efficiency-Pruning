"""
MNIST [1024, 1024] MLP pruner: λ × seed sweep with efficiency curve.

Same protocol as scripts/hypernetwork/train_pruner_cifar_lenet.py — produces
identical output structure (multi-seed plots, summary table, efficiency.png) so
the three points (LeNet 63K → MNIST 1.86M → CIFAR_big 10.4M) can be combined
on one scaling-law plot via scripts/hypernetwork/efficiency_compare.py.

Base model: experiments/checkpoints/mnist_model.pt
  MLP: 784 → 1024 → 1024 → 10, dropout 0.1, 1.86M params
  Prunable: hidden1 (1024 neurons), hidden2 (1024 neurons). Output unmasked.

Run from project root:
  venv/bin/python scripts/hypernetwork/train_pruner_mnist_sweep.py
"""

import os
import sys
import time
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(".")
from src.model import MLP
from src.dataset import get_mnist_loaders
from src.pruners.bilstm import Pruner
from src.prune_train import pruner_step as base_pruner_step
from src.prune_train import get_hidden_weights, masked_forward
import torch.nn.functional as F


DEFAULT_CKPT     = "experiments/checkpoints/mnist_model.pt"
DEFAULT_OUT_ROOT = "experiments/latest/hypernetwork/mnist_lambda_sweep_15ep"


def load_mnist_model(ckpt_path: str, device) -> MLP:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = MLP(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def pruner_step_with_per_layer(pruner, base_model, optimizer, x, y, sparsity_weight):
    """Wraps src.prune_train.pruner_step but returns per-layer keep fractions too
    (the upstream version returns only avg_gate). Re-implements the body — minor
    duplication, but lets us trace fc1 vs fc2 commitment over time."""
    optimizer.zero_grad()
    hidden_weights = get_hidden_weights(base_model)
    gates = pruner(hidden_weights)

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
# Plotting (same shape as LeNet script — minor differences for n hidden layers)
# ─────────────────────────────────────────────────────────────────────────────

def _smooth(values, window=50):
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo:i + 1]) / (i - lo + 1))
    return out


def plot_one_run(history, save_path, title, n_hidden):
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

    palette = ["#c0392b", "#2980b9", "#27ae60", "#8e44ad"]
    for i in range(n_hidden):
        per = [(1 - k) * 100 for k in history["per_layer_keep"][i]]
        axes[2].plot(steps, per, alpha=0.2, color=palette[i])
        axes[2].plot(steps, _smooth(per), color=palette[i], lw=2,
                     label=f"h{i+1}")
    axes[2].set_title("per-layer % pruned"); axes[2].set_xlabel("step")
    axes[2].set_ylabel("% pruned"); axes[2].set_ylim(0, 100)
    axes[2].grid(alpha=0.3); axes[2].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_efficiency(per_lambda_stats, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    lambdas = [s["lambda"] for s in per_lambda_stats]
    eff = []
    for s in per_lambda_stats:
        drop_pp = (s["orig_test_acc"] - s["pruned_test_acc_mean"]) * 100
        eff.append(s["pct_pruned_mean"] / max(drop_pp, 0.5))
    ax.plot(lambdas, eff, "o-", color="darkorange", markersize=10, lw=2)
    for lam, e in zip(lambdas, eff):
        ax.annotate(f"{e:.1f}", (lam, e), xytext=(6, 4),
                    textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("λ (log scale)")
    ax.set_ylabel("efficiency  =  (% pruned)  /  max(drop pp, 0.5)")
    ax.set_title("MNIST [1024,1024] pruning efficiency vs λ",
                 fontweight="bold")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_multiseed_comparison(per_lambda_stats, save_path):
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
    ax.set_xlabel("% MLP neurons pruned (avg)")
    ax.set_ylabel("pruned test accuracy (%)")
    ax.set_title("MNIST [1024,1024] — multi-seed Pareto (mean ± stdev)",
                 fontweight="bold")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_run_summary(path, lam, seed, layer_shapes, history, per_layer_kept,
                      orig_test, pruned_test, total_time):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    final_gate = history["avg_gate"][-1]
    pct_pruned = (1 - final_gate) * 100
    lines = [
        f"MNIST MLP pruner — λ = {lam}, seed = {seed}",
        f"layers : {layer_shapes}",
        f"steps  : {len(history['loss'])}",
        f"time   : {total_time:.1f}s",
        "-" * 56,
        f"final avg keep gate          : {final_gate:.4f}",
        f"final % MLP neurons pruned   : {pct_pruned:.2f}%",
        f"per-layer kept               : {per_layer_kept}",
        "-" * 56,
        f"FULL test set:",
        f"  original (unpruned) acc    : {orig_test*100:.2f}%",
        f"  pruned              acc    : {pruned_test*100:.2f}%",
        f"  drop                       : {(orig_test - pruned_test)*100:+.2f}pp",
    ]
    with open(path, "w") as f: f.write("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Per-(λ, seed) training run
# ─────────────────────────────────────────────────────────────────────────────

def train_one(lam, seed, base_model, train_loader, test_loader, args, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    hidden_weights = get_hidden_weights(base_model)
    layer_shapes = [(w.shape[0], w.shape[1]) for w in hidden_weights]
    n_hidden = len(layer_shapes)
    pruner = Pruner(layer_shapes, embed_dim=args.embed_dim, lstm_hidden=args.lstm_hidden).to(device)
    opt = torch.optim.Adam(pruner.parameters(), lr=args.lr)

    tag = f"λ={lam} seed={seed}"
    print(f"\n── {tag} ── pruner params: "
          f"{sum(p.numel() for p in pruner.parameters()):,}", flush=True)

    history = {"loss": [], "orig_acc": [], "pruned_acc": [], "acc_drop": [],
               "avg_gate": [], "per_layer_keep": [[] for _ in range(n_hidden)]}
    t0 = time.time()
    step = 0
    for ep in range(1, args.epochs + 1):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            m = pruner_step_with_per_layer(pruner, base_model, opt, x, y, lam)
            history["loss"].append(m["loss"])
            history["orig_acc"].append(m["orig_acc"])
            history["pruned_acc"].append(m["pruned_acc"])
            history["acc_drop"].append(m["acc_drop"])
            history["avg_gate"].append(m["avg_gate"])
            for i, k in enumerate(m["per_layer_keep"]):
                history["per_layer_keep"][i].append(k)
            step += 1
            if step % args.log_every == 0:
                per_p = [(1 - k) * 100 for k in m["per_layer_keep"]]
                per_str = " ".join(f"h{i+1}={v:5.1f}" for i, v in enumerate(per_p))
                print(f"  [{tag}] step {step:>4} ep{ep} | loss {m['loss']:+.3f} | "
                      f"orig {m['orig_acc']:.3f} pruned {m['pruned_acc']:.3f} | "
                      f"pruned% {per_str} "
                      f"(avg={sum(per_p)/len(per_p):5.1f})", flush=True)
    total_time = time.time() - t0

    pruner.eval()
    with torch.no_grad():
        final_gates = pruner(get_hidden_weights(base_model))
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
            "layer_shapes": layer_shapes, "n_hidden": n_hidden}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas",     type=float, nargs="+",
                    default=[0.04, 0.06, 0.08, 0.10, 0.15, 0.20])
    ap.add_argument("--seeds",       type=int,   nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs",      type=int,   default=15)
    ap.add_argument("--batch_size",  type=int,   default=256)
    ap.add_argument("--embed_dim",   type=int,   default=64)
    ap.add_argument("--lstm_hidden", type=int,   default=128)
    ap.add_argument("--lr",          type=float, default=0.001)
    ap.add_argument("--log_every",   type=int,   default=100)
    ap.add_argument("--device",      type=str,   default="cpu")
    ap.add_argument("--ckpt",        type=str,   default=DEFAULT_CKPT,
                    help="Path to MNIST checkpoint. The model's hidden_dims drive layer-shape detection.")
    ap.add_argument("--out_dir",     type=str,   default=DEFAULT_OUT_ROOT)
    args = ap.parse_args()
    out_root = args.out_dir

    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}  λs={args.lambdas}  seeds={args.seeds}  epochs={args.epochs}")

    base_model = load_mnist_model(args.ckpt, device)
    n_base = sum(p.numel() for p in base_model.parameters())
    hidden_str = [h.shape[0] for h in get_hidden_weights(base_model)]
    print(f"Loaded {args.ckpt}: hidden={hidden_str}, {n_base:,} params, frozen.")

    train_loader, test_loader = get_mnist_loaders(data_dir="./data",
                                                  batch_size=args.batch_size)
    os.makedirs(out_root, exist_ok=True)

    all_results = []
    for lam in args.lambdas:
        for seed in args.seeds:
            res = train_one(lam, seed, base_model, train_loader, test_loader, args, device)
            all_results.append(res)

            if len(args.seeds) == 1:
                run_dir = os.path.join(out_root, f"lambda_{lam}")
            else:
                run_dir = os.path.join(out_root, f"lambda_{lam}", f"seed_{seed}")
            plot_one_run(res["history"], os.path.join(run_dir, "plot.png"),
                         title=f"MNIST MLP pruner — λ={lam} seed={seed} — "
                               f"{res['pct_pruned']:.1f}% pruned, "
                               f"test {res['pruned_test_acc']*100:.2f}%",
                         n_hidden=res["n_hidden"])
            write_run_summary(os.path.join(run_dir, "summary.txt"), lam, seed,
                              res["layer_shapes"], res["history"],
                              res["per_layer_kept"], res["orig_test_acc"],
                              res["pruned_test_acc"], res["total_time"])

    # Aggregate
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

    plot_multiseed_comparison(per_lambda_stats, os.path.join(out_root, "comparison.png"))
    plot_efficiency(per_lambda_stats, os.path.join(out_root, "efficiency.png"))

    lines = [f"MNIST [1024,1024] MLP pruner — λ sweep, seeds={args.seeds}, epochs={args.epochs}",
             f"base   : 1.86M-param MLP, hidden_dims=[1024,1024]; output untouched",
             f"pruner : BiLSTM embed_dim={args.embed_dim} lstm_hidden={args.lstm_hidden}",
             f"train  : batch {args.batch_size} on {device}",
             "-" * 96,
             f"{'lambda':>7} {'seed':>5} | {'% pruned':>10} | {'orig acc':>9} | {'pruned acc':>11} | "
             f"{'drop':>7} | {'efficiency':>11} | per-layer kept",
             "-" * 96]
    for s in per_lambda_stats:
        for r in s["runs"]:
            kept = r["per_layer_kept"]
            drop = (r["orig_test_acc"] - r["pruned_test_acc"]) * 100
            eff  = r["pct_pruned"] / max(drop, 0.5)
            kept_str = " ".join(f"{k}/{n}" for k, n in kept)
            lines.append(
                f"{r['lambda']:>7} {r['seed']:>5} | "
                f"{r['pct_pruned']:>9.2f}% | "
                f"{r['orig_test_acc']*100:>8.2f}% | {r['pruned_test_acc']*100:>10.2f}% | "
                f"{drop:>+6.2f}pp | {eff:>10.2f}  | {kept_str}")
        if len(args.seeds) > 1:
            drop_m = (s["orig_test_acc"] - s["pruned_test_acc_mean"]) * 100
            eff_m  = s["pct_pruned_mean"] / max(drop_m, 0.5)
            lines.append(
                f"{s['lambda']:>7} {'MEAN':>5} | "
                f"{s['pct_pruned_mean']:>7.2f}±{s['pct_pruned_std']:>4.2f}% | "
                f"{s['orig_test_acc']*100:>8.2f}% | "
                f"{s['pruned_test_acc_mean']*100:>7.2f}±{s['pruned_test_acc_std']*100:>3.2f}% | "
                f"{drop_m:>+6.2f}pp | {eff_m:>10.2f}  | -")
            lines.append("-" * 96)
    with open(os.path.join(out_root, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nResults → {out_root}/")


if __name__ == "__main__":
    main()
