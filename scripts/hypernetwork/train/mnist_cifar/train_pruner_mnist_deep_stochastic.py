"""
Stochastic layer-subset pruner training vs. normal joint training — MNIST
784->512->512->512->512->10 (4 hidden layers, the deepest existing MNIST net;
chosen over the standard [1024,1024] net specifically because it's the
richest available testbed for a "subset of layers" mechanism -- [1024,1024]
only has 2 layers, too thin for k=2-of-N to mean much).

THE IDEA BEING TESTED
----------------------
Each pruner training step, instead of the pruner scoring and gating all 4
hidden layers jointly (the normal/baseline design), stochastically sample a
subset S of k=2 layers (uniform, out of 4) and:
  - the Pruner computes gates ONLY for the layers in S
    (Pruner.forward(weights_S, layer_indices=S) -- see src/pruners/bilstm.py)
  - those gates are APPLIED only to layers in S; the other 4-k layers pass
    through UNGATED (full width) for this step
  - the sparsity penalty only sums over S's gates this step
Every --full_eval_every steps (default 250; --steps default 1000, matching
the existing non-stochastic baseline at experiments/latest/hypernetwork/
shape_deep4x512/) AND at the final step, gates are computed for all 4 layers
TOGETHER and evaluated on the full test set via evaluate_with_gates -- that
full-joint number is the only one that means "the actual pruned model";
every other step is a partial-pruning training signal, not a result.

This is randomized block-coordinate descent (Nesterov 2012; Richtarik &
Takac 2014) applied to the pruner's 4 per-layer gate decisions as blocks.
Uniform sampling (not a depth-weighted schedule a la Stochastic Depth,
Huang et al. 2016) per Single-Path-One-Shot NAS's finding that uniform
sampling is a fine simple default for this kind of search-space training
(Guo et al. 2019, arxiv 1904.00420).

KNOWN, ACCEPTED TRADEOFF (see src/pruners/bilstm.py's _node_scores
docstring): the BiLSTM's cross-layer context path only sees the layers
actually passed in, so a layer's "neighbors" in the BiLSTM sequence change
depending on which other layers happen to be sampled alongside it that step.
This is a real difference from the baseline (which always sees the true
full-layer sequence) -- the hypothesis is that this still works, possibly
even benefits from the extra stochasticity (skip connections between
non-adjacent layers effectively get explored), not that it's risk-free.

DOES NOT test a training-speed hypothesis. Gating is activation-masking, not
dimension reduction -- the frozen base model's forward/backward FLOPs are
the same either way. This tests optimization dynamics / final mask quality
at MATCHED total steps, not wall-clock efficiency.

PER-LAYER %KEPT TRACKING: full per-step history is kept for all N_LAYERS.
A layer not sampled this step has NO fresh gate computed for it and is
simply ungated (100% kept) for that step's forward pass -- see
masked_forward_partial -- so its history entry for that step is recorded
as 1.0 (100% kept), not carried forward from its last sampled value. This
is a per-step snapshot, not a memory of past gates.

Base model: experiments/checkpoints/mnist_deep4x512.pt (784->512x4->10,
1.86M-ish? no -- 4x512 hidden, dropout 0.1). Comparison baseline: identical
protocol (same steps/lambda/seeds/pruner config), but the normal joint
src.prune_train.pruner_step every step (imported directly, not
reimplemented, so the baseline is guaranteed identical to the existing
non-stochastic convention, not a re-derived approximation of it).

DELIBERATELY NOT RUN as part of writing this file -- see the module's
if __name__ == "__main__" guard. Run manually:
    venv/bin/python scripts/hypernetwork/train/train_pruner_mnist_deep_stochastic.py
"""

import os
import sys
import time
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(".")
from src.model import MLP
from src.dataset import get_mnist_loaders
from src.pruners.bilstm import Pruner
from src.prune_train import get_hidden_weights, masked_forward
from src.interpretability import evaluate_with_gates


DEFAULT_CKPT     = "experiments/checkpoints/mnist_deep4x512.pt"
DEFAULT_OUT_ROOT = "experiments/latest/hypernetwork/mnist_deep_stochastic"
N_LAYERS         = 4   # 784->512->512->512->512->10, 4 prunable hidden layers


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_frozen(ckpt_path: str, device) -> MLP:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MLP(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Partial-gate forward — like src.prune_train.masked_forward, but only the
# layers in gates_by_layer get scaled; the rest pass through ungated.
# ─────────────────────────────────────────────────────────────────────────────

def masked_forward_partial(model, gates_by_layer: dict[int, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    import torch.nn as nn
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    x = x.view(x.size(0), -1)
    for i, linear in enumerate(linears[:-1]):
        if i in gates_by_layer:
            gate = gates_by_layer[i]
            w = linear.weight.detach() * gate.unsqueeze(1)
            b = linear.bias.detach() * gate
        else:
            w = linear.weight.detach()
            b = linear.bias.detach()
        x = F.relu(F.linear(x, w, b))
    out = linears[-1]
    return F.linear(x, out.weight.detach(), out.bias.detach())


# ─────────────────────────────────────────────────────────────────────────────
# Baseline (joint, all layers every step) training step. Reimplements
# src.prune_train.pruner_step's body rather than calling it directly, ONLY to
# additionally expose per_layer_keep -- the shared version only returns a
# scalar avg_gate. Same workaround train_pruner_mnist_sweep.py already uses
# (its pruner_step_with_per_layer) for the identical reason; math is
# unchanged from the shared function, just also returns the per-layer list.
# ─────────────────────────────────────────────────────────────────────────────

def baseline_pruner_step(pruner, model, optimizer, x, y, sparsity_weight: float) -> dict:
    optimizer.zero_grad()

    hidden_weights = get_hidden_weights(model)
    gates = pruner(hidden_weights)   # full joint call, all N_LAYERS, every step

    with torch.no_grad():
        orig_logits = model(x)
        ce_orig  = F.cross_entropy(orig_logits, y)
        orig_acc = (orig_logits.argmax(1) == y).float().mean().item()

    pruned_logits = masked_forward(model, gates, x)
    ce_pruned = F.cross_entropy(pruned_logits, y)

    with torch.no_grad():
        pruned_acc = (pruned_logits.argmax(1) == y).float().mean().item()

    sparsity_loss = sum(g.mean() for g in gates) / len(gates)
    loss = (ce_pruned - ce_orig) + sparsity_weight * sparsity_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(pruner.parameters(), max_norm=1.0)
    optimizer.step()

    per_layer_keep = [g.mean().item() for g in gates]
    return {
        "loss":            loss.item(),
        "orig_acc":        orig_acc,
        "pruned_acc":      pruned_acc,
        "acc_drop":        orig_acc - pruned_acc,
        "per_layer_keep":  per_layer_keep,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stochastic layer-subset training step
# ─────────────────────────────────────────────────────────────────────────────

def stochastic_pruner_step(pruner, model, optimizer, x, y, sparsity_weight: float, k: int) -> dict:
    optimizer.zero_grad()

    S = sorted(random.sample(range(N_LAYERS), k))
    all_weights = get_hidden_weights(model)
    weights_S = [all_weights[i] for i in S]
    gates_S = pruner(weights_S, layer_indices=S)   # only S scored this step

    with torch.no_grad():
        orig_logits = model(x)
        ce_orig  = F.cross_entropy(orig_logits, y)
        orig_acc = (orig_logits.argmax(1) == y).float().mean().item()

    gates_by_layer = dict(zip(S, gates_S))
    pruned_logits = masked_forward_partial(model, gates_by_layer, x)
    ce_pruned = F.cross_entropy(pruned_logits, y)

    with torch.no_grad():
        pruned_acc = (pruned_logits.argmax(1) == y).float().mean().item()

    sparsity_loss = sum(g.mean() for g in gates_S) / len(gates_S)   # mean over S ONLY
    loss = (ce_pruned - ce_orig) + sparsity_weight * sparsity_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(pruner.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        "loss":            loss.item(),
        "orig_acc":        orig_acc,
        "pruned_acc":      pruned_acc,
        "acc_drop":        orig_acc - pruned_acc,
        "sampled_layers":  S,
        "fresh_keep":      {i: g.mean().item() for i, g in zip(S, gates_S)},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full joint evaluation — the only number that means "the actual pruned
# model." Used periodically (every --full_eval_every steps) and at the end,
# for both modes (baseline already recomputes all 4 every step, so its
# "periodic full eval" is just a checkpoint of what it's already doing;
# stochastic's periodic full eval is the only place all 4 layers are ever
# scored together during that run).
# ─────────────────────────────────────────────────────────────────────────────

def full_eval(pruner, model, test_loader, device):
    pruner.eval()
    with torch.no_grad():
        weights = get_hidden_weights(model)
        gates = pruner(weights)          # full joint call, all N_LAYERS
    test_acc = evaluate_with_gates(model, gates, test_loader, device)
    per_layer_kept = [g.mean().item() for g in gates]
    pruner.train()
    return test_acc, per_layer_kept


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _smooth(values, window=50):
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo:i + 1]) / (i - lo + 1))
    return out


def plot_one_run(history, full_eval_history, save_path, title):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    steps = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(title, fontsize=11, fontweight="bold")

    axes[0].plot(steps, history["loss"], alpha=0.15, color="steelblue")
    axes[0].plot(steps, _smooth(history["loss"]), color="steelblue", lw=2)
    axes[0].axhline(0, color="gray", ls="--", lw=0.8)
    axes[0].set_title("Pruner loss"); axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss"); axes[0].grid(alpha=0.3)

    axes[1].plot(steps, _smooth(history["orig_acc"]),   color="steelblue", lw=2, label="orig")
    axes[1].plot(steps, _smooth(history["pruned_acc"]), color="tomato",    lw=2, label="pruned (partial/proxy)")
    fe_steps = [e["step"] for e in full_eval_history]
    fe_acc   = [e["test_acc"] for e in full_eval_history]
    axes[1].scatter(fe_steps, fe_acc, color="darkgreen", zorder=5, s=40,
                    label="full-joint eval (real)")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("step")
    axes[1].set_ylabel("acc"); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=7)

    cmap = plt.cm.tab10(np.linspace(0, 1, N_LAYERS))
    for i in range(N_LAYERS):
        per = [(1 - k) * 100 for k in history["per_layer_keep"][i]]
        axes[2].plot(steps, per, color=cmap[i], lw=1.2, label=f"L{i}")
    for e in full_eval_history:
        for i in range(N_LAYERS):
            axes[2].scatter(e["step"], (1 - e["per_layer_kept"][i]) * 100,
                            color=cmap[i], marker="x", s=50, zorder=5)
    axes[2].set_title("per-layer % pruned (line=per-step, 0% when unsampled; x=full eval)")
    axes[2].set_xlabel("step"); axes[2].set_ylabel("% pruned"); axes[2].set_ylim(0, 100)
    axes[2].grid(alpha=0.3); axes[2].legend(ncol=2, fontsize=7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_efficiency_vs_lambda(results, save_path):
    """results: list of dicts with mode, lambda, seed, efficiency (= pct_pruned / acc_drop_pp)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = {"baseline": "steelblue", "stochastic": "tomato"}
    for mode in colors:
        rows = [r for r in results if r["mode"] == mode]
        if not rows:
            continue
        by_lam = {}
        for r in rows:
            by_lam.setdefault(r["lambda"], []).append(r)
        lams = sorted(by_lam)
        ys   = [np.mean([r["efficiency"] for r in by_lam[l]]) for l in lams]
        yerr = [np.std([r["efficiency"]  for r in by_lam[l]]) for l in lams]
        ax.errorbar(lams, ys, yerr=yerr, fmt="o-", color=colors[mode],
                    markersize=8, capsize=4, lw=1.5, label=mode)
    ax.set_xscale("log")
    ax.set_xlabel("λ (sparsity weight)")
    ax.set_ylabel("efficiency = % pruned / accuracy drop (pp)")
    ax.set_title("MNIST deep4x512 — efficiency vs λ, baseline vs. stochastic-subset",
                fontweight="bold")
    ax.grid(alpha=0.3, which="both"); ax.legend()
    fig.tight_layout(); fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mode_comparison(results, save_path):
    """results: list of dicts with mode, lambda, seed, pct_pruned, test_acc."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = {"baseline": "steelblue", "stochastic": "tomato"}
    for mode in colors:
        rows = [r for r in results if r["mode"] == mode]
        if not rows:
            continue
        by_lam = {}
        for r in rows:
            by_lam.setdefault(r["lambda"], []).append(r)
        lams = sorted(by_lam)
        xs   = [np.mean([r["pct_pruned"] for r in by_lam[l]]) for l in lams]
        xerr = [np.std([r["pct_pruned"] for r in by_lam[l]])  for l in lams]
        ys   = [np.mean([r["test_acc"]   for r in by_lam[l]]) for l in lams]
        yerr = [np.std([r["test_acc"]    for r in by_lam[l]]) for l in lams]
        ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, fmt="o-", color=colors[mode],
                    markersize=8, capsize=4, lw=1.5, label=mode)
        for l, x, y in zip(lams, xs, ys):
            ax.annotate(f"λ={l}", (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("% pruned (full-joint eval, mean over seeds)")
    ax.set_ylabel("pruned test accuracy")
    ax.set_title("MNIST deep4x512 — baseline vs. stochastic-subset training",
                fontweight="bold")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Per-(mode, λ, seed) training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one(mode, lam, seed, model, train_loader, test_loader, args, device, run_dir):
    assert mode in ("baseline", "stochastic")
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    layer_shapes = [(w.shape[0], w.shape[1]) for w in get_hidden_weights(model)]
    pruner = Pruner(layer_shapes, embed_dim=args.embed_dim, lstm_hidden=args.lstm_hidden).to(device)
    opt = torch.optim.Adam(pruner.parameters(), lr=args.lr)

    tag = f"{mode} λ={lam} seed={seed}"
    print(f"\n── {tag} ── pruner params: {sum(p.numel() for p in pruner.parameters()):,}",
          flush=True)

    history = {
        "loss": [], "orig_acc": [], "pruned_acc": [],
        "per_layer_keep": [[] for _ in range(N_LAYERS)],
    }
    full_eval_history = []

    it = iter(train_loader)
    t0 = time.time()
    pbar = tqdm(total=args.steps, desc=tag, unit="step", dynamic_ncols=True)

    for step in range(1, args.steps + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)
        x = x[:args.samples_per_step].to(device)
        y = y[:args.samples_per_step].to(device)

        if mode == "baseline":
            m = baseline_pruner_step(pruner, model, opt, x, y, lam)
            step_keep = m["per_layer_keep"]
        else:
            m = stochastic_pruner_step(pruner, model, opt, x, y, lam, args.k)
            # Unsampled layers get NO fresh gate this step -- they're ungated
            # (100% kept) in the forward pass, so record 1.0, not a carried-
            # forward value from whenever they were last sampled.
            step_keep = [1.0] * N_LAYERS
            for i, v in m["fresh_keep"].items():
                step_keep[i] = v

        history["loss"].append(m["loss"])
        history["orig_acc"].append(m["orig_acc"])
        history["pruned_acc"].append(m["pruned_acc"])
        for i in range(N_LAYERS):
            history["per_layer_keep"][i].append(step_keep[i])

        pbar.update(1)
        pbar.set_postfix(loss=f"{m['loss']:+.3f}", refresh=False)

        if step % args.full_eval_every == 0 or step == args.steps:
            test_acc, per_layer_kept_full = full_eval(pruner, model, test_loader, device)
            full_eval_history.append({
                "step": step, "test_acc": test_acc, "per_layer_kept": per_layer_kept_full,
            })
            tqdm.write(f"  [{tag}] step {step:>5}/{args.steps} FULL EVAL | "
                       f"test_acc {test_acc:.4f} | "
                       f"per_layer_kept {[f'{v:.2f}' for v in per_layer_kept_full]}")

    pbar.close()
    total_time = time.time() - t0

    final = full_eval_history[-1]
    pct_pruned = (1 - sum(final["per_layer_kept"]) / N_LAYERS) * 100

    plot_one_run(
        history, full_eval_history,
        os.path.join(run_dir, "plot.png"),
        title=f"MNIST deep4x512 — {mode} λ={lam} seed={seed} — "
              f"{pct_pruned:.1f}% pruned, acc {final['test_acc']:.4f}",
    )

    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        f.write(f"MNIST deep4x512 pruner — mode={mode} λ={lam} seed={seed}\n")
        f.write(f"layers : {N_LAYERS} hidden blocks, 512 neurons each\n")
        if mode == "stochastic":
            f.write(f"k      : {args.k} of {N_LAYERS} layers sampled per step (uniform)\n")
        f.write(f"steps  : {args.steps}\n")
        f.write(f"time   : {total_time:.1f}s\n")
        f.write("-" * 60 + "\n")
        f.write(f"final % pruned (full joint)  : {pct_pruned:.2f}%\n")
        f.write(f"final per-layer kept          : {final['per_layer_kept']}\n")
        f.write(f"final test accuracy            : {final['test_acc']:.4f}\n")
        f.write("-" * 60 + "\n")
        f.write("full-eval history (step, test_acc, per_layer_kept):\n")
        for e in full_eval_history:
            f.write(f"  {e['step']:>5}  {e['test_acc']:.4f}  {e['per_layer_kept']}\n")

    torch.save({
        "pruner_state_dict": pruner.state_dict(),
        "mode": mode, "lambda": lam, "seed": seed, "k": args.k if mode == "stochastic" else None,
        "embed_dim": args.embed_dim, "lstm_hidden": args.lstm_hidden,
        "final_per_layer_kept": final["per_layer_kept"],
        "final_test_acc": final["test_acc"],
        "full_eval_history": full_eval_history,
    }, os.path.join(run_dir, "pruner.pt"))
    print(f"  [saved] {run_dir}/  (plot.png, summary.txt, pruner.pt)", flush=True)

    return {
        "mode": mode, "lambda": lam, "seed": seed,
        "pct_pruned": pct_pruned, "test_acc": final["test_acc"],
        "per_layer_kept": final["per_layer_kept"], "total_time": total_time,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes",         nargs="+", default=["baseline", "stochastic"],
                    choices=["baseline", "stochastic"])
    ap.add_argument("--lambdas",       type=float, nargs="+", default=[0.5],
                    help="Default matches the existing shape_deep4x512 baseline (sw=0.5).")
    ap.add_argument("--seeds",         type=int, nargs="+", default=[0, 1, 2],
                    help="Default matches the existing shape_deep4x512 baseline (3 seeds).")
    ap.add_argument("--steps",         type=int, default=1000,
                    help="Default matches the existing shape_deep4x512 baseline (PRUNER_STEPS=1000).")
    ap.add_argument("--k",             type=int, default=2,
                    help="Layers sampled per step in stochastic mode (out of 4). Ignored for baseline.")
    ap.add_argument("--full_eval_every", type=int, default=250,
                    help="Full joint 4-layer eval cadence. 1000 steps / 250 = 4 checkpoints.")
    ap.add_argument("--samples_per_step", type=int, default=64,
                    help="Matches configs/config.yaml's pruner.samples_per_step convention.")
    ap.add_argument("--embed_dim",     type=int, default=64)
    ap.add_argument("--lstm_hidden",   type=int, default=128)
    ap.add_argument("--lr",            type=float, default=0.001)
    ap.add_argument("--ckpt",          type=str, default=DEFAULT_CKPT)
    ap.add_argument("--out_dir",       type=str, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--data_dir",      type=str, default="./data")
    ap.add_argument("--batch_size",    type=int, default=256)
    ap.add_argument("--device",        type=str, default="cpu")
    args = ap.parse_args()

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device} | modes={args.modes} | λs={args.lambdas} | "
          f"seeds={args.seeds} | steps={args.steps} | k={args.k}/{N_LAYERS}")

    model = load_frozen(args.ckpt, device)
    print(f"Loaded {args.ckpt} — frozen, {sum(p.numel() for p in model.parameters()):,} params.")

    train_loader, test_loader = get_mnist_loaders(args.data_dir, args.batch_size)

    # Unpruned frozen-model accuracy, computed once (model never changes) --
    # denominator for efficiency = %pruned / accuracy-drop-pp. all-ones gates
    # through the same evaluate_with_gates codepath used everywhere else, so
    # it's directly comparable to every "final test accuracy" number below.
    all_ones_gates = [torch.ones(w.shape[0], device=device) for w in get_hidden_weights(model)]
    orig_test_acc = evaluate_with_gates(model, all_ones_gates, test_loader, device)
    print(f"Unpruned frozen-model test accuracy: {orig_test_acc:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    all_results = []
    total_runs = len(args.modes) * len(args.lambdas) * len(args.seeds)
    run_num = 0

    for mode in args.modes:
        for lam in args.lambdas:
            for seed in args.seeds:
                run_num += 1
                tqdm.write(f"\n{'='*70}\nRun {run_num}/{total_runs}\n{'='*70}")
                run_dir = os.path.join(args.out_dir, mode, f"lambda_{lam}", f"seed_{seed}")
                res = train_one(mode, lam, seed, model, train_loader, test_loader,
                                args, device, run_dir)
                acc_drop_pp = (orig_test_acc - res["test_acc"]) * 100
                res["acc_drop_pp"] = acc_drop_pp
                # acc_drop_pp <= 0 (pruned model matches/beats the unpruned one on
                # this seed, plausible noise at light pruning) makes the ratio
                # blow up or flip sign -- flagged, not silently clamped.
                res["efficiency"] = res["pct_pruned"] / acc_drop_pp if acc_drop_pp > 0 else float("nan")
                all_results.append(res)

    plot_mode_comparison(all_results, os.path.join(args.out_dir, "mode_comparison.png"))
    plot_efficiency_vs_lambda(all_results, os.path.join(args.out_dir, "efficiency_vs_lambda.png"))

    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(f"MNIST deep4x512 — baseline vs stochastic (k={args.k}/{N_LAYERS}) | "
                f"steps={args.steps} | seeds={args.seeds}\n")
        f.write(f"unpruned frozen-model test accuracy: {orig_test_acc:.4f}\n")
        f.write("-" * 96 + "\n")
        f.write(f"{'mode':>10} {'lambda':>7} {'seed':>5} | {'% pruned':>9} | "
                f"{'test_acc':>9} | {'drop_pp':>8} | {'efficiency':>10}\n")
        f.write("-" * 96 + "\n")
        for r in all_results:
            f.write(f"{r['mode']:>10} {r['lambda']:>7} {r['seed']:>5} | "
                    f"{r['pct_pruned']:>8.2f}% | {r['test_acc']:>9.4f} | "
                    f"{r['acc_drop_pp']:>7.2f}p | {r['efficiency']:>10.2f}\n")
    print(f"\nResults → {args.out_dir}/")


if __name__ == "__main__":
    main()
