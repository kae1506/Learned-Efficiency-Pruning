"""
Interpretability metrics for a trained pruner applied to an MLP.

The single entry point `analyze_pruner` returns a structured dict and
optionally saves a multi-panel plot showing per-layer prune density,
the alive/dead bitmap per layer, and the activation distribution split
by kept-vs-pruned under the *original* (un-pruned) model.

The "under original model" choice is deliberate: pruned-model activations
are zero by construction for dead neurons. We want to know whether the
pruner is killing intrinsically low-firing neurons, or making a more
sophisticated joint decision.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.prune_train import get_hidden_weights


@torch.no_grad()
def _collect_mean_activations(model: nn.Module, loader, n_batches: int, device):
    """Per-layer mean post-ReLU activation on the ORIGINAL (un-pruned) model."""
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    hidden  = linears[:-1]
    sums    = [torch.zeros(L.out_features, device=device) for L in hidden]
    count   = 0
    for i, (x, _) in enumerate(loader):
        if i >= n_batches:
            break
        x = x.to(device).view(x.size(0), -1)
        count += x.size(0)
        h = x
        for k, L in enumerate(hidden):
            h = F.relu(L(h))
            sums[k] += h.sum(dim=0)
    return [s / count for s in sums]


@torch.no_grad()
def evaluate_with_gates(
    model: nn.Module,
    gates: list[torch.Tensor],
    test_loader,
    device,
) -> float:
    """
    Apply binary `gates` (1 = keep, 0 = prune) to a deep copy of `model`
    by zeroing the corresponding weight rows and biases, then evaluate
    classification accuracy on the full test set.
    """
    import copy as _copy
    pruned = _copy.deepcopy(model)
    linears = [m for m in pruned.modules() if isinstance(m, nn.Linear)]
    hidden  = linears[:-1]
    for L, g in zip(hidden, gates):
        g = g.to(device).bool()
        mask = ~g
        L.weight[mask, :] = 0.0
        L.bias[mask]      = 0.0
    pruned.eval()
    correct, total = 0, 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        preds = pruned(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
    return correct / total


def print_report(result: dict) -> None:
    """Print a clean human-readable summary of `analyze_pruner`'s output."""
    bar  = "=" * 64
    line = "-" * 64
    n_layers = len(result["per_layer"])
    total      = sum(p["n_total"]  for p in result["per_layer"])
    pruned_n   = sum(p["n_pruned"] for p in result["per_layer"])
    kept_n     = total - pruned_n

    print()
    print(bar)
    print("PRUNER INTERPRETABILITY REPORT")
    print(bar)
    print(f"  Total hidden neurons     : {total}")
    print(f"  Kept                     : {kept_n} ({(kept_n/total)*100:5.2f}%)")
    print(f"  Pruned                   : {pruned_n} ({(pruned_n/total)*100:5.2f}%)")
    print()
    print("  Per-layer breakdown")
    print(f"  {'Layer':>6} | {'Total':>7} | {'Kept':>7} | {'Pruned':>7} | {'% Pruned':>9}")
    print("  " + line[:58])
    for i, p in enumerate(result["per_layer"]):
        print(f"  {i+1:>6} | {p['n_total']:>7} | {p['n_kept']:>7} | "
              f"{p['n_pruned']:>7} | {p['frac_pruned']*100:>8.2f}%")
    print()
    print("  Mean post-ReLU activation (under ORIGINAL, un-pruned model)")
    print(f"  {'Layer':>6} | {'Kept (alive)':>16} | {'Pruned (dead)':>16} | {'Ratio dead/alive':>18}")
    print("  " + line[:64])
    for i in range(n_layers):
        a = result["mean_act_alive_per_layer"][i]
        d = result["mean_act_dead_per_layer"][i]
        ratio = (d / a) if (a and not math.isnan(a) and a != 0) else float("nan")
        a_str = f"{a:.4f}" if not math.isnan(a) else "n/a"
        d_str = f"{d:.4f}" if not math.isnan(d) else "n/a"
        r_str = f"{ratio:.3f}" if not math.isnan(ratio) else "n/a"
        print(f"  {i+1:>6} | {a_str:>16} | {d_str:>16} | {r_str:>18}")
    print()
    print(f"  Overall mean activation — alive : {result['mean_act_alive_overall']:.4f}")
    print(f"  Overall mean activation — dead  : {result['mean_act_dead_overall']:.4f}")
    if result["mean_act_alive_overall"]:
        ratio = result["mean_act_dead_overall"] / result["mean_act_alive_overall"]
        print(f"  Ratio (dead / alive)            : {ratio:.3f}")
        if ratio < 0.3:
            interp = "pruner targets low-firing neurons (activation-like behaviour)"
        elif ratio < 0.7:
            interp = "pruner partially correlates with activation magnitude"
        else:
            interp = "pruner is killing high-firing neurons too — joint reasoning at work"
        print(f"  Interpretation                  : {interp}")
    print(bar)
    print()


@torch.no_grad()
def analyze_pruner(
    model: nn.Module,
    pruner: nn.Module | None = None,
    calib_loader=None,
    device=None,
    n_calib_batches: int = 5,
    save_plot: str | None = None,
    verbose: bool = False,
    gates: list[torch.Tensor] | None = None,
) -> dict:
    """
    Compute interpretability metrics for `pruner` applied to `model`.

    Returns a dict with:
      - frac_pruned_total          float, fraction of all hidden neurons gated off
      - per_layer                  list of {n_total, n_pruned, frac_pruned}
      - mean_act_alive_per_layer   list[float], mean activation of kept neurons (under original model)
      - mean_act_dead_per_layer    list[float], mean activation of pruned neurons (under original model)
      - mean_act_alive_overall     float, neuron-count-weighted average
      - mean_act_dead_overall      float, neuron-count-weighted average
      - gates                      list of binary numpy arrays per layer  (1 = kept, 0 = pruned)
      - mean_activations           list of numpy arrays per layer  (per-neuron mean activation)

    If `save_plot` is given, a multi-panel PNG is written to that path.
    """
    if gates is None and pruner is None:
        raise ValueError("Provide either `pruner` (called on model weights) or `gates` directly.")

    pruner_was_training = pruner.training if pruner is not None else False
    model_was_training  = model.training
    if pruner is not None:
        pruner.eval()
    model.eval()

    # ── gates ────────────────────────────────────────────────────────────────
    if gates is None:
        hidden_weights = get_hidden_weights(model)
        gates = pruner(hidden_weights)
    gates = [g.detach().cpu().bool() for g in gates]

    # ── activations under ORIGINAL model ─────────────────────────────────────
    mean_acts = _collect_mean_activations(model, calib_loader, n_calib_batches, device)
    mean_acts = [a.detach().cpu() for a in mean_acts]

    # ── per-layer metrics ────────────────────────────────────────────────────
    per_layer    = []
    alive_means  = []
    dead_means   = []
    weighted_alive_num, weighted_alive_den = 0.0, 0
    weighted_dead_num,  weighted_dead_den  = 0.0, 0

    for g, a in zip(gates, mean_acts):
        n_total  = int(g.numel())
        n_kept   = int(g.sum())
        n_pruned = n_total - n_kept

        a_alive = a[g].mean().item()  if n_kept   > 0 else float("nan")
        a_dead  = a[~g].mean().item() if n_pruned > 0 else float("nan")

        per_layer.append({
            "n_total"     : n_total,
            "n_kept"      : n_kept,
            "n_pruned"    : n_pruned,
            "frac_pruned" : n_pruned / n_total,
        })
        alive_means.append(a_alive)
        dead_means.append(a_dead)

        if not math.isnan(a_alive):
            weighted_alive_num += a_alive * n_kept
            weighted_alive_den += n_kept
        if not math.isnan(a_dead):
            weighted_dead_num  += a_dead * n_pruned
            weighted_dead_den  += n_pruned

    total          = sum(p["n_total"]   for p in per_layer)
    total_pruned   = sum(p["n_pruned"]  for p in per_layer)
    frac_total     = total_pruned / total
    overall_alive  = weighted_alive_num / max(weighted_alive_den, 1)
    overall_dead   = weighted_dead_num  / max(weighted_dead_den,  1)

    result = {
        "frac_pruned_total"        : frac_total,
        "per_layer"                : per_layer,
        "mean_act_alive_per_layer" : alive_means,
        "mean_act_dead_per_layer"  : dead_means,
        "mean_act_alive_overall"   : overall_alive,
        "mean_act_dead_overall"    : overall_dead,
        "gates"                    : [g.numpy() for g in gates],
        "mean_activations"         : [a.numpy() for a in mean_acts],
    }

    # ── optional plot ────────────────────────────────────────────────────────
    if save_plot is not None:
        _plot_analysis(result, save_plot)

    if verbose:
        print_report(result)

    if pruner_was_training and pruner is not None:
        pruner.train()
    if model_was_training:
        model.train()

    return result


def _plot_analysis(result: dict, save_path: str) -> None:
    n_layers = len(result["per_layer"])
    n_cols   = max(2, n_layers)
    fig = plt.figure(figsize=(4.5 * n_cols, 7))
    gs  = fig.add_gridspec(2, n_cols, hspace=0.45, wspace=0.30)

    # ── Row 1: per-layer % pruned (one bar plot spanning all columns) ─────
    ax_bars = fig.add_subplot(gs[0, :])
    layers   = [f"L{i+1}\n({p['n_total']} units)" for i, p in enumerate(result["per_layer"])]
    pruned_p = [p["frac_pruned"] * 100 for p in result["per_layer"]]
    bars = ax_bars.bar(layers, pruned_p, color="#c0392b", alpha=0.85)
    ax_bars.axhline(result["frac_pruned_total"] * 100, color="k", ls="--", alpha=0.6,
                    label=f"overall {result['frac_pruned_total']*100:.1f}%")
    for b, p in zip(bars, pruned_p):
        ax_bars.text(b.get_x() + b.get_width()/2, p + 1, f"{p:.1f}%",
                     ha="center", fontsize=10)
    ax_bars.set_ylim(0, 105)
    ax_bars.set_ylabel("Neurons pruned (%)")
    ax_bars.set_title("Pruning fraction per layer", fontweight="bold")
    ax_bars.legend(loc="upper right")
    ax_bars.grid(axis="y", alpha=0.3)

    # ── Row 2: activation histogram split by alive/dead (per layer) ───────
    for i, (g, a, p) in enumerate(zip(result["gates"],
                                      result["mean_activations"],
                                      result["per_layer"])):
        ax = fig.add_subplot(gs[1, i])
        a_alive = a[g.astype(bool)]
        a_dead  = a[~g.astype(bool)]
        bins = np.linspace(0, max(a.max(), 1e-6), 40)
        if a_alive.size:
            ax.hist(a_alive, bins=bins, color="#27ae60", alpha=0.7,
                    label=f"kept (n={a_alive.size}, mean={a_alive.mean():.3f})")
        if a_dead.size:
            ax.hist(a_dead, bins=bins, color="#c0392b", alpha=0.7,
                    label=f"pruned (n={a_dead.size}, mean={a_dead.mean():.3f})")
        ax.set_xlabel("Mean post-ReLU activation (original model)")
        ax.set_ylabel("Neurons")
        ax.set_title(f"Layer {i+1} activation distribution", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Pruner interpretability — {result['frac_pruned_total']*100:.1f}% pruned overall  "
        f"|  mean act. alive={result['mean_act_alive_overall']:.3f}  "
        f"vs dead={result['mean_act_dead_overall']:.3f}",
        fontsize=13, fontweight="bold", y=0.995,
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
