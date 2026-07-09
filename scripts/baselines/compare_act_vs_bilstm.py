"""
Side-by-side comparison: activation pruning vs BiLSTM hypernetwork pruning.

Both methods start from the SAME trained MNIST MLP checkpoint
(experiments/checkpoints/mnist_model.pt) and are evaluated on the full
MNIST test set after pruning.

Activation pruning:
  Sweep the threshold lambda in LAMBDAS. For each lambda, copy the model,
  zero out neurons whose mean post-ReLU activation (over CALIB_BATCHES of
  training data) falls below lambda, then evaluate.

BiLSTM hypernetwork pruning:
  Sweep the sparsity weight in SPARSITY_WEIGHTS. For each weight, train a
  fresh BiLSTM pruner for PRUNER_STEPS steps on the same base model, apply
  the final binary gates, then evaluate.

Output: experiments/latest/baselines/activation_vs_bilstm/{plot.png, summary.txt}
Run from project root: venv/bin/python scripts/compare_act_vs_bilstm.py
"""

import os
import sys
import copy
import datetime
import yaml
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(".")
from src.model import MLP
from src.dataset import get_mnist_loaders
from src.pruners.bilstm import Pruner
from src.prune_train import pruner_step, get_hidden_weights


# ── config ────────────────────────────────────────────────────────────────────
CONFIG_PATH    = "configs/config.yaml"
CKPT_PATH      = "experiments/checkpoints/mnist_model.pt"
OUT_DIR        = "experiments/latest/baselines/activation_vs_bilstm"

LAMBDAS           = [0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00, 1.50]
SPARSITY_WEIGHTS  = [0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50]

CALIB_BATCHES   = 5
PRUNER_STEPS    = 600
PRUNER_LR       = 1e-3
PRUNER_SAMPLES  = 64


# ── helpers ───────────────────────────────────────────────────────────────────

def load_model(device) -> tuple[MLP, dict]:
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    model = MLP(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt["config"]


@torch.no_grad()
def eval_full(model: MLP, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
    return correct / total


@torch.no_grad()
def collect_mean_activations(model: MLP, loader, n_batches: int, device):
    linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    hidden  = linears[:-1]
    sums    = [torch.zeros(L.out_features, device=device) for L in hidden]
    n = 0
    for i, (x, _) in enumerate(loader):
        if i >= n_batches:
            break
        x = x.to(device).view(x.size(0), -1)
        n += x.size(0)
        h = x
        for k, L in enumerate(hidden):
            h = F.relu(L(h))
            sums[k] += h.sum(dim=0)
    return [s / n for s in sums]


def apply_activation_prune(base_model: MLP, mean_acts, lam: float):
    """Return (pruned_model_copy, frac_pruned, layer_breakdown)."""
    m = copy.deepcopy(base_model)
    linears = [x for x in m.modules() if isinstance(x, torch.nn.Linear)]
    hidden  = linears[:-1]
    total   = sum(L.out_features for L in hidden)
    per_layer = []
    pruned_n = 0
    with torch.no_grad():
        for L, a in zip(hidden, mean_acts):
            mask = a < lam
            L.weight[mask, :] = 0.0
            L.bias[mask]      = 0.0
            per_layer.append((int(mask.sum()), L.out_features))
            pruned_n += int(mask.sum())
    return m, pruned_n / total, per_layer


def train_bilstm_pruner(base_model: MLP, train_loader, sparsity_w: float,
                        steps: int, device):
    hidden_weights = get_hidden_weights(base_model)
    layer_shapes = [(w.shape[0], w.shape[1]) for w in hidden_weights]
    pruner = Pruner(layer_shapes).to(device)
    opt = torch.optim.Adam(pruner.parameters(), lr=PRUNER_LR)

    it = iter(train_loader)
    for step in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)
        x, y = x[:PRUNER_SAMPLES].to(device), y[:PRUNER_SAMPLES].to(device)
        pruner_step(pruner, base_model, opt, x, y, sparsity_w)

    pruner.eval()
    with torch.no_grad():
        gates = pruner(get_hidden_weights(base_model))
    return gates


def apply_bilstm_prune(base_model: MLP, gates):
    m = copy.deepcopy(base_model)
    linears = [x for x in m.modules() if isinstance(x, torch.nn.Linear)]
    hidden  = linears[:-1]
    total   = sum(L.out_features for L in hidden)
    per_layer = []
    pruned_n = 0
    with torch.no_grad():
        for L, g in zip(hidden, gates):
            mask = (g == 0)
            L.weight[mask, :] = 0.0
            L.bias[mask]      = 0.0
            per_layer.append((int(mask.sum()), L.out_features))
            pruned_n += int(mask.sum())
    return m, pruned_n / total, per_layer


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    base, mcfg = load_model(device)
    train_loader, test_loader = get_mnist_loaders(**cfg["data"])

    baseline_acc = eval_full(base, test_loader, device)
    print(f"Baseline test accuracy (full set): {baseline_acc*100:.2f}%\n")

    # ── activation sweep ──────────────────────────────────────────────────────
    print("=== Activation pruning sweep ===")
    mean_acts = collect_mean_activations(base, train_loader, CALIB_BATCHES, device)
    print(f"  Calibration: {CALIB_BATCHES} batches\n")

    act_results = []
    for lam in LAMBDAS:
        pruned_model, frac, per_layer = apply_activation_prune(base, mean_acts, lam)
        acc = eval_full(pruned_model, test_loader, device)
        drop = baseline_acc - acc
        act_results.append({
            "lambda": lam, "frac_pruned": frac, "acc": acc, "drop": drop,
            "per_layer": per_layer,
        })
        print(f"  lambda={lam:>5.2f} | pruned={frac*100:5.1f}% | acc={acc*100:5.2f}% | drop={drop*100:5.2f}%")
    print()

    # ── BiLSTM sweep ──────────────────────────────────────────────────────────
    print("=== BiLSTM hypernetwork pruning sweep ===")
    print(f"  Steps per run: {PRUNER_STEPS}\n")

    bilstm_results = []
    for sw in SPARSITY_WEIGHTS:
        t0 = datetime.datetime.now()
        gates = train_bilstm_pruner(base, train_loader, sw, PRUNER_STEPS, device)
        pruned_model, frac, per_layer = apply_bilstm_prune(base, gates)
        acc = eval_full(pruned_model, test_loader, device)
        drop = baseline_acc - acc
        dt = (datetime.datetime.now() - t0).total_seconds()
        bilstm_results.append({
            "sparsity_w": sw, "frac_pruned": frac, "acc": acc, "drop": drop,
            "per_layer": per_layer,
        })
        print(f"  sw={sw:>5.2f} | pruned={frac*100:5.1f}% | acc={acc*100:5.2f}% | drop={drop*100:5.2f}%  [{dt:5.1f}s]")
    print()

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Activation pruning vs BiLSTM hypernetwork pruning  (same trained model)",
                 fontsize=13, fontweight="bold")

    # Pareto: acc vs % pruned
    ax = axes[0]
    ax.plot([r["frac_pruned"]*100 for r in act_results],
            [r["acc"]*100         for r in act_results],
            "o-", color="#c0392b", lw=2, ms=8, label="Activation (sweep λ)")
    ax.plot([r["frac_pruned"]*100 for r in bilstm_results],
            [r["acc"]*100         for r in bilstm_results],
            "s-", color="#2980b9", lw=2, ms=8, label="BiLSTM (sweep sparsity wt)")
    ax.axhline(baseline_acc*100, color="k", ls="--", alpha=0.5, label=f"baseline {baseline_acc*100:.2f}%")
    ax.set_xlabel("Neurons pruned (%)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Pareto frontier")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")

    # Accuracy drop vs % pruned
    ax = axes[1]
    ax.plot([r["frac_pruned"]*100 for r in act_results],
            [r["drop"]*100        for r in act_results],
            "o-", color="#c0392b", lw=2, ms=8, label="Activation")
    ax.plot([r["frac_pruned"]*100 for r in bilstm_results],
            [r["drop"]*100        for r in bilstm_results],
            "s-", color="#2980b9", lw=2, ms=8, label="BiLSTM")
    ax.axhline(0,   color="k",       ls="--", alpha=0.4)
    ax.axhline(1.0, color="#e67e22", ls=":",  alpha=0.6, label="1% drop")
    ax.axhline(3.0, color="#c0392b", ls=":",  alpha=0.6, label="3% drop")
    ax.set_xlabel("Neurons pruned (%)")
    ax.set_ylabel("Accuracy drop (pp)")
    ax.set_title("Accuracy drop vs sparsity")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")

    fig.tight_layout()
    plot_path = f"{OUT_DIR}/plot.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {plot_path}")

    # ── summary ───────────────────────────────────────────────────────────────
    arch = (f"{mcfg['input_dim']} -> "
            + " -> ".join(str(d) for d in mcfg['hidden_dims'])
            + f" -> {mcfg['output_dim']}")
    total_hidden = sum(mcfg['hidden_dims'])

    def best_under(rows, drop_pct):
        kept = [r for r in rows if r["drop"]*100 <= drop_pct]
        return max(kept, key=lambda r: r["frac_pruned"]) if kept else None

    lines = [
        "=" * 78,
        "ACTIVATION vs BiLSTM PRUNING — same trained model",
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 78,
        "",
        "BASE MODEL",
        f"  Architecture        : MLP {arch}",
        f"  Checkpoint          : {CKPT_PATH}",
        f"  Total hidden neurons: {total_hidden}",
        f"  Baseline test acc   : {baseline_acc*100:.2f}%  (full MNIST test set)",
        "",
        "ACTIVATION PRUNING (zero out neurons with mean post-ReLU < lambda)",
        f"  Calibration batches : {CALIB_BATCHES}",
        f"  Lambdas swept       : {LAMBDAS}",
        "",
        f"  {'Lambda':>7} | {'% Pruned':>9} | {'Test Acc':>9} | {'Drop':>7} | Layer breakdown",
        "  " + "-" * 70,
    ]
    for r in act_results:
        lb = "  ".join(f"L{i}:{p[0]}/{p[1]}" for i, p in enumerate(r["per_layer"]))
        lines.append(
            f"  {r['lambda']:>7.2f} | {r['frac_pruned']*100:8.2f}% | "
            f"{r['acc']*100:8.2f}% | {r['drop']*100:6.2f}% | {lb}"
        )

    lines += [
        "",
        "BiLSTM HYPERNETWORK PRUNING (STE binary gates from trained pruner)",
        f"  Pruner steps        : {PRUNER_STEPS}",
        f"  Pruner samples/step : {PRUNER_SAMPLES}",
        f"  Pruner LR           : {PRUNER_LR}",
        f"  Sparsity weights    : {SPARSITY_WEIGHTS}",
        "",
        f"  {'Spars wt':>8} | {'% Pruned':>9} | {'Test Acc':>9} | {'Drop':>7} | Layer breakdown",
        "  " + "-" * 70,
    ]
    for r in bilstm_results:
        lb = "  ".join(f"L{i}:{p[0]}/{p[1]}" for i, p in enumerate(r["per_layer"]))
        lines.append(
            f"  {r['sparsity_w']:>8.2f} | {r['frac_pruned']*100:8.2f}% | "
            f"{r['acc']*100:8.2f}% | {r['drop']*100:6.2f}% | {lb}"
        )

    lines += ["", "=" * 78, "BEST PRUNE AT FIXED ACCURACY-DROP TOLERANCES", "=" * 78]
    for tol in (0.5, 1.0, 3.0):
        a = best_under(act_results,    tol)
        b = best_under(bilstm_results, tol)
        lines += [f"\n  Drop tolerance: <= {tol}%"]
        if a:
            lines.append(f"    Activation : lambda={a['lambda']:.2f}  pruned={a['frac_pruned']*100:5.2f}%  acc={a['acc']*100:5.2f}%")
        else:
            lines.append(f"    Activation : no setting under {tol}% drop")
        if b:
            lines.append(f"    BiLSTM     : sw={b['sparsity_w']:.2f}      pruned={b['frac_pruned']*100:5.2f}%  acc={b['acc']*100:5.2f}%")
        else:
            lines.append(f"    BiLSTM     : no setting under {tol}% drop")

    lines += ["", "=" * 78]

    summary_path = f"{OUT_DIR}/summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
