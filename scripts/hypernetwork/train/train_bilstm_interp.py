"""
Train the BiLSTM hypernetwork pruner at sw = 0.3, then run
interpretability analysis on the resulting gates.

Outputs (under experiments/latest/hypernetwork/interp_sw<sw>/):
  - training_curves.png   pruner training (loss, accuracies, % pruned)
  - interp.png            interpretability panel (per-layer % pruned,
                          density bitmap per layer, activation histograms)
  - summary.txt           text report (also printed to stdout)

Run from project root:
    venv/bin/python scripts/train_bilstm_interp.py
"""

import os
import sys
import argparse
import datetime
import yaml
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(".")
from src.model import MLP
from src.dataset import get_mnist_loaders
from src.pruners.bilstm import Pruner
from src.prune_train import pruner_step, get_hidden_weights
from src.interpretability import analyze_pruner, print_report, evaluate_with_gates


# ── config ────────────────────────────────────────────────────────────────────
CONFIG_PATH      = "configs/config.yaml"
CKPT_PATH        = "experiments/checkpoints/mnist_model.pt"
OUT_ROOT         = "experiments/latest/hypernetwork"

PRUNER_STEPS     = 1000
PRUNER_LR        = 1e-3
PRUNER_SAMPLES   = 64
CALIB_BATCHES    = 5


def load_model(device) -> tuple[MLP, dict]:
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    model = MLP(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt["config"]


def _smooth(values, window=20):
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo : i + 1]) / (i - lo + 1))
    return out


def plot_training(history: dict, sw: float, path: str):
    steps = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"BiLSTM pruner training (sw = {sw})",
                 fontsize=13, fontweight="bold")

    axes[0].plot(steps, history["loss"], alpha=0.25, color="steelblue")
    axes[0].plot(steps, _smooth(history["loss"]), color="steelblue", lw=2)
    axes[0].axhline(0, color="gray", ls="--", lw=0.8)
    axes[0].set_title("Pruner loss")
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)

    orig   = [a*100 for a in history["orig_acc"]]
    pruned = [a*100 for a in history["pruned_acc"]]
    axes[1].plot(steps, orig,   alpha=0.2, color="steelblue")
    axes[1].plot(steps, pruned, alpha=0.2, color="tomato")
    axes[1].plot(steps, _smooth(orig),   color="steelblue", lw=2, label="orig")
    axes[1].plot(steps, _smooth(pruned), color="tomato",    lw=2, label="pruned")
    axes[1].set_title("Mini-batch accuracy")
    axes[1].set_xlabel("Step"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    pct = [(1-g)*100 for g in history["avg_gate"]]
    axes[2].plot(steps, pct, alpha=0.25, color="darkorange")
    axes[2].plot(steps, _smooth(pct), color="darkorange", lw=2)
    axes[2].set_title("Neurons pruned (%)")
    axes[2].set_xlabel("Step"); axes[2].set_ylim(0, 100)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sw", type=float, default=0.30,
                        help="Sparsity weight in pruner loss.")
    args = parser.parse_args()
    sw = args.sw
    out_dir = f"{OUT_ROOT}/interp_sw{sw}"
    os.makedirs(out_dir, exist_ok=True)

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    base, mcfg = load_model(device)
    train_loader, test_loader = get_mnist_loaders(**cfg["data"])

    hidden_weights = get_hidden_weights(base)
    layer_shapes = [(w.shape[0], w.shape[1]) for w in hidden_weights]
    print(f"Base model: MLP {mcfg['input_dim']} -> "
          + " -> ".join(str(d) for d in mcfg['hidden_dims'])
          + f" -> {mcfg['output_dim']}")
    print(f"Prunable layers: {layer_shapes}")
    print(f"Sparsity weight: {sw}  |  Steps: {PRUNER_STEPS}\n")

    pruner = Pruner(layer_shapes).to(device)
    opt    = torch.optim.Adam(pruner.parameters(), lr=PRUNER_LR)

    history = {k: [] for k in ["loss", "orig_acc", "pruned_acc", "acc_drop", "avg_gate"]}
    it = iter(train_loader)
    for step in range(1, PRUNER_STEPS + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader); x, y = next(it)
        x, y = x[:PRUNER_SAMPLES].to(device), y[:PRUNER_SAMPLES].to(device)
        metrics = pruner_step(pruner, base, opt, x, y, sw)
        for k in history:
            history[k].append(metrics[k])
        if step % 100 == 0:
            print(f"  step {step:>4} | loss {metrics['loss']:+.4f}  "
                  f"pruned_acc {metrics['pruned_acc']:.4f}  "
                  f"avg_gate {metrics['avg_gate']:.4f}  "
                  f"% pruned {(1-metrics['avg_gate'])*100:5.2f}")

    plot_training(history, sw, f"{out_dir}/training_curves.png")

    # ── full-test-set evaluation of the final mask ────────────────────────────
    pruner.eval()
    with torch.no_grad():
        final_gates = pruner(get_hidden_weights(base))
    test_acc = evaluate_with_gates(base, final_gates, test_loader, device)
    baseline_acc = evaluate_with_gates(base,
                                       [torch.ones_like(g) for g in final_gates],
                                       test_loader, device)
    acc_drop = baseline_acc - test_acc
    print(f"\nFull test set:  baseline={baseline_acc*100:.2f}%  "
          f"pruned={test_acc*100:.2f}%  drop={acc_drop*100:.2f}pp\n")

    # ── interpretability ──────────────────────────────────────────────────────
    print("Running interpretability analysis...")
    result = analyze_pruner(
        model=base,
        pruner=pruner,
        calib_loader=train_loader,
        device=device,
        n_calib_batches=CALIB_BATCHES,
        save_plot=f"{out_dir}/interp.png",
        verbose=True,
    )
    print(f"Saved interpretability plot to {out_dir}/interp.png")

    # ── summary text file ─────────────────────────────────────────────────────
    summary_path = f"{out_dir}/summary.txt"
    lines = [
        "=" * 64,
        "BiLSTM PRUNER + INTERPRETABILITY",
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 64,
        "",
        f"Base model       : MLP {mcfg['input_dim']} -> "
            + " -> ".join(str(d) for d in mcfg['hidden_dims'])
            + f" -> {mcfg['output_dim']}",
        f"Checkpoint       : {CKPT_PATH}",
        f"Sparsity weight  : {sw}",
        f"Pruner steps     : {PRUNER_STEPS}",
        f"Calib batches    : {CALIB_BATCHES}",
        "",
        f"Full test set    : baseline={baseline_acc*100:.2f}%  "
            f"pruned={test_acc*100:.2f}%  drop={acc_drop*100:.2f}pp",
        "",
        f"OVERALL  pruned = {result['frac_pruned_total']*100:5.2f}%  "
            f"(kept {sum(p['n_kept'] for p in result['per_layer'])}, "
            f"pruned {sum(p['n_pruned'] for p in result['per_layer'])} of "
            f"{sum(p['n_total'] for p in result['per_layer'])})",
        "",
        "Per-layer breakdown:",
    ]
    for i, p in enumerate(result["per_layer"]):
        lines.append(
            f"  Layer {i+1}: {p['n_pruned']:>5}/{p['n_total']} pruned "
            f"({p['frac_pruned']*100:5.2f}%)"
        )
    lines += ["", "Mean post-ReLU activation (under ORIGINAL model):"]
    for i in range(len(result["per_layer"])):
        lines.append(
            f"  Layer {i+1}: alive {result['mean_act_alive_per_layer'][i]:.4f}  "
            f"dead {result['mean_act_dead_per_layer'][i]:.4f}"
        )
    lines += [
        "",
        f"OVERALL  alive mean activation = {result['mean_act_alive_overall']:.4f}",
        f"OVERALL  dead  mean activation = {result['mean_act_dead_overall']:.4f}",
    ]
    if result["mean_act_alive_overall"]:
        ratio = result["mean_act_dead_overall"] / result["mean_act_alive_overall"]
        lines.append(f"OVERALL  ratio (dead/alive)    = {ratio:.3f}")
    lines += ["", "=" * 64]
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
