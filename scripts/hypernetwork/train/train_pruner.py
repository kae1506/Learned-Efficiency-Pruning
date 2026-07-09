import os
import sys
import copy
import yaml
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.append(".")
from src.model import MLP
from src.dataset import get_mnist_loaders
from src.pruners.bilstm import Pruner
from src.prune_train import pruner_step, get_hidden_weights


def load_mnist_model(path: str, device) -> MLP:
    ckpt = torch.load(path, map_location=device)
    model = MLP(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _smooth(values: list, window: int = 20) -> list:
    """Simple moving average for noisy step-level curves."""
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo : i + 1]) / (i - lo + 1))
    return out


def plot_pruner(history: dict, save_path: str = "experiments/latest/hypernetwork/training/plot.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    steps = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Pruner Training", fontsize=13, fontweight="bold")

    # --- Loss ---
    axes[0].plot(steps, history["loss"], alpha=0.25, color="steelblue")
    axes[0].plot(steps, _smooth(history["loss"]), color="steelblue", linewidth=2, label="smoothed")
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_title("Pruner Loss  (CE_pruned − CE_orig)")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Accuracy ---
    orig_pct   = [a * 100 for a in history["orig_acc"]]
    pruned_pct = [a * 100 for a in history["pruned_acc"]]
    axes[1].plot(steps, orig_pct,   alpha=0.2, color="steelblue")
    axes[1].plot(steps, pruned_pct, alpha=0.2, color="tomato")
    axes[1].plot(steps, _smooth(orig_pct),   color="steelblue", linewidth=2, label="Original")
    axes[1].plot(steps, _smooth(pruned_pct), color="tomato",    linewidth=2, label="Pruned")
    axes[1].set_title("Accuracy on Mini-Batch")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # --- % Pruned (1 - avg_gate) ---
    pct_pruned = [(1 - g) * 100 for g in history["avg_gate"]]
    axes[2].plot(steps, pct_pruned, alpha=0.25, color="darkorange")
    axes[2].plot(steps, _smooth(pct_pruned), color="darkorange", linewidth=2)
    axes[2].set_title("Nodes Pruned (%)")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("% nodes shut off")
    axes[2].set_ylim(0, 100)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {save_path}")


def main(config_path: str = "configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    mnist_model = load_mnist_model("experiments/checkpoints/mnist_model.pt", device)
    print("Loaded MNIST model.")

    train_loader, _ = get_mnist_loaders(**cfg["data"])

    pcfg = cfg["pruner"]
    N = pcfg["samples_per_step"]
    steps = pcfg["steps"]
    lr = pcfg["lr"]
    sparsity_weight = pcfg["sparsity_weight"]

    # Build pruner from MNIST model's hidden layer shapes
    hidden_weights = get_hidden_weights(mnist_model)
    layer_shapes = [(w.shape[0], w.shape[1]) for w in hidden_weights]
    print(f"Pruning layers with shapes: {layer_shapes}")

    pruner = Pruner(layer_shapes).to(device)
    optimizer = torch.optim.Adam(pruner.parameters(), lr=lr)

    history = {k: [] for k in ["loss", "orig_acc", "pruned_acc", "acc_drop", "avg_gate"]}
    data_iter = iter(train_loader)

    for step in range(1, steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x[:N].to(device), y[:N].to(device)
        metrics = pruner_step(pruner, mnist_model, optimizer, x, y, sparsity_weight)

        for k in history:
            history[k].append(metrics[k])

        if step % 50 == 0:
            print(
                f"Step {step:>4} | loss {metrics['loss']:+.4f} | "
                f"orig_acc {metrics['orig_acc']:.4f} | "
                f"pruned_acc {metrics['pruned_acc']:.4f} | "
                f"acc_drop {metrics['acc_drop']:+.4f} | "
                f"avg_gate {metrics['avg_gate']:.4f}"
            )

    # Apply final gates to a copy of the model and collect post-prune activations
    pruner.eval()
    with torch.no_grad():
        final_gates = pruner(get_hidden_weights(mnist_model))
    post_prune_acts = collect_post_prune_activations(
        mnist_model, final_gates, train_loader, n_batches=5, device=device
    )

    plot_pruner(history, save_path="experiments/latest/hypernetwork/training/plot.png")
    write_summary(cfg, layer_shapes, history, post_prune_acts=post_prune_acts)


def collect_post_prune_activations(
    model: MLP,
    gates: list[torch.Tensor],
    loader,
    n_batches: int,
    device,
) -> list[dict]:
    """
    Apply final binary gates to a copy of the model, run n_batches of data
    through it, and return per-layer mean post-ReLU activation statistics.
    Pruned neurons (gate=0) will naturally have activation=0.
    """
    pruned_model = copy.deepcopy(model)
    linears = [m for m in pruned_model.modules() if isinstance(m, torch.nn.Linear)]
    with torch.no_grad():
        for linear, gate in zip(linears[:-1], gates):
            shut = gate == 0
            linear.weight[shut, :] = 0.0
            linear.bias[shut]      = 0.0

    pruned_model.eval()
    hidden_linears = linears[:-1]
    sums   = [torch.zeros(L.out_features, device=device) for L in hidden_linears]
    counts = 0

    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= n_batches:
                break
            x = x.to(device).view(x.size(0), -1)
            counts += x.size(0)
            h = x
            for idx, linear in enumerate(hidden_linears):
                h = F.relu(linear(h))
                sums[idx] += h.sum(dim=0)

    layer_stats = []
    for s, gate in zip(sums, gates):
        mean_act = s / counts                          # [num_nodes]
        kept_mask = gate.bool()
        layer_stats.append({
            "mean_all":   mean_act.mean().item(),
            "mean_kept":  mean_act[kept_mask].mean().item() if kept_mask.any() else 0.0,
            "mean_pruned": mean_act[~kept_mask].mean().item() if (~kept_mask).any() else 0.0,
            "per_node":   mean_act.tolist(),
            "num_kept":   int(kept_mask.sum()),
            "num_pruned": int((~kept_mask).sum()),
        })
    return layer_stats


def write_summary(cfg: dict, layer_shapes: list, history: dict,
                  post_prune_acts: list[dict] | None = None,
                  path: str = "experiments/latest/hypernetwork/training/summary.txt"):
    import datetime
    final_gate   = history["avg_gate"][-1]
    pct_pruned   = (1 - final_gate) * 100
    avg_orig_acc = sum(history["orig_acc"]) / len(history["orig_acc"]) * 100
    avg_pruned_acc = sum(history["pruned_acc"]) / len(history["pruned_acc"]) * 100
    avg_drop     = sum(history["acc_drop"]) / len(history["acc_drop"]) * 100
    min_gate     = min(history["avg_gate"])
    final_loss   = history["loss"][-1]

    lines = [
        "=" * 60,
        "PRUNER EXPERIMENT SUMMARY",
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "BASE MODEL",
        f"  Architecture : MLP {cfg['model']['input_dim']} -> "
            + " -> ".join(str(d) for d in cfg['model']['hidden_dims'])
            + f" -> {cfg['model']['output_dim']}",
        f"  Dropout      : {cfg['model']['dropout']}",
        f"  Train epochs : {cfg['training']['epochs']}",
        f"  LR           : {cfg['training']['lr']}",
        "",
        "PRUNER CONFIG",
        f"  Steps              : {cfg['pruner']['steps']}",
        f"  Samples / step     : {cfg['pruner']['samples_per_step']}",
        f"  LR                 : {cfg['pruner']['lr']}",
        f"  Sparsity weight    : {cfg['pruner']['sparsity_weight']}",
        f"  Prunable layers    : {layer_shapes}",
        f"  Gate type          : Hard binary (Straight-Through Estimator)",
        "",
        "RESULTS",
        f"  Final avg gate     : {final_gate:.4f}  (1.0 = all kept, 0.0 = all pruned)",
        f"  Nodes pruned       : {pct_pruned:.1f}%",
        f"  Peak pruning       : {(1 - min_gate)*100:.1f}%",
        f"  Avg orig acc       : {avg_orig_acc:.2f}%",
        f"  Avg pruned acc     : {avg_pruned_acc:.2f}%",
        f"  Avg acc drop       : {avg_drop:.2f}%",
        f"  Final pruner loss  : {final_loss:.4f}",
    ]

    if post_prune_acts:
        lines += ["", "POST-PRUNE ACTIVATIONS  (mean over 5 train batches)"]
        for i, stats in enumerate(post_prune_acts):
            lines += [
                f"  Layer {i+1}:",
                f"    Kept nodes    : {stats['num_kept']}   mean activation = {stats['mean_kept']:.4f}",
                f"    Pruned nodes  : {stats['num_pruned']}   mean activation = {stats['mean_pruned']:.4f}  (expected 0)",
                f"    All nodes avg : {stats['mean_all']:.4f}",
            ]

    lines += ["", "=" * 60]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved summary to {path}")


if __name__ == "__main__":
    main()
