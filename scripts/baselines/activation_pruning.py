"""
Activation-based pruning.

Algorithm:
  1. Run the frozen MNIST model over N calibration batches, recording
     the post-ReLU activation of every hidden neuron.
  2. Compute each neuron's mean activation across those batches.
  3. Shut off any neuron whose mean activation < lambda (default 0.4)
     by zeroing its incoming weight row and bias entry.
  4. Evaluate the pruned model on 5 test batches and report accuracy
     and number of neurons pruned.
"""

import os
import sys
import datetime
import torch
import torch.nn.functional as F
import yaml

sys.path.append(".")
from src.model import MLP
from src.dataset import get_mnist_loaders


# ── helpers ──────────────────────────────────────────────────────────────────

def load_mnist_model(path: str, device) -> MLP:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = MLP(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def collect_mean_activations(
    model: MLP, loader, n_batches: int, device
) -> list[torch.Tensor]:
    """
    Run `n_batches` through the model and return the mean post-ReLU
    activation per hidden neuron for each hidden layer.

    Returns a list of tensors [mean_act_layer0, mean_act_layer1, ...]
    each of shape [num_neurons].
    """
    linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    hidden_linears = linears[:-1]

    # Accumulate sum and count separately so we don't need to store all activations
    sums   = [torch.zeros(L.out_features, device=device) for L in hidden_linears]
    counts = 0

    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= n_batches:
                break
            x = x.to(device).view(x.size(0), -1)
            counts += x.size(0)

            # Manual forward, capturing post-ReLU activations
            h = x
            for idx, linear in enumerate(hidden_linears):
                h = F.relu(linear(h))
                sums[idx] += h.sum(dim=0)

    return [s / counts for s in sums]


def apply_activation_mask(model: MLP, mean_acts: list[torch.Tensor], lam: float):
    """
    Zero out weights and bias for every hidden neuron whose mean
    activation is below `lam`. Modifies the model in-place.

    Returns a list of boolean masks (True = pruned) per hidden layer.
    """
    linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    hidden_linears = linears[:-1]
    masks = []

    with torch.no_grad():
        for linear, mean_act in zip(hidden_linears, mean_acts):
            pruned = mean_act < lam          # [out_features]  True = shut off
            linear.weight[pruned, :] = 0.0
            linear.bias[pruned]      = 0.0
            masks.append(pruned)

    return masks


@torch.no_grad()
def evaluate_batches(model: MLP, loader, n_batches: int, device) -> float:
    correct, total = 0, 0
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
    return correct / total


# ── main ─────────────────────────────────────────────────────────────────────

def main(
    config_path: str  = "configs/config.yaml",
    ckpt_path: str    = "experiments/checkpoints/mnist_model.pt",
    lam: float        = 0.5,
    calib_batches: int = 5,
    eval_batches: int  = 5,
):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_mnist_model(ckpt_path, device)
    print("Loaded MNIST model.")

    train_loader, test_loader = get_mnist_loaders(**cfg["data"])

    # ── Step 1: baseline accuracy (before pruning) ────────────────────────────
    baseline_acc = evaluate_batches(model, test_loader, eval_batches, device)
    print(f"Baseline accuracy (first {eval_batches} test batches): {baseline_acc*100:.2f}%")

    # ── Step 2: collect mean activations on calibration batches ─────────────
    print(f"\nCollecting activations over {calib_batches} calibration batches...")
    mean_acts = collect_mean_activations(model, train_loader, calib_batches, device)

    for i, ma in enumerate(mean_acts):
        print(f"  Layer {i+1}: min={ma.min():.4f}  mean={ma.mean():.4f}  max={ma.max():.4f}")

    # ── Step 3: apply threshold mask ─────────────────────────────────────────
    print(f"\nPruning neurons with mean activation < {lam} ...")
    masks = apply_activation_mask(model, mean_acts, lam)

    total_neurons = sum(m.numel() for m in masks)
    pruned_neurons = sum(m.sum().item() for m in masks)
    for i, m in enumerate(masks):
        print(f"  Layer {i+1}: {int(m.sum())}/{m.numel()} neurons pruned "
              f"({m.float().mean()*100:.1f}%)")

    # ── Step 4: evaluate pruned model ────────────────────────────────────────
    pruned_acc = evaluate_batches(model, test_loader, eval_batches, device)
    print(f"\nPruned accuracy  (first {eval_batches} test batches): {pruned_acc*100:.2f}%")
    print(f"Accuracy drop: {(baseline_acc - pruned_acc)*100:.2f}%")

    # ── Step 5: save summary ──────────────────────────────────────────────────
    summary_path = "experiments/latest/baselines/activation/summary.txt"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    arch = (f"{cfg['model']['input_dim']} -> "
            + " -> ".join(str(d) for d in cfg['model']['hidden_dims'])
            + f" -> {cfg['model']['output_dim']}")

    lines = [
        "=" * 60,
        "ACTIVATION PRUNING SUMMARY",
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "ALGORITHM",
        "  For each hidden neuron, compute its mean post-ReLU activation",
        f"  over {calib_batches} calibration batches of training data.",
        f"  Shut off any neuron with mean activation < lambda={lam}",
        "  by zeroing its incoming weight row and bias.",
        "",
        "BASE MODEL",
        f"  Architecture : MLP {arch}",
        f"  Checkpoint   : {ckpt_path}",
        "",
        "RESULTS",
        f"  Calibration batches : {calib_batches}",
        f"  Evaluation batches  : {eval_batches}",
        f"  Lambda (threshold)  : {lam}",
        f"  Baseline accuracy   : {baseline_acc*100:.2f}%",
        f"  Pruned accuracy     : {pruned_acc*100:.2f}%",
        f"  Accuracy drop       : {(baseline_acc - pruned_acc)*100:.2f}%",
        f"  Neurons pruned      : {int(pruned_neurons)} / {total_neurons} "
            f"({pruned_neurons/total_neurons*100:.1f}%)",
    ]
    for i, m in enumerate(masks):
        lines.append(f"    Layer {i+1}: {int(m.sum())}/{m.numel()} "
                     f"({m.float().mean()*100:.1f}%)")
    lines += ["", "=" * 60]

    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
