"""
Sequential pruning experiment.

Stage 1 — Activation pruning:
  Shut off neurons whose mean post-ReLU activation over calibration
  batches falls below lambda. Modifies model weights in-place.

Stage 2 — Learned pruning (on top of Stage 1):
  Train the Pruner network on the already-activation-pruned model.
  The learned pruner finds additional neurons to gate off via STE.

Reports cumulative neurons pruned and accuracy after each stage.
"""

import os
import sys
import datetime
import torch
import yaml

sys.path.append(".")
sys.path.insert(0, "scripts")

from src.model import MLP
from src.dataset import get_mnist_loaders
from src.pruners.bilstm import Pruner
from src.prune_train import pruner_step, get_hidden_weights

# Import activation-pruning helpers directly from the sibling script
from activation_pruning import (
    load_mnist_model,
    collect_mean_activations,
    apply_activation_mask,
    evaluate_batches,
)


# ── pruned-neuron counting ────────────────────────────────────────────────────

def count_zero_rows(model: MLP) -> tuple[int, int]:
    """Count hidden-layer neurons whose entire incoming weight row is zero."""
    linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    pruned, total = 0, 0
    for linear in linears[:-1]:
        zero_rows = (linear.weight.abs().sum(dim=1) == 0)
        pruned += zero_rows.sum().item()
        total  += linear.weight.shape[0]
    return int(pruned), total


def gates_to_pruned(gates: list[torch.Tensor]) -> tuple[int, int]:
    """Count neurons gated to 0 by the learned pruner."""
    pruned = sum((g == 0).sum().item() for g in gates)
    total  = sum(g.numel() for g in gates)
    return int(pruned), total


# ── learned pruning (single forward to get final gates, no training) ──────────

def train_and_apply_learned_pruner(
    model: MLP,
    train_loader,
    device,
    pcfg: dict,
) -> list[torch.Tensor]:
    """
    Train the Pruner on `model` (which may already be activation-pruned)
    and return the final binary gates.
    """
    hidden_weights = get_hidden_weights(model)
    layer_shapes   = [(w.shape[0], w.shape[1]) for w in hidden_weights]

    pruner    = Pruner(layer_shapes).to(device)
    optimizer = torch.optim.Adam(pruner.parameters(), lr=pcfg["lr"])

    N            = pcfg["samples_per_step"]
    steps        = pcfg["steps"]
    sparsity_w   = pcfg["sparsity_weight"]
    data_iter    = iter(train_loader)

    for step in range(1, steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x[:N].to(device), y[:N].to(device)
        pruner_step(pruner, model, optimizer, x, y, sparsity_w)

        if step % 200 == 0:
            with torch.no_grad():
                gates = pruner(get_hidden_weights(model))
            p, t = gates_to_pruned(gates)
            print(f"  [learned pruner] step {step:>4} | "
                  f"nodes pruned so far: {p}/{t} ({p/t*100:.1f}%)")

    # Final gates
    pruner.eval()
    with torch.no_grad():
        gates = pruner(get_hidden_weights(model))
    return gates


# ── main ──────────────────────────────────────────────────────────────────────

def main(
    config_path:   str   = "configs/config.yaml",
    ckpt_path:     str   = "experiments/checkpoints/mnist_model.pt",
    lam:           float = 0.5,
    calib_batches: int   = 5,
    eval_batches:  int   = 5,
):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    model = load_mnist_model(ckpt_path, device)
    train_loader, test_loader = get_mnist_loaders(**cfg["data"])

    total_hidden = sum(cfg["model"]["hidden_dims"])

    # ── Baseline ──────────────────────────────────────────────────────────────
    baseline_acc = evaluate_batches(model, test_loader, eval_batches, device)
    print(f"Baseline accuracy : {baseline_acc*100:.2f}%\n")

    # ── Stage 1: activation pruning ───────────────────────────────────────────
    print(f"[Stage 1] Activation pruning (lambda={lam}) ...")
    mean_acts = collect_mean_activations(model, train_loader, calib_batches, device)
    act_masks = apply_activation_mask(model, mean_acts, lam)

    act_pruned = int(sum(m.sum().item() for m in act_masks))
    act_pct    = act_pruned / total_hidden * 100
    acc_after_act = evaluate_batches(model, test_loader, eval_batches, device)

    for i, m in enumerate(act_masks):
        print(f"  Layer {i+1}: {int(m.sum())}/{m.numel()} pruned "
              f"({m.float().mean()*100:.1f}%)")
    print(f"  Total pruned  : {act_pruned}/{total_hidden} ({act_pct:.1f}%)")
    print(f"  Accuracy      : {acc_after_act*100:.2f}%  "
          f"(drop: {(baseline_acc-acc_after_act)*100:.2f}%)\n")

    # ── Stage 2: learned pruning on activation-pruned model ───────────────────
    print(f"[Stage 2] Learned pruning ({cfg['pruner']['steps']} steps) ...")
    gates = train_and_apply_learned_pruner(
        model, train_loader, device, cfg["pruner"]
    )

    # Apply gates: zero out any neuron the learned pruner shut off
    linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    with torch.no_grad():
        for linear, gate in zip(linears[:-1], gates):
            shut = gate == 0                      # neurons the pruner closed
            linear.weight[shut, :] = 0.0
            linear.bias[shut]      = 0.0

    # Count neurons pruned by learned pruner that weren't already zero
    learned_extra, _ = count_zero_rows(model)
    learned_extra   -= act_pruned   # additional neurons beyond Stage 1
    total_pruned     = act_pruned + max(learned_extra, 0)
    total_pct        = total_pruned / total_hidden * 100

    acc_after_learned = evaluate_batches(model, test_loader, eval_batches, device)

    lp, lt = gates_to_pruned(gates)
    print(f"  Learned pruner gated off : {lp}/{lt} ({lp/lt*100:.1f}%) neurons")
    print(f"  Additional (new) pruned  : {max(learned_extra,0)}/{total_hidden}")
    print(f"  Total pruned (cumulative): {total_pruned}/{total_hidden} ({total_pct:.1f}%)")
    print(f"  Accuracy                 : {acc_after_learned*100:.2f}%  "
          f"(drop from baseline: {(baseline_acc-acc_after_learned)*100:.2f}%)\n")

    # ── Summary file ──────────────────────────────────────────────────────────
    arch = (f"{cfg['model']['input_dim']} -> "
            + " -> ".join(str(d) for d in cfg['model']['hidden_dims'])
            + f" -> {cfg['model']['output_dim']}")

    lines = [
        "=" * 60,
        "SEQUENTIAL PRUNING SUMMARY",
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "BASE MODEL",
        f"  Architecture  : MLP {arch}",
        f"  Total hidden  : {total_hidden} neurons",
        "",
        "STAGE 1 — Activation Pruning",
        f"  Lambda        : {lam}",
        f"  Calib batches : {calib_batches}",
    ]
    for i, m in enumerate(act_masks):
        lines.append(f"  Layer {i+1}       : {int(m.sum())}/{m.numel()} pruned "
                     f"({m.float().mean()*100:.1f}%)")
    lines += [
        f"  Total pruned  : {act_pruned}/{total_hidden} ({act_pct:.1f}%)",
        f"  Accuracy      : {acc_after_act*100:.2f}%",
        f"  Accuracy drop : {(baseline_acc-acc_after_act)*100:.2f}%",
        "",
        "STAGE 2 — Learned Pruning (on activation-pruned model)",
        f"  Steps         : {cfg['pruner']['steps']}",
        f"  Sparsity wt   : {cfg['pruner']['sparsity_weight']}",
        f"  Gated off     : {lp}/{lt} ({lp/lt*100:.1f}%)",
        f"  New pruned    : {max(learned_extra,0)}/{total_hidden}",
        f"  Total pruned  : {total_pruned}/{total_hidden} ({total_pct:.1f}%)",
        f"  Accuracy      : {acc_after_learned*100:.2f}%",
        f"  Accuracy drop : {(baseline_acc-acc_after_learned)*100:.2f}%",
        "",
        "CUMULATIVE SUMMARY",
        f"  Baseline acc  : {baseline_acc*100:.2f}%",
        f"  After Stage 1 : {acc_after_act*100:.2f}%  "
            f"({act_pruned}/{total_hidden} neurons, {act_pct:.1f}% pruned)",
        f"  After Stage 2 : {acc_after_learned*100:.2f}%  "
            f"({total_pruned}/{total_hidden} neurons, {total_pct:.1f}% pruned)",
        "",
        "=" * 60,
    ]

    out_path = "experiments/latest/baselines/sequential/summary.txt"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
