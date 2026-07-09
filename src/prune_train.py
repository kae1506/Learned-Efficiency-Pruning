import torch
import torch.nn as nn
import torch.nn.functional as F


def get_hidden_weights(model: nn.Module) -> list[torch.Tensor]:
    """Returns detached weight matrices of all hidden (non-output) Linear layers."""
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    return [L.weight.detach() for L in linears[:-1]]


def masked_forward(model: nn.Module, gates: list[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """
    Run the model forward pass with per-node gates applied to hidden layers.
    For each hidden node j: its incoming weight row and bias are scaled by gates[i][j].

    Gradients flow to `gates` (and thus to the Pruner's parameters).
    The MNIST model's own weights are detached — they are NOT updated here.
    """
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    x = x.view(x.size(0), -1)

    for i, linear in enumerate(linears[:-1]):
        gate = gates[i]                              # [out_nodes]
        w = linear.weight.detach() * gate.unsqueeze(1)  # [out, in]: scale each row
        b = linear.bias.detach() * gate              # [out]: scale each bias
        x = F.relu(F.linear(x, w, b))

    # Output layer — unmasked
    out = linears[-1]
    x = F.linear(x, out.weight.detach(), out.bias.detach())
    return x


def pruner_step(
    pruner: nn.Module,
    mnist_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    sparsity_weight: float = 0.1,
) -> dict:
    """
    One training step for the Pruner.

    Loss = (CE of pruned model − CE of original model) + sparsity_weight * mean(gates)

    The first term minimises the accuracy drop from pruning.
    The sparsity term encourages more nodes to be shut off.
    """
    optimizer.zero_grad()

    hidden_weights = get_hidden_weights(mnist_model)
    gates = pruner(hidden_weights)

    # Baseline cross-entropy (no grad needed — MNIST model is frozen)
    with torch.no_grad():
        orig_logits = mnist_model(x)
        ce_orig = F.cross_entropy(orig_logits, y)
        orig_acc = (orig_logits.argmax(1) == y).float().mean().item()

    # Pruned model cross-entropy (differentiable through gates → pruner)
    pruned_logits = masked_forward(mnist_model, gates, x)
    ce_pruned = F.cross_entropy(pruned_logits, y)

    # Accuracy drop (not used for grad, just for logging)
    with torch.no_grad():
        pruned_acc = (pruned_logits.argmax(1) == y).float().mean().item()

    # Sparsity: fraction of nodes kept (lower = more pruned). Gates are binary (STE).
    sparsity_loss = sum(g.mean() for g in gates) / len(gates)

    loss = (ce_pruned - ce_orig) + sparsity_weight * sparsity_loss
    loss.backward()
    # Clip gradients — essential for LSTM pruners (long sequences → exploding grads)
    torch.nn.utils.clip_grad_norm_(pruner.parameters(), max_norm=1.0)
    optimizer.step()

    avg_gate = sum(g.mean().item() for g in gates) / len(gates)
    return {
        "loss": loss.item(),
        "ce_pruned": ce_pruned.item(),
        "ce_orig": ce_orig.item(),
        "orig_acc": orig_acc,
        "pruned_acc": pruned_acc,
        "acc_drop": orig_acc - pruned_acc,
        "avg_gate": avg_gate,
    }
