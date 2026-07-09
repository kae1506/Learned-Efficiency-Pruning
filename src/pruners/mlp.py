import torch
import torch.nn as nn


def binary_ste(logits: torch.Tensor) -> torch.Tensor:
    """
    Straight-Through Estimator for hard binary gates.
    Forward:  hard 0/1 threshold at 0.5
    Backward: gradient flows through sigmoid (non-zero everywhere)
    """
    soft = torch.sigmoid(logits)
    hard = (soft > 0.5).float()
    # STE trick: substitute hard for soft in forward, keep soft in backward
    return hard - soft.detach() + soft


class RowPruner(nn.Module):
    """
    Maps each row of a weight matrix (incoming weights to one node) to a scalar
    logit. Rows are processed independently with shared weights.
    """
    def __init__(self, in_features: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            # No activation — outputs raw logit for STE
        )
        # Start with positive bias so gates initialise to 1 (keep all nodes).
        # This prevents the sparsity loss from collapsing all gates to 0 before
        # the task loss has a chance to learn which nodes are safe to prune.
        nn.init.constant_(self.net[-1].bias, 2.0)

    def forward(self, weight_matrix: torch.Tensor) -> torch.Tensor:
        # weight_matrix: [num_nodes, in_features]
        return self.net(weight_matrix).squeeze(-1)  # [num_nodes] logits


class Pruner(nn.Module):
    """
    Takes the weight matrices of the MNIST model's hidden layers as input.
    For each hidden layer, a RowPruner produces per-node logits which are
    binarised via STE — gates are exactly 0 (pruned) or 1 (kept).

    Output: list of hard binary gate tensors (one per hidden layer).
    """
    def __init__(self, layer_shapes: list[tuple[int, int]], hidden: int = 64):
        # layer_shapes: [(out_nodes, in_features), ...] for each prunable hidden layer
        super().__init__()
        self.row_pruners = nn.ModuleList([
            RowPruner(in_features, hidden)
            for _, in_features in layer_shapes
        ])

    def forward(self, weight_matrices: list[torch.Tensor]) -> list[torch.Tensor]:
        # weight_matrices: list of [out_nodes, in_features] tensors (detached)
        logits = [rp(w) for rp, w in zip(self.row_pruners, weight_matrices)]
        return [binary_ste(l) for l in logits]
