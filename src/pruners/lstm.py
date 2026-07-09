import torch
import torch.nn as nn
import torch.nn.functional as F

from src.pruners.mlp import binary_ste


class Pruner(nn.Module):
    """
    Hybrid RowEncoder + LSTM pruner.

    Two separate computation paths are combined:

    1. Per-node path (same as MLP pruner):
       RowEncoder_i maps each weight row W_i[j,:] to a scalar logit,
       capturing individual node importance.

    2. Cross-layer context path (new):
       For each layer, the mean weight row is projected to an embedding
       and fed into an LSTM as a single timestep. The LSTM hidden state
       carries context across layers (e.g. "layer 1 was pruned heavily,
       so adjust layer 2 decisions accordingly"). Its output is projected
       to a scalar bias that shifts all node logits in that layer.

    The LSTM processes only num_hidden_layers timesteps (here: 2), not
    num_nodes (1024) — this avoids the vanishing/exploding gradient
    problem of unrolling through thousands of LSTM steps.

    Final gate for node j in layer i:
        logit  = RowEncoder_i(W_i[j,:])  +  LSTM_context_bias_i
        gate_j = binary_ste(logit)
    """

    def __init__(
        self,
        layer_shapes: list[tuple[int, int]],
        embed_dim: int   = 64,
        lstm_hidden: int = 64,
    ):
        super().__init__()

        # ── Per-node path ─────────────────────────────────────────────────────
        # One RowEncoder per layer: weight row → scalar logit
        self.row_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, 1),
            )
            for _, in_features in layer_shapes
        ])
        # Positive bias → all gates start at 1 (keep all nodes)
        for enc in self.row_encoders:
            nn.init.constant_(enc[-1].bias, 2.0)

        # ── Cross-layer context path ──────────────────────────────────────────
        # Project each layer's mean weight row to a fixed-size embedding
        self.layer_projectors = nn.ModuleList([
            nn.Linear(in_features, lstm_hidden)
            for _, in_features in layer_shapes
        ])

        # LSTM over the layer sequence (num_layers timesteps, not num_nodes)
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            batch_first=True,
        )

        # Project LSTM output → scalar bias applied to all nodes in that layer
        self.context_head = nn.Linear(lstm_hidden, 1)
        # Zero init: context starts at 0 so behaviour identical to MLP pruner at step 0
        nn.init.zeros_(self.context_head.weight)
        nn.init.zeros_(self.context_head.bias)

    def forward(self, weight_matrices: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        weight_matrices: list of [out_nodes, in_features] detached tensors.
        Returns: list of hard binary gate tensors [out_nodes] per layer.
        """
        # ── Per-node logits ───────────────────────────────────────────────────
        node_logits = [
            enc(W).squeeze(-1)                       # [out_nodes]
            for enc, W in zip(self.row_encoders, weight_matrices)
        ]

        # ── Layer-level embeddings for LSTM ───────────────────────────────────
        layer_embeds = [
            F.relu(proj(W.mean(dim=0)))              # [lstm_hidden]
            for proj, W in zip(self.layer_projectors, weight_matrices)
        ]

        # LSTM over layer sequence: [1, num_layers, lstm_hidden]
        seq      = torch.stack(layer_embeds, dim=0).unsqueeze(0)
        lstm_out, _ = self.lstm(seq)                 # [1, num_layers, lstm_hidden]

        # Scalar context bias per layer: [num_layers]
        context_biases = self.context_head(lstm_out.squeeze(0)).squeeze(-1)

        # ── Combine and binarise ──────────────────────────────────────────────
        return [
            binary_ste(logits + ctx)
            for logits, ctx in zip(node_logits, context_biases)
        ]
