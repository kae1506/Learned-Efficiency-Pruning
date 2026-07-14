import torch
import torch.nn as nn
import torch.nn.functional as F

from src.pruners.mlp import binary_ste


class Pruner(nn.Module):
    """
    Hybrid RowEncoder + Bidirectional LSTM pruner.

    Identical to the LSTM pruner (pruner_lstm.py) except the cross-layer
    context path uses a BiLSTM instead of a unidirectional LSTM.

    Unidirectional LSTM: when computing the context bias for layer i,
      only layers 0..i-1 have been seen.

    Bidirectional LSTM: the context bias for layer i is informed by ALL
      layers in both directions. For layer 1 (out of 2), the backward
      pass has already seen layer 2's weight statistics before producing
      layer 1's context. This lets the pruner reason globally — e.g.
      "layer 2 has many high-norm nodes, so I can prune layer 1 more
      aggressively without losing capacity".

    Implementation detail: BiLSTM outputs [forward; backward] concatenated
    at each timestep, giving hidden size 2 * lstm_hidden. The context_head
    projects this down to a scalar bias per layer.
    """

    def __init__(
        self,
        layer_shapes: list[tuple[int, int]],
        embed_dim: int   = 64,
        lstm_hidden: int = 64,
    ):
        super().__init__()

        # ── Per-node path (same as LSTM pruner) ──────────────────────────────
        self.row_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, 1),
            )
            for _, in_features in layer_shapes
        ])
        for enc in self.row_encoders:
            nn.init.constant_(enc[-1].bias, 2.0)

        # ── Cross-layer context path (BiLSTM) ─────────────────────────────────
        self.layer_projectors = nn.ModuleList([
            nn.Linear(in_features, lstm_hidden)
            for _, in_features in layer_shapes
        ])

        # bidirectional=True doubles the effective hidden size
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )

        # LayerNorm stabilises the 2*lstm_hidden BiLSTM output before the gate head.
        # Without it, the 2x wider output causes Adam to push context biases down
        # twice as fast as the LSTM pruner, collapsing all gates before the task
        # loss can react.
        self.context_norm = nn.LayerNorm(lstm_hidden * 2)

        # Input to context_head is 2*lstm_hidden (forward + backward concat)
        self.context_head = nn.Linear(lstm_hidden * 2, 1)
        nn.init.zeros_(self.context_head.weight)
        nn.init.zeros_(self.context_head.bias)

    def _node_scores(
        self,
        weight_matrices: list[torch.Tensor],
        layer_indices: list[int] | None = None,
    ) -> list[torch.Tensor]:
        """
        Continuous per-node importance logits (pre-threshold). Higher = more
        likely kept; the keep decision in forward() is sigmoid(score) > 0.5,
        i.e. score > 0. Exposed via scores() for threshold/ranking sweeps.

        layer_indices: optional list, same length as weight_matrices, giving
        each matrix's true layer index into self.row_encoders /
        self.layer_projectors -- lets a caller score a SUBSET of layers (e.g.
        stochastic layer-subset training: sample k of N layers per step,
        score only those) instead of the full ordered list. Defaults to
        range(len(weight_matrices)) -- i.e. "weight_matrices is already the
        complete ordered list" -- identical to the original behavior; every
        existing caller passes only weight_matrices and is unaffected.

        Note: the BiLSTM only sees the layers actually passed in, in the
        order given. If layer_indices is a strict subset, "cross-layer
        context" is computed over that subset's sequence, not the true full
        network sequence -- a real property of subset training (a layer's
        BiLSTM neighbors change depending on which other layers happen to be
        sampled alongside it that step), not a bug. See
        diary/engineering_decisions.md for the tradeoff this was accepted for.
        """
        if layer_indices is None:
            layer_indices = range(len(weight_matrices))

        # ── Per-node logits ───────────────────────────────────────────────────
        node_logits = [
            self.row_encoders[i](W).squeeze(-1)
            for i, W in zip(layer_indices, weight_matrices)
        ]

        # ── Layer-level embeddings ─────────────────────────────────────────────
        layer_embeds = [
            F.relu(self.layer_projectors[i](W.mean(dim=0)))
            for i, W in zip(layer_indices, weight_matrices)
        ]

        # BiLSTM over layer sequence: [1, num_layers, lstm_hidden]
        # Output: [1, num_layers, 2*lstm_hidden]  (forward + backward concatenated)
        seq      = torch.stack(layer_embeds, dim=0).unsqueeze(0)
        lstm_out, _ = self.lstm(seq)

        # tanh bounds context to (-1, 1): it can modulate but never override the
        # per-node logit (which starts at +2.0), preventing runaway collapse.
        context_biases = torch.tanh(
            self.context_head(self.context_norm(lstm_out.squeeze(0))).squeeze(-1)
        )

        return [logits + ctx for logits, ctx in zip(node_logits, context_biases)]

    def forward(
        self,
        weight_matrices: list[torch.Tensor],
        layer_indices: list[int] | None = None,
    ) -> list[torch.Tensor]:
        """
        weight_matrices: list of [out_nodes, in_features] detached tensors.
        layer_indices: see _node_scores -- optional subset support.
        Returns: list of hard binary gate tensors [out_nodes] per layer.
        """
        return [binary_ste(s) for s in self._node_scores(weight_matrices, layer_indices)]

    @torch.no_grad()
    def scores(
        self,
        weight_matrices: list[torch.Tensor],
        layer_indices: list[int] | None = None,
    ) -> list[torch.Tensor]:
        """Continuous per-node importance logits (no grad) for ranking sweeps."""
        return self._node_scores(weight_matrices, layer_indices)
