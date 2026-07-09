"""
Top-K variant of the BiLSTM weight-conditioned pruner.

This is the SAME architecture as src/pruners/bilstm.py (per-node row encoder +
cross-layer BiLSTM context), but adapted for the STE-top-K curriculum, which
removes the need for the λ (sparsity-weight) collapse-babysitting hacks:

  DROPPED (existed only to lose the one-shot λ collapse race to all-zero):
    - the +2.0 final-bias init on the row encoder
        Under top-K exactly K gates survive every step by construction, so there
        is no all-zero attractor to defend against; gates do not need to "start
        open".
    - the tanh bound on the cross-layer context
        Its only job under λ was to stop the context from overriding the +2.0
        per-node logit and forcing collapse. With no collapse to prevent we let
        the context range freely — and under GLOBAL top-K an unbounded per-layer
        context bias is exactly the knob the pruner uses to allocate the keep
        budget across layers.

  KEPT:
    - LayerNorm on the 2x-wide BiLSTM output (toggle: use_layernorm)
        Not collapse-babysitting under top-K — it is optimisation conditioning:
        (a) a sane common scale across layers at init (so the starting global
        ranking is not a degenerate "keep one whole layer"), and (b) it keeps
        scores off the sigmoid saturation tails so the STE revival gradient
        (sigmoid'(s) ~ 0 when |s| large) does not vanish for neurons far from
        the threshold. We keep it ON by default but expose the toggle so the
        ablation ("were tanh/LayerNorm load-bearing or just λ-babysitting?") is
        a one-line change.
    - zero-init context_head (neutral start: step-0 context = 0 -> plain
      per-node ranking), LSTM-over-LAYERS (2-4 steps, not neurons), and the
      grad-clip in the training step (anti-explosion, unrelated to collapse).

The pruner exposes:
    node_scores(weights)  -> list of continuous per-node scores (WITH grad)
    forward(weights, k)   -> list of hard {0,1} gates, exactly K kept GLOBALLY
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def topk_ste(scores: list[torch.Tensor], k: int,
             temp: float = 1.0, center: bool = True) -> list[torch.Tensor]:
    """
    Global top-K straight-through gates with a THRESHOLD-CENTERED surrogate.

    scores : list of [n_layer] continuous score tensors (with grad).
    k      : number of neurons to KEEP across ALL layers combined.
    temp   : temperature T of the centered sigmoid (width of the active band).
    center : if True, soft = σ((s - thresh)/T)  (centered, recommended);
             if False, soft = σ(s)              (plain — saturates, kept for ablation).

    Forward : the K highest-scoring neurons (pooled across layers) -> 1.0, rest
              -> 0.0. Exact budget every step => collapse-to-zero impossible, no
              per-model λ sweep. NOTE: the hard mask does NOT depend on T or
              `center` (it's just s >= thresh) — those only shape the backward.

    Backward: gradient = σ'(·) per neuron. PLAIN puts σ's gradient peak at s=0,
              but the keep/kill boundary under top-K is `thresh` (the K-th largest
              score), which can be far from 0 -> the borderline neurons sit on
              σ's saturated tail -> ~zero gradient -> frozen ranking (this is what
              made plain top-K fail at aggressive K). CENTERED slides the sigmoid
              onto `thresh`, so σ' peaks exactly on the neurons that might flip in
              /out of the top-K, scale-independently. (Same fix λ got for free:
              under λ the boundary WAS 0, coinciding with σ's center.)
    """
    flat = torch.cat([s.reshape(-1) for s in scores])
    n = flat.numel()
    k = max(1, min(int(k), n))                      # clamp into [1, N]
    thresh = torch.topk(flat, k).values.min()       # K-th largest = keep threshold
    gates = []
    for s in scores:
        if center:
            soft = torch.sigmoid((s - thresh) / temp)   # grad peak at the boundary
        else:
            soft = torch.sigmoid(s)                      # grad peak at 0 (plain)
        hard = (s >= thresh).float()                     # exact top-K (forward value)
        gates.append(hard - soft.detach() + soft)        # value=hard, grad=d(soft)
    return gates


class TopKPruner(nn.Module):
    """RowEncoder + BiLSTM context pruner, top-K-ready (no λ collapse hacks)."""

    def __init__(
        self,
        layer_shapes: list[tuple[int, int]],
        embed_dim: int = 64,
        lstm_hidden: int = 64,
        use_layernorm: bool = True,
        node_norm: str = "std",      # "none" | "std" | "std_detach" | "center"
        bound_context: bool = False, # tanh-bound the per-layer context bias?
    ):
        super().__init__()
        self.use_layernorm = use_layernorm
        self.node_norm = node_norm
        self.bound_context = bound_context

        # ── Per-node path ────────────────────────────────────────────────────
        # One small MLP per layer mapping a weight ROW (incoming weights of one
        # node) -> scalar score. NOTE: no +2.0 bias init (dropped — see header).
        self.row_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, 1),
            )
            for _, in_features in layer_shapes
        ])

        # ── Cross-layer context path (BiLSTM over the LAYER sequence) ─────────
        self.layer_projectors = nn.ModuleList([
            nn.Linear(in_features, lstm_hidden)
            for _, in_features in layer_shapes
        ])
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )
        # Kept (toggleable): conditions the 2x-wide BiLSTM output. See header.
        self.context_norm = nn.LayerNorm(lstm_hidden * 2) if use_layernorm else nn.Identity()

        # Project BiLSTM output (forward+backward concat) -> scalar per-layer
        # context bias. Zero-init => neutral start (step-0 context = 0).
        self.context_head = nn.Linear(lstm_hidden * 2, 1)
        nn.init.zeros_(self.context_head.weight)
        nn.init.zeros_(self.context_head.bias)

    def node_scores(self, weight_matrices: list[torch.Tensor]) -> list[torch.Tensor]:
        """Continuous per-node scores (WITH grad). Higher = more likely kept."""
        # Per-node scores from each layer's row encoder.
        node_logits = [
            enc(W).squeeze(-1)
            for enc, W in zip(self.row_encoders, weight_matrices)
        ]

        # Per-layer normalisation of the node scores: stops the unbounded
        # row-encoder logits from drifting large + saturating the STE gradient,
        # and puts every layer on a common spread for fair GLOBAL top-K.
        #   "std"        : (l-mean)/std  — backprops through std; UNSTABLE when the
        #                  encoder output has low variance (std->0 amplifies grads).
        #   "std_detach" : same forward, but mean/std are detached -> no exploding
        #                  gradient through the division (recommended).
        #   "none"       : raw logits (relies on centered STE alone for scale).
        # Ranking within a layer is preserved (monotone affine); cross-layer
        # allocation is carried by the per-layer context bias added below.
        if self.node_norm == "std":
            node_logits = [(l - l.mean()) / (l.std() + 1e-5) for l in node_logits]
        elif self.node_norm == "std_detach":
            node_logits = [(l - l.mean().detach()) / (l.std().detach() + 1e-5) for l in node_logits]
        elif self.node_norm == "center":
            # Subtract per-layer mean ONLY (no division by std). Keeps each layer
            # zero-mean -> cross-layer allocation still rides on the (tanh-bounded)
            # context bias. Crucially this is NOT scale-invariant: y=l-μ scales with
            # l, so L(αl)≠L(l) -> the encoder scale σ_ℓ is loss-constrained (no free
            # blow-up) -> no 1/σ gradient decay -> the encoder does not freeze.
            node_logits = [l - l.mean() for l in node_logits]

        # Per-layer embedding token = projected mean weight row, fed to the BiLSTM.
        layer_embeds = [
            F.relu(proj(W.mean(dim=0)))
            for proj, W in zip(self.layer_projectors, weight_matrices)
        ]
        seq = torch.stack(layer_embeds, dim=0).unsqueeze(0)   # [1, n_layers, H]
        lstm_out, _ = self.lstm(seq)                          # [1, n_layers, 2H]

        # Per-layer context bias c_ell. This is the cross-layer keep-budget
        # ALLOCATION knob under global top-K: since the node scores are per-layer
        # standardised (zero-mean/unit-var), the bias is the ONLY thing that
        # shifts one layer's score cloud relative to another. Unbounded, the gap
        # Δ = c_2 - c_1 can diverge → one layer's cloud clears the other → that
        # layer takes all K slots → the other is severed (layer-starvation, F12).
        # bound_context=True applies tanh → c_ell ∈ (-1,1), Δ ∈ (-2,2): bounded,
        # so with unit within-layer spread the clouds always overlap → no severance.
        context_biases = self.context_head(
            self.context_norm(lstm_out.squeeze(0))
        ).squeeze(-1)                                         # [n_layers]
        if self.bound_context:
            context_biases = torch.tanh(context_biases)

        return [logits + ctx for logits, ctx in zip(node_logits, context_biases)]

    def forward(self, weight_matrices: list[torch.Tensor], k: int,
                temp: float = 1.0, center: bool = True) -> list[torch.Tensor]:
        """Hard {0,1} gates with exactly K kept globally (centered-STE backward)."""
        return topk_ste(self.node_scores(weight_matrices), k, temp=temp, center=center)
