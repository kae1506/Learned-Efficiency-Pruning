"""
Training step for the STE-top-K curriculum pruner.

Differs from src/prune_train.py's pruner_step in exactly two ways:
  1. The mask is produced by GLOBAL top-K (exactly K neurons kept) instead of a
     fixed sigmoid>0.5 threshold.
  2. The loss is purely (CE_pruned - CE_orig) — there is NO sparsity term and NO
     λ. "How many to keep" is set by K; the pruner only learns "which" (the
     ranking). This is what eliminates the per-model sparsity-weight sweep.

The base model is frozen (its weights are detached in masked_forward); gradients
flow only into the pruner.
"""

import torch
import torch.nn.functional as F

from src.prune_train import get_hidden_weights, masked_forward
from src.pruners.bilstm_topk import topk_ste


def topk_pruner_step(pruner, model, optimizer, x, y, k: int, temp: float = 1.0) -> dict:
    """One curriculum step at keep-budget K (centered STE, temperature `temp`)."""
    optimizer.zero_grad()

    # Continuous scores (with grad) -> global top-K STE gates at budget K.
    hidden_weights = get_hidden_weights(model)
    scores = pruner.node_scores(hidden_weights)
    gates = topk_ste(scores, k, temp=temp, center=True)

    # Baseline CE (frozen model, no grad).
    with torch.no_grad():
        orig_logits = model(x)
        ce_orig = F.cross_entropy(orig_logits, y)

    # Pruned CE (differentiable through gates -> pruner).
    pruned_logits = masked_forward(model, gates, x)
    ce_pruned = F.cross_entropy(pruned_logits, y)

    # Pure accuracy-drop proxy. No sparsity term.
    loss = ce_pruned - ce_orig
    loss.backward()
    # Anti-explosion clip (unrelated to collapse — kept for all LSTM pruners).
    torch.nn.utils.clip_grad_norm_(pruner.parameters(), max_norm=1.0)
    optimizer.step()

    return {"ce_drop": (ce_pruned - ce_orig).item(), "loss": loss.item()}
