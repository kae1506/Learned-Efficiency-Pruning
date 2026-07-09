"""
Structured (neuron) OBC — the STRUCTURED sibling of SparseGPT.

SparseGPT (sparsegpt.py) is the unstructured, fast-approximate member of the
Optimal Brain family. Its structured predecessor is OBC (Frantar & Alistarh,
"Optimal Brain Compression", NeurIPS 2022): the SAME OBS/Hessian math, but it
removes whole input COLUMNS (= whole upstream neurons) instead of scattered
weights. Because the layers here are small (input dim ≤ 1024) we run the EXACT
greedy version — no fixed-order/lazy approximation needed.

Removing input column j of a layer with weight W ∈ ℝ^{rows × d} and input Hessian
H = 2 X Xᵀ costs, across all output rows (H⁻¹ is shared):

        saliency(j) = ( Σ_i W[i,j]² ) / [H⁻¹]_{jj}

Greedy loop: pick the min-saliency column, apply the OBS compensation to the
survivors and rank-1 downdate H⁻¹, repeat until `keep` columns remain.

        W        ←  W − W[:,j] ⊗ ( H⁻¹[j,:] / [H⁻¹]_{jj} )     (zeros col j, fixes rest)
        H⁻¹      ←  H⁻¹ − H⁻¹[:,j] ⊗ H⁻¹[j,:] / [H⁻¹]_{jj}

A column of THIS layer = an output neuron of the UPSTREAM layer, so the caller
also zeroes the matching rows of the upstream weight (those neurons are dead).

All maths in float64 (853 sequential rank-1 downdates accumulate error in fp32).
"""

import torch
import torch.nn as nn


class StructuredOBC:
    """Exact greedy structured (input-column) OBC for one nn.Linear layer."""

    def __init__(self, layer: nn.Linear):
        self.layer = layer
        self.rows, self.columns = layer.weight.shape
        self.H = torch.zeros((self.columns, self.columns), dtype=torch.float64)
        self.nsamples = 0

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor):
        """Accumulate H = 2 X Xᵀ from this layer's (post-activation) inputs."""
        if inp.dim() > 2:
            inp = inp.reshape(-1, inp.shape[-1])
        inp = inp.double().t()                       # (d, N)
        n = inp.shape[1]
        self.H *= self.nsamples / (self.nsamples + n)
        self.nsamples += n
        inp = (2.0 / self.nsamples) ** 0.5 * inp
        self.H += inp.matmul(inp.t())

    @torch.no_grad()
    def prune(self, keep: int, percdamp: float = 0.01):
        """Prune input columns down to `keep`, compensating survivors. Returns the
        boolean keep-mask over columns (True = column kept)."""
        W = self.layer.weight.data.double().clone()
        H = self.H.clone()

        dead = torch.diag(H) == 0
        H[dead, dead] = 1.0
        W[:, dead] = 0.0

        damp = percdamp * torch.mean(torch.diag(H))
        idx = torch.arange(self.columns)
        H[idx, idx] += damp
        Hinv = torch.linalg.inv(H)

        alive = torch.ones(self.columns, dtype=torch.bool)
        alive[dead] = False                          # dead cols pruned for free
        n_remove = (alive.sum() - keep).item()

        for _ in range(max(n_remove, 0)):
            diagHinv = torch.diag(Hinv).clamp_min(1e-12)
            sal = (W ** 2).sum(0) / diagHinv         # saliency per column
            sal[~alive] = float("inf")
            j = int(torch.argmin(sal))

            d = Hinv[j, j].clamp_min(1e-12)
            W -= torch.outer(W[:, j], Hinv[j, :] / d)   # zero col j, compensate rest
            Hinv -= torch.outer(Hinv[:, j], Hinv[j, :]) / d
            W[:, j] = 0.0
            Hinv[j, :] = 0.0
            Hinv[:, j] = 0.0
            alive[j] = False

        self.layer.weight.data = W.to(self.layer.weight.dtype)
        return alive

    def free(self):
        self.H = None
