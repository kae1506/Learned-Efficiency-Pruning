"""
Faithful re-implementation of SparseGPT (Frantar & Alistarh, ICML 2023,
"SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot")
applied to the *linear layers* of our CIFAR conv-nets.

----------------------------------------------------------------------------
THE MATH (why this is not just magnitude pruning)
----------------------------------------------------------------------------
For one linear layer y = W x with W ∈ ℝ^{d_row × d_col}, given a calibration
set of inputs stacked as X ∈ ℝ^{d_col × N}, SparseGPT solves the *layer-wise
reconstruction* problem

        min_{Ŵ, mask}  ‖ W X − Ŵ X ‖²_F      s.t.  Ŵ obeys the sparsity mask.

This is Optimal Brain Surgeon (OBS) applied per layer. The Hessian of the
squared error w.r.t. one output row's weights is the SAME for every row:

        H = 2 X Xᵀ  ∈ ℝ^{d_col × d_col}                 (shared across rows)

OBS gives, for pruning a single weight w_j (column j of a row):
  • saliency (error incurred)      :  L_j = w_j² / [H⁻¹]_{jj}
  • optimal update to survivors    :  δ  = − (w_j / [H⁻¹]_{jj}) · H⁻¹_{:,j}

Naively re-inverting H after every removed weight is O(d_col⁴) per row and the
optimal removal order differs per row, so H⁻¹ cannot be shared. SparseGPT's two
key tricks make it one-shot and fast:

  1. FIXED column order + shared H⁻¹. Prune columns strictly left→right in the
     SAME order for every row. Then the sequence of inverse Hessians of the
     "remaining" submatrices is exactly the sequence of Gaussian-elimination
     partial inverses — obtainable from ONE Cholesky of H⁻¹. Row-independent, so
     H⁻¹ is computed once for the whole layer.

  2. Adaptive mask + iterative update in blocks. Within a block of `blocksize`
     columns the mask is chosen once (keep the (1−p) fraction of largest
     w²/[H⁻¹]²_{jj} per row), then each column is zeroed-or-kept and the OBS
     error err_j = (w_j − q_j)/[H⁻¹]_{jj} is propagated to all columns to the
     right (lazily: inside-block immediately, cross-block once per block).

The compensation step (1) is what separates SparseGPT from plain magnitude
pruning: every surviving weight is nudged to absorb the error of the pruned
ones, so a 50% mask barely moves the layer output.

----------------------------------------------------------------------------
RELATION TO OUR RESEARCH
----------------------------------------------------------------------------
Our learned hypernetwork does STRUCTURED pruning (drops whole neurons / rows of
W) with a soft-λ penalty and NO weight update. SparseGPT here does UNSTRUCTURED
pruning (drops individual weights) WITH a second-order weight update and NO
retraining. It is the strong post-training baseline on the same CIFAR FC heads:
"how far can second-order, data-aware, one-shot pruning go without learning a
mask and without gradient descent on the mask?" See README.md.
"""

import math
import torch
import torch.nn as nn

DEBUG = False


class SparseGPT:
    """One-shot SparseGPT pruning for a single nn.Linear (or nn.Conv2d) layer.

    Usage:
        gpt = SparseGPT(layer)
        # feed calibration activations (the layer's own inputs):
        for x in calib_inputs: gpt.add_batch(x)
        gpt.fasterprune(sparsity=0.5)          # unstructured 50%
        gpt.fasterprune(prunen=2, prunem=4)    # 2:4 semi-structured
        gpt.free()
    """

    def __init__(self, layer: nn.Module):
        self.layer = layer
        W = layer.weight.data.clone()
        if isinstance(layer, nn.Conv2d):
            # flatten conv weight to (out_channels, in_channels*kh*kw); each output
            # channel is a "row", each unrolled input position a "column".
            W = W.flatten(1)
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        # running estimate of H = 2 X Xᵀ (accumulated in add_batch), on CPU/float64
        # for numerical stability of the Cholesky.
        dev = layer.weight.device
        self.H = torch.zeros((self.columns, self.columns), device=dev)
        self.nsamples = 0

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor):
        """Accumulate H = 2 X Xᵀ from a batch of this layer's inputs.

        inp: for Linear, shape (batch, d_col) or (batch, seq, d_col); for Conv2d
        the pre-unfolded input (batch, C, Hh, Ww) — we unfold it to match the
        flattened weight. H is kept as a running mean over samples so batches of
        different size compose correctly.
        """
        if isinstance(self.layer, nn.Linear):
            if inp.dim() > 2:
                inp = inp.reshape(-1, inp.shape[-1])
            inp = inp.t()                                    # (d_col, N)
        elif isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(self.layer.kernel_size, dilation=self.layer.dilation,
                               padding=self.layer.padding, stride=self.layer.stride)
            inp = unfold(inp)                                # (batch, d_col, L)
            inp = inp.permute(1, 0, 2).reshape(inp.shape[1], -1)   # (d_col, N)
        else:
            raise TypeError(f"unsupported layer {type(self.layer)}")

        n = inp.shape[1]
        self.H *= self.nsamples / (self.nsamples + n)
        self.nsamples += n
        # scale so that after accumulation H = (2/N) Σ x xᵀ = 2 · empirical E[x xᵀ].
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    @torch.no_grad()
    def fasterprune(self, sparsity: float = 0.0, prunen: int = 0, prunem: int = 0,
                    blocksize: int = 128, percdamp: float = 0.01):
        """Run SparseGPT (Algorithm 1 of the paper) on this layer, in place.

        sparsity: unstructured target fraction of weights to zero (used when
                  prunen == 0).
        prunen, prunem: n:m semi-structured sparsity (e.g. 2:4). If prunen>0 the
                  `sparsity` argument is ignored and each contiguous group of m
                  columns keeps exactly m−n weights per row.
        blocksize: column block size for the lazy update (128 = paper default).
        percdamp: Hessian dampening λ = percdamp · mean(diag H), the Tikhonov
                  ridge that keeps H⁻¹ well-conditioned (the "λI" in (2XXᵀ+λI)).
        """
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        W = W.float()

        H = self.H
        # dead input columns (never activated) → their diag is 0. Set to 1 and
        # zero the corresponding weights so they are pruned for free.
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=W.device)

        # ── Tikhonov dampening, then H⁻¹ as an UPPER-triangular Cholesky factor ──
        # The upper-Cholesky of H⁻¹ is exactly the row-independent sequence of
        # partial inverses needed for the fixed left→right elimination order.
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=W.device)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)                # quantised/pruned block
            Err1 = torch.zeros_like(W1)              # OBS errors, for cross-block update
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            # choose the mask for this block (unstructured case)
            if prunen == 0:
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    # saliency w² / [H⁻¹]²_{jj}; keep the largest (1−sparsity) fraction.
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh            # True = prune
            else:
                mask1 = torch.zeros_like(W1, dtype=torch.bool)

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                # n:m: every m columns, pick the n smallest-saliency to prune per row.
                if prunen != 0 and i % prunem == 0:
                    tmp = (W1[:, i:(i + prunem)] ** 2
                           / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2)
                    mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1,
                                                     largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0                   # apply the mask

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                # OBS error and its propagation to the remaining columns of the block
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            # lazy cross-block update: push this block's accumulated error to the
            # columns to the right of the block, once.
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if DEBUG:
            print(f"  reconstruction error (sum over rows): {torch.sum(Losses).item():.4f}")

        W = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        self.layer.weight.data = W
        return torch.sum(Losses).item()

    def free(self):
        self.H = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
