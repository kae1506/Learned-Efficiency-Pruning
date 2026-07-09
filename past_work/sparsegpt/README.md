# SparseGPT for CIFAR conv-nets

A faithful re-implementation of **SparseGPT** (Frantar & Alistarh, ICML 2023 —
*"Massive Language Models Can Be Accurately Pruned in One-Shot"*), applied to the
**linear (FC-head) layers** of the CIFAR conv-nets used in this project
(`cifar_cnn.pt` LeNet, `cifar_mid.pt`, `cifar_big.pt`).

It is here as a **strong post-training pruning baseline** for our learned-pruner
research: second-order, data-aware, one-shot, no retraining, no learned mask.

## Files
- `sparsegpt.py` — the core `SparseGPT` class (Algorithm 1 of the paper): Hessian
  accumulation `add_batch` + the Cholesky-based `fasterprune`. Layer-agnostic
  (Linear + Conv2d).
- `models.py`  — self-contained copies of the three CIFAR architectures so a
  checkpoint reloads without importing from `scripts/`.
- `prune_cifar.py` — driver: load ckpt → collect calibration activations → prune
  the FC head → report acc before/after and achieved sparsity.

## The algorithm (why it is not magnitude pruning)

For one linear layer `y = W x`, `W ∈ ℝ^{d_row × d_col}`, and calibration inputs
stacked as `X ∈ ℝ^{d_col × N}`, SparseGPT solves the layer-wise reconstruction

```
min_{Ŵ, mask}  ‖ W X − Ŵ X ‖²_F      s.t. Ŵ obeys the sparsity mask.
```

This is **Optimal Brain Surgeon** per layer. The Hessian is the same for every
output row:

```
H = 2 X Xᵀ  ∈ ℝ^{d_col × d_col}
```

OBS for removing weight `w_j`:

```
saliency (error) :  L_j = w_j² / [H⁻¹]_{jj}
optimal update   :  δ   = − (w_j / [H⁻¹]_{jj}) · H⁻¹_{:,j}     (nudges survivors)
```

The δ step is the whole point: surviving weights **absorb** the error of the
pruned ones, so the layer output barely moves. Two tricks make it one-shot:

1. **Fixed left→right column order** ⇒ the sequence of partial inverse Hessians
   is row-independent and comes from **one Cholesky** of `H⁻¹`. `H⁻¹` is computed
   once per layer, shared across all rows.
2. **Block-adaptive mask + lazy update** ⇒ mask chosen once per 128-column block
   (keep the largest `w²/[H⁻¹]²_{jj}` per row); errors propagated inside-block
   immediately and cross-block once per block.

When `X` is isotropic, `H` is diagonal and SparseGPT **collapses to magnitude
pruning**; its advantage grows with feature correlation (real activations).

## Usage (run from project root; **not run automatically** — see note)

```bash
# unstructured 50% on the LeNet FC head (fc1, fc2)
venv/bin/python past_work/sparsegpt/prune_cifar.py \
    --ckpt experiments/checkpoints/cifar_cnn.pt --sparsity 0.5

# 70% on the big net (fc1, fc2, fc3), larger calibration set
venv/bin/python past_work/sparsegpt/prune_cifar.py \
    --ckpt experiments/checkpoints/cifar_big.pt --sparsity 0.7 --n-calib 1024

# 2:4 semi-structured
venv/bin/python past_work/sparsegpt/prune_cifar.py \
    --ckpt experiments/checkpoints/cifar_cnn.pt --nm 2:4
```

Key flags: `--sparsity`, `--nm n:m`, `--n-calib`, `--blocksize` (128),
`--percdamp` (0.01 Hessian ridge), `--layers` (override which FC layers).
The classifier (last `fc`) is excluded by default — we prune the hidden MLP head,
matching the research convention (F13, Appendix D).

## Verification done

`scratchpad/check_sparsegpt.py` (synthetic Linear, correlated inputs) confirms:
- unstructured sparsity hits the target (0.501 @ 50%), 2:4 hits exactly 0.5;
- SparseGPT reconstruction MSE = **0.66×** magnitude pruning at the same 50%
  sparsity (34% lower error) — the second-order compensation works.

The full CIFAR pruning experiment has **not** been run (per request — code written
and checked only). To run it, use the commands above; needs the checkpoints in
`experiments/checkpoints/` (present) and CIFAR-10 in `./data`.

## Results on cifar_big.pt (run 2026-07-04, CPU, n_calib=10000)

`run_comparison.py` — SparseGPT swept over FC-head weight-sparsity vs the SAVED
LEP numbers (F13/App. D; no LEP was retrained). Full log: `results.txt`.

```
method                   weight-sp  test-acc   drop pp
SparseGPT                    51.7%    87.38%    +0.01
SparseGPT                    70.5%    87.36%    +0.03
SparseGPT                    80.0%    87.40%    -0.01
SparseGPT  <- iso-LEP        84.0%    87.54%    -0.15
SparseGPT                    90.0%    87.11%    +0.28
LEP (F13, 3-seed)            84.0%    85.91%    -1.48   (=71% neurons)
```

**This is NOT "SparseGPT beats LEP" — it is the structured/unstructured gap made
quantitative (~1.3pp here).** SparseGPT solves the easier objective (which few
*weights* reconstruct `WX`); on the hugely redundant CIFAR_big FC head (fc1 =
8.4M params for 10 classes) second-order compensation is near-lossless to 90%
weight-sparsity, but leaves a still-1024-wide dense-shaped matrix (no FLOP
saving without sparse kernels). LEP pays 1.48pp because it removes whole neurons
(rank-1 blocks → kills activation directions that can't be compensated) and in
return yields a genuinely smaller dense net (fc1 1024→171). Different questions.

An apples-to-apples comparison still needs one of: structured SparseGPT (OBS
saliency summed over rows), a 2:4 point, or an iso-FLOP comparison — none run yet.

## ISO-FLOP results (run 2026-07-04, `run_isoflop.py`, `structured.py`)

Structured OBC (the STRUCTURED sibling of SparseGPT — whole-neuron OBS pruning)
at LEP's exact per-layer neuron budget → identical dense architecture → identical
FLOPs. Full log: `results_isoflop.txt`.

```
method                       FLOP%  neurons_pruned  test-acc  drop pp
structured-OBC  <- iso-LEP   16.0%           75.4%    87.28%   +0.11
structured-OBC (7.9-20.9%)   ...sweep 0.5x-2x...      (Pareto curve)
LEP (F13, 3-seed)            16.0%           70.9%    85.91%   -1.48
```

At iso-FLOP, structured OBC loses only 0.11pp removing 75% of FC neurons, vs LEP's
1.48pp. **Predicted OBC would LOSE here — it won by ~1.4pp. Prediction wrong.**

**DOMINANT CONFOUND (not a clean criterion comparison):** OBC = select + OBS
weight REFIT; LEP (F13) = select-only on FROZEN weights (mask, no re-fit). OBC's
win largely reflects the least-squares refit LEP forgoes, not necessarily better
neuron selection. To isolate selection: no-compensation ablation (zero columns,
no OBS update → both select-only) and/or LEP + least-squares refit. NOT yet run.

Minor caveats: (1) LEP saved numbers internally inconsistent (70.9% headline vs
75.4% from per-layer keeps 171/177/92); we matched the per-layer keeps, so OBC ran
the harder budget. (2) fc1 has only ~230 ever-active (post-ReLU) neurons of 1024,
so f>=1.5 keeps plateau; iso point f=1.0 is unaffected.

## How this contrasts with our learned pruner

| | our hypernetwork pruner | SparseGPT (this) |
|---|---|---|
| granularity | **structured** (whole neurons / rows) | **unstructured** weights (+ n:m) |
| uses data? | yes (activations, forward+backward) | yes (activations, forward only) |
| weight update? | **no** (mask only) | **yes** (OBS second-order) |
| learns a mask? | yes (soft-λ gradient descent) | no (closed-form saliency) |
| retraining? | no | no |

So SparseGPT answers a different, complementary question: *how far does one-shot
second-order weight surgery go on the same FC heads, with no learned mask?* Its
unstructured sparsity is not directly FLOP-comparable to our structured neuron
counts, but the accuracy-vs-sparsity curve is the reference a learned structured
pruner should be measured against.

## Reference
Frantar, Alistarh. *SparseGPT: Massive Language Models Can Be Accurately Pruned
in One-Shot.* ICML 2023. Core follows the authors' `fasterprune`.
