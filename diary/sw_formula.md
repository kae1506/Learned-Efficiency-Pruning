# Sparsity weight λ — formulation attempts

**Problem:** We sweep λ for every new base model. **Goal:** predict λ_opt from one cheap measurement instead.

---

## Data — 5 base models (single seed = `MNIST_NARROW` excluded; rest are 3-seed × 15ep efficiency peaks)

| net | N_L | S_L | params | base acc | observed λ_opt | peak eff | λ·N_L | λ·S_L |
|---|:-:|---:|---:|---:|---:|---:|---:|---:|
| MNIST Wide [2048]      | 1 | 2048 | 1.63M | 98.16% | **0.03** | 159 | 0.03 | 61.4 |
| LeNet (CIFAR, narrow)  | 2 |   96 |   63K | 64.84% | **0.04** | 5.4 | 0.08 | 3.8 |
| MNIST Medium [1024×2]  | 2 | 1024 | 1.86M | 98.06% | **0.06** | 136 | 0.12 | 61.4 |
| CIFAR_big (3L FC head) | 3 |  597 | 10.4M | 87.39% | **0.03** |  48 | 0.09 | 17.9 |
| MNIST Deep [512×4]     | 4 |  512 | 1.19M | 98.32% | **0.25** | 109 | 1.00 | 128 |

Spread of observed λ_opt: **8×** (0.03 → 0.25) across nominally similar setups. Not constant.

---

## Hypotheses tested (chronological)

### H2-original — `λ_opt ∝ CE_orig / mean_layer_size`
Predicts MNIST has λ_opt ~250× smaller than LeNet; observed: MNIST is 1.5× LARGER.
**REFUTED — off by 2 orders of magnitude.**

### H2-monotonic-in-size — `λ_opt ∝ 1/N_params` or `∝ 1/mean_layer_size`
Predicts monotone decline with model size; observed sequence is non-monotone (LeNet 0.04 → MNIST 0.06 → CIFAR_big 0.03).
**REFUTED — wrong direction at one of three boundaries.**

### N_layers law — `λ_opt · N_L ≈ const`
Predicted 1L wide λ_opt ≈ 0.10; observed 0.03. λ·N_L spans 0.03 → 1.00 (33× spread).
**REFUTED by the wide point.**

### `λ_opt · S_L ≈ const`
1L wide and 2L medium both give λ·S = 61.4 — but only because they share the same 2048-neuron budget. 4L deep gives 128 (2×).
**REFUTED in general — looks like a law only in a coincidental 2-point slice.**

### Within-MNIST same-budget: `λ_opt ≈ k · 2^(N_L − 1)` with k=0.015
- 1L wide: pred 0.030, obs 0.030 ✓ (within 0%)
- 2L medium: pred 0.060, obs 0.060 ✓ (0%)
- 4L deep: pred 0.240, obs 0.250 ✓ (4%)
- 2L LeNet (CIFAR): pred 0.060, obs 0.040 ✗ (saturated narrow + diff task)
- 3L CIFAR_big: pred 0.120, obs 0.030 ✗ (diff task, k~half of MNIST)
**CONSISTENT only in narrow regime (MNIST + over-parameterized + equal-width + same total neurons).**

### Refined H2 (current candidate): `λ_opt ≈ k_task · 2^(N_L − 1) · saturation_factor`
Decomposes into:
- `k_task`: per-task constant (MNIST ≈ 0.015, CIFAR ≈ 0.0075 from CIFAR_big fit)
- `saturation_factor`: ≥ 1 when net is saturated; ≈ 1 over-parameterized
- `2^(N_L−1)`: structural, free

Status: **CANDIDATE — fits the 3 MNIST shape models within 4%, but both `k_task` and `saturation_factor` need calibration on a new setup → still a sweep.** This form is the best we have but not yet predictive.

---

## Current direction: gradient probe (scripts/hypernetwork/lambda_scaling/gradient_probe.py)

**Mechanism:** at λ_opt, the per-gate sparsity gradient `λ/(N_L · S_ℓ)` balances the per-gate CE gradient `|∂CE/∂g_i|` at the boundary gate.

**Probe:** one forward+backward pass on each frozen base with all gates = 1 (= unmasked output), capture `g.grad.abs()`, average over 2 TRAINING batches. ~1 minute total compute.

**Candidate predictors** (one constant `c` fit across all 5 nets in log-LS):
- P1: `c · N_L · S_avg · ⟨|grad|⟩_overall`
- P2: `c · max_ℓ (N_L · S_ℓ · ⟨|grad|⟩_ℓ)` — bottleneck layer
- P3: `c · median_ℓ (N_L · S_ℓ · ⟨|grad|⟩_ℓ)` — robust
- P4: `c · max_ℓ (N_L · S_ℓ · median(|grad|_ℓ))` — marginal-gate at q=50
- P5: `c · N_L · Σ_ℓ S_ℓ · ⟨|grad|⟩_ℓ` — sum-weighted

**Pass criterion:** any one of P1-P5 fits all 5 nets within ±2× residual → gradient probe becomes a sweep-free λ predictor.

**If it works:** the `k_task` and `saturation_factor` variation we attribute to "task" and "saturation" should collapse into the measured `⟨|∂CE/∂g|⟩` — these are the things the gradient probe directly captures.

**If it fails:** the failure mode tells us what's missing (operating-point mismatch, layer aggregation, or task-specific scaling we haven't modelled).

### Result (2026-06-11): probe FAILED in this form
- All 5 predictors P1–P5 have **R²(log) < 0** (worse than mean). Best (P2) still 4.6× off on the worst point.
- **Cross-task anti-correlation:** LeNet (hard task, 64% acc) has 60× larger ⟨|grad|⟩ than MNIST Medium (98% acc) but a SMALLER λ_opt (0.04 vs 0.06). Within-task (MNIST shape sweep) the gradient and λ_opt correlate cleanly; across tasks they don't.
- **Heterogeneity (med/mean) partially explains MNIST sweep** (0.015 → 0.142 → 0.233 ↔ λ_opt 0.03 → 0.06 → 0.25) but LeNet breaks it: highest med/mean (0.795, near-uniform gradients = saturated) → predicts largest λ_opt; observed 0.04 (smallest). Saturated nets don't *need* high λ because there's no redundancy to push.
- **Diagnosis:** CE gradient at gates=1 measures "how wrong per neuron" (task-coupled). λ_opt responds to "redundancy slack at the operating point" — a different physical quantity. They correlate only when task is fixed.
- Source: [hypernetwork/gradient_probe/](../experiments/latest/hypernetwork/gradient_probe/) (summary.txt, scaling_fits.png, probe_results.json).

### Refinements that could rescue the probe direction
1. **Operating-point probe.** Random 50%-keep mask gradients — measures the regime the pruner actually operates in, not the unpruned starting point.
2. **Two-state difference.** ⟨|∂CE/∂g|⟩ at gates=1 minus at gates=0.5. Direct measure of "neurons that don't matter when their neighbors are killed" = redundancy.
3. **Drop the probe, accept a short pilot.** One 2-epoch pruner training at λ=0.1, read per-layer dynamics, binary-search. ~3× a probe, ~5× cheaper than a full sweep.

---

## What we won't try again

- Monotonic scalar-of-size scalings — refuted by the non-monotone λ_opt sequence.
- Single universal λ across tasks — 8× spread observed.
- `λ × N_L` product laws — refuted by 1L wide point.
- `λ × S_L = const` — refuted by 4L deep point (the apparent law was a coincidence).
- Soft-λ → hard-budget reformulation (top-K) — F12 closed this with a structural negative result.

---

## Connections to other findings

- **F4 / F10 / F14** (width = redundancy): saturation factor is the data-side manifestation of this. Narrow saturated nets need higher per-gate pressure because each neuron is more load-bearing.
- **F13 / F15** (BiLSTM transfers, efficiency curves): provides the observed λ_opt values; F15 set up the H2 refutation cleanly with 3 nets, this expands to 5.
- **Refined H3** (λ_sim vs λ_Pareto on saturated nets): on saturated LeNet they diverge (0.25 vs 0.04); on over-parameterized nets they coincide. The gradient probe might predict λ_sim more cleanly than λ_Pareto since λ_sim is a regime boundary, not an optimum.
