# Methodology Review — faults & naivities in the experimental program

A code-level audit of how the experiments are set up and what the headline findings
actually rest on. Ordered by how load-bearing the fault is. Each item: **what — why it
biases the conclusion — fix.** Source files cited inline. PDF version:
`diary/methodology-review.pdf` (generator: `docs/methodology_review_pdf.py`).

Scope note: many of these do **not** mean a finding is wrong — they mean the finding is
not *yet* isolated from a confound, or is reported more strongly than the protocol earns.
The robust results (transfer-fails F7, the qualitative width>depth trend) are flagged as
such at the end.

---

## CRITICAL — these undercut headline findings

### C1 — One base checkpoint per architecture; the shape→prunability program has no base-net variance
`scripts/hypernetwork/shape_studies/iso_accuracy_retrain.py` loads a single fixed `.pt`
per shape (wide / medium / deep / narrow). The "2 seeds" reseed only the **pruner**
(`set_seed(seed)` at line 167, then a fresh `BiLSTMPruner`). The base net is never
retrained.

- **Consequence.** F4 / F10 / F11 ("wide 84.6% > medium 76.1% > deep 60.2%") compare
  *one* wide training run against *one* deep run. The width/depth gap could be the
  particular init + SGD trajectory of those four checkpoints, not the architecture.
- The reported ± is the half-range over pruner inits of a **near-deterministic** readout,
  so it structurally cannot see the dominant variance source (base-net training).
- **Fix.** ≥3 independently trained base nets per shape; report mean ± std over base
  seeds, with the pruner retrained on each. This is the single most important change.

### C2 — No held-out validation anywhere: the test set selects the operating point AND reports it
`best_within_budget()` picks, per seed, the most-aggressive `sw` whose **test** drop ≤ 2pp,
then reports that same test number as "survivors@2pp":

```
survivors@2pp = min_sw { survivors(sw) : drop_test(sw) ≤ 2pp }
```

- **Consequence.** An argmin over a noisy constraint *checked on the reported set* is
  optimistically biased — you keep whichever `sw`'s test-noise dipped under 2pp. The same
  pattern recurs in `dim_sweep.py` and the λ sweeps: the test set is the only held-out data.
- **Fix.** MNIST has 60k train images — carve a 5–10k validation split, select `sw` / λ on
  it, report drop on the untouched test set.

### C3 — The RL agent is trained and selected on the test set
In `src/rl/env.py`, `eval_x, eval_y` come from `test_loader`
(`multi_seed_compare.build_fixed_batches(test_loader, ...)`). `orig_acc`, the per-step
reward, and `final_acc` are all computed on those 256 **test** samples. `best_greedy_gates`
is then chosen by `g_info["final_acc"]` (greedy episode selected by test accuracy) and
reported on the full test set.

- **Consequence.** The BiLSTM-vs-RL comparison is not apples-to-apples: BiLSTM's signal is
  train minibatches, RL's signal *is* the test set. RL has strictly more information and
  still loses — so the **direction** of F5 is conservative/safe, but it must not be
  described as a fair head-to-head. Reward + greedy selection on only 256 points → noisy
  objective and noisy selection, reported on 10k.
- **Fix.** Calibrate/reward on a train-derived split; reserve test purely for the final number.

### C4 — γ = 1.0 is hardcoded, and the F6 "MDP is degenerate" theorem rests on exactly that
`RL_GAMMA = 1.0` in every RL script. With the telescoping reward rₜ = accₜ − accₜ₋₁ the
return is Σrₜ = acc_final − acc_orig, which is **path-independent only because γ = 1**. With
γ < 1 the discounted return Σ γᵗ rₜ no longer telescopes — ordering matters and the MDP is
non-degenerate.

- **Consequence.** F6 is really "REINFORCE with γ = 1 and a telescoping reward has no
  ordering gradient" — a statement about a degenerate reward/discount *pair*, not about RL
  pruning in general. The diary already flags γ as the load-bearing default (CLAUDE.md), but
  the scripts producing F5/F6 never run γ < 1, and "RL track is CLOSED" overstates what was
  tested.
- **Fix.** A γ-sweep on the **frozen** net (not just prune-during-training) is the missing
  control that would actually close, or reopen, the RL line.

---

## SIGNIFICANT — wrong or misleading metrics

### S1 — `train_pruner.py` reports trajectory-averaged train-minibatch numbers as "RESULTS"
`write_summary` computes `avg_drop`, `avg_orig_acc`, `avg_pruned_acc` as means over **all
1000 steps** on 64-sample **train** batches (lines 200–202), averaging in the early steps
where nothing was pruned yet, on training data, under a "RESULTS / Avg acc drop" heading.
Only `avg_gate[-1]` is a converged quantity.
- **Fix.** Headline = final mask, evaluated on the test set. (Sweep scripts already do this;
  the main documented entry point does not.)

### S2 — dim_sweep's "larger nets are more prunable" (F1's 51%) has no accuracy-drop control
`dim_sweep.py` prunes every dim at fixed `SPARSITY_W = 0.05` and reports `pct_pruned` next
to **pre-pruning** `te_acc`; post-prune accuracy is never measured.
- **Consequence.** "dim=2048 → 51% pruned" is "% pruned at a fixed penalty," not "%
  prunable at iso-accuracy." At fixed absolute sparsity weight a bigger net has more free
  capacity to surrender at the *same* penalty regardless of redundancy structure, so the
  trend partly measures the λ-vs-capacity interaction. Single seed, single base-train, no
  error bars, no drop axis.
- **Fix.** Re-run at iso-accuracy (the F4 protocol). The qualitative trend may survive, but
  as run the numbers don't isolate prunability.

### S3 — The efficiency metric is dominated by a clamp floor, and cross-net comparisons ride on it
`efficiency_compare.py`: `eff = (% pruned) / max(drop_pp, 0.5)`. Whenever drop < 0.5pp the
denominator pins at 0.5, so eff ≈ 2·(% pruned) — it stops measuring the trade-off. Most
MNIST points have drop < 0.5 (medium λ=0.04 → 0.12pp).
- **Consequence.** MNIST's "peak eff 135.8" vs CIFAR_big's "47.9" (F15) is largely the
  clamp: MNIST is in the floored regime, CIFAR isn't. Comparing those two numbers as the
  same quantity, and concluding "task-driven not size-driven," compares a clamped ratio to
  an unclamped one.
- **Fix.** Report the Pareto curve (% pruned at fixed drop budgets), not a single ratio with
  a floor.

### S4 — The λ_opt·N_layers ≈ 0.097 "law" is curve-fit on argmax-over-noise, n≈3
λ_opt is the argmax of S3's clamped efficiency over a **coarse** grid
({0.04, 0.06, 0.08, …}), read to a single grid value per net. When drops sit near the 0.5
floor that argmax is decided by sub-noise % differences. Fitting `λ_opt·N_layers ≈ const`
across 3 effective datapoints (N_L ∈ {2,2,3}) and using it to *predict* 1- and 4-layer
λ_opt is overfitting a relation with more freedom than constraints.
- **Fix.** Treat as a hypothesis (n≈3), not a result; the 1-layer and 4-layer datapoints the
  diary names as "cheap to test" are exactly the missing constraints.

### S5 — "RL loses decisively" (F5) is not supported by the reported stats
BiLSTM 3.68 ± 0.79 vs actor-critic 4.71 ± 1.55, n = 5 each. Welch SE ≈
√(0.79²/5 + 1.55²/5) ≈ 0.78, t ≈ 1.03/0.78 ≈ 1.3 → not significant at 5%; the ±ranges
overlap heavily.
- **Consequence.** The broader RL-loses *pattern* across many levers is plausibly real, but
  "decisively" on this specific comparison isn't earned by n = 5. No significance test is
  run anywhere in the project.
- **Fix.** More seeds and/or a paired test (same base, paired BiLSTM/RL per seed → paired t
  or Wilcoxon).

---

## MODERATE — hygiene / smaller confounds

- **M1 — "2 seeds, BiLSTM near-deterministic" conflates two variances.** Determinism is over
  pruner init for a *fixed* checkpoint; it says nothing about base-net training variance (C1)
  or operating-point selection variance (C2). A 2-point half-range is not an error bar.
- **M2 — Cross-net efficiency mixes training budgets.** `efficiency_compare` overlays
  CIFAR_big (5 epochs) with MNIST (15 epochs) on one axis, and 1-seed "extras" with 3-seed
  points on one curve. Cross-net λ_opt/eff claims compare unequal compute and seed counts.
- **M3 — Hardcoded transcribed numbers.** `efficiency_compare.py` embeds all per-λ results as
  literals copied from `summary.txt`. These can silently drift from the runs they summarize;
  read the summaries at runtime instead.
- **M4 — RL state is stale 4 steps of 5.** `_update_activations` runs every
  `recalibrate_every = 5` steps but the reward is computed every step, so the `mean_act`
  feature the policy conditions on lags the actual masked network most of the time.
- **M5 — Per-sw reseeding correlates the "independent" runs.** In iso_accuracy_retrain
  `set_seed(seed)` is re-called before every `sw`, so within a seed all `sw` points share
  identical init and minibatch order — the 5 points are perfectly correlated; only 2 truly
  independent draws exist.
- **M6 — `train_epoch` runs the model twice per batch with dropout on.** `model(x)` at line
  12 (loss) and again at line 16 (accuracy) — the second is a different stochastic pass, so
  logged train-acc is dropout-perturbed and base training costs ~1.5× more than needed.
  Cosmetic (test eval is clean) but it's in every base-train.

---

## What is done well (calibration)

- `evaluate_with_gates` correctly zeros weight **rows and biases** and evaluates on the full
  10k test set — final reported drops, *where measured at the converged mask*, are clean.
- Activation interpretability deliberately measures alive/dead firing under the **original**
  model — the right counterfactual.
- The transfer test (F7) is structurally sound: same layer shapes, frozen-P_A vs oracle-P_B,
  full-test eval. The transfer-fails conclusion (28pp vs 4pp) is a large effect well outside
  any of the above biases — **the most robust result in the set.**
- Gradient clipping, tanh-bounded context, +2.0 bias init are reasonable and documented.

---

## The three that matter most

1. **C1** — re-run shape studies with ≥3 base-net seeds per architecture before trusting
   width > depth quantitatively.
2. **C2** — add a validation split; never select `sw`/λ on the test set you report.
3. **C3 / C4** — the RL "closed" verdict is scoped to γ=1 + test-set-as-reward; a γ<1
   frozen-net control with a train-derived reward is the honest missing experiment.

---

*Created 2026-06-20. Companion PDF: `diary/methodology-review.pdf`.*
