# Claude's notes on this research

Qualitative conclusions accumulated across the experiments in this repo, with links to source data and code. Updated as runs land. Sister file to [my-notes.md](my-notes.md).

---

## 1 · Weight-conditioned hypernetwork pruning (BiLSTM)

### Core conclusion
**Weights alone carry a learnable signature of redundancy.** A small BiLSTM pruner that only sees the trained weight matrices can prune 60% of hidden neurons at <0.1% accuracy drop. The pruner is not memorising a fixed mask — different runs converge to consistent prune fractions.
- Source: [experiments/latest/hypernetwork/](../experiments/latest/hypernetwork/training/) (sw=0.05, 1000 steps)
- Code: [src/pruners/bilstm.py](../src/pruners/bilstm.py)

### Architectural stability
BiLSTM is only stable with three concrete tricks together: **tanh-bounded cross-layer context + LayerNorm on the 2× wider BiLSTM output + +2.0 bias init on the final gate layer**. Drop any one and the policy collapses gates to zero before the task loss can react.
- Source: design comments in [src/pruners/bilstm.py](../src/pruners/bilstm.py)

### Prunability scales with width
Monotonic across `dim ∈ {32, 64, …, 2048}`: dim=32 → 7.8% pruned, dim=2048 → 51.3% pruned at 5 epochs of training and 600 pruner steps. Consistent with the Lottery Ticket Hypothesis. Wider networks expose more redundancy.
- Source: [experiments/latest/hypernetwork/dim_sweep/](../experiments/latest/hypernetwork/dim_sweep/)

---

## 2 · Activation pruning vs BiLSTM

### Pareto-dominated, not competitive
BiLSTM **strictly dominates** activation pruning at every accuracy-drop tolerance:

| Drop tolerance | Activation best | BiLSTM best |
|---|---|---|
| ≤0.5% | 34.5% pruned | **64.7%** pruned |
| ≤1.0% | 45.8% pruned | **72.1%** pruned |
| ≤3.0% | 54.7% pruned | **75.1%** pruned |

Activation pruning falls off a cliff past 50% pruning (12.6% drop at 70% pruning). BiLSTM holds ≤2% drop up to 75% pruning.
- Source: [experiments/latest/baselines/activation_vs_bilstm/](../experiments/latest/baselines/activation_vs_bilstm/)

### "Activation pruned more" in earlier sequential run was misleading
At fixed hyperparameters (λ=0.5 vs sw=0.05), activation removed 69.5% vs BiLSTM 60.9% — but lost 3.3% accuracy vs BiLSTM's ~0%. Different operating points on different curves; not a fair comparison.
- Source: discussion in [experiments/latest/baselines/sequential/](../experiments/latest/baselines/sequential/) summary

### Sequential (activation → BiLSTM) is decent but dominated by pure BiLSTM
73.3% pruned at 1.56% drop. Comparable to BiLSTM alone at sw=0.2 (72.1% at 1.07% drop) but with no architectural benefit.
- Source: [experiments/latest/baselines/sequential/](../experiments/latest/baselines/sequential/)

---

## 3 · RL / MDP pruning (REINFORCE)

### Works, with caveats
REINFORCE achieves +58 points over a random policy at 80% pruning (95.31% sampled / 92.19% greedy vs 36.72% random). The MDP framing captures ordering effects that static scorers cannot.
- Source: [experiments/latest/rl/reinforce/80/](../experiments/latest/rl/reinforce/80/)

### The 80% → 90% cliff
| Target | Best greedy acc | Random floor |
|---:|---:|---:|
| 70% | **98.44%** | 79.69% |
| 80% | 92.19% | 36.72% |
| 90% | 65.62% | 54.30% |

The accuracy cliff sits between 80% and 90%. Past 78% target, run-to-run variance dominates the signal.
- Source: [experiments/latest/rl/reinforce/70/](../experiments/latest/rl/reinforce/70/), [80/](../experiments/latest/rl/reinforce/80/), [90/](../experiments/latest/rl/reinforce/90/)

### Fine-grained sweep pinpoints the 3%-drop threshold
Sweeping 10 fractions across [0.65, 0.85]: **71.67% pruning at 98.05% accuracy (1.56% drop)** is the largest sparsity meeting the ≤3% greedy-drop budget. Threshold crosses between 71.67% and 73.89%.
- Source: [experiments/latest/rl/reinforce/sweep_65_85/](../experiments/latest/rl/reinforce/sweep_65_85/)

### REINFORCE's failure mode: ep-200 collapse
Return drops to −0.68 around ep 200 in the 80% runs. Two causes:
1. **EMA scalar baseline lags** the policy's improved return distribution → mis-estimated advantages.
2. **Late-training high entropy** — greedy ≪ sampled (69.5% vs 81.6% at ep 300) means the policy is hedging across many similar-probability neurons rather than committing.
- Source: [rl/reinforce/80/run.log](../experiments/latest/rl/reinforce/80/run.log)

### Sampled vs greedy is a diagnostic, not a result
- **greedy ≈ sampled**: confident policy, deployment-ready.
- **greedy ≪ sampled**: hedging, too much entropy.
- **greedy ≫ sampled**: entropy regularisation too low (we haven't seen this).
- Source: discussion in [docs/rl_experiment.md](../docs/rl_experiment.md)

---

## 4 · PPO attempt (failed at default hyperparameters)

### Diagnosis
PPO underperformed REINFORCE at 80% (best greedy 89.06% vs REINFORCE 92.19%). **Cause is hyperparameter, not algorithmic.**
- `entropy_coef=0.01` is sized for small action spaces. With ~2048 alive neurons, `log(2048) ≈ 7.6` makes the entropy bonus 0.076 — comparable to the policy-loss magnitude. Entropy dominates the gradient.
- Policy entropy stayed at ~6.5–7.0 across all 300 episodes — essentially uniform over the alive set. Never committed.
- Value loss stayed ≤0.001 — critic learned to predict 0 (correct in expectation) but advantages collapsed → noise.
- Source: [experiments/latest/rl/ppo/80/](../experiments/latest/rl/ppo/80/)

### Fixes to try (not yet run)
- Drop `entropy_coef` to `1e-4`, OR normalize entropy by `log(N_alive)` to make it action-space-invariant.
- Bump LR from 3e-4 back to 1e-3 — PPO's clipping makes this safe.
- Multi-episode rollouts (4–8 per update) for sharper value targets.

---

## 5 · Interpretability

### BiLSTM partially correlates with activation magnitude
Dead/alive activation ratio is stable across BiLSTM sparsity settings:

| Run | Pruned | Alive act | Dead act | Ratio |
|---|---:|---:|---:|---:|
| BiLSTM sw=0.3 | 75.83% | 0.8138 | 0.2547 | **0.313** |
| BiLSTM sw=0.5 | 78.81% | 0.8567 | 0.2663 | **0.311** |

The 0.31 ratio across sparsity levels means the pruner's *strategy* doesn't change with sw — it just kills more neurons of the same activation profile. The BiLSTM is closer to a sophisticated activation-magnitude scorer than to a true joint reasoner.
- Source: [hypernetwork/interp_sw0.3/](../experiments/latest/hypernetwork/interp_sw0.3/), [hypernetwork/interp_sw0.5/](../experiments/latest/hypernetwork/interp_sw0.5/)
- Code: [src/interpretability.py](../src/interpretability.py)

### REINFORCE prunes less activation-aligned than BiLSTM
At matched sparsity (~80%):

| Method | Pruned | Alive act | Dead act | Ratio |
|---|---:|---:|---:|---:|
| BiLSTM sw=0.5 | 78.81% | 0.8567 | 0.2663 | 0.311 |
| REINFORCE 80% | 80.47% | 0.7315 | 0.2924 | **0.400** |

REINFORCE kills neurons that fire ~30% more on average than BiLSTM's kills. Because each RL decision is conditioned on the *current* state of the network (recalibrated activations, fraction pruned), it can afford to remove moderately-active neurons whose role is filled by others. The hypernetwork only sees weights and so leans more on what is essentially an activation-proxy via weight magnitude.
- Source: [rl/reinforce_interp/80/](../experiments/latest/rl/reinforce_interp/80/) vs [hypernetwork/interp_sw0.5/](../experiments/latest/hypernetwork/interp_sw0.5/)

### Full test-set drops at matched sparsity — 5-seed comparison

| Method | % Pruned | Drop (pp) | Dead/alive ratio |
|---|---:|---:|---:|
| BiLSTM sw=0.5 | 79.17 ± 0.16 | **3.68 ± 0.79** | **0.308 ± 0.007** |
| REINFORCE 80% | 80.47 ± 0.00 | **6.01 ± 4.17** | **0.486 ± 0.144** |

Per-seed REINFORCE drops: 3.64, 5.75, **13.27**, 4.06, 3.32 (seed 2 is an outlier collapse). BiLSTM: 3.19, 2.83, 3.80, 3.68, 4.90 (tight).

Key takeaways from the 5-seed runs:
- **BiLSTM dominates REINFORCE in expectation** at matched sparsity.
- **BiLSTM's strategy is essentially seed-invariant.** Ratio σ = 0.007 means the pruner finds the same prune policy each time — strong evidence that the weight matrix contains a near-unique optimal mask for the hypernetwork to find.
- **Reproduced exactly on independent 5-seed re-run** ([hypernetwork/bilstm_5seed/](../experiments/latest/hypernetwork/bilstm_5seed/)): 3.68±0.79, ratio 0.308±0.007, per-seed values identical to 2dp. The BiLSTM pruning is **near-deterministic** given the frozen checkpoint — the redundancy readout is a property of the network, not of pruner init. This is a precondition for the "pruner-as-instrument" and "transfer" research framings (§11).
- **REINFORCE's variance is the real bottleneck.** σ = 4.17pp on accuracy drop means single-seed comparisons are nearly meaningless; you need ≥3 seeds to draw any conclusion. This is the strongest empirical argument so far for moving to actor-critic / properly-tuned PPO.
- Source: [rl/variance_study/reinforce_vs_bilstm/summary.txt](../experiments/latest/rl/variance_study/reinforce_vs_bilstm/summary.txt), [rl/variance_study/reinforce_vs_bilstm/plot.png](../experiments/latest/rl/variance_study/reinforce_vs_bilstm/plot.png)
- Code: [scripts/multi_seed_compare.py](../scripts/rl/multi_seed_compare.py)

### Actor-critic (learned V(s) + reward-to-go) cuts variance 2.7× but doesn't catch BiLSTM
5-seed run, identical config to the REINFORCE run except the EMA scalar baseline is replaced with a learned value net and reward-to-go advantages (entropy_coef left at 0.01 for isolation).

| Method | Drop (pp) | σ (drop) | Dead/alive ratio |
|---|---:|---:|---:|
| BiLSTM sw=0.5 | 3.68 | **0.79** | 0.308 ± 0.007 |
| REINFORCE 80% | 6.01 | **4.17** | 0.486 ± 0.144 |
| Actor-Critic 80% | 4.71 | **1.55** | 0.454 ± 0.123 |

Key takeaways:
- **The variance fix worked as the theory predicted.** Drop σ fell 4.17 → 1.55 (2.7×); mean improved 6.01 → 4.71. No seed collapsed (REINFORCE worst seed 13.27pp → AC worst seed 6.13pp). This is the `E[R²] → E[δ²]` reduction from the variance derivation, isolated to the baseline change.
- **Still loses to BiLSTM** on both mean and σ. The gap narrowed but didn't close — the value baseline addresses one of four variance sources (the baseline), leaving entropy mis-scaling and multi-categorical action noise. Residual σ=1.55 is the budget for the next two fixes (entropy normalization, per-neuron Bernoulli).
- **RL-family prunes higher-firing neurons regardless of optimiser.** AC ratio 0.454 sits with REINFORCE's 0.486, far from BiLSTM's 0.308. So the activation-alignment difference is a property of the *sequential MDP framing*, not optimiser variance — strengthens the joint-reasoning claim.
- Source: [rl/variance_study/actor_critic/summary.txt](../experiments/latest/rl/variance_study/actor_critic/summary.txt), [rl/variance_study/actor_critic/](../experiments/latest/rl/variance_study/actor_critic/)
- Code: [scripts/multi_seed_ac.py](../scripts/rl/multi_seed_ac.py), [src/rl_ac_train.py](../src/rl/ac_train.py), [src/pruners/rl_value.py](../src/pruners/rl_value.py)

### Normalised entropy REGRESSED the result — entropy bonus was load-bearing exploration
3-seed, 500-episode AC run with entropy normalised by log(N_alive) (so bonus ∈ [0,1] × coef).

| Method | Drop (pp) | σ |
|---|---:|---:|
| AC (raw entropy) | 4.71 | 1.55 |
| AC + normalised entropy | **6.40** | **3.73** |

**Surprising negative result — the earlier diagnosis was backwards for this setup.** Normalising shrank the bonus from ~0.076 (raw, `0.01·log(2048)`) to ~0.01, i.e. **8× weaker exploration pressure**. The policy committed earlier; seed 2 collapsed to 10.66pp, reviving the REINFORCE-style tail the value baseline had suppressed. The 0.076 bonus wasn't a bug — it was the exploration keeping unlucky seeds from locking onto bad masks.
- **Raw-entropy AC (4.71 ± 1.55) remains the best RL result.**
- **The residual gap to BiLSTM is NOT an entropy problem** — both entropy settings match-or-worsen it. Next lever is the multi-categorical action (per-neuron Bernoulli), not entropy tuning.
- Source: [rl/variance_study/actor_critic_norment/summary.txt](../experiments/latest/rl/variance_study/actor_critic_norment/summary.txt), [convergence.png](../experiments/latest/rl/variance_study/actor_critic_norment/convergence.png)
- Code: [scripts/multi_seed_ac_norment.py](../scripts/rl/multi_seed_ac_norment.py)

### Convergence: 300 episodes is enough; ~270 is the plateau
Mean running-best full-test drop plateaus hard: 33pp (ep10) → 7.1pp (ep130) → flat to ep250 → 6.67pp (ep270) → 6.40pp (ep500). **Only 0.12pp gained from ep 300→500.** Plateau (< 0.3pp improvement remaining) hits at **ep 270**. Future runs: use 300 episodes; 500 is wasted compute.
- Source: [rl/variance_study/actor_critic_norment/summary.txt](../experiments/latest/rl/variance_study/actor_critic_norment/summary.txt)

### Chunk size is NOT the variance lever — smaller k is worse
Tested whether shrinking the prune chunk k (neurons removed per macro-step) reduces variance / improves performance. It does neither.

1-seed sweep (capped, confounded by episode budget):
| k | episodes | drop (pp) |
|---|---:|---:|
| 8 | 250 | 2.68 |
| 4 | 200 | 3.83 |
| 1 | 100 | 4.53 |

But k=1/k=4 were under-trained (each episode = 1 gradient update, smaller k = fewer episodes affordable; return traces showed k=1 still thrashing at ep100). **Key lesson: smaller k → longer horizon → needs MORE gradient updates to converge, fighting the per-episode cost.** "More customizable MDP" is real but untrainable in a reasonable budget.

Clean k=8 3-seed (equal 200-episode budget, no confound):
| Config | drop (pp) |
|---|---:|
| BiLSTM 5-seed | **3.68 ± 0.79** |
| AC k=16 5-seed | 4.71 ± 1.55 |
| **AC k=8 3-seed** | **5.54 ± 3.44** |
| AC k=8 1-seed (cherry) | 2.68 |

k=8 per-seed: 2.47, 4.89, **9.26**. The single-seed 2.68 was the bottom of a wide distribution, not a real win. **k=8 is worse than k=16 on BOTH mean and variance (σ 3.44 vs 1.55).** The cleaner-action benefit is swamped.
- **Variance sources now ruled out as tunable levers:** entropy normalization (worse), chunk size (worse). The residual variance looks structural to the sequential-pruning MDP, not a knob.
- **One-seed RL numbers are meaningless at this variance** — same lesson as the original REINFORCE comparison.
- Source: [rl/chunk_sweep/summary.txt](../experiments/latest/rl/chunk_sweep/summary.txt), [rl/chunk_sweep/k8_3seed/summary.txt](../experiments/latest/rl/chunk_sweep/k8_3seed/summary.txt)
- Code: [scripts/chunk_sweep.py](../scripts/rl/chunk_sweep.py), [scripts/chunk_k8_k16_3seed.py](../scripts/rl/chunk_k8_k16_3seed.py)

### Layer 2 tolerates more "active neuron" pruning than Layer 1 across all methods
Dead/alive ratios are consistently higher in Layer 2:
- BiLSTM sw=0.3: L1 0.264, L2 0.349
- BiLSTM sw=0.5: L1 0.261, L2 0.376
- REINFORCE 80%: L1 0.361, L2 0.444

Layer 2 features are evidently more substitutable — multiple neurons cover similar functions. Layer 1 features tied to raw pixel structure are harder to swap out.

---

## 6 · Design / architecture notes

### Permutation invariance
- Policy net uses **single-query dot-product attention** (pointer-network style): per-neuron features → MLP → keys; global features → MLP → query; logit = key·query. Equivariant by construction.
- Value net uses **mean-pool over the alive set** after a shared encoder. Invariant by construction (Deep Sets recipe, Zaheer 2017).
- This is the minimum architecture needed for a variable-length alive set; full self-attention is the natural next-step upgrade.
- Code: [src/pruners/rl_policy.py](../src/pruners/rl_policy.py), [src/pruners/rl_value.py](../src/pruners/rl_value.py)

### State exposed to the RL policy
8 per-neuron features + 3 global. Per-neuron: in-L1/L2 norms, out-L1/L2 norms (against alive downstream), mean activation, layer one-hot, fraction-pruned-layer. Global: CE-gap from original, fraction pruned, current accuracy.
- Source: [src/rl_env.py](../src/rl/env.py)

---

## 7 · Scaling considerations (not yet experimentally validated)

### Smaller networks (~90k params, e.g. dim=100)
- From dim_sweep: prunability collapses below dim≈128. Targeting 80% on a 100-wide MLP is infeasible regardless of algorithm.
- REINFORCE should actually train *easier* (shorter episodes, smaller policy) once the prune target is set to a feasible level (10–20%, not 80%).

### Larger networks (~90k neurons)
- **PRUNE_CHUNK must scale with N** to keep episode length ~constant. `chunk ≈ 0.8% × N_hidden` keeps ~100 macro-steps regardless of model size.
- **REINFORCE variance grows with horizon** — actor-critic becomes load-bearing past 1k neurons.
- **`entropy_coef` must scale inversely with `log(N_alive)`** or the bonus dominates the policy gradient.
- **Reduce `recalibrate_every`** at scale — recalibration is the most expensive per-step op as model grows.

---

## 8 · Cross-cutting takeaways

1. **The MDP framing itself is the bottleneck, NOT the optimiser** (corrected — earlier belief was wrong). On a *frozen* base model, pruning is static subset selection. The telescoping reward (`acc_t−acc_{t−1}`, γ=1) sums to `acc_final−acc_orig` → path-independent → the sequence is vestigial. No variance-reduction or convergence trick can give RL an edge it structurally doesn't have. See §10.
2. **Width is destiny.** Across every method tried, prunability scales monotonically with hidden width. Underparameterised models will not prune well no matter what algorithm you point at them.
3. **Joint reasoning shows up in measurable interpretability differences.** The 0.31 vs 0.40 alive/dead ratio gap between BiLSTM and REINFORCE is real signal, not noise — it reflects the difference between scoring neurons from static weights vs reasoning about a changing live set.
4. **Sequential pipelines (activation → learned) don't add value** when the learned method is strong enough on its own. Holdover from when only weak baselines existed.

---

## 9 · Deferred ideas (to revisit after current variance-reduction work)

### Richer pruner input: gradients + activations (not just weights)
Feed the BiLSTM per-neuron activation stats (mean/var post-ReLU on calib batch) and/or Taylor importance (|∂L/∂a·a|) alongside weights.
- Activations: cheap (1 fwd pass), modest gains easy-regime, bigger near the cliff. Gradients/Taylor: most informative for hard cliff decisions (1 bwd pass).
- **Best reason = TRANSFER (Q6/F7):** weight values are per-network (permutation/scale); activation & gradient *statistics* are permutation-invariant and comparable across nets → could enable the transfer weight-only failed at. Test as: weights / +act / +grad ablation, measuring Pareto move AND transfer-on.
- **Ceiling caveat (F11):** frozen-net pruning is capped ~240K weights regardless of scorer — richer input approaches the floor more reliably + maybe transfers, but won't break past it.

### Curriculum / staged greedy pruning (prune N least-damaging, re-rank, repeat to threshold)
= iterative pruning + nested/anytime objective + knee-termination, with BiLSTM as the per-stage scorer. **TWO separate axes — don't conflate (I did, user corrected):**

(a) **Solution quality (frozen net):** staging only helps via better mask-space SEARCH; BiLSTM already scores jointly, so modest, still bounded by ~240K floor.

(b) **Optimization difficulty (frozen net) — THIS IS THE REAL WIN even without retrain.** Staging/sw-annealing is a **continuation/curriculum method**: decomposes the hard one-shot "find the 15% to keep" landscape into a sequence of easy, warm-started sub-problems. Precedent = Automated Gradual Pruning (Zhu & Gupta 2017): ramping sparsity beats jumping-to-target. Benefits: (1) far more STABLE — likely lets us DROP the +2.0 bias / tanh / LayerNorm hacks, which exist *only* to tame aggressive one-shot; (2) plausibly fewer steps (early stages converge fast, later warm-started) — but must MEASURE, K·M could exceed 1000 if over-staged; (3) same final floor. **Most valuable at SCALE** (CIFAR/large nets), where aggressive one-shot pruning is genuinely unstable; small win at MNIST (one-shot already converges in ~90s).

(c) **WITH retrain between stages (the other prize):** genuinely sequential (§10 escape), = iterative-magnitude-pruning. **Only path past the ~240K frozen floor toward the ~82K train-small floor (F11).** Yields a nested mask family (aligned anytime objective). Greedy "least-damaging N" is myopic (usually fine).

- **Clean A/B test:** one-shot BiLSTM (sw=0.5) vs sw-annealed (ramp 0→0.5, AGP-style) to same sparsity → measure steps/wall-clock to floor, stability across seeds (does it survive WITHOUT bias/tanh/LayerNorm hacks?), final drop. Prediction: same floor, ≤ steps, much more stable.
- **Synthesis:** staged + retrain + gradient/activation input = learned iterative Taylor pruning, near-SOTA-structured.

### REFINEMENT: enforce TOP-K (not λ) in the curriculum — better formulation
Decided top-K enforcement beats λ-annealing for the sparsity-target curriculum:
- **Kills per-model λ tuning** (λ→K is 8× different across shapes: deep 0.8 vs wide 0.1). Top-K schedule (90%→80%→…) is architecture-portable.
- **Better-posed:** decouples how-many (you set K) from which (pruner learns the ranking). λ has to decide both.
- **Can't collapse → drop the stabilization hacks** (+2.0 bias, sparsity term exist only to babysit soft λ). Clean test of whether tanh/LayerNorm were load-bearing or just λ-babysitting.
- **Natural nested mask family** (anytime deliverable).
- **Implementation:** STE-top-K — move binary_ste's threshold from 0 to the K-th-largest score; forward keeps exactly K, backward flows through soft scores. ~5-line change, no λ.
- **Two design forks (matter more than top-K-vs-λ):** (a) hard nesting (clean family, greedy/myopic) vs re-rank+warm-start each K (best per-K masks, recommended); (b) frozen (capped ~240K, curriculum = optimization-ease only) vs retrain-between-stages (the real win, breaks the floor toward 82K).
- **First build:** STE-top-K BiLSTM, decreasing-K on frozen net, no λ/no +2.0-bias → check it (a) matches λ pruner's 240K floor, (b) survives without the hacks. Cheap validation of the whole reformulation.

### Per-neuron Bernoulli action space — TESTED, COLLAPSED (3 seeds)
Replaced "pick k via multinomial" with independent per-neuron Bernoulli prune decisions (exact factorised log-prob). Result: **62.65 ± 41.20 pp drop** (vs AC k=16 4.71, BiLSTM 3.68) — catastrophic.
- **Failure mode = horizon collapse.** With no per-step prune-count cap, the policy raced to 80% in 2–8 steps (vs intended ~100), dumping 200–800 neurons per step → final mask is a near-random 80% subset → acc craters to ~5–15%.
- **The chunk action was load-bearing scaffolding, not a removable proxy.** Capping each step at k is what kept episodes long and decisions incremental. Removing it broke the MDP.
- Credit assignment across 2048 simultaneous Bernoulli decisions from one scalar return is hopeless — every neuron's gradient gets the same advantage sign, so the policy learns "prune more/less" globally, not *which*.
- Source: [rl/bernoulli_3seed/summary.txt](../experiments/latest/rl/bernoulli_3seed/summary.txt)
- Code: [src/pruners/rl_bernoulli_policy.py](../src/pruners/rl_bernoulli_policy.py), [src/rl_bernoulli_train.py](../src/rl/bernoulli_train.py), [scripts/bernoulli_3seed.py](../scripts/rl/bernoulli_3seed.py)

### Local-state MDP (drop global trajectory features)
Currently the state contains both per-neuron features and global trajectory (`CE gap, frac_pruned, current_acc`). Try **local-only**: each decision based purely on per-neuron features (incoming/outgoing norms, mean activation, layer one-hot, layer-local sparsity).

Why this could matter:
- Cleaner credit assignment per neuron.
- Smaller state → easier transfer across architectures and model sizes (zero-shot from MLP-1024 → MLP-512).
- Tests whether the global context actually carries information or is a memorization shortcut.

Trade-off: loses information about how much budget is left in the episode. Worth running as an ablation.

### Infinite-horizon MDP (sparsity-as-reward, terminate on accuracy drop)
Replace the fixed 80%-prune cutoff with a different MDP formulation:

- Reward: `+1` per neuron pruned (or `+sparsity_gain` per step)
- Termination: when test accuracy falls below threshold (e.g. baseline − 3pp)
- Episode length variable; agent learns to maximise sparsity subject to accuracy constraint

Why this could matter:
- *Discovers* the optimal sparsity level rather than being told.
- Natural answer to "what's the max sustainable pruning?" without baking it into the cutoff.
- Better matches what we actually care about (Pareto-optimal sparsity given accuracy tolerance).

Only worth running once variance-reduction (actor-critic, PPO retuning) is settled.

### Stagnation / knee-termination MDP (variant of infinite-horizon)
Instead of terminating at a fixed accuracy *threshold*, terminate when the **marginal cost of pruning crosses a slope** — i.e. detect the Pareto knee from the trajectory shape:

    terminate when  Δacc / Δpruned  over a recent window  <  −τ

- Reward: `+1` per neuron pruned. Return = neurons pruned before the knee.
- Agent's job collapses to: **choose the prune order that keeps the accuracy slope flat as long as possible** (pushes the cliff rightward). Dense, self-normalising objective.

Key observations:
- **Breaks Markov property.** "Slope over recent window" depends on trajectory history, not current state. Must fold trajectory stats into the state (recent acc slope, drop-since-start, steps-since-expensive-prune) — the current state has absolute accuracy but not its derivative.
- **Might LOWER variance, not raise it.** The fixed-80% target forces the agent past the knee into the cliff zone, which is inherently high-variance (small mask diffs → big acc swings; this is why seed 2 collapsed to 13pp). Knee-termination keeps the whole trajectory in the flat region. Random horizon adds variance but better-behaved rewards may net reduce it. **Testable hypothesis.**
- **τ is the dual of BiLSTM's `sparsity_weight` λ.** BiLSTM solves `max(acc − λ·sparsity)` and sweeps λ to trace the front; this MDP walks the front and stops at slope = τ. Same "how much accuracy is one neuron worth" knob, relocated from a loss weight to a stopping slope. Advantage: τ is more interpretable and needs no pre-committed sparsity level.
- **Reward-hacking risk:** agent doesn't control termination, the detector does. Lenient detector → agent dumps damage right before a late cutoff. Strict detector → episodes end on noise dips. The knee-detector (window size, τ, smoothing, eval-batch size) is a sub-problem with its own bias/variance trade-off — **design and validate the detector on fixed prune trajectories before wiring it into a live MDP.**

Sequencing: only after the existing fixed-target MDP's variance floor is known (AC + normalised entropy + Bernoulli action).

---

## 10 · WHY THE RL/MDP FRAMING IS WRONG (and the one direction that fixes it)

This is the central negative conclusion of the RL track, derived after exhausting every variance/optimiser lever (REINFORCE → AC → entropy scaling → chunk size → Bernoulli action; all match-or-worsen BiLSTM's 3.68±0.79).

### The core argument: frozen model ⇒ static subset selection
With the base model **frozen** (weights never retrained between prunes), the accuracy of a mask is a fixed function of *which* neurons are gone — order is irrelevant: `acc(remove A then B) = acc(remove B then A)`. Pruning is therefore **static subset selection** ("pick the best 80% to delete"), not a sequential decision problem. Any MDP layered on top is a costume.

### No reward design rescues it (on a frozen model)
- **Telescoping reward** `r_t = acc_t − acc_{t−1}`, γ=1 → return = `acc_final − acc_orig`. **Path-independent**: depends only on the final mask. The sequence is vestigial; the per-step rewards carry no aligned credit.
- **Global reward** `r_t = acc_t − acc_orig` → return = `Σ acc_t − T·acc_orig` = (area under the accuracy-vs-step curve) − const. This **IS path-dependent** (intuition that "it's just a sum, order-independent" is WRONG — it's a sum of order-dependent terms; same final mask gives different returns by order). **But** the path-dependence is over *intermediate* accuracies, which are **deployment-irrelevant** if you only ship the final 80% mask. It optimises the wrong thing (smooth descent) and muddies the final-mask value signal (a good mask now has a *range* of returns by order). For single-target pruning it's arguably worse than telescoping.
- **The one aligned case:** if the objective is the *whole curve* — **anytime / nested pruning**, good accuracy at *every* sparsity level — then the integral reward is exactly right and the sequence is genuinely meaningful. Different problem; not what we've optimised.

### Consequence
RL has **no structural advantage** to exploit on a frozen model. The direct differentiable method (BiLSTM hypernetwork — outputs a mask, backprops the exact CE loss) is RL's performance **ceiling**, approached from below at far higher variance. The only theoretical crack is RL's exploration escaping a local minimum the hypernetwork falls into — empirically swamped by variance. **Variance reduction can only claw RL toward the hypernetwork, never past it.** The variance was the symptom; the absent sequential structure is the disease.

### THE FIX → genuine research direction: prune DURING training, not post-hoc
Order only matters if the network **co-evolves with the pruning**. If we fine-tune / continue training the base model *between* prune steps, then removing A genuinely changes what B becomes; `acc(A then B) ≠ acc(B then A)`; the path is real and consequential. This is exactly why iterative-prune-and-retrain beats one-shot in the literature — sequential structure *exists* once weights adapt.

Concretely, integrate pruning into the training loop:
- Interleave training epochs with prune steps; the pruner (RL policy or hypernetwork) acts on a model that is still learning / re-adapting.
- Now the MDP is genuine: state = (weights, training progress), action = prune, transition includes the network's *adaptation* to the prune. Reward can be end-of-training accuracy at target sparsity — and it does NOT reduce to a path-independent function, because the weights' trajectory depends on when/what was pruned.
- This is the setting where a learned sequential pruner has a *principled* reason to beat a one-shot hypernetwork (lottery-ticket / rewinding connections here).
- Open design questions: prune-vs-train cadence, whether to rewind weights (lottery ticket) or continue, whether the pruner sees gradients/optimiser state as part of the state, cost of the inner training loop per env step.

**This is the highest-value next direction for the RL line — it's the only framing where the sequential machinery isn't vestigial.** Everything done so far is the degenerate static case where the hypernetwork should and does win.

---

## 11 · Transfer & shape experiments (what generalizes)

### Shape, not neuron count, determines prunability — MONOTONIC in width
Same 2048-neuron budget, three shapes, BiLSTM sw=0.5:

| Shape | Width | Layers | % pruned | Drop (pp) | Ratio |
|---|---:|---:|---:|---:|---:|
| `[2048]` | 2048 | 1 | 93.47% | 7.32 | 0.288 |
| `[1024,1024]` | 1024 | 2 | 79.17% | 3.68 | 0.308 |
| `[512×4]` | 512 | 4 | 58.20% | 1.05 | 0.331 |

Prunability falls monotonically as you trade width for depth at fixed neuron count (93→79→58%). Width concentrates redundancy; depth and param count anti-correlate with it. The dead/alive ratio stays ~0.31 across all three → the pruner's *selection strategy* is shape-invariant; there's simply **less redundancy to find** in narrow layers (deep model prunes conservatively, only 1pp drop). Consistent with the dim-sweep (narrow=low redundancy) and lottery-ticket (per-layer overparameterization creates winning subnetworks).
- Source: [hypernetwork/transfer_wide2048/](../experiments/latest/hypernetwork/transfer_wide2048/), [hypernetwork/shape_deep4x512/](../experiments/latest/hypernetwork/shape_deep4x512/)

### Prunable fraction is not constant — small nets sit near a task floor
Test: if [1024,1024] keeps ~427, take [205,205] (410 total, ~20% budget) — already ~the survivor size. At sw=0.5 it pruned **52.6% → 194 kept, but at 15.81pp drop and dead/alive ratio 0.645** (vs stable ~0.31). The ratio spike = the pruner exhausted redundant neurons and was forced to cut useful ones; the 52% is coerced by the sparsity weight, not free slack.

| Model | Total | Kept | Drop pp | Ratio |
|---|---:|---:|---:|---:|
| `[2048]` | 2048 | ~134 | 7.32 | 0.288 |
| `[1024,1024]` | 2048 | ~427 | 3.68 | 0.308 |
| `[512×4]` | 2048 | ~856 | 1.05 | 0.331 |
| `[205,205]` | 410 | 194 | 15.81 | 0.645 |

- **Fixed-fraction REJECTED:** [205,205] couldn't prune 79% for free (pays 16pp); [1024,1024] prunes 79% for 3.7pp.
- **Leans task-floor:** narrow net has least slack (only model to hit the accuracy cliff). Suggests a soft absolute minimal-useful-subnetwork (~few hundred for MNIST); "prunability %" mostly measures starting distance above it.
- **Caveat:** sw=0.5 is iso-*pressure*, not iso-*accuracy* (drops span 1–16pp) so survivor counts aren't directly comparable. Decisive follow-up = prune all shapes to a fixed drop (e.g. 2pp) and compare absolute survivors. [Q9] → done below.
- Source: [hypernetwork/shape_narrow205x2/](../experiments/latest/hypernetwork/shape_narrow205x2/)

### Iso-accuracy survivor sweep — no universal floor (Q9 answered)
Threshold-sweep on each pruner's continuous scores (cheap: 1 train + ~33 evals/model, no retraining). Absolute survivors at ≤2pp drop:

| Model | Total | Kept @2pp | % pruned |
|---|---:|---:|---:|
| narrow `[205,205]` | 410 | 360 | 12.2% |
| medium `[1024,1024]` | 2048 | 1179 | 42.4% |
| deep `[512×4]` | 2048 | 744 | 63.7% |
| wide `[2048]` | 2048 | 806 | 60.6% |

- **Q9 answered: NEITHER absolute floor NOR clean width-scaling.** Survivors span 360→1179 — architecture-dependent.
- **Narrow net is near a HARD floor**: razor-sharp cliff (0→5pp in ~50 neurons), only 12% removable at 2pp. The earlier "52% at sw=0.5" was sparsity-weight-coerced destructive pruning, not free redundancy.
- **Wider nets carry large cheaply-removable redundancy.** Prunability is a property of how redundancy is *distributed*, not a fixed task constant.
- **Method caveat:** re-thresholding one pruner's fixed ranking is ~1–2pp pessimistic away from its training sw (checked vs trained sw=0.5 points). Non-convergence is real; exact counts would shift slightly under per-sparsity retraining.
- Source: [hypernetwork/iso_accuracy_sweep/](../experiments/latest/hypernetwork/iso_accuracy_sweep/)

### Rigorous iso-accuracy (retrained per sw, 2 seeds) — width cheap, depth expensive
Retrained a fresh pruner at each sw (per-model grids bracketing 2pp), picked the most-pruned ≤2pp point. Supersedes the pessimistic threshold-sweep.

| Model | Layers | Survivors@2pp | % pruned | sw |
|---|---:|---:|---:|---:|
| wide `[2048]` | 1 | 314 ± 2 | **84.6%** | 0.1 |
| medium `[1024,1024]` | 2 | 490 ± 4 | 76.1% | 0.3 |
| deep `[512×4]` | 4 | 814 ± 0 | 60.2% | 0.8 |
| narrow `[205,205]` | 2 | 296 ± 20 | 27.9% | 0.06–0.1 |

- **Prunability monotonic in WIDTH** (wide 84.6 > medium 76.1 > deep 60.2 > narrow 27.9 %) — widest prunes best. The threshold-sweep's "deep beats wide" was a pessimism artifact (it underestimated wide by 806 vs true 314 survivors). Lesson: threshold-sweeps mislead away from the training operating point; retrain to be sure.
- **Absolute survivors grow with DEPTH** (314→490→814 for 1→2→4 layers). NOT a single universal floor. The 1-layer wide net (2048) compresses to ~314 — nearly the narrow net's ~296 floor — while the 4-layer net needs ~814.
- **Dual claim: width = cheap parallel redundancy; depth = load-bearing capacity.** Every layer is a sequential bottleneck that can't be gutted, so deeper nets need more total neurons at iso-accuracy. Minimal subnetwork is network-determined by depth, near-insensitive to excess width.
- Variance ±0–20 → BiLSTM near-deterministic, 2 seeds ample.
- Source: [hypernetwork/iso_accuracy_retrain/](../experiments/latest/hypernetwork/iso_accuracy_retrain/)

### Weights, not neurons, are the conserved quantity — compute floor is architecture-invariant
The neuron-pruning % is misleading for compute. Pruning a hidden neuron removes its incoming row + outgoing column; input(784×h1)/output(h×10) matrices are one-sided (LINEAR savings), hidden→hidden matrices are two-sided (QUADRATIC, 1−f²).

| Model | Neurons -% | Weights -% | Orig W | Final W @2pp | survivors/layer |
|---|---:|---:|---:|---:|---|
| narrow `[205,205]` | 30.2% | 59.9% | 205K | **82K** | [81, 205] |
| medium `[1024,1024]` | 75.9% | 87.7% | 1862K | **230K** | [213, 281] |
| deep `[512×4]` | 60.3% | 79.7% | 1193K | **242K** | [155,308,135,216] |
| wide `[2048]` | 84.8% | 84.8% | 1626K | **248K** | [312] |

- **Weight floor converges (~240K) for the three 2048-nets** despite neuron survivors 490/814/314. Neurons are packaging (wide = few fat 784-fan-in neurons; deep = many thin quadratic-discounted ones); **~240K MACs is the conserved compute floor for MNIST@2pp, architecture-invariant.** The neuron view (F10) hid this.
- **Weight ranking correction:** medium 1.86M > wide 1.63M > deep 1.19M > narrow 0.20M. Wide has MORE weights than deep (the 784×width input matrix dominates) — counterintuitive but the arithmetic is clear.
- **Nets with hidden→hidden matrices save MORE weights than neurons** (quadratic): deep 60%→80%, medium 76%→88%. Wide (none) is linear (84.8%=84.8%).
- **Pruning ≠ training-small:** narrow reaches MNIST@2pp at 82K weights, ~3× below pruning-from-2048's ~240K floor. Pruning a big net does not reach the true minimum (cf. Liu et al. 2019).
- **sw scales with depth:** 0.1(1L)/0.3(2L)/0.8(4L) to hit 2pp — 8× more pressure for deep. Each neuron in a deep net is more load-bearing → tougher to prune.
- Source: [hypernetwork/iso_accuracy_retrain/weight_summary.txt](../experiments/latest/hypernetwork/iso_accuracy_retrain/weight_summary.txt) + weights.png + curves.png
- Code: [scripts/hypernetwork/shape_studies/weight_analysis.py](../scripts/hypernetwork/shape_studies/weight_analysis.py)

### A trained pruner does NOT transfer; the method does
Frozen P_A (trained on model A) applied to fresh models B (same architecture, different seed), no retraining:

| Condition | Drop (pp) | Ratio |
|---|---:|---:|
| P_A → A (in-distribution) | 3.19 | 0.311 |
| P_A → B (transfer, frozen) | **27.96 ± 12.77** | 0.412 |
| P_B → B (oracle, retrained) | **4.04 ± 0.61** | 0.285 |

The pruner's learned weights **overfit to A's specific weights** — useless on a different network. But the oracle (retrain on each B) reproduces ~4pp every time. **Redundancy structure is idiosyncratic per network; the BiLSTM is a reliable procedure for re-discovering it, not a transferable theory of it.** The near-determinism (σ=0.007) is per-network, not universal.
- Source: [hypernetwork/transfer_frozen_pruner/](../experiments/latest/hypernetwork/transfer_frozen_pruner/)

---

## 14 · STE-top-K curriculum (B1) — cleaner formulation, but BRITTLE (partial negative)

Goal: replace the λ sparsity penalty with a hard GLOBAL top-K budget (set K directly), drop the +2.0-bias/tanh collapse hacks, and walk K down on a warm-started decreasing-K curriculum. Validate it (a) matches λ's ~490-survivor/2pp floor on medium [1024,1024], (b) trains without the hacks. **Outcome: it does NOT cleanly validate — top-K is markedly more brittle than λ.** Baseline 98.06%, λ reference = 490 survivors (76.1% pruned) @2pp.

### The central failure: GLOBAL TOP-K STARVES A LAYER TO DEATH (proven, collapse_probe)
Every collapsing run lands on the **exact same 88.32pp** (≈chance) on medium [1024,1024] — bit-identical across all configs. Per-layer survivor logging (scripts/hypernetwork/topk/topk_collapse_probe.py) nails the cause:

| keep% | K | L1 surv | L2 surv | drop |
|---:|---:|---:|---:|---:|
| 90 | 1843 | 819 | 1024 | 0.17 |
| 75 | 1536 | 512 | 1024 | 2.39 |
| 60 | 1229 | 205 | 1024 | 13.92 |
| **50** | 1024 | **0** | 1024 | **88.32** |
| 42 | 860 | 0 | 973 | 88.32 |

**L2 stays pinned full; L1 drains monotonically 819→512→205→0.** When the budget K reaches one layer's size (1024 at the 50% stage), L1 is fully pruned → **first hidden layer severed → one-class output (88.32pp)**, absorbing (dead layer → no gradient back). Gradual erosion across the warm-started stages, NOT a kick (gradient trace has no spike).

**Why (logically tight):** per-layer standardization centers BOTH layers' score clouds at 0 with unit variance, so the *only* thing positioning one layer relative to the other in the global top-K pool is the per-layer **context bias**. We dropped the **tanh that bounded it** (judged a λ-only hack). Unbounded, it diverges (context_bias[L2] ≫ context_bias[L1]) → global top-K allocates the whole shrinking budget to L2 → starves L1. This is invariant to T-schedule / normalization / Adam-carry because none of them touch the allocation knob — which is exactly why every variant hit the identical 50% wall.
**The dropped tanh was NOT only λ-babysitting — under global top-K it was the guardrail bounding cross-layer allocation. Global top-K needs a per-layer keep-floor or bounded/normalised allocation; without one a shrinking-K curriculum reliably kills a layer.** (Earlier "tie→index-order attractor" story was WRONG: `survivors>K` ties were a minor L2 side-effect *after* L1 had already died, not the cause.)

### Formal model of the starvation + the tanh fix
Layer ℓ standardized node scores z^(ℓ) ~ (0,1); final score s^(ℓ)_j = z^(ℓ)_j + c_ℓ (c_ℓ = context bias). Global top-K keeps s ≥ τ (τ = K-th largest). Survivors n_ℓ ≈ N_ℓ(1−Φ(τ−c_ℓ)); for two equal layers n_1/n_2 = (1−Φ(τ−c_1))/(1−Φ(τ−c_2)) → 0 as Δ=c_2−c_1 → ∞. Since standardization centers both layers at 0/unit-var, the bias is the ONLY cross-layer differentiator, so unbounded Δ ⇒ severance. tanh ⇒ c_ℓ∈(−1,1), Δ∈(−2,2); with unit within-layer spread the clouds always overlap (worst case Δ=2: n_1/N≈0.16, n_2/N≈0.84, both >0) ⇒ no severance.

### CONFIRMED by intervention + a SECOND pathology revealed (collapse_probe_tanh)
Re-adding tanh on c_ℓ (only change vs v4): **the 88.32 collapse is GONE — L1 never hits 0** (floors at ~150–230). Layer-starvation mechanism CONFIRMED by intervention.
**But top-K + tanh STILL loses to λ:** @2pp ≈ 35% pruned vs λ's 76% (drop 5.8pp@50% kept, 11.5pp@24% kept). Cause = **pathology #2, the scale-invariance freeze of std-normalization:** y=(l−μ)/σ is invariant to rescaling l (L(αl)=L(l)) ⇒ the encoder output scale σ is a FREE coordinate ⇒ it random-walks up (σ: 0.7→148 observed). With detached σ, ∂L/∂l = (1/σ)∂L/∂y ⇒ ‖∇_l L‖ ∝ 1/σ → 0 ⇒ the row encoder's effective LR decays to ~0 ⇒ the per-neuron ranking FREEZES at its coarse early-curriculum state and can't refine for aggressive K. (Classic scale-invariance↔effective-LR-decay, van Laarhoven 2017.)
**Refined verdict: top-K has TWO separable failure modes — (1) layer-starvation [FIXED by tanh], (2) encoder-freeze from score normalization [now FIXED by center-only, below].**
- Source: collapse_probe_tanh.{png,txt}, probe_tanh.log; flag `bound_context` in src/pruners/bilstm_topk.py

### Pathology #2 FIXED by center-only normalization (collapse_probe_center)
node_norm="center" = subtract per-layer mean ONLY (no /σ). Keeps zero-mean (allocation still via tanh-bounded bias) but is NOT scale-invariant (L(αl)≠L(l)), so σ is loss-constrained: **σ plateaus ~21–27 instead of →148**, and the loss runs a clean per-stage **sawtooth** (spike on K-drop, encoder relearns it down) = encoder is plastic again. **Aggressive-K drops ~halved vs tanh+std_detach** (76% pruned: 11.5→5.66pp; 50%: 5.8→3.5pp). Both layers stay alive. Best top-K curve yet — but **still ~3× λ** (5.66pp vs ~2pp @76% pruned; @2pp ≈ 32% pruned vs λ's 76%).

### Residual gap = pathology #3 (mild): centered-STE band mismatch
center-only fixes the freeze but lets σ(scores)≈21 while T∈[1,4]. Centered-STE grad ∝ (1/T)σ'((s−τ)/T) is non-negligible only for |s−τ|≲T, so the fraction of neurons getting gradient ≈ T/std(s) ≈ T/21 ∈ [5%,19%] → most neurons saturated. **Tension:** std-norm matches the band to scores but freezes the encoder; center-only unfreezes but lets the scale (band mismatch) drift. Clean decoupling = **adaptive temperature T ∝ std(s)** (band covers a fixed neuron-fraction regardless of scale). UNTESTED.

### AdamW route FAILED at wd=1e-2 (too weak)
std_detach + tanh + AdamW(wd=1e-2): σ STILL blew to ~140 → freeze persists → 11.0pp @76% pruned (≈ no-WD). Reason: AdamW decoupled shrinkage = η·λ_wd = 1e-3·1e-2 = **1e-5 per step**, ~100× too weak to pin the scale-invariant encoder norm in 5100 steps. The van Laarhoven equilibrium strength is set by the PRODUCT η·λ_wd; with small lr you need a large wd (~1.0) to matter. Untested whether wd~1.0 (shrinkage 1e-3) would bound σ. **center-only remains the best top-K variant (5.66pp @76%).**

### AdamW route is a DEAD END (wd=1.0 tested)
wd=1.0 DID bound σ (L2: 141→24, diagnosis confirmed) but drop got WORSE: **15.1pp @76%** (vs 11 at wd=1e-2). The tell: wd=1.0 and center-only reach ~same σ (24 vs 21) yet differ 3× in drop (15.1 vs 5.66) ⇒ **σ-bounding alone is not the win; HOW you bound it matters.** wd=1.0 applies a blunt global penalty η·λ_wd·w to ALL pruner weights (LSTM/context/encoder) → over-regularizes → degrades the ranking. center-only bounds σ by removing the ROOT CAUSE (scale-invariance) with zero weight penalty → no over-reg tax. AdamW dead end: too weak (1e-2)→freeze, strong enough (1.0)→over-reg. **center-only wins by attacking the cause, not the symptom.**

### Standings @76% pruned (24% kept, K=492), λ ≈ 2.0pp
std_detach+tanh 11.48 | AdamW(1e-2) 11.03 | AdamW(1.0) 15.10 | **center-only+tanh 5.66 (best)**.

### B1 FINAL PICTURE — LOCKED (2-seed full curriculum, center+tanh)
Catastrophic top-K collapse = TWO fixable pathologies [unbounded allocation→tanh; scale-invariance freeze→center-only] + a THIRD residual [STE band mismatch]. Fully de-bugged config (centered STE + center-only + tanh + global-T + carried Adam), **2-seed, no collapse, σ(drop) ≤ 0.6pp**:
- **@2pp = 1340 survivors = 34.6% pruned** vs λ's 490 / 76.1%; **5.29 ± 0.36pp at λ's 76%-pruned point** (K=492) vs λ's ~2pp. Stable, monotonic, reproducible.
- **WHY it still loses (structural, beyond the pathologies):** hard-budget + centered-STE = gradient reaches only the ≈T/σ neurons near the moving threshold τ (boundary-LOCAL) and the hard K removes the soft accuracy↔sparsity trade-off; λ's soft penalty (threshold fixed at 0, +2.0 bias holding scores near 0) gives gradient to ALL neurons → joint GLOBAL optimization of the subset. The curriculum sweeps τ but warm-start/carry let the early ranking dominate → under-optimized subset. **Hard-constraint+local-gradient is a weaker subset-selection optimizer than soft-penalty+global-gradient — "sweep-free ≠ better".**
- **Verdict: B1 CLOSED. λ wins ~2.6× @ iso-accuracy.** top-K is recoverable to stability but structurally inferior; needed a guardrail stack (norm/tanh/center) λ never did.
- Source: topk_curriculum/plot_final.png, summary_final.txt, run_final.log; collapse_probe_{tanh,center,adamw,adamw1}.{png,txt}; flags `node_norm`, `bound_context` in src/pruners/bilstm_topk.py

### What we established (the diagnostic ladder)
1. **Plain STE saturates.** `binary_ste` used σ centered at 0; under λ that was fine because the keep/kill boundary WAS 0 (coincided with σ's gradient peak). Top-K moves the boundary to the K-th-largest score (far from 0) → borderline neurons sit on σ's saturated tail → ~0 gradient → frozen ranking. One-shot plain top-K @K=492 = **42.69pp** (≈ the curriculum's 41pp — so the curriculum wasn't the culprit; the estimator was). @K=1024 = 13.88pp.
2. **Centered STE** σ((s−thresh)/T) slides the gradient peak onto the boundary (restoring λ's free property). Fixed the moderate regime: one-shot @K=1024 (50% kept) **13.88→2.09pp** ✓. But @K=492 (24% kept) it **cold-collapses to 88.32pp** ✗ — at aggressive K from a cold start the global ranking is wrong and the narrow band can't repair it.
3. **Curriculum (warm-start) is WORSE than one-shot, not better.** v2/v2b/v3 all collapse at the **50%** stage — earlier than fresh one-shot (which only collapses at 24%). So continuation is actively harmful: the stage transition kicks the committed solution into the tie-attractor. Best curriculum point @2pp ≈ 40% pruned vs λ's 76%.
4. **Isolation @50% kept, FRESH init, 500 steps:** all of {none, std, std_detach}×{T=1, T 4→1} are FINE (0.43–2.14pp; fresh std best at 0.43). → No single ingredient breaks 50%; **warm-start is what breaks it.**
5. **Node normalization matters but doesn't fix it.** `std` (backprop through std) warm-starts BETTER (60%→1.21pp) but has a low-variance gradient-blowup; `std_detach` (detached mean/std) is stabler in theory but warm-starts WORSE (60%→16pp). Both still collapse at 50%.
6. **Global T anneal (4→1 across whole curriculum, no per-stage reset) did NOT fix it.** Collapsed at 50% identically. So it's not the T-reset either.

### Why warm-start hurts (best current theory)
At least two routes into the same tie-attractor: (Route 1, `std`) low-variance-after-commit → 1/std gradient blow-up → scores destabilize → tie; (Route 2, `std_detach`) the stage-boundary kick — fresh Adam's first step on a committed θ is a large normalized step (no 2nd-moment damping yet), and the threshold jump to smaller K knocks the fragile solution into the degenerate basin. Wide band (high T) at the boundary makes it easier, not harder.

### Verdict & contrast
**The "cleaner" λ-free top-K traded a per-model sw sweep for a stability cliff.** λ was robust because the +2.0 bias + soft sparsity penalty kept scores spread — there is no tie-attractor in that regime. Top-K's brittleness (5+ interacting knobs each flip ~2pp ↔ collapse) is the headline result of B1 so far.
- **Next (reasoned, not yet run):** carry Adam state across stages (kill the reset-kick, Route 2) — possibly + tie-break `topk` / add ε-noise to scores (make the attractor non-absorbing, Routes 1&2). Honest prior: ~50/50 that carry-Adam *alone* closes it, because the absorbing tie-state may be reachable by other routes.
- Source: [hypernetwork/topk_curriculum/](../experiments/latest/hypernetwork/topk_curriculum/) — run_v1/v2/v2b/v3 logs, diagnostic.txt, isolation.txt, plot.png/plot_v2.png. Code: src/pruners/bilstm_topk.py, src/topk_train.py, scripts/hypernetwork/topk_curriculum.py + topk_diagnostic.py + topk_isolation.py

---

## 12 · Expansive reframings — 5 mindset shifts (break out of "build a better pruner")

We've been optimizing *within* one frame: find the best mask for this model with this method. Each below **changes the frame**. Stored for pursuit; ordered roughly by cost-to-novelty.

1. **Pruner as scientific instrument, not compression tool.** The mask is near-deterministic (σ=0.007) → it's a *measurement* of the network's redundancy structure. Ask "what is the pruner discovering?" not "how much can we prune?" Probes: do kept neurons form an identifiable functional basis? Is the dead/alive split stable across *retrainings* of the base model (redundancy in the task) or only across pruner seeds (redundancy in the weights)? — Transfer result (§11) already says: it's in the *weights*, idiosyncratic per network.

2. **Transfer as the real result.** Does a weight-conditioned pruner generalize to unseen networks? Tested (§11): frozen weights DON'T transfer, but the *procedure* does. Open: can a pruner trained across a *distribution* of networks learn a transferable readout (meta-learning the redundancy detector)? That would be the strong version.

3. **Prunability as an effective-dimension / scaling-law probe.** % prunable at fixed accuracy = how much capacity the task actually consumed. Sweep dataset size, depth, training time, label noise → does prunability track intrinsic task complexity? Connect quantitatively to double descent & lottery tickets. Deliverable = a curve + a claim about learning, not a method.

4. **Train the network to BE prunable (flip the pipeline).** Training-time regularizer that concentrates redundancy so a clean prunable substructure emerges; then any pruner works better. Different from "prune during training" — here the *base model* is born prunable. Is there a free lunch (prunable + same acc) or a tax? Where's the Pareto front?

5. **Prune in a learned basis, not the neuron basis.** A neuron is an arbitrary coordinate. Learn a rotation R, prune *directions* in the rotated space, rotate back → equivalent network with fewer effective directions. Breaks the deepest unquestioned assumption (neuron = unit). Connects to PCA/activation-covariance pruning but learned & joint across layers.

Common thread: shift from *optimizing within a frame* → measurement (#1,#3), generalization (#2), co-design (#4), representation (#5).

---

## 13 · Generalizable lessons about RL/ML (beyond pruning)

What these experiments taught us that transfers to *other* problems:

### About when RL is the wrong tool
1. **Diagnose sequential structure before reaching for RL.** Test: does the return depend on the trajectory, or only the final state? If `r_t = Φ(s_t) − Φ(s_{t−1})` (a potential difference) on a deterministic env with a fixed endpoint, the return telescopes to `Φ(s_T) − Φ(s_0)` — **path-independent**. By the potential-based shaping theorem (Ng et al. 1999), a reward that is *entirely* a potential difference adds no information beyond the endpoint value: you have an optimization/bandit problem wearing an MDP costume. Many "we used RL for X" setups are secretly this.
2. **RL is a high-variance way to do optimization. If you can get the objective's gradient, use it.** RL earns its keep only for non-differentiable / unknown-dynamics / genuinely-sequential problems. When the objective is differentiable (or relaxable) and dynamics are known/static, direct optimization dominates — here the hypernetwork *is* RL's performance ceiling.
3. **Variance reduction cannot exceed the direct method.** Actor-critic/PPO/more-seeds only claw a policy-gradient estimator back *toward* what direct optimization already gets; they can't manufacture structure that isn't there. Only worth the effort if there's exploitable structure RL can reach that the direct method can't.

### About RL methodology
4. **Single-seed RL numbers are noise.** At σ≈4pp, one seed tells you almost nothing. Demonstrated cleanly: k=8 single-seed 2.68 vs 3-seed 5.54±3.44 — the cherry was the bottom of a wide distribution. Always ≥3 seeds, report variance. (The field routinely violates this.)
5. **Policy-gradient variance has a structural floor you can read off the reward.** `Var ≈ E[R²]·F·T` — the squared *mean* return (not its variance) is paid for free without a baseline, scales linearly with horizon, and multiplies with action-space Fisher info. You can predict a setup will be high-variance before running it.

### About problem/action/exploration design
6. **Action-space constraints can be load-bearing scaffolding, not just restrictions.** Removing the chunk constraint (→ Bernoulli) caused horizon collapse. What looks like a crude proxy may be silently keeping the problem well-posed. Audit what the "messy" part does before cleaning it up.
7. **Exploration/entropy bonuses are scale-coupled and load-bearing, not nuisance knobs.** Entropy ∝ log(action-space size); a coefficient that looks "too high" may be essential exploration. "Fixing" it (normalizing) collapsed the policy. The right value is problem-specific and often counterintuitive.
8. **Credit assignment across simultaneous multi-dim actions from one scalar is ill-posed.** When one reward must explain many concurrent decisions, the gradient pushes them all the same way → the policy learns "more/less" globally, not "which." Factored actions (Bernoulli) don't fix this for free.

### About networks/representation
9. **Overparameterization yields near-deterministic, readable redundancy structure** (σ=0.007) — but it's *idiosyncratic per network* (doesn't transfer), and **its amount is set by width, not param count or depth.** Capacity is not a scalar; its *distribution* governs compressibility.
10. **Meta-lesson on research: a negative result with a mechanism beats a tuning win.** The durable output of the RL track isn't a number — it's the telescoping-reduction theorem and the co-adaptation boundary condition. Knowing *why* and *when* generalizes; a leaderboard entry doesn't.

## 15 · CIFAR_big MLP pruning — first transfer of the BiLSTM pruner to a conv-net's FC head (F13)

This is the first time the soft-λ BiLSTM weight-conditioned pruner has been pointed at anything other than an MNIST MLP. The questions going in were: (a) does the architecture transfer at all to a conv-net's FC head? (b) does F4's width-prunability law hold across architectures? (c) what's the new Pareto knee in λ? (d) does the "near-deterministic mask" property (F1, σ=0.007 on MNIST) hold under more demanding conditions?

### Setup
- **Base model:** `CIFARNetBig` (3 conv blocks 64/256/512 + BN + max-pool, then FC 8192→1024→512→256→10). 10.38M params, 87.39% test accuracy on CIFAR-10, trained 40 epochs with AdamW + cosine + aug + BN. Frozen for pruner training (BN in eval, no grad anywhere except pruner params).
- **Pruner targets:** fc1 (1024×8192, 8.39M), fc2 (512×1024, 0.52M), fc3 (256×512, 0.13M). fc4 (output) untouched. The conv blocks (which together are ~1.34M params) are NOT pruned in this experiment — this run is MLP-only. The convolutional → MLP seam at fc1's 8192-wide input (= 512 conv3 channels × 4×4 spatial) is implicit but not exploited yet (no channel-pruning here).
- **Pruner architecture:** the existing `src/pruners/bilstm.py` Pruner with no changes — the model is layer-shape-agnostic. Instantiated with `embed_dim=64, lstm_hidden=128` (3-step BiLSTM sequence, so doubled hidden compared to MNIST's 4-step). 2.13M params (~20% of base — small but non-trivial).
- **Loss:** L = CE(pruned) − CE(orig) + λ · mean_per_layer(g̅). Soft penalty on per-layer keep fraction averaged across the 3 layers. The +2.0 bias on row-encoder output keeps all gates open at step 0; the tanh-bounded context bias prevents allocation runaway (F12's lesson — the layer-starvation pathology that killed top-K cannot occur here).
- **Training:** 5 epochs of CIFAR train (50k samples, batch 256 → ~196 steps/epoch → ~980 steps total). Adam(lr=1e-3), grad-clip 1.0. No augmentation in the pruner's data pipeline (pruner sees the same dist as test).
- **Sweep:** initial λ ∈ {0.01, 0.03, 0.1, 0.3} single-seed; then a finer sweep λ ∈ {0.02, 0.03, 0.05, 0.07} × seeds {0, 1, 2} to confirm the elbow.

### Result table (3 seeds, fine sweep)

| λ    | % pruned mean ± std | test acc mean ± std | drop  | fc1 kept (~mean / 1024) | fc2 kept (~mean / 512) | fc3 kept (~mean / 256) |
|------|---------------------|---------------------|-------|-------------------------|------------------------|------------------------|
| 0.02 | 64.32 ± 3.95%       | 85.72 ± 0.28%       | −1.67pp | ~162                  | ~263                   | ~102                   |
| 0.03 | **70.90 ± 0.76%**   | **85.91 ± 0.36%**   | **−1.48pp** | ~171              | ~177                   | ~92                    |
| 0.05 | 74.33 ± 0.20%       | 85.27 ± 0.66%       | −2.12pp | ~162                  | ~135                   | ~89                    |
| 0.07 | 76.22 ± 0.71%       | 84.87 ± 0.67%       | −2.52pp | ~147                  | ~108                   | ~92                    |

λ=0.03 dominates Pareto-wise: best mean drop AND best stability AND 6.6% more pruning than λ=0.02. The Pareto knee on CIFAR is just slightly higher than the MNIST winner's λ — the soft penalty needs to be a bit stronger here, consistent with CIFAR's higher CE scale.

### Per-layer dynamics (the new view from the fine sweep)

The new per-step plots track each layer's % pruned independently and reveal a **two-phase transition**:

1. **Phase 1 (steps 0–~300): exploration.** All gates near 100% keep. Loss is small and positive (CE_pruned ≈ CE_orig + λ since average gate is ≈1). The pruner is learning which neurons to attack but has not committed.

2. **Phase 2 (steps ~300–700): commitment cascade.** Layers transition sequentially:
   - fc1 transitions first (steps ~300–500) — the widest layer is easiest to attack because each killed neuron costs little (lots of redundant copies).
   - fc3 transitions next (~400–600) — the smallest layer, but it can be pruned hard at first because the floor isn't hit yet.
   - fc2 transitions last (~500–700) — the most "marginal" layer; its decision is most sensitive to λ.

3. **Phase 3 (steps ~700–980): stabilization.** All three layers are flat. Whatever subset is locked in by step 700 stays locked. This matches the "near-deterministic" finding from MNIST (F1) — the pruner finds a local optimum and sits in it.

### Why fc3 is incompressible

Across λ ∈ {0.03, 0.05, 0.07}, fc3 keeps ~90/256 ≈ 35% with very tight variance. Even raising λ from 0.03 to 0.07 barely moves it (92 → 92 → 90 → 92). Yet fc1 and fc2 continue to compress under stronger λ. **fc3 has hit a load-bearing floor**: the layer immediately before the 10-class output bottleneck cannot lose more capacity without the linear classifier (fc4) failing to discriminate. This is the "depth = required capacity" half of F4, observed *within* a single network's depth dimension. It complements the cross-network finding (deep networks have higher floors): within a network, the layers closer to the output are more load-bearing.

### Why λ=0.02 is unstable

At λ=0.02 the % pruned stdev is 3.95% — 5× the next-worst point. Inspecting per-seed allocation:
- Seed 0 keeps fc2 at 268/512 (52%) → light prune on fc2.
- Seed 1 keeps fc2 at 332/512 (65%) → barely touches fc2.
- Seed 2 keeps fc2 at 188/512 (37%) → hard prune on fc2.

The instability is concentrated in fc2 specifically — fc1 stays at ~160/1024 and fc3 at ~102/256 across seeds, but fc2 swings by 70% of its size. **The mechanism:** at λ=0.02, the marginal cost of keeping a fc2 neuron is roughly the same as the marginal benefit (since CE_pruned barely changes for moderate fc2 pruning when fc1/fc3 are still over-parameterized). The gradient on fc2 gates is small; whichever direction the random init nudges them tends to win. At λ=0.03, the cost is unambiguously high enough to dominate the noise — fc2 commits. This is the "below the penalty threshold the marginal decision is noise" mechanism, made explicit by the per-layer trace.

### Connecting to existing findings

- **F1 (near-deterministic masks, σ=0.007 on MNIST):** at the right λ the property holds here too — λ=0.03 / 0.05 / 0.07 all have ≤0.8% stdev across seeds. The transfer of "near-deterministic" is contingent on being above the stability threshold; F1's σ=0.007 was always measured at well-tuned λ on MNIST.
- **F4 (width → prunability):** confirmed within a CIFAR conv-net's FC head. fc1 (widest) prunes hardest; fc3 (narrowest) prunes least. This is the first replication of F4 across architectures.
- **F11 (weights, not neurons, are conserved):** un-tested here directly because we didn't compute weight savings or compare against a from-scratch baseline. fc1 is 8.39M, fc2 0.52M, fc3 0.13M — saving 83% of fc1 alone saves ~7M weights. Future work could compute this and compare to a hand-trained smaller MLP head.
- **F12 (top-K is brittle):** the soft-λ choice paid off here. fc2's marginal nature (the wobble at λ=0.02) is exactly the kind of layer that hard top-K would have starved given F12's layer-starvation mechanism. λ's per-layer-independent thresholding is more robust by design.

### What this opens up

- **Conv channel pruning** is the natural next step. The same architecture should drop in: replace "row encoder" with "filter encoder" (Conv2d weight slice as input, flattened to C_in·k·k), one gate per output channel, plus a coupling rule at the conv3→fc1 seam (pruning conv3 channel c removes fc1 input columns [16c:16c+16]). MetaPruning (Liu 2019) is direct prior art.
- **Post-prune finetune.** The −1.48pp drop should largely close with a few epochs of base-model finetune at the pruned mask. Worth measuring as a baseline before claiming the pruner alone is the bottleneck.
- **Longer pruner training.** 5 epochs is short. The two-phase plot suggests pruning is "done" by step 700, but the per-step loss has not flattened — more training might tighten the final mask.

### Code / artifacts

- Scripts: `scripts/hypernetwork/train/mnist_cifar/train_pruner_cifar.py` (initial 4-λ wide sweep), `scripts/hypernetwork/train/mnist_cifar/train_pruner_cifar_fine.py` (4-λ × 3-seed fine sweep, with per-layer live log + per-layer plot panels).
- Outputs: `experiments/latest/hypernetwork/cifar_lambda_sweep/` (1-seed), `experiments/latest/hypernetwork/cifar_lambda_fine/` (3-seed, comparison.png, per-(λ, seed) plot.png + summary.txt).
- One bug found and fixed mid-run: `masked_forward` originally used `model.conv2(x)` and `model.conv3(x)` (should have been `(h)` to chain). Pre-fix run crashed at step 1; post-fix sweep completed cleanly.
