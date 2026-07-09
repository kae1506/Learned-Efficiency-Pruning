# Crisp Findings

Terse index of what we know, what we're answering, what's next. Each finding is one statement + one or two lines of proof + intuition. Full detail per finding lives in its appendix below; original full reasoning in [claude-notes.md](claude-notes.md).

---

## Findings

**F1 — Weights alone encode redundancy.** A BiLSTM reading only weight matrices prunes ~60–79% of an MNIST MLP at <0.1–3.7pp drop. No data, no gradients needed.

**F2 — The pruner is near-deterministic.** Same checkpoint → fresh pruner seeds → same mask (dead/alive ratio σ=0.007). Redundancy is a fixed property of the network's weights.

**F3 — BiLSTM strictly dominates classic pruning.** At every accuracy budget, BiLSTM prunes more than activation/magnitude (≤1% drop: 72% vs 46%). Holds because the row-encoder sees full weight vectors, not scalar summaries.

**F4 — Prunability is set by WIDTH, not neuron count or depth.** Same 2048-neuron budget, monotonic in width: wide [2048]→93%, medium [1024,1024]→79%, deep [512×4]→58%. Width concentrates redundancy; depth and param count anti-correlate with it.

**F5 — RL loses to the hypernetwork, decisively.** Best RL (actor-critic) 4.71±1.55pp vs BiLSTM 3.68±0.79 at ~80%. Every RL lever (PPO, entropy scaling, chunk size, Bernoulli action) matched-or-worsened it; Bernoulli collapsed (62pp).

**F6 — The RL MDP is structurally degenerate.** Telescoping reward on a frozen model makes return path-independent → pruning is static subset selection → RL provably dominated by direct differentiable optimization. *Theorem, not just empirics.*

**F7 — A trained pruner does NOT transfer; the method does.** Frozen pruner from model A → model B = 28pp drop (catastrophic). Retrain on B = 4pp (fine). Redundancy is idiosyncratic per network (permutation frame differs); the BiLSTM is a reliable *procedure*, not a portable *answer*.

**F8 — Single-seed RL numbers are noise.** σ≈4pp. k=8 single-seed 2.68 → 3-seed 5.54±3.44. Always ≥3 seeds.

**F9 — Prunable fraction is NOT constant; small nets sit near a task floor.** [205,205] (410 neurons) could only prune 52% and paid 16pp for it (vs [1024,1024]: 79% for 3.7pp). Dead/alive ratio spiked to 0.645 (vs stable ~0.31) — pruner ran out of redundant neurons. Fixed-fraction REJECTED.

**F10 — Width is cheap, depth is expensive.** Survivors @≤2pp (rigorous retrain) grow with depth (314/490/814 for 1/2/4 layers); prunability monotonic in width (wide 84.6% > medium 76.1% > deep 60.2% > narrow 27.9%). Width concentrates redundancy, depth is load-bearing capacity. See [Appendix B].

**F11 — Weights, not neurons, are the conserved quantity.** Three 2048-neuron MNIST architectures all compress to ~240K weights @2pp (medium 230K, deep 242K, wide 248K; σ=8%) despite neuron survivors 490/814/314. Pruning a big net does NOT reach training-small's true minimum (narrow trained at 410 reaches 82K weights — 3× fewer). See [Appendix C].

**F12 — STE-top-K is brittle and loses to λ by ~2.6× at iso-accuracy.** Fully de-bugged center+tanh top-K (2-seed, no collapse) reaches 34.6% pruned @2pp vs λ's 76%. Structural: hard budget + boundary-local STE gradient under-optimizes the subset vs λ's soft global gradient (active fraction ~0.077 at λ's operating K). B1 CLOSED. See [Appendix A].

**F13 — BiLSTM transfers to CIFAR_big's MLP head; F4 holds.** λ=0.03 Pareto knee: 70.9 ± 0.8% pruned at −1.48 ± 0.36pp test drop (3 seeds, frozen CIFAR_big @87.39%). Widest layer (fc1, 1024 neurons) prunes hardest (~17% kept); narrowest hidden (fc3, 256) hits a load-bearing floor at ~35% kept — depth/width axis appears within a single network. See [Appendix D].

**F14 — Smaller base = less prunable on same task.** CIFAR LeNet (63K) vs CIFAR_big (10.4M) on CIFAR-10: at matched λ=0.03, LeNet 13.7% pruned at −5.22pp vs big 70.9% / −1.48pp — ~5× less prunable per-neuron. Efficiency curve (% pruned / drop pp) on LeNet shows a sharp dip at λ=0.10 that persists with 15-ep training — a structural half-committed dynamical pathology, not under-training. See [Appendix E].

**F15 — Across 3 base nets (LeNet 63K → MNIST 1.86M → CIFAR_big 10.4M, CIFAR ↔ MNIST), λ_opt lives in [0.03, 0.06] — refutes any monotonic λ_opt-vs-size hypothesis.** Peak efficiencies (% pruned / max(drop pp, 0.5)): LeNet 5.4 (λ=0.04), MNIST 135.8 (λ=0.06), CIFAR_big 47.9 (λ=0.03). λ_opt is NOT monotonic in N — refutes H2 in its original form. Empirical regularity worth testing: λ_opt · N_layers ≈ 0.097 ± 0.02 across all 3 datapoints (0.08, 0.12, 0.09). See [Appendix F].

---

## Questions we're answering

**Q1 — Can a small net learn to prune from weights alone?** → Yes (F1, F2).
**Q2 — Is learned pruning better than heuristics?** → Yes, strictly (F3).
**Q3 — Does RL's sequential framing help?** → No, and we proved why (F5, F6).
**Q4 — What determines how prunable a network is?** → Width / redundancy distribution, not size (F4, F14).
**Q5 — Does redundancy structure generalize across networks?** → No (location is per-network); the *amount* and the *method* do (F7).

## Questions still open

**Q6 — Is there a TRANSFERABLE redundancy detector?** Meta-train across a distribution of networks with permutation-invariant features only.
**Q7 — Does RL win when weights co-adapt?** The one regime where the MDP is non-degenerate (F6). Untested.
**Q8 — What is the pruner measuring?** Treat the mask as an instrument, not a tool.
**Q9 — Is the minimal subnetwork an ABSOLUTE task floor or network-relative?** → ANSWERED: network-determined by DEPTH (F10).
**Q10 — Is there a predictable λ_opt formula?** Original H2 size-scaling REFUTED by F15. λ_opt · N_layers ≈ const is a candidate (needs 4-layer + 1-layer datapoints to confirm).

---

## Research directions (ranked by payoff/effort)

1. **Prune DURING training** (co-adapting weights). The only regime where RL is principled; converts F6's negative into a characterized boundary. *Highest value for the RL line.*
2. **Meta-train a transferable pruner** (Q6). Permutation-invariant features, distribution of source networks. Tests if F7 is escapable.
3. **Prunability as effective-dimension probe.** Sweep data/width/depth/training → prunability vs task complexity; tie to double descent & lottery tickets. *Science, not method.*
4. **Train-to-be-prunable.** Training-time regularizer concentrating redundancy; flip pruning to an inductive bias.
5. **Prune in a learned basis.** Neuron = arbitrary coordinate; rotate, prune directions, rotate back.
6. **Pruner-capacity scaling law (meta-architecture).** What pruner size does a base of N neurons / max-fan-in C need? Methodology counterpart to F4 — lets us predict pruner sizing instead of guessing.
7. **λ-scaling law: which form fits?** F15 refuted monotonic size-scaling but suggests `λ_opt · N_layers ≈ const`. Test with 1-layer and 4-layer MNIST bases (both already trained as `mnist_wide2048.pt`, `mnist_deep4x512.pt`). If law holds: predict λ_opt from architecture alone, no sweep needed.
8. **λ_sim formula (refined H3) — could be true, needs proving.** λ_sim ≈ k · ⟨|∂CE/∂g|⟩ · mean_layer_size. Measurable in one forward+backward pass. F14's data partially supports this as a "regime boundary" vs the noisier "Pareto-optimal λ".
9. **BiLSTM MLP-portion universality.** Prove the soft-λ BiLSTM works on the MLP portions of arbitrary architectures: (a) deeper conv backbones (ResNet+BN), (b) **transformer MLP/FFN blocks**. If universal, the pruner becomes a generic MLP-head prunability detector.

(Full mindset rationale: claude-notes §12. Generalizable RL/ML lessons: §13. MDP-degeneracy theorem: §10.)

---

## The one-liner

**Weight-conditioned hypernetworks read out a network's redundancy near-deterministically and beat both heuristics and RL; the RL framing is provably vestigial on a frozen model; and redundancy is real & findable in every network but idiosyncratic to each one's weights.**

---

## Appendix A — STE-top-K iterative fix chain (F12 / B1)

The full sequence of fixes for the λ-free top-K pruner. Each row is `problem → fix → new problem it introduced`. The pattern (most fixes patch the previous fix's side-effect) IS the brittleness, made concrete. λ needed NONE of this. (Detail: claude-notes §14; medium [1024,1024], drop @76% pruned in parens.)

0. **Base** = global top-K + plain STE. Plain `binary_ste` uses soft $=\sigma(s)$ → gradient peaks at $s=0$, but the top-K keep/kill boundary is $\tau$ (the $K$-th-largest score), far from 0 → borderline neurons sit on $\sigma$'s saturated tail → ~0 gradient → ranking frozen. **(42.7pp)**
1. **Centered STE** soft $=\sigma((s-\tau)/T)$ → moves the gradient peak ONTO $\tau$. *Introduces:* effective band has width $\sim T$ → needs scores on a scale comparable to $T$.
2. **Node-norm (std)** $y=(l-\mu)/\sigma$ → unit-variance scores ⇒ band matched, AND per-layer zero-mean ⇒ layers cross-comparable. *Introduces:* non-detached $/\sigma$ backward blows up when $\sigma$ small (committed scores) → instability/collapse.
3. **std_detach** (detach $\mu,\sigma$) → kills the blow-up; forward still unit-var. *Introduces:* forward now scale-invariant $\mathcal L(\alpha l)=\mathcal L(l)$ → $\sigma$ is a free coordinate → drifts to $\infty$ → $\partial\mathcal L/\partial l\propto 1/\sigma\to0$ → **encoder FREEZE (#2)**.
4. **tanh on context bias** $c_\ell=\tanh(\hat c_\ell)$ → bounds gap $\Delta=c_2-c_1\in(-2,2)$ ⇒ no **layer-starvation (#1)** (else one layer's cloud clears the other → severed → 88.32pp). [Orthogonal fix; confirmed: L1 stops draining to 0.]
5. **center-only** $y=l-\mu$ (no $/\sigma$) → not scale-invariant ⇒ $\sigma$ loss-constrained ⇒ no freeze. *Introduces:* gives up step-2 band-matching → $\sigma$ drifts to ~21, band too narrow ($\approx T/\sigma$ of neurons get gradient) = **mild residual (#3)**. **(5.66pp — best top-K)**
6. **AdamW** (std_detach + weight decay) → pin $\sigma$ at the van Laarhoven equilibrium so band stays matched (std_detach) AND no freeze (WD floors $\eta_\text{eff}$). **FAILED at $\lambda_{wd}=10^{-2}$**: shrinkage $=\eta\lambda_{wd}=10^{-3}\cdot10^{-2}=10^{-5}$/step, ~100× too weak; $\sigma\to140$, freeze persists. **(11.0pp)** — testing $\lambda_{wd}=1$.

**Three pathologies:** #1 layer-starvation (tanh ✓), #2 scale-invariance freeze (center-only ✓ / WD ✗: too-weak 1e-2 freezes, strong 1.0 over-regularizes), #3 STE band-mismatch (residual).

### F12 FINAL (locked — full 2-seed curriculum, center+tanh)
Fully de-bugged STE-top-K — centered STE + center-only norm + tanh-bounded allocation + global-T anneal + carried Adam; **no collapse, 2-seed stable (σ(drop) ≤ 0.6pp)** — reaches **@2pp = 1340 survivors = 34.6% pruned**, paying **5.29 ± 0.36pp at λ's 76%-pruned operating point** (24% kept, K=492) vs **λ's ~2pp**. So even at its best, λ-free top-K prunes **~2.6× fewer** neurons than λ at iso-accuracy.

### F12 WHY top-K loses even when stable (the deep reason)
The 3 pathologies were *fixable symptoms*. The residual ~2.6× gap is **structural — hard-budget + boundary-local gradient is a weaker optimizer of the subset-selection problem than soft-penalty + global gradient.** λ optimizes a smooth Lagrangian with the STE threshold fixed at 0 (scores held near 0 by +2.0 bias) → gradient reaches **every** neuron → joint/global optimization. top-K hard-fixes K and applies centered STE at the moving threshold τ → gradient reaches only the ≈T/σ fraction of neurons near τ (boundary-local). **General principle: swapping a soft global penalty for a hard constraint with local-only gradient trades a hyperparameter sweep for (a) a stability-guardrail stack and (b) worse optimization of the underlying selection. Sweep-free ≠ better.**

### F12 PROOF (quant + qual, center+tanh)
- **Boundary-local CONFIRMED.** Active fraction = neurons with $\sigma'((s-\tau)/T)>0.05$ falls as $K$ shrinks: 90% kept 1.00 → 60% 0.60 → 50% 0.38 → **24% kept 0.077 (~38/492)** → 18% 0.037. ~92% of neurons saturated each step at the target sparsity — only the threshold neighbourhood learns.
- **Ties REFUTED as cause.** 0/5100 steps with ≥50%·K scores tied at τ.
- **Gradients STABLE.** Param grad-norm: median 0.010, max 0.734, 0/5100 clipped → neither exploding nor vanishing.
- Source: topk_proof.{png,txt}, scripts/hypernetwork/topk/topk_proof.py

---

## Appendix B — F10 detail (depth/width prunability table)

Iso-accuracy survivors @ ≤2pp, pruners RETRAINED per sw, 2 seeds (rigorous version; supersedes the pessimistic threshold-sweep proxy):

| Model | Layers | Survivors@2pp | % pruned |
|---|---:|---:|---:|
| wide [2048] | 1 | 314 | **84.6%** |
| medium [1024,1024] | 2 | 490 | 76.1% |
| deep [512×4] | 4 | 814 | 60.2% |
| narrow [205,205] | 2 | 296 | 27.9% |

- Prunability monotonic in WIDTH: wide 84.6 > medium 76.1 > deep 60.2 > narrow 27.9. Widest prunes best (confirms intuition; the threshold-sweep proxy's "deep beats wide" was a per-sw-untuned artifact hitting wide hardest).
- Absolute survivors grow with DEPTH (314→490→814 for 1→2→4 layers), NOT a single universal floor. A 1-layer wide net compresses to ~314, nearly the narrow's ~296 floor; a 4-layer net needs ~814.
- Dual statement: width = cheap parallel redundancy; depth = load-bearing capacity (every layer is a sequential bottleneck).
- Variance ±0–20 (BiLSTM near-deterministic; 2 seeds ample).
- Threshold-sweep proxy [iso_accuracy_sweep] gave 806/1179/744/360 — overturned for medium & wide by retraining.
- Source: [hypernetwork/iso_accuracy_retrain/](../experiments/latest/hypernetwork/iso_accuracy_retrain/) (rigorous), [hypernetwork/iso_accuracy_sweep/](../experiments/latest/hypernetwork/iso_accuracy_sweep/) (proxy).

---

## Appendix C — F11 detail (weights ≠ neurons)

- **Neuron% ≠ weight%.** Pruning a neuron drops its in-row + out-column. Input(784×h1)/output(h×10) matrices prune one-sided → LINEAR; hidden→hidden matrices prune two-sided → QUADRATIC (1−f²). So nets with hidden→hidden matrices save far more weights than neurons: deep 60.3% neurons → **79.7% weights**; medium 75.9%→87.7%; narrow 30.2%→59.9%. Wide (no hidden→hidden) is the only one where they're equal (84.8%=84.8%).
- **Final weights @2pp converge** for the three 2048-nets: medium 230K, deep 242K, wide 248K (~8% spread) despite neuron survivors 490/814/314 and starting sizes 1.19–1.86M. Wide keeps few fat neurons (784 fan-in); deep keeps many thin ones (quadratic-discounted); both land at ~240K weights. **~240K MACs ≈ the conserved compute floor for MNIST@2pp regardless of architecture.**
- **Param ranking premise correction:** wide (1.63M) has MORE weights than deep (1.19M) — the 784×width input matrix dominates. Order: medium 1.86M > wide 1.63M > deep 1.19M > narrow 0.20M.
- **Pruning ≠ training-small.** narrow (trained at 410 neurons) reaches MNIST@2pp with **82K weights** — ~3× fewer than pruning-from-2048's ~240K floor, at comparable accuracy. Pruning a big net does NOT reach the true minimal solution (echoes Liu et al. 2019, "Rethinking the Value of Network Pruning").
- **sw needed scales with depth:** wide(1L) sw=0.1, medium(2L) 0.3, deep(4L) 0.8 — 8× more sparsity pressure to prune deep to the same budget. Deeper = each neuron more load-bearing.
- Source: [hypernetwork/iso_accuracy_retrain/weight_summary.txt](../experiments/latest/hypernetwork/iso_accuracy_retrain/weight_summary.txt), weights.png, curves.png.

---

## Appendix D — F13 detail (CIFAR_big MLP head sweep)

- **Layer allocation mirrors F4.** At λ=0.03 the pruner keeps fc1 ~17% (171/1024), fc2 ~35% (177/512), fc3 ~36% (92/256). Widest layer (fc1, 1024×8192, 8.4M params alone) prunes hardest in fraction terms.
- **fc3 hits a load-bearing floor.** fc3 (the last hidden layer, 256) keeps ~35% across all λ∈{0.03,0.05,0.07} — pruning more costs accuracy fast. The layer right before the output is incompressible — depth-as-capacity at the *layer* level inside one network.
- **Two-phase per-layer dynamics.** Gates barely move for ~300 steps, then sharp transition; by step ~700 all three layers committed and stay flat. fc1 transitions first, fc3 next, fc2 last (the marginal layer).
- **λ=0.03 = the stability threshold.** At λ=0.02 fc2 wobbles 37%→65% across seeds (3.95% stdev in overall % pruned); at λ≥0.03 every seed locks the same equilibrium (≤0.8% stdev) — the soft penalty needs to be strong enough to commit on the marginal fc2.
- **λ≥0.05 = diminishing returns.** 70.9→76.2% pruned for −1.04pp extra accuracy cost. MLP cliff sits ~75%.
- **Accuracy floor for clean MLP pruning ~1.5pp.** Even the winner has a real gap from 87.39%. MNIST winner was <0.5pp at 80% — CIFAR is genuinely harder.
- Source: [hypernetwork/cifar_lambda_fine/](../experiments/latest/hypernetwork/cifar_lambda_fine/) (4 λ × 3 seeds), [cifar_lambda_sweep/](../experiments/latest/hypernetwork/cifar_lambda_sweep/) (initial 1-seed wide sweep).

---

## Appendix E — F14 detail (cross-architecture, same task)

- **At matched λ=0.03:** LeNet 13.7% / −5.22pp, CIFAR_big 70.9% / −1.48pp. **~5× less prunable per-neuron at the same penalty.**
- **λ* shifts:** big λ*=0.03; LeNet first-pass wide-sweep λ*=0.3 (52.3%/−8.1pp) — looked like a 10× shift. **Refined by F15:** the LeNet 15-ep multi-seed fine sweep showed λ*=0.04–0.08 plateau; the 10× was an artifact of single-seed-5-epoch under-training.
- **Per-layer dynamics:** same two-phase pattern as CIFAR_big. fc1 first (~step 400), fc2 lags (~step 700). fc2 = marginal layer again.
- **Sequential→simultaneous regime is a sharp dynamical boundary.** LeNet @ λ≤0.08 = sequential; λ=0.10 = STUCK half-committed; λ≥0.20 = simultaneous. Mechanism: a gate closes iff `λ/(N·S_ℓ) > −∂CE/∂g_i` — low λ only satisfies inequality for the most-redundant layer's gates → greedy layer-by-layer; high λ satisfies it everywhere at once → parallel commitment.
- **Efficiency curve (% pruned / drop pp vs λ) is the cleanest λ-selection diagnostic.** CIFAR_big has a sharp peak at λ=0.03 (eff 47.9, inverted-U). LeNet has a plateau at λ∈{0.04,0.06,0.08} ≈ 5.4, deep dip at λ=0.10 (3.94 — the half-committed state), recovery at λ=0.25 (5.04). **The dip persists with 15 epochs — dynamical, not under-training.**
- **F4 in pure numbers:** ~9× efficiency gap (CIFAR_big 47.9 vs LeNet 5.4) at matching protocol — direct numerical handle on width = redundancy.
- **Refined H3:** λ_Pareto ≠ λ_sim on saturated nets. Wide nets: λ_Pareto = λ_sim. Narrow saturated nets: λ_Pareto < λ_sim (LeNet Pareto ≈ 0.04–0.08, sim ≈ 0.20). Generalized: λ_Pareto = min(λ_sim, λ_saturation).
- Source: [hypernetwork/cifar_lenet_lambda_sweep/](../experiments/latest/hypernetwork/cifar_lenet_lambda_sweep/), [cifar_lenet_lambda_fine/](../experiments/latest/hypernetwork/cifar_lenet_lambda_fine/), [cifar_lenet_lambda_fine_15ep/](../experiments/latest/hypernetwork/cifar_lenet_lambda_fine_15ep/), [cifar_lenet_lambda_extra_15ep/](../experiments/latest/hypernetwork/cifar_lenet_lambda_extra_15ep/), [efficiency_compare.png](../experiments/latest/hypernetwork/efficiency_compare.png).

---

## Appendix F — F15 detail (3-net efficiency comparison, λ_opt scaling)

3-net efficiency table at matched protocol (15-ep × 3-seed multi-seed where available; CIFAR_big uses 5-ep 3-seed fine + 5-ep 1-seed wide-sweep extras for tail):

| net | params | N_layers | base test acc | λ_opt | peak eff | λ_opt · N_layers |
|---|---:|:-:|---:|---:|---:|---:|
| CIFAR LeNet | 63K | 2 | 64.84% | 0.04 | 5.4 | 0.08 |
| MNIST [1024,1024] | 1.86M | 2 | 98.06% | 0.06 | 135.8 | 0.12 |
| CIFAR_big | 10.4M | 3 | 87.39% | 0.03 | 47.9 | 0.09 |
| **mean** | | | | | | **0.097 ± 0.02** |

**Key observations:**
- **λ_opt is NOT monotonic in model size.** 63K→0.04, 1.86M→0.06, 10.4M→0.03. **Original H2 hypothesis "λ_opt ∝ 1/N_params" or "∝ 1/mean_layer_size" — REFUTED.** See main-text H2 analysis.
- **λ_opt sits in a narrow [0.03, 0.06] range** despite 165× model-size shrink and a task swap (CIFAR↔MNIST). Something is keeping it near-constant.
- **MNIST peak efficiency (135.8) > CIFAR_big (47.9) despite MNIST being 5.6× smaller.** Task-driven, not size-driven: MNIST is easy enough that the pruner pays almost no accuracy cost (0.12pp at λ=0.04) → ratio explodes.
- **LeNet's efficiency curve is bimodal** (plateau + dip at λ=0.10 + partial recovery at λ=0.25), unlike MNIST/CIFAR_big's clean inverted-U. The dip is the half-committed dynamical pathology (F14, Appendix E) — only narrow nets show it.
- **Empirical regularity worth testing:** `λ_opt · N_layers ≈ 0.097 ± 0.02` across all 3 datapoints. 2-layer nets (LeNet, MNIST) need λ_opt ≈ 0.05; 3-layer (CIFAR_big) needs 0.03. **Predicts: 4-layer needs ≈ 0.025, 1-layer needs ≈ 0.10.** Cheap to test on `mnist_wide2048.pt` (1L) and `mnist_deep4x512.pt` (4L) — already trained.
- **Mechanism (hypothesis):** sparsity loss = `λ · (1/N_layers) · Σ_ℓ g̅_ℓ` → per-gate sparsity gradient = `λ/(N_layers · S_ℓ)`. The 1/N_layers factor pulls the effective penalty down as layers are added; to keep the effective penalty constant at the gate level, λ has to grow with N_layers. The opposite of what we see — UNLESS the loss term itself has a competing 1/N or N² in the CE gradient. Needs derivation.
- **MNIST pruning actually *improves* test acc by 0.12pp at λ=0.04** (97.94 vs 98.06 unpruned, 3-seed stable). 1.86M params is heavily over-parameterized for MNIST → removing 64% of hidden neurons recovers a slightly better generalization — a small lottery-ticket / dropout-style effect.
- Source: [efficiency_compare.png](../experiments/latest/hypernetwork/efficiency_compare.png), [mnist_lambda_sweep_15ep/](../experiments/latest/hypernetwork/mnist_lambda_sweep_15ep/), [cifar_lenet_lambda_fine_15ep/](../experiments/latest/hypernetwork/cifar_lenet_lambda_fine_15ep/) + extras, [cifar_lambda_fine/](../experiments/latest/hypernetwork/cifar_lambda_fine/) + wide. Code: scripts/hypernetwork/efficiency_compare.py.
