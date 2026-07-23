# Ideas & Research Directions

Consolidated backlog of everything we've proposed. Status tags: **[LIVE]** untested & worth doing · **[NEXT]** recommended soon · **[DEAD]** tested, negative · **[OPEN-Q]** question to answer. Crisp format. Detail in [claude-notes.md](claude-notes.md) (§ refs), findings in [crisp-findings.md](crisp-findings.md).

​

### LEARNING BY UNLEARNING:

how and why some LLMs can GAIN in performance by Learned Efficiency Pruning.  or at least be more fine-tuned? this makes more sense, because we are getting rid of things that arent being rewarded by this behaviour&#x20;

---

## A · Pruner input / signal

**A1. Richer input: activations + gradients/Taylor** [LIVE, cheap first step]
Feed the BiLSTM per-neuron activation stats (mean/var post-ReLU) and/or Taylor importance `|∂L/∂a·a|`, not just weights.
- *Why:* more signal at the hard cliff decisions. **Bigger reason = TRANSFER** — activation/gradient *statistics* are permutation-invariant & comparable across nets (raw weights aren't, which is why transfer failed, F7).
- *Test:* ablation weights / +act / +grad → measure Pareto move AND whether transfer turns on.
* _Ceiling:_ frozen net capped \~240K weights (F11); richer input approaches floor + maybe transfers, won't break it.

**A2. Better layer pooling** [LIVE, trivial]
Replace `W.mean(dim=0)` (signed mean cancels) with `abs().mean` / `std` / norm for the per-layer LSTM token. Cheap upgrade to a crude hack.

---

## B · Curriculum / iterative pruning

**B1. Top-K curriculum (decreasing K)** [CLOSED — NEGATIVE, mechanism proven; see F12 / claude-notes §14]
STE-top-K (set K directly, drop λ + +2.0-bias/tanh). **Negative result with a proven cause.**
- *What held:* removes the per-model sw sweep; centered-STE σ((s−thresh)/T) matches λ at moderate K (50% kept → 2.1pp one-shot).
- *What broke (PROVEN):* GLOBAL top-K **starves a layer to death** — L1 drains 819→512→205→0 while L2 stays full; at K=one-layer-size (50% stage) L1 severed → one-class 88.32pp (bit-identical, absorbing). Cause: per-layer standardization centers both layers at 0/unit-var → cross-layer allocation is governed only by the per-layer context bias → dropping tanh left it unbounded → diverges → starves L1.
- *Tested & failed to fix:* global T-anneal, std/std_detach, carry-Adam — none touch the allocation knob → same 50% wall.
- *Would-be fix (not pursued):* per-layer keep-floor / per-layer top-K, or re-bound the context allocation — but that surrenders "pruner freely allocates across layers". Verdict: λ wins; λ-free top-K trades a sweep for a structural collapse.
- *Forks (moot now):* re-rank+warm-start vs hard-nesting; frozen vs retrain.

**B2. sw-annealing A/B (continuation method)** [LIVE]
One-shot sw=0.5 vs ramped sw 0→0.5 (AGP-style) to same sparsity. Measure steps-to-floor, stability, whether it survives without bias/tanh/LayerNorm. Prediction: same floor, ≤steps, more stable. Win is optimization-ease, biggest at scale.

**B3. Staged prune + RETRAIN between stages** [LIVE, highest-value for breaking the floor]
Iterative-magnitude-pruning with BiLSTM scorer. **Only path past the \~240K frozen floor toward the \~82K train-small floor (F11).** Genuinely sequential (weights co-adapt). Synthesis with A1+B1 = learned iterative Taylor pruning, near-SOTA.

**B4. Stochastic layer-subset training (randomized block-coordinate descent on the gate decisions)** [DONE, 2026-07-14 — `scripts/hypernetwork/train/mnist_cifar/train_pruner_mnist_deep_stochastic.py`] — Sample k of N layers uniformly per pruner step, score/apply/penalize only those (rest pass through ungated, recorded as 100% kept that step, not carried forward); full N-layer joint eval periodically + at the end. Grounded in Nesterov (2012)/Richtárik & Takáč (2014) randomized block-coordinate descent theory, structurally close to Stochastic Depth/LayerDrop. Testbed: `mnist_deep4x512.pt` (4 layers, k=2), λ-swept (0.1/0.25/0.5/1.0/2.0/4.0) against the `shape_deep4x512` baseline protocol, both modes, 3 seeds. **Result: NOT a speed win, confirmed not just predicted** (measured 34.2s/run baseline vs 31.0s/run stochastic, only ~10%, because gating doesn't reduce base-model FLOPs; the mechanistic reason generalizes — pruner-to-base compute ratio = embed_dim/tokens_per_step, shape-independent, and *shrinks* at real LLM batch sizes, so this gets less useful at scale, not more). **Comparing at matched λ is misleading** — stochastic needs ~2× less λ pressure to reach the same sparsity (sparsity_loss averages over k gates not N, so per-sampled-layer pressure is stronger at equal λ) — comparing at matched *actual sparsity* instead, stochastic is competitive with or slightly better than baseline in the 60–70%-pruned overlap region (e.g. 60.9% pruned/97.36% acc vs baseline's 58.1%/97.03%), with no excess noise once you're not sampling near either mode's own collapse cliff. See mode_comparison.png / efficiency_vs_lambda.png in the run directory.

**B5. Task-specific pruning as a specialization/deployment technique** [NEW, 2026-07-14, motivated by F19] — F19 (OPT-125M FFN pruning) doesn't hold up as general-purpose "free" pruning — the WikiText-2 improvement doesn't transfer to out-of-domain data (C4), confirmed via direct OOD eval, not assumed. But reframed, it's a different and still potentially useful result: given a pretrained base model and a *specific* target domain/task (not general-purpose deployment), the same `(CE_pruned − CE_orig) + λ·sparsity` training procedure finds a smaller, structured-sparse subnetwork that is *both* cheaper to run *and* better-performing on that specific domain than the original dense model — a combined compression + specialization step, needing only domain-matched unlabeled text (next-token prediction, no task labels). Use case: narrow-domain deployments (a company's internal docs, a specific support corpus, a codebase) where the base model's full general-purpose capability isn't needed or wanted. Open questions before this is a real proposal, in priority order: (a) **does a plain dense fine-tune on the same domain-matched data match or beat this, with or without the sparsity?** — the load-bearing unresolved question; if yes, sparsity isn't the active ingredient here, task-adaptation is, and this idea collapses into "fine-tuning, dressed up." (b) Does the favorable-ratio-at-low-λ trend (F19 Appendix J, 1.7× at λ=0.01) keep improving below the current floor, or has it already plateaued? (c) Compute cost is a full ~18,750-step training run per operating point (~20-25 min GPU per λ), vs. SparseGPT/Wanda's few-minute one-shot calibration — only worth it if (a) comes back genuinely in sparsity's favor.

**B6. Calibration-gap hypothesis, controlled test: fix the model, swap the dataset** [NEW, 2026-07-14, motivated by the GPT-2-vs-OPT-125M comparison] — F18 (GPT-2, pruning hurts monotonically on WikiText-2) and F19 (OPT-125M, pruning helps in-domain on WikiText-2) look contradictory given an identical setup, but GPT-2's unpruned WikiText-2 CE (3.218 nats) is far better than OPT-125M's (3.893 nats) on the same eval — hypothesis: how much in-domain gain a WikiText-2-trained pruner can find is predicted by how poorly-calibrated the base model already is to that domain, not by which model it is. **This is currently a hypothesis fit to one (model, model) pair, not a proven law** — GPT-2 vs OPT-125M differ in a dozen other ways (pretraining data, era, tokenizer details), any of which could be the real driver instead of calibration gap specifically. The clean test: hold the model fixed (GPT-2), swap the dataset to one GPT-2 is zero-shot *worse* at than OPT-125M is (candidates, given GPT-2's WebText vs OPT's Pushshift-Reddit/CC-News/Pile-CC-heavier mix: book/narrative-style text, or raw Reddit-comment-style casual text) — first just measure GPT-2's zero-shot CE there to confirm the calibration gap actually exists in the predicted direction, *then* run the λ-sweep pruning procedure on GPT-2 against that corpus. Predicted result if the hypothesis holds: GPT-2 shows the F19 in-domain-improvement-that-doesn't-transfer pattern on this dataset, which it did NOT show on WikiText-2. This isolates the calibration-gap variable from model-identity in a way the current 2-point (GPT-2, OPT-125M) × (WikiText-2) comparison structurally cannot. See crisp-findings.md F19 appendix J for the full comparison numbers.

**B7. Dynamic step budget (convergence-based stopping) + pruner-capacity ablation** [NEW, 2026-07-17, HIGHEST PRIORITY, motivated by F20] — Two related, empirically-motivated questions raised by actually reading the training curves across every λ tested, not just spot-checking:

(a) **Steps-needed varies by λ and by model, so a single fixed budget can't be right for all of it.** F20: GPT-2 converges in ~500-1000 steps across its *entire* tested range (18,750 is ~15-20x oversized, wasted compute on every GPT-2 run so far). OPT-125M converges just as fast in the middle of its range but is still visibly drifting at both its tested extremes (λ=0.01, λ=1.8) even at the full 18,750 — meaning the same fixed budget is simultaneously wasteful in most cases and possibly *insufficient* in the cases that matter most (the boundary/knee-finding runs). Proposed: a dynamic, convergence-based stopping criterion — monitor a smoothed per-layer-%pruned or pruner-loss delta over a trailing window, stop once it's been under some threshold for N consecutive checks — instead of a hand-set step count, so each (model, λ) run trains exactly as long as it needs to, no more, no less.

(b) **Pruner capacity (embed_dim=64, lstm_hidden=128, ~2M params, ~1.6% of the 125M base model) has never been revisited since project inception, and may be the actual cause of (a)'s slow-converging extreme-λ cases** rather than those cases simply needing more steps. Mechanistic story: moderate λ has an easy, obvious redundancy pattern any reasonably-sized pruner finds fast; extreme λ (very light or very heavy pruning pressure) requires a more delicate/coordinated decision that a capacity-limited pruner may need to search longer for, or may never fully reach. Proposed controlled test: rerun a known-slow case (OPT-125M λ=1.8) at a larger pruner (e.g. embed_dim=128, lstm_hidden=256) at the *same* step budget. If it converges within-budget where the smaller pruner was still drifting at step 18,000, capacity was the bottleneck. If convergence speed is unchanged, it's an inherently harder optimization landscape at that λ, independent of pruner size. Pruner-side compute cost is negligible either way even at 2x scale (embed_dim/tokens_per_step ratio, established earlier) — the only real cost of scaling up is a bigger `pruner.pt` file.

Both are cheap relative to their payoff: if (a) resolves, every future sweep (pg19 and beyond) runs at a fraction of current compute without sacrificing correctness; if (b) resolves in capacity's favor, every future sweep also needs a resized pruner to get correct results at the operating points that matter most.

**B8. Full corrected OPT-125M WikiText-2 re-sweep, reconciled against F18** [NEW, 2026-07-20, CO-HIGHEST PRIORITY, motivated by F21] — F21: the WikiText-2 tokenization bug fix (per-line BOS artifact) roughly halved OPT-125M's baseline ppl (49.038→23.941) and the 4 mid-range λ re-run under the fix now show the *same qualitative shape* as GPT-2's F18 curve (small free region at the lightest λ, then monotonic real cost) rather than F19's original "improves everywhere" story. This is only a partial check (4 λ, 8000 steps, 1 seed, no data below λ=0.2) — needs the same treatment F16 got on its way to becoming F18: full original λ grid (0.01→1.8), full 18,750 steps (or whatever B7 resolves the correct budget to be), 2 seeds, then reconciled directly against GPT-2's F18 numbers on the same plot. Open question this would settle: does *any* real GPT-2-vs-OPT difference survive once both are measured cleanly, or was the entire F18/F19 divergence — and by extension B5's specialization framing and B6's calibration-gap hypothesis, both built on F19's original story — substantially an artifact from the start? This also determines whether B6's pg19 comparative sweep (in progress) is still worth finishing in its current form, or needs to be re-anchored once the WikiText-2 picture is settled. Natural to run alongside B7 (same sweep, same compute) rather than as a separate pass.

**B9. Plateau-triggered cosine LR decay for the convergence-based training loop** [NEW, 2026-07-23, motivated by F3/F5, NOT YET BUILT — logged as a design only, per explicit instruction] — `check_converged`'s v2 (block-mean) fix filters Adam's non-decaying per-step noise by averaging over it; it doesn't shrink the noise itself, since `lr=0.001` stays flat for the entire run regardless of length. F3 found a case (opt125m λ=0.3, 2x-capacity pruner) where the block-mean check declared convergence mid-genuine-monotonic-climb — a case block-averaging can't distinguish from "settled but noisy" without more information. Proposed fix: the first time `check_converged` returns True (at step `t_trigger`), don't stop — cosine-decay `lr` from `lr_0` to `lr_min` over the next `lr_decay_window` steps (proposed default: reuse `window*check_every=250`, no new timescale introduced), then run `check_converged` again on the now lower-noise trailing block means. Passes → true convergence, stop. Fails → the first signal was noise-masking real movement (the F3 case) — hold `lr` at `lr_min` (don't re-decay from scratch, avoids thrashing) and keep running the normal loop, now at reduced noise. Free parameters not yet decided: `lr_decay_window` (proposed 250), `lr_min` (proposed `lr_0/10`). **What this mitigates**: false convergence calls caused by noise specifically — the more fundamental fix of the three options tried/discussed (block-mean averages over noise, window-lengthening waits inside noise, this shrinks the noise itself at the exact decision point). **What it does NOT mitigate**: genuinely slow structural dynamics at high λ (if the equilibrium is just far from the +2.0 init and many neurons need to cross threshold, decay doesn't make the mask travel faster, only measures it more precisely) — complementary to, not a replacement for, a possible λ-scaled window (a separate, also-untested hypothesis: higher λ plausibly needs a longer confirmation window independent of pruner capacity, since gap-diagnostic trajectories consistently showed larger `max_layer_delta_pct` at λ=0.8/1.6 than at λ=0.05/0.3 across every run checked so far). Target file when built: `train_pruner_opt125m_converge.py` first (already-validated base case), then port to the pg19 scripts.

---

## C · Making RL principled

**C1. Prune DURING training (co-adaptation)** [LIVE, the RL escape]
Interleave training epochs with prune steps; base weights co-evolve. The ONLY regime where the sequential MDP is non-degenerate (§10) — order genuinely matters, telescoping no longer collapses. State=(weights, training progress), reward=end-of-training acc at target sparsity. Overlaps B3.

**C2. RL variance fixes (if pursuing fixed-target RL at all)** [LIVE, but low priority — RL loses on frozen net]
- Batched A2C: K parallel rollouts, gradients averaged → 1/K variance. Most direct fix.
- Auto-tuned entropy (SAC α): learn the coefficient, target an entropy level.
- PPO retune: raw (un-normalised) entropy + the clipping we already have.

**C3. Alternative MDP framings** [OPEN-Q / mostly superseded]
- Conditional / per-input dynamic pruning: genuinely sequential over depth; reward = acc − λ·compute. **The one framing where RL beats a static hypernetwork by construction** (different problem: per-input compute, not a fixed mask).
- Infinite-horizon (sparsity-as-reward, terminate on acc-drop) & stagnation/knee-termination (τ = dual of λ): discover sparsity instead of fixing it. Sound but only worth it post-variance-floor.
- Local-state MDP (drop global features): cleaner credit, easier transfer; ablation.

---

## D · Transfer / generalization

**D1. Meta-train a transferable pruner** [DEMOTED to optional-science, as of 2026-06]
_Reconsidered: transfer is likely NOT the goal._ The deliverable is a smaller model deployed at scale; the pruner is a one-time \~90s cost, dwarfed by lifetime inference savings → retrain-per-model is fine. Transfer only earns its keep if you prune MANY models or want a universal-redundancy-detector science result. Keep the focus on best-mask-per-net + breaking the 240K floor (B3). Original idea below:
Train across a DISTRIBUTION of networks (many seeds/widths) with permutation-invariant features only (A1's signals). Tests if a *portable* redundancy detector exists. (Single-network transfer FAILED — F7 — because raw-weight rankings are per-network; perm-invariant features + cross-net training might fix it.)

---

## E · Reframings (mindset shifts — stop optimizing within the frame)

**E1. Pruner as scientific instrument** [LIVE] — mask is near-deterministic (σ=0.007) → a *measurement* of the net's redundancy. Ask what it's discovering (functional basis? task vs weight redundancy?), not how much it prunes.

**E2. Prunability as effective-dimension / scaling-law probe** [LIVE] — % prunable (or weight-floor) at fixed acc = capacity the task consumed. Sweep data/width/depth/training; tie to double descent & lottery tickets. Deliverable = curve + claim, not method.

**E3. Train-to-be-prunable** [LIVE] — training-time regularizer that concentrates redundancy so any pruner works better. Flips pruning to an inductive bias. Free lunch or tax?

**E4. Prune in a learned basis** [LIVE] — neuron = arbitrary coordinate; learn a rotation R, prune *directions* in rotated space, rotate back. Breaks the neuron=unit assumption.

---

## F · Scale-up & architecture

**F1. CIFAR-10 + conv backbone** [NEXT — validates everything] — does any of the MNIST story (width=redundancy, weight floor, BiLSTM dominance) hold for conv channels & a harder task? Per-channel reformulation of the row encoder.

**F2. Self-attention pruner** [NEXT — the BiLSTM-representer fix] — The BiLSTM has two proven weaknesses: (1) only cross-LAYER context (one scalar/layer) — a neuron sees only its own row + a shared layer scalar, **NO neuron↔neuron reasoning**; (2) layer token = crude signed mean-pool of rows (cancels). Fix: **self-attention over neurons** (within+across layers) → explicit pairwise redundancy ("A,B duplicate → drop one"), perm-equivariant, O(N²) but N≈2048 fine. This is THE principled representer upgrade. Cheaper sub-fix: attention-pool the layer token (Set-Transformer/Deep-Sets) for weakness (2). Autoencoder on weight rows = richer per-neuron embedding but adds no neuron↔neuron reasoning → lower priority.

---

## H · Pruner methodology / meta-scaling laws (the pruner ITSELF, not its targets)

These are methodology directions: instead of asking "what does the pruner reveal about base models?", they ask "what does the pruner *need* to be?" — turning hyperparameter guesses into predictions.

**H1. Pruner-capacity scaling law: how does required pruner size scale with base-model size?** \[EXTREMELY HIGH PRIORITY — methodology unlock] — All experiments so far have used essentially the same pruner config (BiLSTM, embed\_dim=64, lstm\_hidden=128) across vastly different bases: MNIST 1024 → CIFAR\_big 8192-fan-in fc1 → tiny LeNet 400-fan-in. It "works" — but we don't know _what determines the minimum capacity needed_. Hypothesis: required pruner params scale roughly as max(C\_in) · embed\_dim (row encoder dominates) + #layers · lstm\_hidden² (BiLSTM cost). Beyond some #layers, BiLSTM-over-layers may itself become inadequate → switch to attention (ties into F2). **Architecture transitions to test:** at what base width does row encoder's linear projection break (need MLP / autoencoder)? At what base depth does BiLSTM-over-layers degrade (need self-attention over layer tokens)? _Methodology counterpart to F4 — F4 asks "how prunable is the base?", H1 asks "how big does the pruner need to be to find that out?"._ Could become a small but very clean methodology paper. Protocol: hold task fixed (MNIST or CIFAR), sweep base width ∈  × pruner size ∈ , measure pruning quality, fit the surface.

**H2. λ-vs-task/model scaling law: predicting λ* without sweeping.** [EXTREMELY HIGH PRIORITY — removes the sweep] — Pareto-optimal λ* observed so far: MNIST winners λ ∈ [0.05, 0.3], CIFAR_big winner λ = 0.03, CIFAR LeNet TBD. The variation has to come from somewhere — almost certainly from (a) CE scale (CIFAR's higher CE = harder task = soft penalty needs to be relatively smaller to compete with CE gradient) and (b) per-layer keep granularity (averaging over more layers vs fewer changes effective penalty weight). Hypothesis: λ* ≈ k · (CE_orig / mean_layer_size)^α — calibrated so the sparsity-loss gradient lives on the same scale as the task-loss gradient on the gates. Protocol: vary task (MNIST / CIFAR10 / CIFAR100 / SVHN) × base architecture × base width, sweep λ around the predicted point, fit λ*(task, model). If a clean invariant exists, we can compute λ* from a single forward pass on a new model + dataset — no sweep needed. **Even partial success is publishable methodology.** Connects to H1: a pruner-size + λ joint scaling law would be the closest thing this project has to a "law".

**H3. λ_sim formula — the dynamical-regime view of λ_*.** [COULD BE TRUE, NEEDS PROVING — refines H2] — From the LeNet vs CIFAR_big sweeps, every base architecture has a sharp "sequential → simultaneous commitment" transition at some λ_sim. Below λ_sim, the pruner attacks layers one at a time, greedy by per-layer redundancy, and often runs out of training budget before fc2 (the marginal layer) commits. Above λ_sim, all layers transition together in one phase. The Pareto-winning λ_* sits at or just above λ_sim. **Hypothesis (could be true, needs proving):**
$\lambda_{\mathrm{sim}} \approx k \cdot \langle |\partial \mathrm{CE} / \partial g| \rangle \cdot \mathrm{mean\_layer\_size}$where ⟨|∂CE/∂g|⟩ is the typical gate-gradient magnitude under CE on a small validation batch. **Mechanism:** at λ\_sim, the soft-penalty gradient per gate (λ / (N\_layers · S\_ℓ)) matches the typical CE gradient per gate, so all gates feel comparable downward pressure simultaneously and no layer's gates dominate the priority queue. **Protocol:** measure ⟨|∂CE/∂g|⟩ on the held-out base model in one forward+backward pass (no pruner training needed), predict λ\_sim, then sweep ±50% around the prediction. If predicted λ\_sim is within 2× of empirical λ\__: huge win (no sweep needed). Even an order-of-magnitude correct prediction is a usable heuristic. Cleaner than H2's "Pareto λ" because λ\_sim is a regime boundary (sharp) instead of an optimization optimum (smooth, noisy at small scale)._

**H4. BiLSTM MLP-portion universality across architectures (incl. transformers).** [HIGH PRIORITY — turns the method into a finding] — Prove the soft-λ BiLSTM weight-conditioned pruner works on the MLP portions of *any* base architecture, not just standalone MLPs and not just CIFAR_big's FC head. The explicit targets:
- **(a) Conv backbones beyond LeNet/CIFAR_big.** ResNet-18/34/50 on CIFAR-10/100 — does the same protocol (freeze backbone, prune fc head with our BiLSTM) hold? Single-FC-layer heads (ResNet has only one fc) collapse the BiLSTM sequence to length 1 — interesting edge case (does the BiLSTM still help?).
- **(b) Transformer MLP blocks** — every transformer layer has a 2-layer FFN (e.g. d_model → 4·d_model → d_model). These are exactly the kind of MLP head the BiLSTM should drop into. Per-transformer-block FFN pruning could be a clean compression target. Open question: do we prune each block's FFN independently (N independent BiLSTMs) or jointly (one BiLSTM over a 2N-token sequence)? The latter lets the pruner reason about cross-block redundancy (probably massive in transformers).
- **Why this matters:** If universal, the pruner becomes a *generic MLP-head prunability detector* for arbitrary architectures, not a recipe tied to specific topologies. This is the framing that turns "we built a pruner for our MLPs" into "we built a pruner for all MLPs" — much stronger paper framing. Connects to F4/F13: if width-prunability holds across transformer FFN widths (768/1024/4096/...), we have a clean width-scaling law on top of the universality result.
- **H4b landed a first result (F16):** GPT-2 small FFN pruning works, and unexpectedly *improves* WikiText-2 perplexity at every λ tested (0.01–0.4, 12–22% pruned) rather than trading it off. Two follow-ups this opened, both proposed not yet run:

**H4b-i. Extend the λ sweep upward to find the actual knee.** \[DONE, 2026-07-11 — see F17] — Ran λ ∈  (finer 1.35× steps, not the original doubling proposal) under the newly-fixed sliding-window eval protocol. Result: pruning hurts monotonically at every point (opposite direction from F16), no knee found because there's no improvement region in this range at all under the corrected eval — see F17. Opened a NEW higher-priority item below (H4b-iii) instead of resolving cleanly, because v1's original λ range wasn't tested under the same protocol.

**H4b-iii. Re-evaluate v1's checkpoints under the new eval protocol (no retraining).** [DONE, 2026-07-11 — see F18] — Ran `reeval_gpt2_checkpoints.py`, merged with v2 via new `reconcile_gpt2_sweep.py`. F16 was almost entirely a protocol artifact (ΔCE at λ=0.01/0.02 is statistically zero under the fixed protocol, not the reported improvement). Reconciled λ=0.01→3.2 curve now exists in one consistent protocol; peak efficiency at λ=1.35 (49.16% pruned, broad plateau λ≈0.75–2.4).

**H4b-ii. Re-evaluate on a broader/matched-distribution dataset (e.g. OpenWebText) to test the domain-narrowing hypothesis.** \[NEXT — tests F16's central open question] — F16's "pruning improves ppl" result might be WikiText-2-specific (narrow Wikipedia-prose eval vs GPT-2's broad WebText pretraining) rather than a general effect. Direct test: eval the same trained pruners (`pruner.pt` checkpoints already saved per run) on an OpenWebText-style held-out set. If the improvement shrinks/vanishes on the broader set, F16 is an eval-domain-narrowing artifact, not a real capability gain — important distinction for how the result gets framed. Practical note: OpenWebText (`Skylion007/openwebtext` on HF) is \~40GB and has no official train/test split — would need either a small held-out shard carved out manually, or a lighter existing proxy (e.g. `stas/openwebtext-10k`) to avoid a full 40GB download just for eval. Storage/scope tradeoff to decide before running.

---

## G · Closed / tested-negative (do NOT re-propose)

- **[DEAD]** RL on frozen net beats hypernetwork — no (§10 theorem; every variant ≥ BiLSTM's 3.68pp).
- **[DEAD]** Per-neuron Bernoulli action — collapsed (62pp; horizon collapse, F-notes).
- **[DEAD]** Normalised entropy — regressed RL (raw bonus was load-bearing).
- **[DEAD]** Smaller chunk k reduces variance — no, worse (k=8 5.54±3.44).
- **[DEAD]** Single-network frozen pruner transfer — fails (F7).
* **\[KNOWN]** Threshold-sweep ≈ retrain only at the trained operating point (\~1–2pp pessimistic elsewhere).

---

## Priority (my ranking)

-1. **B8 full corrected OPT-125M WikiText-2 re-sweep, reconciled against F18** (CO-HIGHEST PRIORITY, 2026-07-20 — F21 shows F19's own headline result may have been substantially a tokenization artifact; this determines whether F18/F19's divergence is real at all, which B5 and B6 both currently assume without this having been confirmed under clean numbers).
0. **B7 dynamic step budget + pruner-capacity ablation** (CO-HIGHEST PRIORITY, 2026-07-17 — every sweep run so far, on every model, has used an unvalidated fixed 18,750-step / embed_dim=64/lstm_hidden=128 convention; F20 shows this is both ~15-20x wasteful in most cases and possibly insufficient at the operating points that matter most — blocks getting trustworthy results cheaply from here on, including the in-progress pg19/B6 sweep; run alongside B8, same sweep).
1. **B1 STE-top-K validation** (cheap, validates the cleaner formulation + tests if hacks are removable).
2. **A1 richer-input ablation** (cheap; tests transfer-enablement).
3. **B3 / C1 staged-prune-+-retrain** (the only path past the 240K floor — the real result).
4. **F1 CIFAR scale-up** (does the story generalize?).
5. **D1 meta-train transferable pruner** (turns "a tool" into "a finding").
6. **E2 prunability scaling-law** (science framing, high novelty).
7. **H1 pruner-capacity scaling law** (methodology unlock — predicts pruner sizing instead of guessing; complements F4 / E2).
8. **H2 λ-vs-task/model scaling law** (removes the λ sweep; clean methodology result if invariant exists).
9. **H3 λ_sim formula** (refines H2 via dynamical-regime view; sharper target than Pareto-λ*).
10. **H4 BiLSTM MLP-portion universality** (conv backbones + transformer FFNs — turns "our method" into "a generic MLP-head detector").
11. ~~H4b-i extend GPT-2 λ sweep upward~~ — **done, see F17.**
12. ~~H4b-iii re-eval v1 checkpoints under the fixed eval protocol~~ — **done, see F18.**
13. **H4b-ii re-eval GPT-2 pruners on OpenWebText-style data** (F18 already shows the "improvement" was a protocol artifact, not a domain-narrowing effect — this idea's original motivation is mostly moot now; could still be worth checking whether the λ=1.35 knee *location* shifts on a broader eval set, but lower priority than before).
15. **Fine-grained λ sweep around 0.75–2.4** to pin down whether the peak-efficiency plateau (F18) has real structure (the λ=1.8 dip) or is 2-seed noise — cheap, same script, same protocol.
14. **Fix the same `max(x, 0.5)` efficiency-metric bug in `efficiency_compare.py`** (low priority — affects only how F14/F15's existing plots are read, not new work; flagged in engineering_decisions.md, needs its own fix design since accuracy-pp isn't a log-ratio like CE).
16. **B5 dense-fine-tune control** (highest-value next step for the task-specific-pruning framing — F19's OPT-125M result isn't a usable claim until this runs: does plain dense fine-tuning on the same WikiText-2 data match the pruned model's improvement, with no sparsity at all? If yes, B5 collapses into "this is just fine-tuning" and the sparsity isn't buying anything beyond what a cheaper dense fine-tune already would).
