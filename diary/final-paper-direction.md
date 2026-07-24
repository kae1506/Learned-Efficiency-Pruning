# Final Paper Direction

Scope-setting doc, not a findings log — this is the skeleton for what we're actually going to write up, filtered from everything in `crisp-findings.md`/`ideas.md` down to the parts that belong in the paper. Read this before proposing new experiments: anything not in scope here needs a case for why it should be added, not just "it's interesting."

## Hard scope constraints (decided, not up for silent reinterpretation)

1. **NOT pursuing STE-top-K.** Closed, F12/B1. Loses to λ-penalty by ~2.6× at iso-accuracy. Not discussed except as a one-line "considered and rejected" note if a reviewer would ask "why not a hard-budget formulation."
2. **NOT pursuing RL.** Closed, F5/F6/F8. F6 is a theorem (telescoping reward on a frozen model → path-independent return → RL is provably dominated by direct differentiable optimization here) — worth one sentence in related work as a reason the paper doesn't bother with an RL baseline, nothing more.
3. **MNIST/CIFAR results are QUALITATIVE ONLY.** No numbers from F1–F15 appear in the paper. They motivate hypotheses (width > depth, mask determinism, a LTH-flavored "stable discoverable subnetwork" story) that the GPT-2/OPT-125M experiments then test for real. Toy-scale is a pilot, not a result.
4. **GPT-2 / OPT-125M results are the paper's actual empirical content.** Quantitative, cited precisely, held to full scrutiny (seeds, protocol, sanity checks).
5. **We are NOT claiming a performance increase, and NOT framing this as fine-tuning-via-pruning.** The story is: take a frozen pretrained model, find a sparse subnetwork, pay some accuracy cost, get a cheaper-to-run model. Full stop. This explicitly excludes the F19/B5/B6 thread (OPT-125M pruning "improving" WikiText-2 CE, task-specialization-as-deployment-technique) from the paper's spine — see Discussion §8 for how that thread gets mentioned (as a methodological cautionary tale, not a result).

---

## Working title

*"Learning to Prune: A BiLSTM Hypernetwork for Structured Neuron Pruning in Frozen Language Models"* (placeholder — revisit once §6 has final numbers)

---

## 1. Abstract (skeleton)

- Structured neuron pruning of frozen, pretrained models via a learned hypernetwork (row-encoder + BiLSTM) that reads only weight matrices and outputs binary keep/prune gates.
- No retraining of the base model; no fine-tuning; no data beyond what's needed to score the pruning loss. Goal is a smaller, cheaper-to-run model at a controlled, characterized accuracy cost — not a better one.
- Toy-scale pilot (MLPs on MNIST/CIFAR) establishes qualitative regularities: prunability tracks width, not depth or raw parameter count; the discovered mask is a near-deterministic property of the trained weights, echoing the lottery-ticket-hypothesis intuition that a small, effective, already-present subnetwork explains a dense net's performance — but discoverable directly from weights, no iterative reinit/retrain cycle needed.
- Main results: extend the same pruner architecture, unmodified, to transformer FFN blocks in GPT-2 small and OPT-125M. Report a clean λ-controlled Pareto frontier (sparsity vs. perplexity cost), an efficiency metric characterizing the frontier's operating point, and a head-to-head against a standard activation-magnitude baseline at matched sparsity.
- [PLANNED] Scaling-law characterization: how pruner capacity, training steps, and λ* should scale with base-model size — currently open, see §7.

## 2. Introduction (skeleton)

- Motivation: inference cost of deployed LLMs; structured (not just unstructured) sparsity is what actually saves wall-clock/memory on commodity hardware, unlike unstructured magnitude pruning.
- Gap: most structured-pruning heuristics (magnitude, activation-magnitude/APoZ, gradient-based) score neurons independently or with a hand-designed rule; they don't learn a scoring function, and they don't share information across layers when deciding a layer's budget.
- Our approach: a small hypernetwork that (a) scores each neuron from its own weight row via a shared, permutation-respecting encoder, and (b) shares that information across the depth axis via a BiLSTM, so a layer's implied redundancy can be informed by every other layer's before gates are decided. Trained end-to-end against a soft sparsity penalty (λ) on the frozen base model's own loss — no external labels beyond next-token prediction, no architecture-specific tuning needed to port across GPT-2 ↔ OPT-125M.
- Contributions list (fill in as results firm up): (1) architecture + training regime, (2) toy-scale qualitative motivation, (3) LLM-scale Pareto/efficiency results across two architecturally-distinct transformers, (4) baseline comparison at matched sparsity, (5) [PLANNED] scaling laws.

## 3. Related work (skeleton, pointers only)

- **Lottery Ticket Hypothesis** (Frankle & Carbin) — existence of sparse, trainable-from-scratch subnetworks via iterative magnitude pruning + rewind. Our F1–F4 pilot connects to this (see §5) but differs mechanically: we never retrain/rewind, we read off a mask directly from the trained weights via a learned scorer, and we care about *frozen-model inference-time* sparsity, not *trainability from a lottery ticket init*.
- **Structured pruning baselines**: magnitude pruning, activation-magnitude/APoZ (Hu et al. 2016) — this is what §6.4's baseline comparison is measured against.
- **One-shot LLM pruning**: SparseGPT, Wanda (both already in `docs/papers/`) — unstructured/semi-structured, calibration-based, no training loop. Worth a paragraph contrasting: they're one-shot and cheap (~minutes); ours is a trained procedure (~20-25 min GPU per λ) that learns a scoring function rather than applying a closed-form rule — the tradeoff is compute-for-quality, and §6.4 is the evidence for whether that tradeoff pays off.
- **Recent structured LLM pruning — TODO closed, 2026-07-23.** PDFs stored in `docs/papers/` (`slimllm_guo2025.pdf`, `gisp_wang2025.pdf`), alongside `sparsegpt_frantar2023.pdf`/`wanda_sun2023.pdf`. Comparison below is from an abstract-level read (arxiv 2505.22689 / 2510.18030) — the summary is directionally reliable but revisit against the full PDF text before finalizing any citation claims:
  - **SlimLLM** (Guo et al., "Accurate Structured Pruning for LLMs") — channel + attention-head pruning, calibration-based (one-pass importance score, not trained), linear-regression output-matrix repair after pruning (philosophically closer to SparseGPT's OBS-style compensation than to anything here). No neural scorer, no learned cross-layer context.
  - **GISP** (Wang et al., "From Local to Global: Revisiting Structured Pruning Paradigms for LLMs") — global (not per-layer) structured pruning of attention heads + MLP channels, iterative but still calibration-based (first-order loss-based importance, block-wise normalized). Tests on Llama2-7B/13B, **Llama3-8B, Mistral-7B** (directly overlapping with §6's scaling targets), Qwen3-8B, DeepSeek-R1-Distill; calibrates on WikiText-2 (+ GSM8K for a task-aligned variant). "Prune-once, deploy-many" nested subnetworks — a capability this method doesn't have (one pruner per λ, not a single multi-sparsity artifact).
  - **The load-bearing contrast, unchanged by either paper**: neither trains a scoring network — both are calibration/importance-score methods (closed-form or iterative-but-analytic), same category as SparseGPT/Wanda in that respect. This method's actual distinguishing claim (a hypernetwork that *learns* the scoring function end-to-end against a soft sparsity loss, rather than applying a hand-designed importance formula) has no direct analogue in either. Both are broader in scope than this method though (attention heads + MLP channels, vs. FFN-neurons-only here) — an honest scope gap, not to paper over.
  - **Two actionable connections**: (1) GISP's importance signal (gradient×activation-style) is close to what `ideas.md` A1 proposes and this project has never actually tested (activation/gradient input instead of raw weights) — external evidence that direction is worth running. (2) GISP reports real numbers on Llama3-8B/Mistral-7B — once §6's scaling sweeps land, GISP's published sparsity/ppl figures on the *same models* are a natural external benchmark, not just a self-built magnitude baseline (F22).

## 4. Method

### 4.1 Problem setup (scope statement — put this early, explicitly)

Frozen pretrained model `f_θ` (θ never updated). A pruner `g_φ` reads `f_θ`'s own weight matrices and outputs per-neuron binary gates `m`. Objective: find `φ` minimizing task loss on the gated model `f_θ(x; m)` subject to a sparsity penalty on `m`. **No fine-tuning of θ at any point, before or after pruning.** The deliverable is `(θ, m)` — a smaller effective model — not a better `θ`.

### 4.2 Pruner architecture

- **Row encoder**: each neuron's incoming weight row `w ∈ ℝ^{d_in}` maps to a scalar logit via a shared 2-layer MLP (`Linear(d_in, embed_dim) → ReLU → Linear(embed_dim, 1)`), applied identically to every row in a layer. Output bias initialized to +2.0 (STE gate starts near-fully-open, so pruning is *learned in*, not defaulted).
- **Cross-layer context (BiLSTM)**: per-layer weight matrices are summarized (mean over the neuron axis) into one embedding per layer, then run through a bidirectional LSTM over the depth sequence. The resulting context vector, passed through `tanh` (bounded to (−1,1), so it can *modulate* but never override the per-node logit), is added back to every neuron's logit in that layer.
- **Gate**: straight-through estimator — hard threshold at 0.5 forward, sigmoid gradient backward.
- Architecture is base-model-agnostic: same code, same hyperparameters (`embed_dim=64, lstm_hidden=128`, ≈2M params) ported from MLP hidden layers → CIFAR conv-net FC head → GPT-2 Conv1D FFN → OPT-125M nn.Linear FFN, with only the weight-matrix-extraction plumbing changed per architecture.

### 4.3 Permutation invariance (why the architecture is shaped this way)

Neurons within one layer have no canonical order — any permutation of a layer's rows (with the matching permutation applied downstream) computes an identical function. A pruning criterion that isn't invariant to this symmetry is fitting an artifact of storage order, not the network's actual redundancy structure.

The row encoder is exactly invariant to this symmetry by construction: it is one shared function `h: ℝ^{d_in} → ℝ` applied independently per row, so permuting the rows just permutes which score goes with which row — no information about row *position* ever enters the score. The layer-context summary (`W.mean(dim=0)`, over the neuron axis) is likewise permutation-invariant — a sum over rows divided by a constant is unaffected by row order. So the pruner's neuron-level output is provably equivariant to within-layer neuron permutation: `g_φ(PW) = P·g_φ(W)` for any permutation matrix `P` acting on the neuron axis. This is a structural guarantee, not a trained-in behavior — worth stating as a small formal lemma in the paper rather than an empirical claim.

*(Practical corollary, discovered while investigating why a trained pruner does not transfer across independently-trained networks: this invariance is already complete — there's no missing permutation-invariance to add via e.g. random-shuffling data augmentation. The transfer failure is a different symmetry problem — see §8 Limitations.)*

### 4.4 Cross-layer context via BiLSTM (why a sequence model here is *not* the same mistake)

Depth, unlike within-layer neuron order, is a genuine, non-arbitrary axis — layer `i`'s output literally feeds layer `i+1`. So a sequence model over the depth axis is the structurally correct choice, in contrast to a sequence model over neurons-within-a-layer (which would impose false order on a permutation-symmetric set — exactly the mistake §4.3 rules out). The BiLSTM lets a layer's implied budget depend on the redundancy profile of every other layer in both directions — e.g., a layer with unusually many high-norm rows can inform neighboring layers that they can be pruned harder without a capacity bottleneck. This is the mechanism given for F4/F10-style toy-scale findings (width concentrates redundancy, depth is load-bearing) — depth-aware context is what lets the pruner discover that asymmetry rather than treating every layer identically.

### 4.5 Training regime / loss

```
loss = (CE_pruned − CE_orig) + λ · sparsity_loss
sparsity_loss = mean(gate) across all neurons  (fraction kept, so loss ↓ as sparsity ↑)
```

Frozen base model forward pass runs twice per step (once ungated for `CE_orig`, once gated for `CE_pruned`); only `φ` (the pruner) receives gradients. λ is the sole knob controlling the sparsity/accuracy operating point, swept externally (§4.6).

**⚠ Open design question, not yet decided — flagging per house rule, this is load-bearing for the paper's framing:** the loss as currently implemented has *no floor at zero* on `(CE_pruned − CE_orig)` — gradient descent is free to push `CE_pruned` below `CE_orig` if the training data allows it. This is exactly the mechanism that produced the F19 "improvement" result the paper is now explicitly *not* claiming (§0.5). Given the "compression only, no improvement claim" framing, should the loss be changed to `max(CE_pruned − CE_orig, 0) + λ·sparsity_loss` for the paper's actual experiments — so the objective structurally cannot claim credit for improvement, only for cheap-as-possible accuracy preservation? This changes what gets trained, not just how a result is described, so it needs a decision before the LLM sweeps that go into the paper are (re-)run. Current sweeps (F16–F22) all used the unfloored version.

### 4.6 λ sweep, Pareto curve, efficiency metric

- λ swept on a grid (typically log-spaced, ~6-8 points per model), 2 seeds per point where feasible (F8: single-seed numbers are noise at this kind of task).
- **Pareto curve**: % neurons pruned (x) vs. pruned-model perplexity (y), with the unpruned baseline as a horizontal reference line.
- **Efficiency metric**: `efficiency(λ) = (% pruned) / exp(ΔCE)`, `ΔCE = ln(pruned_ppl / orig_ppl)`. Single number per λ, converts the nats-scale cost back to a perplexity-ratio scale so it's comparable across models/datasets with different baseline CE. Used to identify the recommended operating point (peak efficiency), while the full Pareto curve is reported alongside it since the peak is typically a broad plateau, not a sharp optimum (F18).

## 5. Toy-scale pilot (MNIST / CIFAR) — qualitative only, no numbers

Purpose: motivate the hypotheses tested for real in §6, and connect to the Lottery Ticket Hypothesis. State results as directional claims, not figures.

- **Weights alone are sufficient to find a good mask** — no data, no gradients, no retraining needed to discover a highly-prunable subnetwork from a trained model's weight matrices (F1). This is a stronger claim than classic LTH's iterative-reinit-and-retrain procedure: the mask is read off directly.
- **The mask is a near-deterministic property of the trained weights**, not an artifact of the pruner's own random init/training noise (F2) — supports reading "prunability" as a real, measurable property of the base network, not noise in the measurement procedure.
- **The learned pruner strictly dominates classical scoring rules** at every accuracy budget tested, because it conditions on the full weight vector rather than a hand-designed scalar summary (F3) — the toy-scale precedent for §6.4's LLM-scale baseline comparison.
- **Prunability is governed by width, not depth or raw parameter count** (F4, F10) — wide layers concentrate redundancy; depth is load-bearing capacity that resists pruning. This directly motivates §4.4's cross-layer context design (a pruner that can't see across layers can't discover this asymmetry).
- **Compression appears to converge toward a similar effective-capacity floor regardless of how a fixed neuron budget is initially distributed across width vs. depth** (F11, qualitative reading only) — a LTH-flavored regularity: many different starting architectures at the same nominal capacity seem to bottom out near the same *effective* capacity when pruned, though a genuinely smaller network trained from scratch on the same task is not reliably matched by pruning a bigger one down to it (i.e., pruning finds *a* small subnetwork, not necessarily *the* smallest one achievable by training small directly) — state this carefully, it's the closest thing to a LTH-hedge in the toy-scale data and shouldn't be overclaimed.

Do not include: any specific sparsity %, any specific accuracy-drop pp figure, any specific weight-count floor number, RL results, STE-top-K results. Section should read as "here's the qualitative shape of what we found at toy scale, motivating the LLM experiments" in well under a page.

## 6. LLM-scale experiments (GPT-2 small, OPT-125M) — the paper's real content

### 6.1 Setup

Frozen GPT-2 small / OPT-125M, prune the FFN intermediate neurons (3072 per block × 12 blocks, both architecturally compute-equivalent: 12 layers, hidden=768, 12 heads). WikiText-2 train/test, standard sliding-window CE evaluation protocol (matches GPT-2 paper / SparseGPT / Wanda convention — no non-overlapping-block penalty, see F17/F18 for why that distinction mattered). Same pruner architecture and hyperparameters as the toy-scale pilot, no per-model tuning beyond the (currently unresolved, see §4.5) architecture-appropriate eval window.

### 6.2 GPT-2 small — Pareto curve and efficiency (DONE, F16→F18, clean result)

Reconciled λ=0.01→3.2 sweep under the corrected eval protocol: cost is monotonic and positive across the *entire* range — no free-lunch region, no crossover. Peak efficiency at λ=1.35 (49.16% pruned), broad plateau from roughly λ=0.75 to λ=2.4. This is a clean, standard compression-tradeoff story and the paper's primary Pareto-curve figure. One unresolved wrinkle (λ=1.8's local dip below its neighbors) — worth a finer grid point or acknowledged as noise, not investigated further unless it recurs.

### 6.3 OPT-125M — status: mid-reconciliation, NOT yet paper-ready (F19 → F21 → B8, open)

The original OPT-125M sweep (F19) reported an in-domain WikiText-2 improvement that does not belong in this paper under the §0.5 scope constraint even if real — but it also turned out to be substantially a tokenization artifact (F21: a WikiText-2 loader bug scattered ~4,358 spurious BOS tokens through the corpus, roughly halving the reported baseline perplexity once fixed). A partial re-run under the fix (4 mid-range λ, reduced steps/seeds) now shows the *same qualitative shape* as GPT-2's curve — small/near-zero cost at the lightest λ, monotonic real cost beyond that — which is actually the right shape for this paper's framing, but isn't yet a full, reconciled, paper-grade sweep (missing low-λ points, reduced step budget, 1 seed not 2). **This is the main open item before §6 is complete** — see `ideas.md` B7/B8 for the exact next-sweep protocol (full λ grid, convergence-appropriate step count, 2 seeds), already scoped, not yet run.

### 6.4 Baseline comparison — activation-magnitude pruning at matched sparsity (DONE, F22)

At OPT-125M's λ=0.75 operating point (42.56% pruned), the trained pruner's perplexity cost is ~6.3× smaller than a standard activation-magnitude baseline (mean post-ReLU activation, global threshold, same gating mechanism, same eval protocol, matched neuron count exactly). This is the LLM-scale analog of the toy-scale F3 result and is a clean, single-number headline comparison for the paper — learned scoring beats a standard heuristic by a wide margin at fixed sparsity, not just "prunes more before breaking."

Worth one sentence noting the *shape* of the difference: the baseline's failure mode is concentrating almost all its cuts in a few middle layers (down to ~13% kept in its worst layer) while the trained pruner stays much flatter across depth — plausibly the actual mechanism behind the gap, not just "better per-neuron scores." This directly supports the §4.4 cross-layer-context design argument with LLM-scale evidence, not just toy-scale motivation.

## 7. Scaling laws — [PLANNED, none of this is run yet]

Frame as future work / a second paper section contingent on compute, not claimed results:

- **H1 — pruner-capacity scaling**: does the minimum pruner size needed to find a good mask scale with base-model size (row-encoder cost ~ `max(d_in)·embed_dim`, BiLSTM cost ~ `#layers·lstm_hidden²`)? Not tested — every experiment so far (MLP → CIFAR conv → GPT-2 → OPT-125M) reused the same fixed pruner config regardless of base size.
- **H2/H3 — λ\* prediction**: can the Pareto-optimal λ be predicted from cheap properties of the base model/task (baseline CE scale, layer count) instead of swept? Toy-scale data (F15) already refutes any simple monotonic λ\*-vs-size relationship; a more structured hypothesis (λ\* ≈ k·(CE_orig/mean_layer_size)^α, or the "sequential vs. simultaneous commitment" dynamical-regime view, H3) is proposed but unproven.
- **Step-budget scaling**: F20 found the (inherited, never re-derived) fixed step count is badly and non-uniformly mismatched to actual convergence across λ and models — any scaling-law claim needs a convergence-based stopping rule first (B7), or the "how does X scale" question is confounded by "did training actually finish."
- **H4 — architecture universality**: the MLP→CIFAR→GPT-2→OPT-125M portability already demonstrated is evidence for this, but it's anecdotal (4 architectures, not a controlled sweep) — a real test would hold task/data fixed and sweep architecture family deliberately.

This section should be written as a roadmap with a clear "not yet run" label on everything, not blended with §6's completed results.

## 8. Discussion / Limitations

- **Explicit scope reminder**: this paper reports a compression method for frozen models — a controlled accuracy-for-sparsity trade, characterized via Pareto curves and an efficiency metric — not a fine-tuning technique and not a claim that pruning ever improves the base model. State this plainly, early in the discussion, not just in scope-setting.
- **The F19 thread as a cautionary tale, not a result**: early OPT-125M experiments appeared to show pruning *improving* in-domain perplexity. Worth one paragraph explaining what that turned out to be (substantially a tokenization bug, F21) and why, even setting the bug aside, an unfloored `(CE_pruned − CE_orig)` loss term will always be *capable* of producing this kind of result on a data distribution correlated between train/test (WikiText-2's own train/test split) — and why it doesn't generalize (the original out-of-domain C4 check showed real, monotonic degradation at every λ). This is useful precisely *because* it explains why the paper's scope constraint (§0.5) is the right one, not a limitation to apologize for.
- **Transfer does not work**: a pruner trained on one network does not transfer to a different (even architecturally identical) independently-trained network (F7) — each deployment needs its own ~20-25 min training run. Explain the mechanism from §4.3/§4.4's discussion: the architecture is already exactly permutation-invariant within a layer, so the failure isn't a missing invariance — it's that raw weight *values* aren't comparable across independently-trained networks' weight-space geometry. Note activation/gradient-based inputs (not yet tested) as the natural fix, without claiming it works.
- **Compute cost honesty**: ~20-25 min GPU per (λ, seed) operating point vs. one-shot calibration methods (SparseGPT/Wanda, minutes total) — the paper should state this tradeoff plainly rather than let the Pareto curve imply this is free.

## 9. Conclusion (skeleton)

Restate: learned, weight-conditioned, depth-context-aware pruning of frozen models produces a clean, controllable sparsity/accuracy Pareto frontier that beats a standard activation-magnitude baseline by a wide margin at matched sparsity, ported without architecture-specific changes across two different transformer implementations. No claim of improved capability — the value proposition is a better compression method, characterized rigorously (protocol bugs found and fixed in public, transfer failure mode explained mechanistically, baseline comparison run fairly).

---

## Open questions before this is actually finalized (need your call, not mine)

1. **§4.5**: floor the training loss at zero (`max(CE_pruned − CE_orig, 0)`) to match the "no improvement claim" framing structurally, or keep the current unfloored loss and handle the framing purely in how results are described? This affects whether existing checkpoints/sweeps (F16–F22) are reusable for the paper or need re-running.
2. **§6.3**: does the paper wait for B8's full OPT-125M reconciliation, or ship with the partial 4-λ result clearly caveated? B8 isn't run yet.
3. **§5**: is a qualitative toy-scale section even worth a full section, or should F1–F4/F11's LTH connection be compressed to a paragraph in the Introduction/Related Work instead, saving the section budget for §6/§7?
4. Target venue/length (workshop paper vs. full paper) — affects how much §7 (currently 100% unrun) can realistically be more than a "future work" paragraph.

---

## TODO — before this is submittable

### Experiments to complete
- [ ] **B8**: full OPT-125M WikiText-2 re-sweep, corrected tokenizer, λ=0.01→1.8 full grid, 2 seeds — replace the current partial check (4λ, 1 seed, 8000 steps, no λ<0.2)
- [ ] **B7**: resolve step budget before B8 — convergence-based stopping vs. fixed 18,750; rerun OPT-125M λ=1.8 at 2× pruner capacity to isolate capacity vs. steps as the slow-convergence cause
- [ ] Fine λ grid 0.75→2.4 on GPT-2 reconciled sweep — resolve whether the λ=1.8 efficiency dip is real curvature or 2-seed noise
- [ ] Numeric comparison against ≥1 trained-mask baseline beyond activation-magnitude — L0 regularization (Louizos 2017) or CoFi (Xia 2022), both already in `docs/papers/`, at matched sparsity on GPT-2 and/or OPT-125M

### Comparisons / numbers to pin down
- [ ] GPT-2-small WikiText-2 baseline ppl vs. literature (SparseGPT/Wanda GPT-2-124M numbers) — trace to a primary source, or drop the comparison and state non-comparability explicitly
- [ ] Structured-vs-unstructured iso-sparsity / iso-FLOP numbers (SparseGPT/OBC, currently only in engineering_decisions.md) — decide whether they appear as supporting context in §6.4 and pull the final numbers into the draft
- [ ] Consolidated compute/cost table — GPU-hours and $ per experiment family (toy-scale, GPT-2 sweep, OPT-125M sweep, baseline runs), currently scattered across engineering_decisions.md

### Needs formalizing
- [ ] §4.3 permutation-equivariance claim → stated as an actual lemma + one-line proof, not prose assertion
- [ ] §4.6 efficiency metric (`%pruned / exp(ΔCE)`) → short formal definition + justification for the methods subsection
- [ ] §4.5 loss-floor decision (floored vs. unfloored `CE_pruned − CE_orig`) → resolved and written up as a fixed methods choice, not left as an open question

### Figures needed (submission-grade, not experiment-tracking pngs)
- [ ] GPT-2 reconciled Pareto curve (%pruned vs. ppl) + efficiency-vs-λ, one consistent style, captioned
- [ ] OPT-125M Pareto curve (post-B8), same style as GPT-2's, for direct side-by-side comparison
- [ ] Baseline comparison figure — trained pruner vs. activation-magnitude, per-layer keep-% at matched sparsity (the "flatter across depth" claim from F22)
- [ ] One summary toy-scale figure for §5 (width vs. prunability across MNIST/CIFAR architectures), replacing the numeric appendix tables if §5 stays a full section
- [ ] Reproducibility table: steps, seeds, λ grids, GPU/precision per experiment family

### Writing
- [ ] Prose draft of Abstract, Introduction, Discussion, Conclusion (currently bullet skeletons)
- [ ] §3 Related Work — integrate papers already in `docs/papers/` (L0, CoFi, FLAP, LLM-Pruner, DISP-LLM, SDS, Network Slimming, MetaPruning) with explicit differentiation, not placeholder pointers
- [ ] Reproducibility / hyperparameter appendix, drawn from engineering_decisions.md's hack list
- [ ] Resolve working title

### Decisions needed (yours, not research)
- [ ] §5 scope — full section vs. paragraph-in-intro
- [ ] Target venue/length — governs how much of §7 is real vs. future work
- [ ] §6.3 — ship with caveated partial OPT-125M result, or hold for B8
