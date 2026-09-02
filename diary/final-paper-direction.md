# Final Paper Direction

Scope-setting doc, not a findings log — this is the skeleton for what we're actually going to write up, filtered from everything in `crisp-findings.md`/`ideas.md` down to the parts that belong in the paper. Read this before proposing new experiments: anything not in scope here needs a case for why it should be added, not just "it's interesting."

## Hard scope constraints (decided, not up for silent reinterpretation)

1. **NOT pursuing STE-top-K.** Closed, F12/B1. Loses to λ-penalty by ~2.6× at iso-accuracy. Not discussed except as a one-line "considered and rejected" note if a reviewer would ask "why not a hard-budget formulation."
2. **NOT pursuing RL.** Closed, F5/F6/F8. F6 is a theorem (telescoping reward on a frozen model → path-independent return → RL is provably dominated by direct differentiable optimization here) — worth one sentence in related work as a reason the paper doesn't bother with an RL baseline, nothing more.
3. **MNIST/CIFAR results are QUALITATIVE ONLY.** No numbers from F1–F15 appear in the paper. They motivate hypotheses (width > depth, mask determinism, a LTH-flavored "stable discoverable subnetwork" story) that the GPT-2/OPT-125M experiments then test for real. Toy-scale is a pilot, not a result.
4. **GPT-2 / OPT-125M results are the paper's actual empirical content.** Quantitative, cited precisely, held to full scrutiny (seeds, protocol, sanity checks).
5. **Main compression results (§4–7) make NO performance-increase claim, framed strictly as: frozen model → sparse subnetwork → accuracy cost → cheaper model.** Full stop, unconditionally, for the GPT-2/OPT-125M/Mistral-7B/Llama-2-7B compression sweeps — this part of the ruling is unchanged. **REVISED 2026-07-29** (previously: flat exclusion of the F19/B5/B6 thread from the paper entirely, 2026-07-14): that thread is reopened as §7.3, a conditionally-scoped case study — reopened because F23 (CNN/DailyMail summarization, Llama-2-7B) is a second, independent, and arguably stronger occurrence of the same "pruned beats dense" pattern than F19 was (ROUGE, a real generation-quality task metric, improved alongside perplexity — not just in-domain CE — and this run isn't exposed to F19's original tokenization-bug confound, F21, since its CE-delta training path was correct from the start). **Gated on B5's still-unrun dense-fine-tune control** (does plain fine-tuning on the same domain/task data match or beat this, with no sparsity at all?) — until that control runs, §7.3 stays labeled "case study in progress," not promoted to a headline result, and Discussion (§9) still carries the "why this isn't confused with the main compression claim" caveat.

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
  - **UPDATE 2026-07-31 (F24) — checked GISP against a real comparison, doesn't clear the bar yet.** Full-text-verified (not abstract-level) numbers from `gisp_wang2025.pdf` Tables 4–5: GISP DOES report Llama2-7B/WikiText-2 (dense ppl **12.19**; 20/30/40/50% → 17.01/24.27/34.54/64.07), and Llama3-8B + Mistral-0.3-7B/WikiText-2 (dense 14.14 / 15.14; 20% → 24.18/18.17, 30% → 31.73/25.58, 40% → 46.10/34.31, 50% → 79.42/58.16). But GISP's Llama2-7B dense baseline (12.19) is **~2.5× our own (4.903) and DISP-LLM's (5.12)** on the same model/dataset — searched the full paper (main text + appendix) for the eval context-length/stride that would explain this; not stated anywhere ("we follow the standard setup" is the only description given). **Not comparable until this gap is explained or re-measured under a shared protocol** — right model, unverifiable eval, exactly the kind of confound item 3 in the venue-gap-analysis (below) already warns about for DISP-LLM. Do not cite GISP's absolute ppl numbers against ours without resolving this first.
  - **UPDATE 2026-07-31 — SparseGPT checked, PARTIALLY overlaps (correction of a same-day error below).** Full-text search of `sparsegpt_frantar2023.pdf`: zero occurrences of "LLaMA"/"Llama"/"Mistral" — no Llama-2-7B/Mistral-7B comparison possible, that part stands. But an earlier pass here wrongly stated "there is no SparseGPT number on any model this project has ever run" — false. SparseGPT's Table 1 DOES report OPT-125M/WikiText2 (missed first time by a regex that didn't survive their table's `OPT - 50% 125M 350M 1.3B` layout). Checked properly: dense 27.66, SparseGPT@50% → 36.85. Two real gaps remain even for this overlapping model: (1) eval protocol — SparseGPT uses non-overlapping 2048-token WikiText2 blocks (their own text: "split it into non-overlapping segments of 2048 tokens"), ours is sliding-window (`eval_max_length=2048, eval_stride=1024`); dense baselines 27.66 (theirs) vs 23.941 (ours), ~15.5% gap, direction consistent with the known non-overlap-inflates-ppl bias this project already documented, so at least explicable, unlike GISP's. (2) sparsity basis — SparseGPT's "50%" is "uniform layer-wise sparsity" over the attention+MLP linear-layer pool (embeddings untouched, inferred from standard convention, not an explicit formula in the paper), which converts to **~33.94% of OPT-125M's true total params** (computed from `AutoConfig.from_pretrained("facebook/opt-125m")`'s real dims: attn+mlp pool 85,017,600 / full total 125,239,296). Our method's max reachable (100% FFN-neuron-pruned) is 45.25% of total — so this operating point is *structurally reachable*, unlike what an earlier pass here concluded — but our actual swept range tops out at λ=1.6 → 63.39% FFN-neurons-pruned → 28.69%-of-total, short of the 33.94% target (would need ~75% FFN-neurons-pruned, untested, and SparseGPT gives only this single point for OPT-125M, nothing to interpolate against). Net: closer to comparable than either prior framing suggested, still not actually comparable without a higher-λ run.
  - **NEW — unstructured+learned exemplar finally identified: LEAP** (Mozaffari, Hourri, Rastegari, Najibi — "Learnable End-to-End Adaptive Pruning of Large Language Models," arXiv 2605.17289, 2026). Per-weight Gumbel-Sigmoid-relaxed learned mask via gradient descent, base weights frozen ("only P is trained; W is held fixed") — the right shape for the unstructured+learned cell this project's related-work research had previously found empty. Tests Qwen-2.5-0.5B, Gemma-3-1B, LLaMA-3.2-1B/3B, LLaMA-3.1-8B, none overlapping our model set, and evaluates ppl at sequence length 4096 (vs. our sliding-window 2048/1024) — not comparable either, for the same two reasons SparseGPT and GISP each fail one of. Real citation now exists for this cell regardless; useful independent of any comparison: LEAP reports 22–276 GPU-hours (4×H100) to train its mask depending on model size (2000 steps, batch 256, seqlen 4096) — external evidence that trained-mask methods cost real compute generally, not a weakness specific to this project's approach. One LEAP number flagged, not yet independently verified: Qwen-2.5-0.5B's reported pruned ppl (11.89 @ 50%, 13.16 @ 60%) reads as *below* its own dense baseline (14.17) in the fetched summary — same shape as F19/F23's "pruned beats dense" pattern, not confirmed against the primary table directly.

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

## 7. 7B-scale validation (DONE, partial — Mistral-7B + Llama-2-7B, promoted out of "planned")

Real results now exist at 7B scale, not just the roadmap below. Three convergence-based sweeps ran (`train_pruner_mistral7b.py` on C4, `train_pruner_mistral7b_wikitext2.py` on WikiText-2, `train_pruner_llama2_7b_wikitext2.py` on WikiText-2), all reusing the identical Pruner architecture/hyperparameters (`embed_dim=64, lstm_hidden=128`) and training regime (block-mean convergence check + B9 plateau-triggered LR decay) as the GPT-2/OPT-125M sweeps — same "no per-model tuning beyond the weight-extraction plumbing" claim §4.2 already makes, now tested at 56x the parameter count.

**Headline: Llama-2-7B/WikiText-2 vs. DISP-LLM's own published numbers, matched model, matched dataset, matched no-weight-update setup.** DISP-LLM's Table 1 (LLaMA-2-7B, no weight update, dense=5.12 ppl) gives 20%→6.10, 30%→6.85, 40%→8.11, 50%→9.84 (total-param pruning ratio). Our 9-λ sweep (dense=4.903 ppl), converted to the same total-param basis via Llama-2-7B's 64.24% MLP parameter share, **beats the interpolated DISP-LLM curve at all 8 converged points** (5.4% to 28.8% total pruned) — by −0.45 ppl at the low end, narrowing to a near-tie (+0.016 ppl) right around 28.8%. This is the most direct, confound-free comparison in the paper: same base model, same eval dataset, same "frozen weights, no retraining" framing DISP-LLM itself uses for its own headline row. Full table in F6/`past-work.md`.

One point (λ=1.4, 32.6% total pruned) loses to DISP-LLM by +0.49 ppl, but never converged (18,000-step safety cap hit, `check_converged` never fired once) — excluded from the headline claim, diagnosed as a genuine training-dynamics failure (§7.1 below), not reported as a real comparison point.

**Mistral-7B** (Apache-licensed, used as the non-gated stand-in before Llama-2-7B access was arranged): C4 sweep shows the same free-region-then-monotonic-cost shape as every smaller model, plus a λ=0.3→0.4 non-monotonicity (more pruning, less cost) attributable to λ=0.4 simply getting far more optimization steps before its convergence trigger fired (4200 vs. 1150) — the same "window-validity depends on how much of the trajectory has actually completed" mechanism as F3, here appearing along the λ axis instead of the capacity axis. WikiText-2 sweep on Mistral-7B shows a much smaller absolute cost curve than C4's (dense ppl 4.740 vs. 8.325) — consistent with Mistral already being well-calibrated to clean Wikipedia text, less genuine redundancy to find there specifically, exactly the concern flagged before running it.

### 7.1 Open failure mode: λ=1.4 (Llama-2-7B) never converges, and pruner capacity is NOT the fix

Worth a paragraph in Discussion/Limitations, not just a footnote — it's a genuine, mechanistically-diagnosed failure, not a "ran out of budget" note. At λ=1.4, `pct_pruned` in the last 20 checkpoints (steps 14200-18000 of an 18,000-step run) still swings an 8.2-point band with barely reduced noise versus the *first* 20 checkpoints, and individual layers (`max_layer_delta_pct`) are still moving by ~30 points between 200-step checkpoints at the very end — a sustained oscillating equilibrium, not a slow monotonic approach to a plateau. Likely mechanism: at this λ the sparsity term and the CE-cost term are comparable in magnitude, and the learning rate is held at a constant `1e-3` regardless of λ (the B9 decay mechanism never gets to intervene, since it requires a raw convergence trigger to fire first, and none ever does here) — plausibly enough to sustain oscillation around a ridge rather than settle to a fixed point.

Directly tested "increase pruner capacity" as the fix against this project's own prior evidence (F3/F4: an 8-point matched grid at half/base/2.26x capacity, GPT-2+OPT-125M/pg19) and it doesn't hold — no monotonic capacity effect was found there on %pruned, ppl, or convergence speed. Capacity controls the row-encoder/BiLSTM's representational expressiveness; it has no obvious mechanistic connection to the STE gate's threshold-oscillation dynamics under a fixed-LR gradient signal. The indicated fix is LR/λ-coupling (scale the initial LR down for high-sparsity operating points) — an optimization-dynamics diagnosis, not a capacity one. Not yet tested at 7B scale; F3/F4's evidence is a reasonable prior from a much smaller model family, not a proven transfer.

### 7.2 Downstream zero-shot accuracy vs. DISP-LLM: the ppl advantage does not transfer (F7)

Ran the full lm-evaluation-harness downstream suite (PIQA/HellaSwag/WinoGrande/ARC-e/ARC-c/BoolQ/OpenBookQA, 0-shot) on all 8 converged Llama-2-7B/WikiText-2 checkpoints from §7 plus dense, via a RunPod H100 pod (`eval_downstream_llama2_7b.py` / `run_downstream_eval_pod.py`). This closes venue-gap-analysis item 1 (no downstream eval) — but the result itself is a genuine, unresolved divergence from §7's ppl headline, not a clean second win, and needs reporting as such.

Restricting to the 5-task subset DISP-LLM's own Table 3 reports (WinoGrande acc, HellaSwag/ARC-e/ARC-c/PIQA acc-norm — matching metric convention exactly) and comparing against DISP-LLM's own published LLaMA-2-7B numbers:

| % total params pruned | LEP avg acc (5-task) | DISP-LLM avg acc (interpolated) | LEP − DISP-LLM |
|---|---|---|---|
| 0% (dense) | 68.46 | 68.99 | −0.53 |
| 5.39% | 66.59 | 67.03 | −0.44 |
| 6.57% | 66.91 | 66.61 | +0.30 |
| 8.89% | 66.28 | 65.76 | +0.52 |
| 11.58% | 64.37 | 64.79 | −0.42 |
| 15.05% | 62.87 | 63.53 | −0.66 |
| 19.79% | 60.29 | 61.81 | −1.52 |
| 24.38% | 59.28 | 60.14 | −0.86 |
| 28.77% | 55.51 | 58.55 | **−3.04** |

(% total params pruned = each checkpoint's own training-time "final % FFN neurons pruned" — identical figures to §7's Table, not a fresh eval-time reconstruction; see reconstruction-discrepancy note below.)

Important caveat on the "interpolated" column: DISP-LLM's Table 3 (downstream) has only **2** real Llama-2-7B operating points (30%, 50%), unlike Table 1 (ppl)'s 5 points (0/20/30/40/50%) — so this interpolation is a straight line between their dense point and their single 30% point, much cruder than §7's ppl interpolation. Our 28.77% comparison point sits almost exactly at their measured 30% checkpoint, so a direct, interpolation-free check is available and more defensible: against DISP-LLM's own actually-measured 30% value (58.10) — at a slightly *more* aggressive ratio than our 28.77%, i.e. charitable to DISP-LLM — LEP is still behind by **2.59pp**.

**Reconstruction discrepancy (flag, not yet resolved):** the downstream eval reconstructs gates fresh from each checkpoint's saved pruner weights rather than reusing a stored mask. For 7 of 8 checkpoints this reproduces the training-time "final % FFN neurons pruned" to <0.05pp — but at λ=1.0 the reconstruction gives 44.23% pruned vs. 44.78% training-logged (0.55pp gap), even though the reconstructed checkpoint's perplexity matches the training log exactly (6.774 both). So the accuracy numbers above for λ=1.0 come from a mask that's very slightly *less* pruned than its own training-time self-report — the table's x-axis label uses the training-time figure for consistency with §7's table, not the reconstruction's. Mechanism unconfirmed: candidates are an eval-mode-vs-training-mode difference somewhere in the pruner forward pass, or a handful of gate values sitting right at the STE threshold flipping under a different numerical path. Small enough (0.55pp on one point) not to change this section's conclusion, but real, and worth checking before trusting the reconstruction mechanism (`load_pruner_and_gates` in `eval_downstream_llama2_7b.py`) at face value elsewhere.

Shape of the result: near-parity through ~9% total-param pruned (within ±0.5pp either direction, consistent with noise on a 7-task harness), then a widening deficit as pruning increases, reaching −3.04pp (interpolated) / −2.59pp (direct) at our most aggressive converged point. This is the *opposite* shape from §7's ppl result — there LEP starts ahead and the margin narrows toward high sparsity; here LEP starts at parity and the gap opens up toward high sparsity.

Mechanism not established. Plausible, untested candidate: ppl is a smooth, dense, next-token-averaged metric measured on text distributionally close to WikiText-2 — the same corpus the pruner was trained against — while the downstream suite is discrete, out-of-domain multiple-choice/QA, and may depend disproportionately on specific circuits that an in-domain-ppl-optimized mask doesn't protect. We have not tested this (e.g., checking whether the gap correlates with per-task distance from WikiText-2's domain) and it should be reported as an open question, not a conclusion. Report this finding plainly in the paper's discussion section — it directly undercuts a reader's temptation to read §7's ppl win as a general compression-quality win, and disclosing it ourselves is much better than a reviewer finding the mismatch first.

Full per-lambda per-task numbers: `experiments/latest/llama2_7b_downstream/summary.json`/`summary.txt`. Raw DISP-LLM Table 3 source: `docs/papers/disp_llm_gao2024.pdf`.

### 7.3 Case study: task-specialization via pruning (F19 → F23 → B5 → F25) — RESOLVED 2026-08-21, written up as §6.6 in `paper.tex`

**STATUS, 2026-08-25: B5's control ran (F25) and came back negative for the specialization claim, positive for a compression claim.** Written up in `paper/paper.tex` §6.6 ("Case study: pruning plus LoRA fine-tuning on CNN/DailyMail summarization") with the compression-only framing constraint #5 anticipated: dense+LoRA and pruned+LoRA reach statistically indistinguishable ROUGE-L (22.99 vs 22.93), dense wins on perplexity (2.774 vs 2.979), so the section's claim is "26.6% fewer params at iso ROUGE-L quality," not "pruning improves task performance." Abstract updated to match (previously said "even increasing performance," which predates F25 and is no longer accurate). All caveats from §7.3 below carried into the write-up (no CI on the point estimate, single seed/λ, non-converged pruner, unverified weak dense baseline, N=300→3000 sample-size correction).

**Framing call flagged, not silently made**: per this doc's own constraint #5, a negative B5 result was supposed to mean "cut back to a footnote," not a full subsection — I judged a full subsection was still warranted since a real matched-control experiment with caveats stated plainly is exactly the paper's established style elsewhere (§6.5, §7.1-7.2), but this is a judgment call on scope/prominence, not just a factual writeup, and should be confirmed rather than treated as final.

<details><summary>Original pre-F25 framing (kept for reference)</summary>

Reopened from a flat exclusion (§0.5 constraint #5, revised 2026-07-29) — this section exists to be written up *if and only if* B5's dense-fine-tune control confirms sparsity is doing something a plain fine-tune doesn't already give for free. Until then, treat everything here as a case study in progress, not a result to cite in the abstract/conclusion.

**STATUS check-in, 2026-07-31: still the single blocking item for this section.** B5's control (TODO list below) has not been run. Do not let this drift — it's the one experiment standing between "case study in progress" and either a real §7.3 or getting cut back to a footnote per constraint #5's original ruling.

</details>

**The pattern, two occurrences:**
- **F19** (OPT-125M, WikiText-2): pruned model showed lower in-domain CE than dense. Later found substantially attributable to a tokenization bug (F21, per-line BOS artifact roughly halving the reported baseline ppl) — not retracted entirely (a smaller, real-shaped effect survived the fix, per B8's partial re-check), but the original headline number doesn't hold as reported.
- **F23** (Llama-2-7B, CNN/DailyMail summarization): same training mechanism (target-only CE-delta objective, `train_pruner_llama2_7b_cnndailymail.py`), single run (λ=0.1, seed=0, non-converged). Pruned beats dense on held-out ppl (3.750→2.881) **and** on ROUGE-via-generation (rouge1 18.80→26.61, rouge2 7.60→10.05, rougeL 13.31→18.71). Not exposed to F19's tokenization confound — this is a cleaner data point on that axis, but a new one of its own: a non-instruction-tuned base model's dense ROUGE-L of 13.31 on a bare completion prompt is low next to typical fine-tuned CNN/DM numbers (30+), so the "improvement" could be adapting a weak, poorly-prompted dense baseline rather than anything pruning-specific.

**What would make this a real result, not two suggestive anecdotes:**
1. **B5's dense-fine-tune control** (the blocking experiment — see `ideas.md` B5): fine-tune the same frozen base model on the same domain/task data, no sparsity penalty at all, same compute budget. If dense fine-tuning matches or beats the pruned-model numbers, sparsity isn't the active ingredient and this section doesn't belong in the paper as a specialization claim — it collapses into "fine-tuning, dressed up," exactly B5's own standing caveat.
2. At least one more (λ, seed) or dataset point — a single non-converged run is not evidence of a stable operating point, just a first look.
3. A prompt/eval sanity check on the dense baseline specifically for the CNN/DailyMail case — ruling out "weak baseline" as the dominant explanation before attributing the gain to pruning.

**Framing discipline, if this section survives to a final draft**: state explicitly, high in the section, that this is compression **plus** specialization, not compression alone, and that it does not licence extrapolating the paper's main §4–7 "controlled accuracy-for-sparsity trade" story onto this case — the objective, the data regime (in-domain, not held-out-general), and the claim being made are all different from the rest of the paper. Keep it visibly separable, not blended into the headline compression narrative.

## 8. Scaling laws — [PLANNED, remainder still not run]

Frame as future work / a second paper section contingent on compute, not claimed results:

- **H1 — pruner-capacity scaling**: does the minimum pruner size needed to find a good mask scale with base-model size (row-encoder cost ~ `max(d_in)·embed_dim`, BiLSTM cost ~ `#layers·lstm_hidden²`)? Partially informed now by §7.1's negative capacity result at 7B — worth folding into H1's writeup as evidence, not just the GPT-2/OPT-125M-scale F3/F4 result. Every experiment so far still reused the same fixed pruner config regardless of base size — no genuine capacity *sweep* has been run at 7B.
- **H2/H3 — λ\* prediction**: can the Pareto-optimal λ be predicted from cheap properties of the base model/task (baseline CE scale, layer count) instead of swept? Toy-scale data (F15) already refutes any simple monotonic λ\*-vs-size relationship; a more structured hypothesis (λ\* ≈ k·(CE_orig/mean_layer_size)^α, or the "sequential vs. simultaneous commitment" dynamical-regime view, H3) is proposed but unproven.
- **Step-budget scaling**: F20 found the (inherited, never re-derived) fixed step count is badly and non-uniformly mismatched to actual convergence across λ and models — any scaling-law claim needs a convergence-based stopping rule first (B7), or the "how does X scale" question is confounded by "did training actually finish."
- **H4 — architecture universality**: the MLP→CIFAR→GPT-2→OPT-125M portability already demonstrated is evidence for this, but it's anecdotal (4 architectures, not a controlled sweep) — a real test would hold task/data fixed and sweep architecture family deliberately. **Mixture-of-Experts is the concrete next test case** — see the dedicated subsection below, the most fully-specified item in this section so far.

This section should be written as a roadmap with a clear "not yet run" label on everything, not blended with §6's completed results.

### 8.1 Extending to Mixture-of-Experts (MoE) — a concrete H4 test case [PLANNED, 2026-07-29, from ideation — no code written, no experiments run]

Why this matters for H4: MoE (Mixtral, DeepSeek-MoE, Qwen-MoE, etc.) is now the dominant architecture for the largest openly-available models — an "architecture universality" claim tested only on dense transformers doesn't say much about whether the method scales to where the field is actually heading.

**Two separable savings, only one of them is this method's contribution.** MoE routes each token through a top-`K` subset of `N` experts: `y = Σ_{i∈TopK(r(x))} g_i(x)·FFN_i(x)`, where `r(x)` is the router. This splits into two different kinds of pruning with two different provenance claims:

1. **Memory-only, zero FLOPs, NOT a contribution of this method.** An expert never selected by the router on the target distribution already contributes zero FLOPs — its weights are never multiplied against any activation. Dropping it saves storage only, and needs no output renormalization (it never held any routing mass to begin with). This is standard MoE deployment hygiene — using a capability MoE already has — and must be reported as such, kept explicitly separate from the paper's actual method.
2. **FLOPs + memory, this method's actual contribution.** Within each *activated* expert, the same learned neuron-level pruning already used for dense FFN blocks (§4) applies unchanged — an activated expert's FFN is structurally identical to a dense FFN block, just one of several running in parallel. Only activated experts' neurons ever contribute FLOPs, so this is where the real compute savings live.

**Procedure, staged:**

- **Stage 0 — measure before deciding anything (no training).** Run the frozen base MoE model forward once over the actual target pruning dataset, log per-expert selection frequency `p_i`. Tests directly whether the dataset is broad enough to exercise every expert non-negligibly, instead of assuming it. Identifies (a) genuinely-dead experts (Stage 1 candidates) and (b) whether Stage 2's training data is adequate before any compute is spent on it.
- **Stage 1 — drop dead experts (memory only).** Remove experts with `p_i ≈ 0` from Stage 0. No renormalization needed (see above). Not a research result — a preprocessing step.
- **Stage 2 — prune neurons within surviving experts (the method, ported, not reinvented).** Apply the existing row-encoder + BiLSTM procedure (§4.2, architecturally unchanged) to every activated expert's FFN neurons, with two adjustments required to stay consistent with the architecture's own permutation-invariance discipline (§4.3/4.4), not two new mechanisms:
  - **Sibling experts within one MoE layer are pooled, not sequenced.** Experts in the same layer run in parallel and are combined by the router — no canonical order between them, the same symmetry §4.3 already establishes for neurons within a layer. Feeding per-expert embeddings through the BiLSTM as a sequence would repeat exactly the mistake §4.4 rules out for neurons, one level up. Fix: mean-pool sibling experts' embeddings into one summary per MoE layer (identical construction to pooling neuron rows into one per-dense-layer embedding today), so the BiLSTM's sequence axis stays depth-only, unchanged.
  - **One row-encoder shared per layer across all its sibling experts**, not one per expert. The row-encoder is already a shared, order-agnostic function over weight rows (§4.3) — sharing it across siblings costs nothing architecturally, and directly helps the risk below: a rarely-selected expert's neurons still get scored by a function shaped by gradient from its more-frequently-selected siblings, partially transferring "what redundancy looks like" across experts instead of leaving a rare expert's scorer under-trained.

**Known open risks, not yet resolved:**
- Stage 0's measurement confirms coverage on the *chosen* dataset only — it doesn't guarantee that dataset matches whatever specialization the router actually learned across the full pretraining mixture. A rare-but-real specialist expert could still look safe to keep-fully-active (Stage 1) or under-scored (Stage 2) if the pruning dataset simply doesn't contain whatever that expert specializes in.
- **Per-expert gradient cadence threatens the existing convergence check.** A training step with zero tokens routed to expert `i` gives that expert's gates exactly zero gradient that step (not small — zero, the computation never happens). `check_converged`'s block-mean flatness check implicitly assumes roughly uniform per-step signal across every tracked unit; a flat gate-history could mean "converged" or "never touched this window" and the current check cannot distinguish the two. Deferred — not blocking Stage 0/1, but needs a step-aware version (gradient-received-count per expert, not raw step-count) before Stage 2's convergence claims can be trusted.
- No target model chosen, no code written, no experiments run — this entire subsection is a plan, not a result, same discipline as the rest of §8.

## 9. Discussion / Limitations

- **Explicit scope reminder**: this paper reports a compression method for frozen models — a controlled accuracy-for-sparsity trade, characterized via Pareto curves and an efficiency metric — not a fine-tuning technique and not a claim that pruning ever improves the base model. State this plainly, early in the discussion, not just in scope-setting.
- **The F19 thread as a cautionary tale, not a result**: early OPT-125M experiments appeared to show pruning *improving* in-domain perplexity. Worth one paragraph explaining what that turned out to be (substantially a tokenization bug, F21) and why, even setting the bug aside, an unfloored `(CE_pruned − CE_orig)` loss term will always be *capable* of producing this kind of result on a data distribution correlated between train/test (WikiText-2's own train/test split) — and why it doesn't generalize (the original out-of-domain C4 check showed real, monotonic degradation at every λ). This is useful precisely *because* it explains why the paper's scope constraint (§0.5) is the right one, not a limitation to apologize for.
- **Transfer does not work**: a pruner trained on one network does not transfer to a different (even architecturally identical) independently-trained network (F7) — each deployment needs its own ~20-25 min training run. Explain the mechanism from §4.3/§4.4's discussion: the architecture is already exactly permutation-invariant within a layer, so the failure isn't a missing invariance — it's that raw weight *values* aren't comparable across independently-trained networks' weight-space geometry. Note activation/gradient-based inputs (not yet tested) as the natural fix, without claiming it works.
- **Compute cost honesty**: ~20-25 min GPU per (λ, seed) operating point at GPT-2/OPT-125M scale, growing to hours at 7B scale, vs. one-shot calibration methods (SparseGPT/Wanda, minutes total) — the paper should state this tradeoff plainly rather than let the Pareto curve imply this is free.
- **The λ=1.4/Llama-2-7B non-convergence (§7.1, F6)**: one operating point in the 7B validation genuinely fails to converge, and the diagnosis (LR/λ-scale mismatch, not capacity) is a real open methods gap — the fixed, un-scaled learning rate used identically across every λ in every sweep so far is itself a load-bearing, never-revisited default. Worth stating plainly as unresolved rather than omitting the point or quietly excluding it without explanation.
- **Single seed at 7B scale**: every 7B-scale sweep (Mistral-7B ×2 datasets, Llama-2-7B) is single-seed, unlike the 2-seed protocol used for GPT-2/OPT-125M — a compute-driven necessity, not a validated claim that seed variance is small at this scale. State this explicitly wherever 7B numbers are reported.
- **Perplexity advantage over DISP-LLM does not transfer to downstream zero-shot accuracy (§7.2, F7)**: near-parity through ~9% total-param pruned, then a widening deficit reaching −3.04pp (interpolated) / −2.59pp (direct) at the most aggressive converged point — the opposite shape from the ppl comparison, where the margin narrows toward high sparsity instead of growing. Mechanism unconfirmed. This is the single most consequential caveat on the paper's headline claim and must be stated plainly in the abstract/discussion, not just in a results table — a reader who only sees §7's ppl table would draw a stronger conclusion than the data supports. Also flagged in §7.2: a fresh eval-time gate reconstruction doesn't exactly reproduce the training-time %-pruned figure at λ=1.0 (44.23% vs. 44.78% logged, 0.55pp gap, despite matching ppl exactly) — small, doesn't change the conclusion, but an unresolved reproducibility question in `load_pruner_and_gates` worth checking before trusting it elsewhere.

## 10. Conclusion (skeleton)

Restate: learned, weight-conditioned, depth-context-aware pruning of frozen models produces a clean, controllable sparsity/accuracy Pareto frontier that beats a standard activation-magnitude baseline by a wide margin at matched sparsity, ported without architecture-specific changes across GPT-2, OPT-125M, Mistral-7B, and Llama-2-7B — and, on the one directly matched comparison available in the literature (Llama-2-7B/WikiText-2 vs. DISP-LLM), beats a recent trained-hypernetwork baseline at every converged operating point. No claim of improved capability — the value proposition is a better compression method, characterized rigorously (protocol bugs found and fixed in public, transfer failure mode explained mechanistically, baseline comparison run fairly, failure modes at scale diagnosed rather than hidden).

---

## Open questions before this is actually finalized (need your call, not mine)

1. **§4.5**: floor the training loss at zero (`max(CE_pruned − CE_orig, 0)`) to match the "no improvement claim" framing structurally, or keep the current unfloored loss and handle the framing purely in how results are described? This affects whether existing checkpoints/sweeps (F16–F22) are reusable for the paper or need re-running.
2. **§6.3**: does the paper wait for B8's full OPT-125M reconciliation, or ship with the partial 4-λ result clearly caveated? B8 isn't run yet.
3. **§5**: is a qualitative toy-scale section even worth a full section, or should F1–F4/F11's LTH connection be compressed to a paragraph in the Introduction/Related Work instead, saving the section budget for §6/§7?
4. Target venue/length (workshop paper vs. full paper) — affects how much §8 (scaling laws, still 100% unrun) can realistically be more than a "future work" paragraph.

---

## TODO — before this is submittable

### Experiments to complete
- [ ] **B8**: full OPT-125M WikiText-2 re-sweep, corrected tokenizer, λ=0.01→1.8 full grid, 2 seeds — replace the current partial check (4λ, 1 seed, 8000 steps, no λ<0.2)
- [ ] **B7**: resolve step budget before B8 — convergence-based stopping vs. fixed 18,750; rerun OPT-125M λ=1.8 at 2× pruner capacity to isolate capacity vs. steps as the slow-convergence cause
- [ ] Fine λ grid 0.75→2.4 on GPT-2 reconciled sweep — resolve whether the λ=1.8 efficiency dip is real curvature or 2-seed noise
- [ ] Numeric comparison against ≥1 trained-mask baseline beyond activation-magnitude — L0 regularization (Louizos 2017) or CoFi (Xia 2022), both already in `docs/papers/`, at matched sparsity on GPT-2 and/or OPT-125M
- [ ] **§7.3 blocking experiment — B5's dense-fine-tune control** (2026-07-29): fine-tune frozen Llama-2-7B on CNN/DailyMail, no sparsity term, same compute budget as the λ=0.1 pruner run — does it match/beat the pruned model's ppl (2.881) and ROUGE (rouge1/2/L 26.61/10.05/18.71)? This is the single experiment gating whether §7.3 is real content or gets cut back down to a footnote.

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
- [x] **Prose draft of Abstract, Introduction, Discussion, Conclusion — DONE (2026-08-17)**: full prose paper written at `paper/paper.tex` (compiles clean, 14pp, natbib bibliography, both figures, all 5 tables, permutation-equivariance lemma+proof). Transcribed from `docs/learned_efficiency_pruning.md`'s content, not a fresh draft — that markdown source is now effectively superseded by `paper/paper.tex` as the canonical version; keep edits going into the `.tex` from here, not the `.md`.
- [x] §3 Related Work — DONE, same file: L0/CoFi/FLAP/LLM-Pruner/DISP-LLM/SDS/SlimLLM/GISP/Network Slimming/MetaPruning all integrated with explicit differentiation (not placeholder pointers).
- [ ] Reproducibility / hyperparameter appendix, drawn from engineering_decisions.md's hack list — NOT yet in `paper.tex`, still open.
- [x] Working title resolved: "Learned Efficiency Pruning" (used as-is in `paper.tex`, no bracketed alternative).

### Decisions needed (yours, not research)
- [ ] §5 scope — full section vs. paragraph-in-intro
- [ ] Target venue/length — governs how much of §8 is real vs. future work
- [ ] §6.3 — ship with caveated partial OPT-125M result, or hold for B8
- [x] §7 — 7B-scale validation promoted from "planned" to real content (2026-07-25): Mistral-7B (C4 + WikiText-2) and Llama-2-7B (WikiText-2) sweeps done, Llama-2-7B vs. DISP-LLM headline comparison in, λ=1.4 failure diagnosed (F6)
- [x] **Constraint #5 — task-specialization thread (F19/B5) reopened as §7.3, conditionally-scoped case study, gated on B5's dense-fine-tune control (2026-07-29)**: was a flat exclusion since 2026-07-14; reopened given F23's second, stronger data point (CNN/DailyMail ROUGE, not just in-domain ppl). NOT yet promoted to a real result — B5's control is unrun and is the explicit blocking item (see TODO above).

---

## Venue gap analysis (2026-07-25) — what's blocking A+, prioritized

Full reasoning behind this list, and the tier verdict, lives in the 2026-07-25 chat log. Verdict: **not A+ (NeurIPS/ICML/ICLR main track) as it stands.** Realistic right now with light polish: a strong workshop paper at a top venue (efficiency/sparsity workshop). Realistic with items 1-3 closed: a second-tier real venue (EMNLP/NAACL Findings, COLM, mid-tier ML conference). Items 1-5 below are the ones that would draw an immediate reject/major-revision at a top main track; 6-10 are "why didn't you do this" gaps, real but secondary.

### Blocking, in priority order

1. [x] **No downstream task evaluation.** DONE (2026-07-28) — ran the full lm-eval-harness suite on all 8 converged Llama-2-7B/WikiText-2 checkpoints + dense (§7.2, F7). Closes the gap, but the result itself opens a new one: downstream accuracy vs. DISP-LLM diverges from the ppl win (near-parity at light pruning, growing deficit at heavy pruning, opposite shape from the ppl comparison) — this must be stated plainly in the paper, not just checked off. See the new §9 bullet.
2. [ ] **No measured wall-clock/memory speedup.** The paper's own stated value proposition ("structured sparsity is what actually saves wall-clock/memory on commodity hardware") is asserted, never measured. Need actual latency/memory numbers, before vs. after pruning, on real hardware, at least at one operating point per model.
3. [ ] **DISP-LLM comparison (§6.3/Table 4) has an unaddressed eval-protocol confound.** We compare our sliding-window ppl against DISP-LLM's own eval script's numbers via linear interpolation, not a re-run under identical conditions — and this project already found a real ~4% gap between sliding-window and non-overlapping-chunk eval protocols earlier this session (the SlimLLM/LLM-Pruner-vs-SparseGPT/DISP-LLM investigation). Either (a) re-eval our dense/pruned Llama-2-7B checkpoints under non-overlapping-chunk eval (`stride = max_length`, no overlap) and report both numbers, or (b) explicitly quantify and state the confound's likely magnitude in §6.3/§7. Currently the paper's text doesn't mention this at all for the headline comparison — a reviewer who knows DISP-LLM's codebase will find it before we disclose it, which is worse than disclosing it ourselves.
4. [ ] **λ=1.4 is diagnosed, not fixed.** Try the LR/λ-coupling fix §6.4 proposes (lower or λ-scaled initial LR for high-λ operating points) and see if it actually converges. If it does, replace the broken point with a real one — a working fix is much stronger than an honest diagnosis of a still-broken headline-scale failure. If it doesn't, that's itself an important negative result worth reporting explicitly rather than leaving the mechanism as a guess.
5. [ ] **Single seed almost everywhere that matters** (all of §6/7B-scale, OPT-125M's headline sweep) — violates this project's own house rule (F8: "single-seed numbers are noise at this kind of task"). At minimum, 2-seed re-runs of the Llama-2-7B operating points nearest the DISP-LLM crossover (λ=0.8, 1.0) would let the headline claim survive a "is this just seed luck" question, which is exactly the kind of question a top reviewer asks first.
6. [ ] **NEW, 2026-07-31 — no rigorous training-cost characterization (wall-clock + GPU-hours to PRODUCE a pruner), distinct from item 2's inference-speedup gap.** Item 2 asks "is the pruned model fast" (a deployment-time question); this asks "what does it cost to get there" (a training-time question) — currently only a scattered handful of numbers (engineering_decisions.md's per-experiment notes, the CNN/DailyMail run's ~3.0 GPU-hr/~\$9 breakdown) and one Discussion bullet asserting the tradeoff in prose without a table (§9, "Compute cost honesty"). Now genuinely comparable to something external: LEAP (F24, arXiv 2605.17289) reports 22–276 GPU-hours on 4×H100 depending on model size for its own trained-mask procedure — a real number from a competing learned-mask method, not just our own internal figures. **What's needed**: one consolidated table, GPU-hours + wall-clock + \$ (at a stated \$/hr) per (model, λ) operating point, across every experiment family that's actually been run (toy-scale, GPT-2, OPT-125M, Mistral-7B ×2, Llama-2-7B ×3 [WikiText-2/HellaSwag/CNN-DailyMail]) — not estimates, pulled from each run's own logged timing. This is the evidence the "compute-for-quality tradeoff" claim in §3/§9 currently asserts but never shows.

### Secondary, real but not blocking

7. [ ] Baseline comparison is thin — one self-implemented baseline (activation-magnitude) + one literature comparison (DISP-LLM). GISP and SparseGPT were checked (F24) and don't qualify for a fair comparison (protocol/model-family mismatches respectively) — a real re-run at matched sparsity on the same model(s) is still the only way to close this, a citation isn't enough.
8. [ ] No LLM-scale ablation isolating the BiLSTM cross-layer context's actual contribution (only argued qualitatively via §5.4's "flatter across depth" note, and only directly demonstrated at toy scale, which the paper's own scope rules out as quantitative evidence). A row-encoder-only-vs-full-pruner ablation at one OPT-125M λ would close this cheaply.
9. [ ] §6.3 (OPT-125M) still mid-reconciliation by this project's own stated bar (B8 not run — see "Experiments to complete" above).
10. [ ] Loss-floor question (§3.5/§4.5) still open, sitting inside a paper whose framing argument depends on it.
11. [ ] No code release / anonymized repo link — close to mandatory for NeurIPS/ICML/ICLR now.
