# Engineering Decisions & Hacks

Crisp reference of every stabilisation hack, normalisation, and design decision in the codebase. Format: **what — why**. Sister to [crisp-findings.md](crisp-findings.md).

---

## SparseGPT baseline (`past_work/sparsegpt/`, run 2026-07-04)

Faithful SparseGPT (OBS per-layer `min‖WX−ŴX‖²`, Cholesky-shared H⁻¹) on the CIFAR_big FC head {fc1,fc2,fc3}, vs SAVED LEP numbers (F13) — no LEP retrained.

- **Comparison axis = FC-head WEIGHT sparsity — why:** LEP is structured (neuron), SparseGPT unstructured (weight); only common currency. LEP 70.9% neurons → 84.0% weight-sparsity (quadratic hidden→hidden, per F11).
- **device=CPU — why:** fc1 Hessian is 8192×8192; MPS linalg (cholesky/inverse) incomplete → CPU is the reliable path.
- **n_calib=10000 — why:** H=2XXᵀ for fc1 needs N ≳ d_col=8192 for full rank; each image = 1 activation vector. fc1 sits at the rank boundary → its compensation is the least over-determined part.
- **uniform per-layer sparsity — flag:** LEP's own allocation is non-uniform (fc1 83/fc2 94/fc3 88%); uniform-per-layer used for SparseGPT, not hand-matched.
- **Result:** iso-84%: SparseGPT −0.15pp vs LEP −1.48pp; SparseGPT ≤0.3pp to 90%. **NOT "beats LEP" — it's the structured/unstructured gap quantified (~1.3pp).** SparseGPT = easier objective (few weights reconstruct WX, dense-shaped output, no FLOP saving); LEP removes whole neurons (rank-1, kills directions) → genuinely smaller dense net.

## ISO-FLOP: structured OBC vs LEP (`run_isoflop.py`, `structured.py`, 2026-07-04)

Structured OBC (structured sibling of SparseGPT; whole-neuron OBS, exact greedy) at LEP's exact per-layer keeps (fc1 171/fc2 177/fc3 92) → identical architecture → identical FLOPs. Prune fcN columns (=upstream neurons), compensate survivors, zero upstream dead rows. Hessians on post-ReLU inputs, dense-model calibration (independent-layer).

- **iso-FLOP == iso per-layer neuron count** (FLOPs of a linear layer = kept_out×kept_in). Unstructured gives NO dense-HW FLOP reduction → iso-FLOP must be structured-vs-structured.
- **Result:** iso-16%-FLOP: structured-OBC −0.11pp vs LEP −1.48pp. **Predicted OBC would LOSE — it WON by ~1.4pp. Prediction wrong.**
- **DOMINANT CONFOUND — flag:** OBC = select + OBS weight REFIT; LEP(F13) = select-only on FROZEN weights. OBC's win largely = the least-squares refit LEP forgoes, NOT proven-better selection. Clean isolation needs: no-compensation ablation (select-only OBC) and/or LEP + LS refit. NOT run.
- **Caveats:** LEP saved numbers inconsistent (70.9% headline vs 75.4% from per-layer keeps — matched keeps, so OBC ran harder budget); fc1 has ~230 live neurons of 1024 (ReLU-dead) → f≥1.5 plateaus, iso f=1.0 clean.

---

## BiLSTM pruner — stabilisation (taming aggressive one-shot pruning)
`src/pruners/bilstm.py`, `src/pruners/mlp.py`

- **+2.0 bias init** on final gate layer — gates start open (σ(2)≈0.88); stops sparsity loss collapsing all gates before task loss reacts.
- **tanh-bounded cross-layer context** (∈ −1..1) — context can modulate but never override per-node logit (+2.0); prevents runaway collapse.
- **LayerNorm on BiLSTM output** — bidirectional output is 2× wider; without it Adam pushes context down 2× too fast → gate collapse.
- **zero-init context_head** (W=0, b=0) — context starts at 0 → step-0 behaviour = plain MLP pruner (neutral start).
- **grad clip max_norm=1.0** — LSTM pruners explode without it.
- **LSTM over LAYER sequence (2–4 steps), not neurons (1024+)** — avoids vanishing/exploding from thousands of unroll steps.
- *Note:* these 4 init/norm hacks exist ONLY to tame aggressive one-shot; a curriculum/sw-anneal may let us drop them (see claude-notes §9).

## Gating / discretisation
- **Straight-Through Estimator** (`binary_ste`): hard 0/1 threshold fwd, sigmoid grad bwd — truly binary inference-ready masks, non-zero grad everywhere. `hard - soft.detach() + soft`.

## STE-top-K (B1) — failure modes & decisions  `src/pruners/bilstm_topk.py`, `src/topk_train.py`
- **Global top-K STE**: `hard = (s >= kth_largest)`, STE backward. Sets K directly → no λ, no sw sweep, collapse-to-zero structurally impossible. BUT brittle (see F12 / claude-notes §14).
- **Plain σ(s) saturates under top-K** — gradient peak at 0, but boundary = K-th score → borderline neurons get ~0 grad. Use **centered STE** `σ((s−thresh)/T)` (gradient peak ON the boundary). Under λ this was free (boundary was 0 = σ's center).
- **Temperature T**: width of the active gradient band. ANNEAL **globally 4→1 across the whole curriculum**, NOT per-stage (per-stage reset to 4 re-blasts a committed ranking → collapse).
- **Node-score normalization**: per-layer. `std` (backprop through) warm-starts better but blows up at low variance; `std_detach` (detached mean/std) is stabler but warm-starts worse. Neither fixes the collapse alone.
- **PROVEN collapse = LAYER-STARVATION** (collapse_probe, per-layer survivors): global top-K drains one layer to 0 (L1 819→512→205→0; L2 pinned 1024) → first layer severed → one-class 88.32pp (bit-identical, absorbing). Per-layer standardization centers both layers at 0/unit-var, so cross-layer allocation is governed ENTIRELY by the per-layer **context bias** — and dropping the **tanh** left it UNBOUNDED → it diverges (L2≫L1) → starves L1. **The tanh was NOT just λ-babysitting; under global top-K it bounds cross-layer allocation.**
- **Tested & FAILED to fix** (all hit the same 50% wall, none touch the allocation knob): global T-anneal, std vs std_detach, carry-Adam-state. NOT a kick (no grad spike) — gradual erosion.
- **The real fix would be**: per-layer keep-floor / per-layer top-K, OR bound/normalise the context allocation (re-add tanh) — but that gives up "pruner freely allocates across layers". B1 declared NEGATIVE instead (λ wins; top-K trades a sweep for a structural collapse).
- **Dropped vs λ pruner**: +2.0 bias, tanh context bound — BUT note tanh was load-bearing for allocation under global top-K (above). KEEP: LayerNorm on context, zero-init context head, grad-clip 1.0 (anti-explosion).

## Pruner training (loss / optimisation)
`src/prune_train.py`

- **CE difference** `(CE_pruned − CE_orig)`, not raw CE — accuracy-drop proxy centred at 0.
- **base model detached** in masked_forward — grads flow only to pruner; base frozen.
- **sparsity term** `sw · mean(gate)` — the single accuracy↔sparsity knob.
- **gate applied per-row**: `W[j]*=gate_j, b[j]*=gate_j` — structured (whole-neuron) pruning.

### Design duality: sparsity-weight λ ↔ top-K (same frontier, different control)
- **λ-penalty** `loss + λ·#kept` (what we use): differentiable, sparsity *emerges*, indirect control. **Top-K** `s.t. #kept≤K`: exact/direct control, non-differentiable (needs soft-topk/STE), enforced every step.
- **Duals**: per K there's ≈a λ with the same solution (non-convex → approximate). Sweeping λ ≈ sweeping K; annealing λ(0→hi) ≈ annealing K(N→target) — both continuation curricula. Same family as "knee-slope τ ↔ λ" (§10 stagnation MDP).
- **λ→K mapping is model-dependent**: to hit ~2pp, deep needed λ=0.8 vs wide λ=0.1 (8×).
- **Tested**: iso-accuracy threshold-sweep = top-K on a λ-trained ranking; matched retrain at the trained operating point but ~1–2pp PESSIMISTIC elsewhere → duals at the optimum, but a fixed-λ ranking isn't calibrated for arbitrary K.

## Normalisations (all of them)
- **Per-neuron feature standardisation** (`src/rl/env.py`): 5 raw features (in/out L1/L2 norms, mean act) → zero-mean/unit-var across alive set each step; norms span 1–2 orders of magnitude. `std.clamp_min(1e-6)`.
- **LayerNorm** on BiLSTM context (see above).
- **Advantage standardisation** (PPO): per-rollout zero-mean/unit-var advantages.
- **Entropy normalisation** — TESTED & REJECTED: dividing entropy by log(N_alive) regressed RL (4.71→6.40pp); the raw scales-with-N bonus is load-bearing exploration.
- **`W.mean(dim=0)` layer pooling** — collapse out-neuron axis → fixed `[in_features]` per-layer LSTM token (perm-invariant). Crude (signed mean cancels); abs/std would carry more.
- **`clamp_min(1e-12)`** on all log/prob — guards log(0).

## RL environment design
`src/rl/env.py`

- **Telescoping reward** `r_t = acc_t − acc_{t−1}` → return = acc_final − acc_orig. (Also the design FLAW: path-independent on frozen net → MDP degenerate, see claude-notes §10.)
- **prune_chunk=16** — load-bearing scaffolding: caps neurons/step, keeps episodes long. Removing it (Bernoulli) → horizon collapse.
- **recalibrate_every=5** — recompute activations every 5 steps (compute vs staleness; minor non-Markovianness).
- **outgoing norms over ALIVE downstream only** — dead downstream zeroed before fan-out features.
- **state**: per-neuron [in_l1,in_l2,out_l1,out_l2,mean_act, layer-1hot, frac_pruned_layer] + global [ce_gap, frac_pruned, cur_acc].

## RL policy / value architecture (permutation symmetry)
- **PolicyNet = single-query key·query attention** (`rl/rl_policy.py`) — perm-EQUIVARIANT per-neuron scoring over variable-length set.
- **ValueNet = mean-pool over alive set** (`rl/rl_value.py`) — perm-INVARIANT V(s) (Deep Sets).
- **Bernoulli policy** (`rl/rl_bernoulli_policy.py`): init_bias=−4.8 (≈16 expected prunes/step), zero-init query head (uniform start), force-prune argmax if 0 sampled (progress), greedy = top round(Σp).

## RL training knobs
- **REINFORCE**: EMA baseline (decay 0.95), entropy_coef 0.01, γ=1.0, grad clip 1.0. (Fails late: ep-200 collapse from baseline lag.)
- **Actor-critic**: learned V(s) + reward-to-go advantages, value_coef 0.5, grad clip 1.0. (Best RL: 4.71±1.55.)
- **PPO**: clip 0.2, GAE λ=0.95, 4 epochs/rollout, value_coef 0.5, lr 3e-4. (Failed at default entropy sizing.)
- **lr**: 1e-3 REINFORCE/AC, 3e-4 PPO.

## GPT-2 pruner — eval metric choice
`scripts/hypernetwork/train/train_pruner_gpt2.py` (canonical; the old `llm/train_pruner_gpt2.py` one-λ-per-process copy and `scripts/runpod/launch_sweep.py` orchestrator were deleted 2026-07 as stale duplicates once checkpoint saving was merged into this one)

- **Final eval uses perplexity (exp(mean CE)), not raw CE** — autonomously chosen (not discussed with user; flagged for confirmation). Rationale: PPL is the standard LM eval metric in the literature (GPT-2 paper reports PPL on WikiText-2/PTB), making results directly comparable to published baselines. Raw CE is equally informative for the pruned-vs-original delta, and simpler. **Training signal is raw CE throughout** — only the reported eval number is PPL. User should decide whether to keep PPL, switch to CE, or report both.

- **Default sweep changed to `--seeds 0 1` (2, not 3) and `--steps 18750` (up from 12500), 2026-07-08, user-confirmed.** Reasoning: [[crisp-findings]] F2 + claude-notes.md 5-seed reruns establish BiLSTM near-determinism (dead/alive ratio σ=0.007) on MNIST/CIFAR-scale layers, with "2 seeds ample" as the established convention there — extrapolated to GPT-2's wider (3072-neuron) MLP layers but **not yet independently confirmed at that width**. The 3rd seed's compute (1/3 of the original 6λ×3×12500=225,000-step budget) was reinvested into steps/run rather than banked, since GPT-2's convergence step count is unmeasured (the only diary saturation evidence — gates commit by step ~700 — is for a 3-layer MNIST net, not comparable). 6λ×2×18,750 = 225,000 steps, same total cost (~8–16 GPU-hr / ~$5.50–$11 on a 4090) as the original plan. Open question: does GPT-2 actually saturate before 18,750 steps? Unverified — check `per_layer_keep` in the per-run `plot.png` for a plateau once the sweep runs.

- **bf16 autocast added around the frozen GPT-2 forward passes, 2026-07-08, user-confirmed.** `autocast_ctx()` wraps both forwards in `pruner_step`/`evaluate` in `torch.autocast(dtype=torch.bfloat16)` on CUDA (no-op on CPU/MPS). Motivation: the script was running pure fp32 with no Tensor Core engagement at all — a bigger, free lever than GPU tier choice (roughly 2x throughput on Ada/Blackwell, $0/hr extra, vs. paying +43% $/hr for a 5090's +27% compute). Pruner itself stays fp32 (negligible compute share, avoids touching the STE gate-threshold numerics). Still unmeasured — `--timing_probe` gives the real number; this workload's small batch (8×512) may be bandwidth/overhead-bound rather than compute-bound, so the realized speedup could differ from the naive 2x.

- **GPU: RTX 4090 chosen for the first `--timing_probe` run over RTX 5090, 2026-07-08.** RunPod pricing: 4090 $0.69/hr vs 5090 $0.99/hr (+43%); 5090 BF16 dense 209.5 TFLOPS vs 4090's 165.2 TFLOPS (+27% compute), +78% memory bandwidth (1792 vs ~1008 GB/s). Cost-per-FLOP is ~13% worse on the 5090, so it trades money for wall-clock time rather than being a strict win — reasonable if bandwidth-bound (plausible at this batch size) makes the realized gain beat the FLOPS ratio, but unverified. Decision: get the bf16-enabled timing number on the cheaper 4090 first; only pay more for the 5090 if that number is still too slow. **Superseded same day** — 4090 hit "does not have the resources to deploy" (RunPod capacity, not our issue) on Secure Cloud; moved to 5090 instead (needs the `torch280`/`cu1281` template — PyTorch 2.4.0 does NOT support Blackwell/sm_120, confirmed via pytorch/pytorch#159207). Measured: 22.2 min/run on 5090 with bf16 → ~4.45hr / ~$4.40-5 for the full 6λ×2seed sweep. Approved, proceeding.

- **`Salesforce/wikitext` fix (not `wikitext`), 2026-07-09.** `get_loaders()` called `load_dataset("wikitext", "wikitext-2-raw-v1")` — the bare unnamespaced alias, since retired by HF (`huggingface_hub.errors.HfUriError: Repository id must be 'namespace/name'`). This was a reconciliation miss: the two original scripts disagreed on this (train/ used the bare name, the deleted llm/ copy used `Salesforce/wikitext`) and I kept the wrong one without checking. Fixed, verified locally (`load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')` → 4358 examples).

- **`os.sync()` added before `stop_pod()`'s actual stop call, 2026-07-09.** Belt-and-suspenders — flushes any buffered writes to the Volume Disk before the container halts. `stop_pod()` was already only called after all per-run and aggregate saves complete, so this doesn't fix a known gap, just removes a theoretical one.

- **Known gap, not fixed (flagged, not asked to fix): no resume/skip-if-completed logic.** If the sweep is interrupted and rerun with the same command, it restarts from λ #1 and overwrites already-completed `run_dir`s with fresh (statistically similar, not identical) results. Not a risk under normal uninterrupted operation.

## Evaluation / interpretability
`src/interpretability.py`

- **`evaluate_with_gates`**: deepcopy model, zero masked rows+biases, eval FULL test set — the honest metric (vs training-minibatch acc, which overstated by ~1.5pp).
- **Activations measured on ORIGINAL un-pruned model** — else dead neurons trivially 0; makes dead/alive ratio meaningful.
- **`param_count = Σ consecutive(in×out)`** = MLP inference MACs → weight% == compute%.
- **n_calib_batches=5** for activation stats.

## Methodology / infra
- **Seeds: ≥3 for RL** (σ≈4pp, single-seed meaningless), **2 for BiLSTM** (near-deterministic, ratio σ=0.007).
- **Threshold-sweep vs retrain-per-sw**: re-thresholding one pruner's ranking is ~1–2pp PESSIMISTIC away from its training sw → retrain for rigor.
- **Per-model sw grids bracket 2pp**: narrow needs tiny sw (~0.06), deep needs large (~0.8) — sw to hit 2pp scales with depth.
- **YAML: write `0.001` NOT `1e-3`** — YAML parses `1e-3` as a string.
- **Always run from project root**, `venv/bin/python scripts/<family>/<name>.py`, scripts use `sys.path.append(".")` + relative output paths.
- **Plots**: embed in chat via Read (no Preview); save to `experiments/latest/<family>/.../plot.png`.

## Base model / standard config
`configs/config.yaml`

- MNIST MLP 784→1024→1024→10, dropout 0.1, Adam lr 0.001, 10 epochs, batch 128.
- Pruner: 1000 steps, 64 samples/step, lr 1e-3. sparsity_weight 0.05 (sweeps) / 0.3 (config) / 0.5 (shape studies).
- Shape-study checkpoints: `mnist_model.pt` [1024,1024], `mnist_wide2048.pt` [2048], `mnist_deep4x512.pt` [512×4], `mnist_narrow205x2.pt` [205,205].
