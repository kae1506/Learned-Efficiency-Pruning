"""
GISP (Global Iterative Structured Pruning, Wang et al. 2025, "From Local to
Global: Revisiting Structured Pruning Paradigms for LLMs") reimplemented
FAITHFULLY (attention heads + MLP channels jointly, matching their published
scope -- not restricted to FFN-only like our own method) on Llama-2-7B,
evaluated under THIS PROJECT'S OWN verified WikiText-2 protocol.

WHY THIS SCRIPT EXISTS -- resolving F24 (diary/crisp-findings.md), not a new
research direction. Two blockers stopped a real GISP-vs-LEP comparison:
  1. Their reported Llama2-7B dense ppl (12.19) is ~2.5x ours (4.903) and
     DISP-LLM's (5.12) on the "same" model/dataset -- eval protocol never
     stated in their paper, unexplained.
  2. Their "pruning ratio" denominator is never defined -- unclear if it's
     against the WHOLE model's params (DISP-LLM's convention, verified) or
     just their own attention+MLP prunable pool.
This script controls BOTH: it reuses train_pruner_llama2_7b_wikitext2.py's
`evaluate()` and `load_llama2_7b()` VERBATIM (imported, not reimplemented --
identical sliding-window ppl protocol, eval_max_length=2048/stride=1024, the
exact numbers already reported for our own sweep and for DISP-LLM), and it
computes its own global-%-of-TOTAL-model-params figure directly from the
loaded model's real dimensions (no hardcoded conversion factor) so the
ratio-denominator question has a verified answer this time, regardless of
what GISP's own paper meant.

ALGORITHM -- Algorithm 1 from the paper, pulled directly from its pseudocode
(not inferred, not a paraphrase):
    Require: θ0, target ratio ρ, iteration steps n, calibration data D
    {ρ_t}_{t=1}^n <- linear ratio schedule, 0 -> ρ, same # structures pruned
                     per step (their own words: "a linear scheduler that
                     gradually increases the pruning ratio across
                     iterations, ensuring that each iteration prunes the
                     same number of structures")
    for k = 1..n:
        I(θ_{k-1}) = |dL(D)/dθ_{k-1} * θ_{k-1}|      # first-order Taylor/
                                                        # OBD-style saliency,
                                                        # elementwise
        I(θ_{k-1}) = sum over each structure's elements (one attention
                     head, or one MLP channel)
        I(θ_{k-1}) = I(θ_{k-1}) / |θ_{k-1}|           # per structure --
                     mean elementwise saliency, NOT raw sum. This is the
                     step that reconciles head-vs-channel group-size
                     differences (a head has ~2.1M elements at this
                     model's dims, a channel has 12,288) -- read literally
                     from Algorithm 1 line 5, not as a separate z-score
                     step layered on top. Section 3.1's prose ("normalize
                     importance scores within attention and MLP blocks
                     separately") is interpreted as describing exactly
                     this per-structure-type mean-normalization, not an
                     additional mechanism -- flagging this reading
                     explicitly since the paper doesn't give a formula
                     beyond Algorithm 1's pseudocode for it.
        τ_k = TOPK threshold hitting cumulative ratio ρ_k, ranked GLOBALLY
              across every structure in the model (heads and channels
              together, all layers together -- not per-layer, that's the
              whole "global" in the method's name)
        m = 1[I(θ_{k-1}) > τ_k]                        # binary keep-mask
        θ_k = θ_{k-1} * m                               # zero pruned
                                                          # structures --
              literal weight zeroing, no STE, no gate, no learned
              parameters anywhere in this script. Re-scored fresh every
              iteration on the partially-pruned model (θ_k feeds into the
              next iteration's importance computation), which is why this
              is "iterative" not "one-shot" -- captures cascading effects
              of earlier pruning decisions on later ones.

SCOPE, confirmed with the user before writing this (not a default) --
faithful GISP: attention heads AND MLP channels, matching the paper. An
MLP-channels-only restricted mode (same prunable set as our own method,
isolating the scoring-ALGORITHM comparison specifically) was considered and
explicitly NOT built -- this script reproduces GISP as published, full stop.

STRUCTURAL UNITS, Llama-2-7B (standard MHA, NOT GQA -- num_key_value_heads
== num_attention_heads == 32 in this model's config, verified via
AutoConfig, unlike Llama-3/Mistral which use GQA):
  - MLP channel i (0..11007), per layer: gate_proj row i, up_proj row i,
    down_proj column i. 3*4096 = 12,288 elements/channel.
  - Attention head h (0..31), per layer: Q/K/V weight ROWS
    [h*128:(h+1)*128, :] (head_dim=128) + O weight COLUMNS
    [:, h*128:(h+1)*128]. 4*128*4096 = 2,097,152 elements/head.
Neither touches embeddings, lm_head, or LayerNorms -- matches the paper
("removes attention heads and MLP channels"), and matches this project's
own DISP-LLM-comparison convention of a verified, model-config-derived
global-%-of-total-params denominator (computed live below, not hardcoded).

CALIBRATION DATA -- C4, matching GISP's own stated convention for the
perplexity-objective variant (the one Table 4/5 report, which is what we're
comparing against -- NOT their margin-based CMQA-task-specific variant,
not implemented here). Sample count/seq_len (128 sequences x 2048 tokens)
is INFERRED from SparseGPT's convention (which GISP cites and which this
whole line of one-shot/iterative calibration-based pruning work generally
follows) -- GISP's own exact calibration-set size for the PPL objective
specifically was not found in the extracted paper text (Appendix A.2/A.3
cover the CMQA/GSM8K calibration sizes in detail but not this one
explicitly). Flagged, not verified -- --calib_samples/--calib_seq_len are
exposed so this can be corrected if the real number surfaces.

RATIO-PER-ITERATION -- 0.625 percentage points/iteration for 7-8B-scale
models, taken directly from their Table 11 hyperparameter listing
("Pruning Ratio/Iter: 0.625%"). n_iters is DERIVED from this (target_ratio
/ 0.00625), not hardcoded to their stated "112" (which appears to correspond
to a ~70% target in some ablation, not the 20-50% range Table 4/5 report --
deriving from the stated per-iteration rate is more faithful than copying a
single iteration count that doesn't obviously match our target ratios).

COMPUTE COST, flagged explicitly -- this is NOT cheap the way SparseGPT/
Wanda are. Every iteration requires a fresh forward+backward pass over the
full calibration set (gradients w.r.t. every attention/MLP weight in the
model, ~6.48B parameters excluding embeddings) to recompute importance on
the newly-pruned θ_{k-1}. At a plausible target ratio (~25-30%, matching our
own sweep's tested range) and n derived from 0.625pp/iter, that's on the
order of 40-50 full calibration passes with backward, EACH one comparable in
cost to a single training step's forward+backward on the calibration batch
size chosen. Memory: enabling requires_grad on all attn+MLP weights
(~6.48B params, excluding the ~260M in embeddings) needs gradient buffers
on top of the frozen bf16 weights -- budget accordingly (single 80GB H100
should be fine at reasonable calib_samples, not verified empirically since
this script has not been run).

*** UNTESTED. *** Never run, no local 7B hardware, same caveat as every
sibling script. Verified: py_compile only. Requires HF_TOKEN with accepted
Llama-2-7b-hf access, same as every sibling script.
"""
import os
import sys
import json
import argparse
import contextlib

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_pruner_llama2_7b_wikitext2 import (
    load_llama2_7b, evaluate, get_loaders, autocast_ctx,
    LLAMA2_REPO, N_LAYERS, N_INTER, HIDDEN,
)

os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")

OUT_ROOT = "/workspace/results/gisp_reimpl_llama2_7b"

N_HEADS = 32
HEAD_DIM = 128          # hidden / n_heads = 4096 / 32, verified against AutoConfig
assert HEAD_DIM * N_HEADS == HIDDEN

C4_REPO = "allenai/c4"
C4_CONFIG = "en"


# ─────────────────────────────────────────────────────────────────────────────
# Calibration data -- C4, matching GISP's stated convention (see docstring
# for the inferred-not-verified sample count/seq_len caveat).
# ─────────────────────────────────────────────────────────────────────────────

def get_calibration_batches(tokenizer, n_samples=128, seq_len=2048, seed=0):
    """Random contiguous seq_len-token windows from C4 train, matching the
    SparseGPT-style calibration convention GISP's own related work cites.
    Streamed (C4 is far too large to load fully) -- draws until n_samples
    windows are collected."""
    from datasets import load_dataset
    import random

    rng = random.Random(seed)
    raw = load_dataset(C4_REPO, C4_CONFIG, split="train", streaming=True)
    batches = []
    buffer_ids = []
    for ex in raw:
        ids = tokenizer(ex["text"])["input_ids"]
        buffer_ids.extend(ids)
        while len(buffer_ids) >= seq_len and len(batches) < n_samples:
            start = rng.randint(0, len(buffer_ids) - seq_len)
            batches.append(torch.tensor(buffer_ids[start:start + seq_len], dtype=torch.long))
            buffer_ids = buffer_ids[start + seq_len:]
        if len(batches) >= n_samples:
            break
    return batches


# ─────────────────────────────────────────────────────────────────────────────
# Structural units -- attention heads + MLP channels, per layer.
# ─────────────────────────────────────────────────────────────────────────────

def get_prunable_tensors(model, layer_idx):
    """Returns the 7 weight tensors GISP's structures live in, for one
    decoder layer: q,k,v,o (attention) + gate,up,down (MLP). No biases in
    Llama-2 (attention_bias=false, mlp_bias=false, verified via config)."""
    blk = model.model.layers[layer_idx]
    return {
        "q": blk.self_attn.q_proj.weight, "k": blk.self_attn.k_proj.weight,
        "v": blk.self_attn.v_proj.weight, "o": blk.self_attn.o_proj.weight,
        "gate": blk.mlp.gate_proj.weight, "up": blk.mlp.up_proj.weight,
        "down": blk.mlp.down_proj.weight,
    }


def enable_weight_grad(model):
    """GISP needs real dL/dW, not gate gradients -- temporarily turn grad
    on for exactly the 7*N_LAYERS prunable tensors (not embeddings/lm_head/
    norms), keeping memory bounded to ~6.48B params' worth of grad buffers
    instead of the full ~6.74B."""
    tensors = []
    for i in range(N_LAYERS):
        for t in get_prunable_tensors(model, i).values():
            t.requires_grad_(True)
            tensors.append(t)
    return tensors


# ─────────────────────────────────────────────────────────────────────────────
# Importance: Algorithm 1, lines 3-5.
# ─────────────────────────────────────────────────────────────────────────────

def compute_importance(model, calib_batches, weight_tensors, device, batch_size=1):
    """|dL/dW * W| accumulated over the calibration set, then aggregated to
    per-structure mean saliency (Algorithm 1 lines 3-5). Standard next-token
    CE loss (the perplexity-objective variant, matching Table 4/5 -- not the
    margin-based task-specific variant)."""
    for t in weight_tensors:
        if t.grad is not None:
            t.grad = None

    n_batches = 0
    for i in tqdm(range(0, len(calib_batches), batch_size), desc="importance: calib fwd/bwd",
                  unit="batch", leave=False, dynamic_ncols=True):
        chunk = torch.stack(calib_batches[i:i + batch_size]).to(device)
        with autocast_ctx(device):
            loss = model(chunk, labels=chunk).loss
        loss.backward()
        n_batches += 1

    raw_importance = {}   # (layer_idx, tensor_name) -> |grad*W| tensor, same shape as W
    for i in range(N_LAYERS):
        for name, t in get_prunable_tensors(model, i).items():
            raw_importance[(i, name)] = (t.grad.float() * t.float()).abs() / n_batches
            t.grad = None
    return raw_importance


def aggregate_structures(raw_importance):
    """Algorithm 1 lines 4-5: sum elements within a structure, divide by
    structure size (mean saliency per structure). Returns two flat tensors
    (all heads across all layers; all channels across all layers) plus
    index maps back to (layer, local_index) for mask application."""
    head_scores = torch.zeros(N_LAYERS, N_HEADS)
    for i in range(N_LAYERS):
        q, k, v, o = raw_importance[(i, "q")], raw_importance[(i, "k")], raw_importance[(i, "v")], raw_importance[(i, "o")]
        for h in range(N_HEADS):
            sl = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
            total = q[sl, :].sum() + k[sl, :].sum() + v[sl, :].sum() + o[:, sl].sum()
            head_scores[i, h] = total / (4 * HEAD_DIM * HIDDEN)   # mean over the head's ~2.1M elements

    channel_scores = torch.zeros(N_LAYERS, N_INTER)
    for i in range(N_LAYERS):
        gate, up, down = raw_importance[(i, "gate")], raw_importance[(i, "up")], raw_importance[(i, "down")]
        per_channel_sum = gate.sum(dim=1) + up.sum(dim=1) + down.sum(dim=0)   # (N_INTER,)
        channel_scores[i] = per_channel_sum / (3 * HIDDEN)   # mean over the channel's 12,288 elements

    return head_scores, channel_scores


# ─────────────────────────────────────────────────────────────────────────────
# Global ranking, thresholding, masking -- Algorithm 1 lines 6-8.
# ─────────────────────────────────────────────────────────────────────────────

def ratio_scheduler(target_ratio, ratio_per_iter=0.00625):
    """Linear schedule, GISP's stated 0.625pp/iteration for 7-8B models.
    n derived from target_ratio, not hardcoded to their stated 112 (see
    module docstring)."""
    n_iters = max(1, round(target_ratio / ratio_per_iter))
    return [target_ratio * (k / n_iters) for k in range(1, n_iters + 1)], n_iters


def apply_global_mask(model, head_scores, channel_scores, head_keep_mask, channel_keep_mask):
    """θ_k <- θ_{k-1} * m (Algorithm 1 line 8) -- literal weight zeroing,
    in place, on the live model. Not reversible, no gates -- this IS the
    pruning, unlike our own project's hook-based apply_gates."""
    with torch.no_grad():
        for i in range(N_LAYERS):
            t = get_prunable_tensors(model, i)
            for h in range(N_HEADS):
                if not head_keep_mask[i, h]:
                    sl = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
                    t["q"][sl, :] = 0; t["k"][sl, :] = 0; t["v"][sl, :] = 0
                    t["o"][:, sl] = 0
            dead = ~channel_keep_mask[i]
            t["gate"][dead, :] = 0
            t["up"][dead, :] = 0
            t["down"][:, dead] = 0


def gisp_prune(model, calib_batches, weight_tensors, target_ratio, device, batch_size=1):
    """Algorithm 1, full loop. Returns final (head_keep_mask, channel_keep_mask)."""
    ratios, n_iters = ratio_scheduler(target_ratio)
    n_total_structures = N_LAYERS * N_HEADS + N_LAYERS * N_INTER

    head_keep_mask = torch.ones(N_LAYERS, N_HEADS, dtype=torch.bool)
    channel_keep_mask = torch.ones(N_LAYERS, N_INTER, dtype=torch.bool)

    print(f"GISP: target_ratio={target_ratio:.4f} -> n_iters={n_iters} "
          f"(0.625pp/iter) over {n_total_structures:,} total structures", flush=True)

    for k, rho_k in enumerate(tqdm(ratios, desc="GISP iterations", unit="iter", dynamic_ncols=True), start=1):
        raw_importance = compute_importance(model, calib_batches, weight_tensors, device, batch_size)
        head_scores, channel_scores = aggregate_structures(raw_importance)

        # already-pruned structures carry zero weights -> zero importance ->
        # naturally stay at the bottom of the global ranking (monotonic mask).
        all_scores = torch.cat([head_scores.flatten(), channel_scores.flatten()])
        n_prune = int(round(rho_k * n_total_structures))
        if n_prune > 0:
            tau_k = torch.kthvalue(all_scores, n_prune).values.item()
        else:
            tau_k = -float("inf")

        head_keep_mask = head_scores > tau_k
        channel_keep_mask = channel_scores > tau_k

        apply_global_mask(model, head_scores, channel_scores, head_keep_mask, channel_keep_mask)

        n_heads_pruned = (~head_keep_mask).sum().item()
        n_channels_pruned = (~channel_keep_mask).sum().item()
        tqdm.write(f"  iter {k}/{n_iters} | rho_k={rho_k:.4f} | tau_k={tau_k:.3e} | "
                  f"heads pruned {n_heads_pruned}/{N_LAYERS*N_HEADS} | "
                  f"channels pruned {n_channels_pruned}/{N_LAYERS*N_INTER}", flush=True)

    return head_keep_mask, channel_keep_mask


# ─────────────────────────────────────────────────────────────────────────────
# Global-%-of-total-params -- computed live from the loaded model, not
# hardcoded, so this resolves F24's ratio-denominator ambiguity for good.
# ─────────────────────────────────────────────────────────────────────────────

def compute_global_pct_pruned(model, head_keep_mask, channel_keep_mask):
    total_params = sum(p.numel() for p in model.parameters())

    pruned_params = 0
    for i in range(N_LAYERS):
        n_heads_pruned = (~head_keep_mask[i]).sum().item()
        pruned_params += n_heads_pruned * (4 * HEAD_DIM * HIDDEN)
        n_channels_pruned = (~channel_keep_mask[i]).sum().item()
        pruned_params += n_channels_pruned * (3 * HIDDEN)

    return 100.0 * pruned_params / total_params, total_params, pruned_params


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_ratios", type=float, nargs="+",
                    default=[0.20, 0.30, 0.40, 0.50],
                    help="Fraction of GISP's OWN structure pool (attention heads + "
                         "MLP channels, NOT total model params) to prune. Default "
                         "matches Table 4/5's own labels (20/30/40/50%%) so the "
                         "reimplementation's output is directly checkable against "
                         "their reported numbers before trusting it for anything else. "
                         "The resulting global-%%-of-total-params (the number this "
                         "project actually needs for comparison) is computed and "
                         "reported for each point, not assumed equal to this input.")
    ap.add_argument("--ratio_per_iter", type=float, default=0.00625,
                    help="0.625pp/iteration, from GISP's Table 11 (7-8B-scale models).")
    ap.add_argument("--calib_samples", type=int, default=128,
                    help="INFERRED from SparseGPT's convention, not verified against "
                         "GISP's own paper for the PPL-objective variant specifically "
                         "-- see module docstring.")
    ap.add_argument("--calib_seq_len", type=int, default=2048)
    ap.add_argument("--calib_batch_size", type=int, default=1,
                    help="Calibration forward/backward batch size -- kept small by "
                         "default given full-model gradient memory (see module "
                         "docstring's compute-cost section).")
    ap.add_argument("--eval_max_length", type=int, default=2048,
                    help="Passed through to the imported evaluate() -- matches our "
                         "own sweep's protocol exactly, the whole point of this script.")
    ap.add_argument("--eval_stride", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_dir", type=str, default=OUT_ROOT)
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"Device: {device} | target_ratios={args.target_ratios}", flush=True)

    if os.environ.get("HF_TOKEN") is None:
        print("WARNING: HF_TOKEN not set -- meta-llama/Llama-2-7b-hf is gate-licensed.", flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(LLAMA2_REPO)

    print(f"Loading C4 calibration data ({args.calib_samples} x {args.calib_seq_len} tokens) ...", flush=True)
    calib_batches = get_calibration_batches(tokenizer, args.calib_samples, args.calib_seq_len, args.seed)
    print(f"Calibration set: {len(calib_batches)} sequences.", flush=True)

    print("Loading WikiText-2 (our own verified sliding-window protocol) ...", flush=True)
    _, _, test_ids = get_loaders(seq_len=512, batch_size=1)   # only test_ids is used here

    os.makedirs(args.out_dir, exist_ok=True)
    all_results = []
    model = None

    for target_ratio in args.target_ratios:
        tag = f"ratio={target_ratio:.2f}"
        print(f"\n{'='*70}\n{tag}\n{'='*70}", flush=True)

        # Fresh model per operating point -- GISP's iterative masking is
        # destructive (real zeros, not reversible gates), so each target
        # ratio needs its own start-from-dense run, unlike our own sweep's
        # single-model-reused-across-lambdas convention.
        if model is not None:
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[{tag}] Loading fresh Llama-2-7B ...", flush=True)
        model = load_llama2_7b(device)
        weight_tensors = enable_weight_grad(model)

        head_keep_mask, channel_keep_mask = gisp_prune(
            model, calib_batches, weight_tensors, target_ratio, device, args.calib_batch_size)

        global_pct, total_params, pruned_params = compute_global_pct_pruned(
            model, head_keep_mask, channel_keep_mask)

        for t in weight_tensors:
            t.requires_grad_(False)

        pruned_ppl = evaluate(model, test_ids, device, gates=None, desc=f"[{tag}] eval",
                              max_length=args.eval_max_length, stride=args.eval_stride)

        result = {
            "gisp_target_ratio": target_ratio,
            "global_pct_of_total_params_pruned": global_pct,
            "total_params": total_params,
            "pruned_params": pruned_params,
            "pruned_ppl": pruned_ppl,
            "n_heads_pruned": int((~head_keep_mask).sum().item()),
            "n_channels_pruned": int((~channel_keep_mask).sum().item()),
        }
        print(f"  -> {tag}: global {global_pct:.2f}% of total params pruned | "
              f"ppl {pruned_ppl:.3f}", flush=True)
        all_results.append(result)

        with open(os.path.join(args.out_dir, f"ratio_{target_ratio:.2f}.json"), "w") as f:
            json.dump(result, f, indent=2)

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    sep = "-" * 90
    print("\n" + sep)
    print(f"{'gisp ratio':>10} | {'global % total':>15} | {'ppl':>10} | {'heads pruned':>13} | {'channels pruned':>16}")
    print(sep)
    for r in all_results:
        print(f"{r['gisp_target_ratio']:>10.2f} | {r['global_pct_of_total_params_pruned']:>14.2f}% | "
              f"{r['pruned_ppl']:>10.3f} | {r['n_heads_pruned']:>13} | {r['n_channels_pruned']:>16}")
    print(f"\nResults -> {args.out_dir}/")


if __name__ == "__main__":
    main()
