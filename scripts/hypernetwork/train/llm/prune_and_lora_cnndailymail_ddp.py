"""
Data-parallel (DDP) variant of prune_and_lora_cnndailymail.py -- same
pipeline (physical surgery from an already-trained pruner.pt -> Dense-LoRA
fine-tune -> eval), replicated across all GPUs on the node instead of using
one.

WHY DATA-PARALLEL, NOT MODEL-PARALLEL (confirmed with the user, 2026-08-19):
Llama-2-7B in bf16 is ~14GB of FROZEN weights; the only thing with an
optimizer state is the LoRA adapter (~40M params with mlp_attn targets,
r=16 -- see the accompanying discussion). A single 48GB GPU already fits
this with room to spare -- there is no memory pressure to relieve, which is
the entire reason tensor/pipeline (model) parallelism exists. Splitting a
model that already fits just to spread it out would ADD an all-reduce/
all-gather after every sharded matmul (dozens of syncs per forward pass,
each blocking the next layer -- latency-bound, not overlappable) for zero
memory benefit, and would complicate .generate() for the ROUGE eval (KV-
cache handling across shards is the standard model-parallel + generation
pain point). Data-parallel instead replicates the whole (already-fits)
model per GPU and only ever syncs the small LoRA gradient once per
optimizer step (~40M params x 4B fp32 ~= 160MB all-reduced 4x less often
than TP's per-layer activation syncs and ~26x less data each time) --
compute-bound, near-linear scaling, and the standard, well-trodden path for
"frozen base + small trainable adapter" (this is exactly what peft +
torch.distributed / accelerate is built for).

*** UNTESTED. *** Same caveat as the sibling script, plus a new failure
surface (distributed init, gradient-checkpointing + DDP interaction, shard
reduction correctness) that single-GPU testing cannot exercise. Run
--sanity_check on a SMALL --nproc_per_node (e.g. 2) before scaling to all 8.

LAUNCH (torchrun, not `python3` directly -- this script exits immediately
if it doesn't see torchrun's env vars):
    torchrun --standalone --nproc_per_node=8 prune_and_lora_cnndailymail_ddp.py \\
        --pruner_ckpt experiments/latest/llama2_7b_cnndailymail/lambda_0.1/pruner.pt \\
        --hf_token <token> --out_dir /workspace/results/llama2_7b_cnndailymail_lora_ddp

DDP-SPECIFIC CHOICES (confirmed with the user, 2026-08-19 -- everything
else, incl. all LoRA hyperparameters, is unchanged from the sibling script):

  - SAME NUMBER OF OPTIMIZER STEPS as the single-GPU script (--max_steps
    still defaults to 500), NOT scaled down. Each rank still runs its own
    batch_size x grad_accum_steps microbatches per step, replicated across
    world_size ranks concurrently -- so total examples processed scales
    with world_size (8x at --nproc_per_node=8: 500*4*4*8 = 64,000 vs the
    single-GPU script's 8,000). This is a genuinely larger training run in
    data-seen terms, not a wall-clock-matched replica of it.

  - LR SCALED LINEARLY WITH GLOBAL BATCH SIZE (Goyal et al. 2017 linear
    scaling rule): --lr is still the same "per effective-batch-16" base
    rate as the sibling script (default 2e-4), but the LR actually handed
    to the optimizer is base_lr * world_size (global effective batch =
    batch_size * grad_accum_steps * world_size = 16 * world_size, vs the
    sibling script's fixed 16). Printed explicitly at startup, not silent.

  - Gradient accumulation uses ddp_model.no_sync() for every microbatch
    except the last one in each optimizer step -- DDP's default is to
    all-reduce gradients on EVERY .backward() call, which is correct
    (averaging commutes with the accumulation sum) but wasteful:
    grad_accum_steps all-reduces per step instead of 1. no_sync() defers
    the sync to the final microbatch.

  - gradient_checkpointing uses use_reentrant=False (transformers'
    gradient_checkpointing_kwargs). Reentrant checkpointing is a documented
    DDP incompatibility -- DDP tracks gradient-ready state via autograd
    hooks, and reentrant checkpointing's re-entrant backward call can mark
    a variable ready out of order / twice, raising a runtime error.
    Non-reentrant is the standard fix and transformers exposes it directly.

  - Eval (CE + ROUGE) is SHARDED round-robin across ranks
    (examples[rank::world_size]) then reduced to reproduce EXACTLY the
    single-GPU metric, not an average-of-per-rank-averages (which would be
    a subtly different, shard-size-unweighted number):
      * CE: each rank accumulates (sum NLL, sum tokens) over its shard,
        both all-reduced (SUM), then divided -- identical to running
        evaluate_ce on the full set in one process.
      * ROUGE: each rank generates over its shard, raw (prediction,
        reference) pairs are all_gather_object'd, rank 0 computes
        rouge.compute() ONCE over the full reunited set (ROUGE's
        per-example-F-measure-then-mean definition makes this exactly
        equivalent to single-GPU, not an approximation), then
        broadcast_object_list's the result back to every rank.

  - HF dataset/model download is gated behind a rank-0-loads-first-then-
    barrier pattern (rank0_first_call) -- avoids 8 processes racing to
    write the same first-time HF cache files.

  - Only rank 0 writes summary.txt/meta.json/plot.png/adapter weights.
    Safe: DDP's constructor broadcasts rank 0's initial parameters to
    every replica at wrap time, and all-reduced gradients keep every
    replica bit-synchronized (up to ordinary floating-point nondeterminism
    across GPUs, the same looseness every DDP job accepts) through
    training -- rank 0's state is representative of all ranks.

Reuses physically_prune_model / load_pruner_final_gates /
lora_forward_backward / build_optimizer_and_scheduler / plot_finetune_run /
OUT_ROOT-pattern verbatim from prune_and_lora_cnndailymail.py, and Pruner /
load_llama2_7b / get_mlp_weights / get_loaders / sample_examples /
build_training_batch / autocast_ctx / _smooth / _rouge_scores_to_floats /
N_LAYERS / N_INTER / LAYER_SHAPE / ARTICLE_PREFIX / SUMMARY_PREFIX verbatim
from train_pruner_llama2_7b_cnndailymail.py via direct import -- no logic
duplicated beyond what DDP sharding/reduction genuinely requires.
"""
import os
import sys
import json
import time
import datetime
import argparse
import contextlib

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_pruner_llama2_7b_cnndailymail import (
    load_llama2_7b, get_mlp_weights, get_loaders, sample_examples,
    build_training_batch, autocast_ctx, _smooth, _rouge_scores_to_floats,
    N_LAYERS, N_INTER, LAYER_SHAPE, ARTICLE_PREFIX, SUMMARY_PREFIX,
)
from prune_and_lora_cnndailymail import (
    load_pruner_final_gates, physically_prune_model, lora_forward_backward,
    build_optimizer_and_scheduler, plot_finetune_run,
)

OUT_ROOT = "/workspace/results/llama2_7b_cnndailymail_lora_ddp"

# train_pruner_llama2_7b_cnndailymail.py's own --rouge_eval_examples default, i.e. the sample
# size the pruner checkpoints' cached rouge_orig/rouge_pruned were measured at. Used only to
# decide whether those cached numbers are comparable to this run's -- see the note where the
# dense baseline is read.
PRUNING_RUN_ROUGE_EVAL_EXAMPLES = 300


# ─────────────────────────────────────────────────────────────────────────────
# Distributed setup / small helpers.
# ─────────────────────────────────────────────────────────────────────────────

def setup_distributed():
    if "WORLD_SIZE" not in os.environ:
        print("ERROR: no torchrun env vars found (WORLD_SIZE unset). Launch this script "
              "with torchrun, e.g.:\n"
              "  torchrun --standalone --nproc_per_node=8 prune_and_lora_cnndailymail_ddp.py ...\n"
              "not `python3 prune_and_lora_cnndailymail_ddp.py ...` directly.", flush=True)
        sys.exit(1)
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available -- this script requires GPUs (nccl backend).", flush=True)
        sys.exit(1)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # Default collective-op timeout is 10 minutes -- too short for the rank0-downloads-
    # first-then-barrier pattern in rank0_first_call: a first-time Llama-2-7B download
    # (~13GB) at this network's observed ~12MB/s sustained throughput takes ~18-20 min,
    # which blew the default timeout on 2026-08-19 (rank1 timed out waiting at the
    # barrier while rank0's download was still progressing, not stuck). 60 min covers a
    # cold-cache first run with margin; subsequent runs hit a warm cache and are fast.
    dist.init_process_group(backend="nccl", init_method="env://",
                            timeout=datetime.timedelta(minutes=60))
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, local_rank, device


def rank0_first_call(rank, fn, *args, **kwargs):
    """Runs fn() on rank 0 first (e.g. a first-time HF hub/dataset download
    that writes shared cache files), barriers, then lets every other rank
    run fn() too (now a cache hit) -- avoids an 8-way write race on the
    same cache files."""
    result = None
    if rank == 0:
        result = fn(*args, **kwargs)
    dist.barrier()
    if rank != 0:
        result = fn(*args, **kwargs)
    dist.barrier()
    return result


def log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# LoRA wrap -- identical to the sibling script's wrap_with_lora, except
# gradient checkpointing is forced non-reentrant (see module docstring's
# DDP-specific-choices section for why this one line differs).
# ─────────────────────────────────────────────────────────────────────────────

def wrap_with_lora_ddp(model, args, target_modules):
    from peft import LoraConfig, get_peft_model, TaskType
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                          target_modules=target_modules, task_type=TaskType.CAUSAL_LM, bias="none")
    return get_peft_model(model, lora_cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Distributed eval -- shard across ranks, reduce to the single-GPU-
# equivalent aggregate (see module docstring). gates is always None here
# (physical surgery already applied, nothing left to gate).
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_ce_ddp(model, examples, device, tokenizer, rank, world_size, desc="eval",
                    batch_size=8, max_article_chars=4000, max_summary_tokens=128):
    from tqdm import tqdm
    shard = examples[rank::world_size]
    total_nll = torch.zeros(1, device=device)
    total_tokens = torch.zeros(1, device=device)
    for i in tqdm(range(0, len(shard), batch_size), desc=desc, unit="batch",
                  leave=False, dynamic_ncols=True, disable=(rank != 0)):
        batch = shard[i:i + batch_size]
        input_ids, attn, labels = build_training_batch(batch, tokenizer, max_article_chars, max_summary_tokens)
        input_ids, attn, labels = input_ids.to(device), attn.to(device), labels.to(device)
        with autocast_ctx(device):
            loss = model(input_ids, attention_mask=attn, labels=labels).loss
        n_tok = (labels != -100).sum()
        total_nll += loss.detach() * n_tok
        total_tokens += n_tok
    dist.all_reduce(total_nll, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
    return (total_nll / total_tokens).item()


@torch.no_grad()
def evaluate_rouge_ddp(model, examples, device, tokenizer, rank, world_size, desc="rouge",
                       batch_size=4, max_new_tokens=128, num_beams=1, max_article_chars=4000):
    import evaluate as hf_evaluate
    from tqdm import tqdm

    shard = examples[rank::world_size]
    gen_max_length = max_article_chars // 2 + 64   # see sibling script's comment on this margin

    prev_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    predictions, references = [], []
    try:
        for i in tqdm(range(0, len(shard), batch_size), desc=desc, unit="batch",
                      leave=False, dynamic_ncols=True, disable=(rank != 0)):
            batch = shard[i:i + batch_size]
            prompts = [ARTICLE_PREFIX + ex["article"][:max_article_chars] + SUMMARY_PREFIX
                      for ex in batch]
            enc = tokenizer(prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=gen_max_length).to(device)
            with autocast_ctx(device):
                gen_ids = model.generate(**enc, max_new_tokens=max_new_tokens,
                                         do_sample=False, num_beams=num_beams,
                                         pad_token_id=tokenizer.pad_token_id)
            gen_only = gen_ids[:, enc["input_ids"].shape[1]:]
            predictions.extend(tokenizer.batch_decode(gen_only, skip_special_tokens=True))
            references.extend(ex["summary"] for ex in batch)
    finally:
        tokenizer.padding_side = prev_padding_side

    gathered = [None] * world_size
    dist.all_gather_object(gathered, list(zip(predictions, references)))

    result_holder = [None]
    if rank == 0:
        all_pairs = [pair for shard_pairs in gathered for pair in shard_pairs]
        preds, refs = zip(*all_pairs)
        rouge = hf_evaluate.load("rouge")
        result_holder[0] = _rouge_scores_to_floats(
            rouge.compute(predictions=list(preds), references=list(refs)))
    dist.broadcast_object_list(result_holder, src=0)
    return result_holder[0]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pruner_ckpt", type=str, required=True,
                    help="Path to a pruner.pt from train_pruner_llama2_7b_cnndailymail.py.")
    ap.add_argument("--lora_target", type=str, choices=["mlp", "mlp_attn"], default="mlp_attn")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-4,
                    help="BASE lr, same convention as the sibling script (per effective-batch-16). "
                         "Actual optimizer lr = --lr * world_size (linear scaling rule) -- see "
                         "module docstring's DDP-specific-choices section.")
    ap.add_argument("--lr_schedule", type=str, choices=["cosine", "linear", "constant"], default="cosine")
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_accum_steps", type=int, default=4,
                    help="PER-RANK micro-batches accumulated per optimizer step -- same as the "
                         "sibling script. Global effective batch = batch_size * grad_accum_steps "
                         "* world_size.")
    ap.add_argument("--gradient_checkpointing", action="store_true", default=True)
    ap.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    ap.add_argument("--max_steps", type=int, default=500,
                    help="OPTIMIZER steps -- SAME default as the sibling single-GPU script, NOT "
                         "scaled down (confirmed with the user, 2026-08-19). Total examples "
                         "processed scales with world_size -- see module docstring.")
    ap.add_argument("--eval_every", type=int, default=125)
    ap.add_argument("--batch_size", type=int, default=4, help="PER-RANK micro-batch size.")
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--gen_batch_size", type=int, default=4)
    ap.add_argument("--rouge_eval_examples", type=int, default=300)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--max_article_chars", type=int, default=4000)
    ap.add_argument("--max_summary_tokens", type=int, default=128)
    ap.add_argument("--gap_eval_examples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hf_token", type=str, required=True,
                    help="HF token with ACCEPTED access to meta-llama/Llama-2-7b-hf. "
                         "REQUIRED CLI arg (not read from HF_TOKEN env var) -- same convention "
                         "as the sibling script, now enforced by argparse rather than a "
                         "print-and-continue warning.")
    ap.add_argument("--out_dir", type=str, default=OUT_ROOT)
    ap.add_argument("--sanity_check", action="store_true",
                    help="Run surgery, verify pct_pruned matches the checkpoint's own record, "
                         "wrap with LoRA+DDP, run ONE training step, then exit. NOT optional "
                         "before a real run -- try this with a SMALL --nproc_per_node first.")
    args = ap.parse_args()

    rank, world_size, local_rank, device = setup_distributed()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    base_lr = args.lr
    args.lr = base_lr * world_size
    log(rank, f"world_size={world_size} | base_lr={base_lr:.2e} -> scaled_lr={args.lr:.2e} "
              f"(linear scaling rule) | global effective batch = "
              f"{args.batch_size}*{args.grad_accum_steps}*{world_size} = "
              f"{args.batch_size * args.grad_accum_steps * world_size}")

    log(rank, "Loading Llama-2-7B (GATE-LICENSED -- requires --hf_token with accepted access) ...")
    model = rank0_first_call(rank, load_llama2_7b, device, hf_token=args.hf_token)

    log(rank, "Loading CNN/DailyMail ...")
    tokenizer, _unused_loader, train_examples, val_examples, test_examples = rank0_first_call(
        rank, get_loaders, args.batch_size, hf_token=args.hf_token)
    log(rank, f"Data: train={len(train_examples):,} val={len(val_examples):,} test={len(test_examples):,}")

    examples_per_step = args.batch_size * args.grad_accum_steps * world_size
    examples_seen = args.max_steps * examples_per_step
    log(rank, f"Train-split coverage: {args.max_steps} steps x {examples_per_step} ex/step = "
              f"{examples_seen:,} examples = {100 * examples_seen / len(train_examples):.1f}% "
              f"of the {len(train_examples):,}-example train split "
              f"({examples_seen / len(train_examples):.2f} epochs)")

    log(rank, f"\nLoading pruner checkpoint + reconstructing final gates: {args.pruner_ckpt}")
    gates, ckpt = load_pruner_final_gates(model, args.pruner_ckpt, device)
    ckpt_pct_pruned = 100 * (1 - sum(g.mean().item() for g in gates) / len(gates))
    log(rank, f"  reconstructed gate: {ckpt_pct_pruned:.2f}% FFN neurons pruned "
              f"(lambda={ckpt.get('lambda')}, converged={ckpt.get('converged')})")

    target_modules = (["gate_proj", "up_proj", "down_proj"] if args.lora_target == "mlp"
                      else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    if args.sanity_check:
        if rank == 0:
            print("\n" + "=" * 70, flush=True)
            print("SANITY CHECK -- surgery + pct_pruned cross-check + 1 DDP LoRA step", flush=True)
            print("=" * 70, flush=True)
        per_layer_kept = physically_prune_model(model, gates)
        actual_pct = 100 * (1 - sum(per_layer_kept) / (N_LAYERS * N_INTER))
        diff = abs(actual_pct - ckpt_pct_pruned)
        log(rank, f"  surgery result   : {actual_pct:.4f}% pruned")
        log(rank, f"  gate reconstruct : {ckpt_pct_pruned:.4f}% pruned")
        log(rank, f"  diff             : {diff:.4f} pp "
                  f"{'PASS' if diff < 1e-3 else 'FAIL -- mismatch, investigate surgery indexing'}")

        peft_model = wrap_with_lora_ddp(model, args, target_modules)
        if rank == 0:
            peft_model.print_trainable_parameters()
        ddp_model = DDP(peft_model, device_ids=[local_rank], output_device=local_rank)
        trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
        opt, sched = build_optimizer_and_scheduler(trainable_params, args)
        ddp_model.train()
        opt.zero_grad()
        sampler = DistributedSampler(train_examples, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
        my_indices = list(sampler)[:args.batch_size]
        batch = [train_examples[i] for i in my_indices]
        loss = lora_forward_backward(ddp_model, tokenizer, batch, device, args.max_article_chars,
                                     args.max_summary_tokens, args.grad_accum_steps)
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        opt.step(); sched.step()
        log(rank, f"  rank {rank} 1 LoRA micro-step loss : {loss:.4f} (finite, no crash -> pipeline OK)")
        dist.barrier()
        if rank == 0:
            print("=" * 70, flush=True)
        dist.destroy_process_group()
        return

    # ── 1) dense baseline -- reuse cached numbers from the original pruning run ──
    dense_test_ppl = ckpt["test_ppl_orig"]
    dense_rouge = ckpt["rouge_orig"]
    log(rank, f"\nDense baseline (cached from pruning-run checkpoint): "
              f"ppl={dense_test_ppl:.3f} rougeL={dense_rouge['rougeL']*100:.2f}%")
    # The checkpoint does not record the ROUGE sample size its rouge_orig/rouge_pruned were
    # measured at -- it is whatever train_pruner_llama2_7b_cnndailymail.py's own
    # --rouge_eval_examples was, default 300. ROUGE is a mean of per-example F-measures, so a
    # 300-example number and a 3000-example number are estimates of the same quantity but NOT
    # the same statistic; mixing them in one table silently compares different sample sizes.
    # The correctness cross-checks below (pre-LoRA vs cached rouge_pruned) are likewise only
    # meaningful at the size the cache was taken at.
    if args.rouge_eval_examples != PRUNING_RUN_ROUGE_EVAL_EXAMPLES:
        log(rank, f"  NOTE: --rouge_eval_examples={args.rouge_eval_examples} != "
                  f"{PRUNING_RUN_ROUGE_EVAL_EXAMPLES} (the pruning run's default, i.e. the sample "
                  f"size the cached dense ROUGE above was measured at). The cached dense number "
                  f"is NOT comparable to this run's ROUGE at this size -- use the dense arm's own "
                  f"LIVE pre-LoRA number (lora_finetune_dense_cnndailymail_ddp.py, same "
                  f"--rouge_eval_examples) as the dense baseline instead. The pre-LoRA "
                  f"correctness-check diffs printed below are expected to be non-zero for the "
                  f"same reason and do not indicate a surgery bug at this setting.")

    # ── 2) physical surgery -- deterministic, identical per rank, no comm needed ──
    log(rank, f"\nPhysically pruning model ({ckpt_pct_pruned:.2f}% target) ...")
    per_layer_kept = physically_prune_model(model, gates)
    actual_pct_pruned = 100 * (1 - sum(per_layer_kept) / (N_LAYERS * N_INTER))
    diff = abs(actual_pct_pruned - ckpt_pct_pruned)
    log(rank, f"  surgery result: {actual_pct_pruned:.4f}% pruned (gate reconstruct said "
              f"{ckpt_pct_pruned:.4f}%, diff={diff:.4f}pp)")
    if diff > 1e-2:
        log(rank, "  WARNING: surgery pct_pruned does not match the gate reconstruction closely -- "
                  "double-check keep_idx indexing before trusting downstream numbers.")

    n_params_pruned = sum(p.numel() for p in model.parameters())
    log(rank, f"  model params after surgery (per replica): {n_params_pruned:,}")

    # ── eval PRE-LoRA (surgery only) -- correctness check vs the pruning run's
    #    own cached numbers, sharded across ranks (see module docstring) ──
    log(rank, "\nEvaluating physically-pruned model BEFORE any LoRA training "
              "(correctness check vs the pruning run's own gated-eval numbers) ...")
    model.eval()
    ce_kw = dict(batch_size=args.eval_batch_size, max_article_chars=args.max_article_chars,
                max_summary_tokens=args.max_summary_tokens)
    pre_lora_test_ce = evaluate_ce_ddp(model, test_examples, device, tokenizer, rank, world_size,
                                       desc="pre-LoRA test CE", **ce_kw)
    pre_lora_ppl = float(np.exp(pre_lora_test_ce))
    rouge_sample = sample_examples(test_examples, args.rouge_eval_examples)   # deterministic, identical per rank
    rouge_kw = dict(batch_size=args.gen_batch_size, max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams, max_article_chars=args.max_article_chars)
    pre_lora_rouge = evaluate_rouge_ddp(model, rouge_sample, device, tokenizer, rank, world_size,
                                        desc="pre-LoRA ROUGE", **rouge_kw)
    ppl_check_diff = abs(pre_lora_ppl - ckpt["test_ppl_pruned"])
    rL_check_diff = abs(pre_lora_rouge["rougeL"] - ckpt["rouge_pruned"]["rougeL"]) * 100
    log(rank, f"  pre-LoRA ppl   : {pre_lora_ppl:.3f}  (gated-eval checkpoint said "
              f"{ckpt['test_ppl_pruned']:.3f}, diff={ppl_check_diff:.4f})")
    log(rank, f"  pre-LoRA R-L   : {pre_lora_rouge['rougeL']*100:.2f}%  (gated-eval checkpoint said "
              f"{ckpt['rouge_pruned']['rougeL']*100:.2f}%, diff={rL_check_diff:.4f}pp)")
    # The ppl check is always valid (full test split either way, no sampling involved). The
    # ROUGE check is only valid when this run's sample size matches the size the checkpoint
    # cached its rouge_pruned at -- otherwise a non-zero diff is just two different sample
    # sizes, not a surgery bug.
    rouge_check_valid = (args.rouge_eval_examples == PRUNING_RUN_ROUGE_EVAL_EXAMPLES)
    if ppl_check_diff > 0.05 or (rouge_check_valid and rL_check_diff > 1.0):
        log(rank, "  WARNING: physical-surgery eval diverges non-trivially from the original "
                  "gated-eval numbers -- expected to be near-identical. Investigate before "
                  "trusting the LoRA results below.")

    # ── 3) LoRA fine-tune, DDP-wrapped ──
    log(rank, f"\nWrapping with LoRA (target_modules={target_modules}, r={args.lora_r}, "
              f"alpha={args.lora_alpha}, dropout={args.lora_dropout}, "
              f"gradient_checkpointing={args.gradient_checkpointing}, use_reentrant=False) ...")
    peft_model = wrap_with_lora_ddp(model, args, target_modules)
    if rank == 0:
        peft_model.print_trainable_parameters()
    ddp_model = DDP(peft_model, device_ids=[local_rank], output_device=local_rank)
    trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
    opt, sched = build_optimizer_and_scheduler(trainable_params, args)
    effective_batch = args.batch_size * args.grad_accum_steps * world_size
    log(rank, f"  optimizer steps={args.max_steps} | grad_accum_steps={args.grad_accum_steps} | "
              f"world_size={world_size} | global effective batch={effective_batch} | "
              f"lr_schedule={args.lr_schedule} (warmup_ratio={args.warmup_ratio}) | "
              f"weight_decay={args.weight_decay}")

    lambda_tag = f"lambda_{ckpt.get('lambda')}" if ckpt.get("lambda") is not None else "run"
    run_dir = os.path.join(args.out_dir, lambda_tag)
    if rank == 0:
        os.makedirs(run_dir, exist_ok=True)

    gap_val_sample = sample_examples(val_examples, args.gap_eval_examples)   # deterministic, identical per rank
    sampler = DistributedSampler(train_examples, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    train_loader = DataLoader(train_examples, batch_size=args.batch_size, sampler=sampler,
                              collate_fn=lambda b: b)
    train_loss_hist, val_points = [], []
    epoch = 0
    sampler.set_epoch(epoch)
    loader_iter = iter(train_loader)
    ddp_model.train()
    t0 = time.time()
    from tqdm import tqdm
    pbar = tqdm(total=args.max_steps, desc="LoRA fine-tune (optimizer steps)", unit="step",
               dynamic_ncols=True, disable=(rank != 0))
    for opt_step in range(1, args.max_steps + 1):
        opt.zero_grad()
        micro_losses = []
        for micro_idx in range(args.grad_accum_steps):
            try:
                batch = next(loader_iter)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                loader_iter = iter(train_loader)
                batch = next(loader_iter)
            is_last_microbatch = (micro_idx == args.grad_accum_steps - 1)
            sync_ctx = contextlib.nullcontext() if is_last_microbatch else ddp_model.no_sync()
            with sync_ctx:
                micro_losses.append(lora_forward_backward(ddp_model, tokenizer, batch, device,
                                                           args.max_article_chars, args.max_summary_tokens,
                                                           args.grad_accum_steps))
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        opt.step(); sched.step()
        loss = sum(micro_losses) / len(micro_losses)
        train_loss_hist.append(loss)
        pbar.set_postfix(loss=f"{loss:.3f}", lr=f"{sched.get_last_lr()[0]:.2e}", refresh=False)
        pbar.update(1)

        if opt_step % args.eval_every == 0 or opt_step == args.max_steps:
            ddp_model.eval()
            val_ce = evaluate_ce_ddp(peft_model, gap_val_sample, device, tokenizer, rank, world_size,
                                     desc=f"val CE @ step {opt_step}", **ce_kw)
            val_points.append((opt_step, val_ce))
            if rank == 0:
                tqdm.write(f"  step {opt_step:>6} | train loss {loss:.4f} | val CE {val_ce:.4f} | "
                          f"lr {sched.get_last_lr()[0]:.2e}")
            ddp_model.train()
    pbar.close()
    total_time = time.time() - t0

    # ── final eval POST-LoRA ──
    log(rank, "\nFinal evaluation (physically-pruned + LoRA fine-tuned) ...")
    ddp_model.eval()
    post_test_ce = evaluate_ce_ddp(peft_model, test_examples, device, tokenizer, rank, world_size,
                                   desc="post-LoRA test CE", **ce_kw)
    post_ppl = float(np.exp(post_test_ce))
    post_rouge = evaluate_rouge_ddp(peft_model, rouge_sample, device, tokenizer, rank, world_size,
                                    desc="post-LoRA ROUGE", **rouge_kw)

    log(rank, f"  -> dense ppl {dense_test_ppl:.3f} | pre-LoRA (surgery only) ppl {pre_lora_ppl:.3f} | "
              f"post-LoRA ppl {post_ppl:.3f}")
    log(rank, f"  -> dense R-L {dense_rouge['rougeL']*100:.2f}% | pre-LoRA R-L "
              f"{pre_lora_rouge['rougeL']*100:.2f}% | post-LoRA R-L {post_rouge['rougeL']*100:.2f}%")

    if rank == 0:
        plot_finetune_run(train_loss_hist, val_points, os.path.join(run_dir, "plot.png"),
                          title=(f"Llama-2-7B — physically-pruned ({actual_pct_pruned:.1f}%) + "
                                f"Dense-LoRA (DDP x{world_size}) — CNN/DailyMail — {lambda_tag}"))

        lines = [
            f"Llama-2-7B — physical surgery + Dense-LoRA (DDP x{world_size}) — CNN/DailyMail — "
            f"{lambda_tag} (from {args.pruner_ckpt})",
            f"lora: target={args.lora_target} r={args.lora_r} alpha={args.lora_alpha} "
            f"dropout={args.lora_dropout} base_lr={base_lr} scaled_lr={args.lr} "
            f"schedule={args.lr_schedule} warmup_ratio={args.warmup_ratio} weight_decay={args.weight_decay}",
            f"optimizer steps : {args.max_steps} | grad_accum_steps={args.grad_accum_steps} | "
            f"world_size={world_size} | global_effective_batch={effective_batch} | "
            f"gradient_checkpointing={args.gradient_checkpointing} | time: {total_time:.1f}s",
            f"train-split coverage : {examples_seen:,} examples seen = "
            f"{100 * examples_seen / len(train_examples):.1f}% of the {len(train_examples):,}-example "
            f"train split ({examples_seen / len(train_examples):.2f} epochs)",
            "-" * 60,
            f"% FFN neurons pruned (surgery) : {actual_pct_pruned:.2f}% "
            f"(gate reconstruct said {ckpt_pct_pruned:.2f}%, diff={diff:.4f}pp)",
            f"model params after surgery     : {n_params_pruned:,}",
            "-" * 60,
            f"cnn/dailymail test set ({len(test_examples)} examples, full split, target-only CE):",
            f"  dense (orig, cached)          ppl : {dense_test_ppl:.3f}",
            f"  pre-LoRA (surgery only)       ppl : {pre_lora_ppl:.3f}  "
            f"(correctness check vs cached pruned-eval: diff={ppl_check_diff:.4f})",
            f"  post-LoRA (surgery + LoRA)    ppl : {post_ppl:.3f}",
            "-" * 60,
            f"ROUGE via generation ({len(rouge_sample)}-example sample of test, "
            f"greedy={args.num_beams == 1}, num_beams={args.num_beams}):",
            f"  dense (orig, cached)      rouge1/rouge2/rougeL/rougeLsum : "
            f"{dense_rouge['rouge1']*100:.2f}/{dense_rouge['rouge2']*100:.2f}/"
            f"{dense_rouge['rougeL']*100:.2f}/{dense_rouge['rougeLsum']*100:.2f}",
            f"  pre-LoRA (surgery only)   rouge1/rouge2/rougeL/rougeLsum : "
            f"{pre_lora_rouge['rouge1']*100:.2f}/{pre_lora_rouge['rouge2']*100:.2f}/"
            f"{pre_lora_rouge['rougeL']*100:.2f}/{pre_lora_rouge['rougeLsum']*100:.2f}  "
            f"(correctness check vs cached pruned-eval R-L: diff={rL_check_diff:.4f}pp)",
            f"  post-LoRA (surgery + LoRA) rouge1/rouge2/rougeL/rougeLsum : "
            f"{post_rouge['rouge1']*100:.2f}/{post_rouge['rouge2']*100:.2f}/"
            f"{post_rouge['rougeL']*100:.2f}/{post_rouge['rougeLsum']*100:.2f}",
        ]
        with open(os.path.join(run_dir, "summary.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        print("\n" + "\n".join(lines), flush=True)

        peft_model.save_pretrained(run_dir)   # peft adapter weights only, a few MB
        with open(os.path.join(run_dir, "meta.json"), "w") as f:
            json.dump({
                "arm": "pruned",
                "pruner_ckpt": args.pruner_ckpt, "lambda": ckpt.get("lambda"),
                "per_layer_kept": per_layer_kept, "pct_pruned": actual_pct_pruned,
                "dense_ppl": dense_test_ppl, "pre_lora_ppl": pre_lora_ppl, "post_lora_ppl": post_ppl,
                "dense_rouge": dense_rouge, "pre_lora_rouge": pre_lora_rouge, "post_lora_rouge": post_rouge,
                "dense_rouge_is_cached_at_300": True,
                "optimizer_steps": args.max_steps, "grad_accum_steps": args.grad_accum_steps,
                "world_size": world_size, "global_effective_batch": effective_batch,
                "train_split_size": len(train_examples), "examples_seen": examples_seen,
                "train_coverage_pct": 100 * examples_seen / len(train_examples),
                "rouge_eval_examples": len(rouge_sample),
                "test_split_size": len(test_examples), "seed": args.seed,
                "n_params_pruned": n_params_pruned,
                "total_time": total_time, "lora_target": args.lora_target, "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha, "lora_dropout": args.lora_dropout,
                "base_lr": base_lr, "scaled_lr": args.lr, "lr_schedule": args.lr_schedule,
                "warmup_ratio": args.warmup_ratio, "weight_decay": args.weight_decay,
                "gradient_checkpointing": args.gradient_checkpointing,
            }, f, indent=2)
        print(f"\n[saved] {run_dir}/ (adapter weights + summary.txt + meta.json + plot.png)", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
