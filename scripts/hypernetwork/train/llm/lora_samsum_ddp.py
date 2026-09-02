"""
SAMSum pruned-vs-dense LoRA control, DDP -- BOTH ARMS IN ONE SCRIPT.

This is the SAMSum analogue of the CNN/DailyMail B5 control that produced
F25. Same question, new task: once BOTH models get the same LoRA fine-tune on
the same data with the same budget, does pruning still buy anything?

STRUCTURAL DEVIATION FROM THE CNN/DAILYMAIL PAIR, deliberate, flagged:
CNN/DailyMail used two scripts (prune_and_lora_cnndailymail_ddp.py +
lora_finetune_dense_cnndailymail_ddp.py) and defended against arm-drift by
having the dense one IMPORT the pruned one's helpers. This script instead
takes `--arm {pruned,dense}` and runs one code path for both, which makes
drift structurally impossible rather than merely discouraged -- there is
exactly one training loop, one eval function, one meta.json writer. The arms
differ ONLY in whether `physically_prune_model` runs before LoRA wrapping.

WHY THIS RUN EXISTS (the SAMSum sweep's own result, 2026-08-27):
The 8-lambda SAMSum pruning sweep produced pruned-beats-dense at EVERY
operating point, including 80.5% of FFN neurons removed (test ppl 3.538 vs
dense 3.668; ROUGE-L 34.14 vs dense 18.98). A generation-length diagnostic
ruled out the obvious artifact -- dense and pruned both generate to the
96-token cap (94.1 vs 94.7 mean tokens), so this is NOT a brevity/precision
effect. Inspecting actual generations showed the real mechanism: the
non-instruction-tuned dense base model does not summarise at all given a bare
"Dialogue: ...\n\nSummary:" prompt -- it continues the dialogue, emits
meta-commentary ("This is a dialogue between two friends"), or emits nothing
but newlines. The pruned model does produce real summaries. So the sweep's
dense baseline measures prompt-following failure, not summarisation quality,
and the whole gap is task adaptation -- exactly B5's hypothesis, and exactly
what F25 showed a plain dense LoRA fine-tune achieves on CNN/DailyMail
WITHOUT any sparsity. This run is the matched control that settles it here.

KNOWN METHODOLOGICAL FLAW, INHERITED, NOT SILENTLY FIXED:
`build_training_batch` does not append EOS to the target span, so nothing in
training ever teaches the model to stop. Every generation runs to
--max_new_tokens and pads with degenerate repetition ("Amanda will text
him." x7) or raw newlines. ROUGE-L F-measure against a ~30-token reference
then mixes "did it summarise" with "how much junk followed", which is a large
part of why the sweep's ROUGE column was non-monotonic in sparsity while its
perplexity column was perfectly monotonic. `--append_eos` is exposed as an
opt-in fix (appends eos_token_id to the supervised target so LoRA learns to
stop, and generation then terminates naturally). It defaults to FALSE to stay
faithful to the CNN/DailyMail convention this is meant to be compared
against. This is a real decision, not an oversight -- flagged for the user.

CHOICES MADE HERE, flagged rather than inherited silently:
  - --max_steps default 345, NOT CNN/DailyMail's 1122. At global effective
    batch 128, SAMSum's 14,731-example train split is 115.1 steps/epoch, so
    345 steps = 3.00 epochs. CNN/DailyMail's own 1122 steps was chosen as
    "50% of a 287,113-example split"; applying that RULE here would give 57
    steps (far too few to adapt a 7B model), while applying that STEP COUNT
    here would give 9.75 epochs (heavy LoRA overfitting on 14.7k examples).
    3 epochs is the standard SAMSum fine-tuning budget in the literature.
  - --rouge_eval_examples default 819 = the ENTIRE test split, not a 300
    prefix. F26 found the 300-example prefix was biased low by ~3pp for
    PRUNED models specifically while being accurate for dense; SAMSum's test
    split is small enough (819) to evaluate whole, removing the sampling
    question rather than managing it.
  - Per-example ROUGE-L scores are DUMPED (per_example_rougeL.json) so
    compare_pruned_vs_dense_lora.py can compute its paired bootstrap CI.
    F25's headline (-0.06pp) had no confidence interval precisely because
    this dump was not saved; that gap does not need repeating.
  - Dense baseline is measured LIVE by the dense arm's own pre-LoRA eval,
    never read from a pruner checkpoint's cached numbers (same lesson as the
    CNN/DailyMail run: cached ROUGE was measured at a different sample size
    and is not the same statistic).

LAUNCH (both arms, sequentially, on one 8-GPU box):
  torchrun --standalone --nproc_per_node=8 lora_samsum_ddp.py \\
      --arm pruned --pruner_ckpt experiments/latest/llama2_7b_samsum/lambda_1.6/pruner.pt \\
      --hf_token $HF_TOKEN --out_dir experiments/latest/llama2_7b_samsum_lora_b5
  torchrun --standalone --nproc_per_node=8 lora_samsum_ddp.py \\
      --arm dense --hf_token $HF_TOKEN \\
      --out_dir experiments/latest/llama2_7b_samsum_lora_b5
"""
import os
import sys
import json
import time
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
from train_pruner_llama2_7b_samsum import (
    load_llama2_7b, get_mlp_weights, get_loaders, sample_examples,
    build_training_batch, autocast_ctx, _rouge_scores_to_floats, Pruner,
    N_LAYERS, N_INTER, LAYER_SHAPE, DIALOGUE_PREFIX, SUMMARY_PREFIX,
)
# Dataset-agnostic helpers -- imported, not reimplemented, so this script and
# the CNN/DailyMail pair cannot drift apart on surgery/optimizer/plot logic.
from prune_and_lora_cnndailymail import (
    physically_prune_model, build_optimizer_and_scheduler, plot_finetune_run,
)
from prune_and_lora_cnndailymail_ddp import (
    setup_distributed, rank0_first_call, log, wrap_with_lora_ddp,
)

OUT_ROOT = "experiments/latest/llama2_7b_samsum_lora_b5"


# ─────────────────────────────────────────────────────────────────────────────
# Gate reconstruction -- uses the SAMSum module's Pruner/get_mlp_weights, NOT
# the CNN/DailyMail one's (identical code, but importing the right one keeps
# the provenance honest if either ever changes).
# ─────────────────────────────────────────────────────────────────────────────

def load_pruner_final_gates(model, pruner_ckpt_path, device):
    ckpt = torch.load(pruner_ckpt_path, map_location=device, weights_only=False)
    pruner = Pruner([LAYER_SHAPE] * N_LAYERS, embed_dim=ckpt["embed_dim"],
                    lstm_hidden=ckpt["lstm_hidden"]).to(device)
    pruner.load_state_dict(ckpt["pruner_state_dict"])
    pruner.eval()
    with torch.no_grad():
        gates = pruner(get_mlp_weights(model))
    del pruner
    torch.cuda.empty_cache()
    return gates, ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Batch construction -- SAMSum fields, with the optional EOS fix.
# ─────────────────────────────────────────────────────────────────────────────

def build_batch(examples, tokenizer, max_dialogue_chars, max_summary_tokens, append_eos):
    """Wraps the pruner script's build_training_batch. With append_eos, the
    EOS token is appended to the supervised target span so LoRA learns to
    stop -- see module docstring's KNOWN METHODOLOGICAL FLAW section."""
    if not append_eos:
        return build_training_batch(examples, tokenizer, max_dialogue_chars, max_summary_tokens)
    rows = []
    for ex in examples:
        dialogue = ex["dialogue"][:max_dialogue_chars]
        ctx_text = DIALOGUE_PREFIX + dialogue + SUMMARY_PREFIX
        ctx_ids = tokenizer(ctx_text)["input_ids"]
        full_ids = tokenizer(ctx_text + ex["summary"])["input_ids"]
        if len(full_ids) - len(ctx_ids) > max_summary_tokens:
            full_ids = full_ids[:len(ctx_ids) + max_summary_tokens]
        full_ids = full_ids + [tokenizer.eos_token_id]
        rows.append((full_ids, len(ctx_ids)))
    max_len = max(len(ids) for ids, _ in rows)
    B = len(rows)
    input_ids = torch.full((B, max_len), tokenizer.pad_token_id, dtype=torch.long)
    attn = torch.zeros((B, max_len), dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)
    for i, (ids, ctx_len) in enumerate(rows):
        L = len(ids)
        t = torch.tensor(ids, dtype=torch.long)
        input_ids[i, :L] = t
        attn[i, :L] = 1
        labels[i, ctx_len:L] = t[ctx_len:]
    return input_ids, attn, labels


def lora_forward_backward(model, tokenizer, batch_examples, device,
                          max_dialogue_chars, max_summary_tokens, grad_accum_steps, append_eos):
    input_ids, attn, labels = build_batch(batch_examples, tokenizer, max_dialogue_chars,
                                          max_summary_tokens, append_eos)
    input_ids, attn, labels = input_ids.to(device), attn.to(device), labels.to(device)
    with autocast_ctx(device):
        loss = model(input_ids, attention_mask=attn, labels=labels).loss
    (loss / grad_accum_steps).backward()
    return loss.item()


# ─────────────────────────────────────────────────────────────────────────────
# Distributed eval -- shard across ranks, reduce to the single-GPU-equivalent
# aggregate. No gating anywhere: the pruned arm has already had physical
# surgery, the dense arm has nothing to prune.
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_ce_ddp(model, examples, device, tokenizer, rank, world_size, desc="eval",
                    batch_size=8, max_dialogue_chars=1600, max_summary_tokens=96,
                    append_eos=False):
    from tqdm import tqdm
    shard = examples[rank::world_size]
    total_nll = torch.zeros(1, device=device)
    total_tokens = torch.zeros(1, device=device)
    for i in tqdm(range(0, len(shard), batch_size), desc=desc, unit="batch",
                  leave=False, dynamic_ncols=True, disable=(rank != 0)):
        batch = shard[i:i + batch_size]
        input_ids, attn, labels = build_batch(batch, tokenizer, max_dialogue_chars,
                                              max_summary_tokens, append_eos)
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
                       batch_size=4, max_new_tokens=96, num_beams=1, max_dialogue_chars=1600):
    """Returns (aggregate_rouge_dict, per_example_rougeL_list, mean_generated_tokens).

    The per-example list is what compare_pruned_vs_dense_lora.py needs for its
    paired bootstrap CI -- F25's headline lacked one purely because this was
    never dumped. mean_generated_tokens is recorded because the SAMSum
    generation-length diagnostic showed every arm saturating max_new_tokens;
    tracking it makes any future length shift visible instead of invisible.
    """
    import evaluate as hf_evaluate
    from tqdm import tqdm

    shard = examples[rank::world_size]
    gen_max_length = max_dialogue_chars // 2 + 64
    prev_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    predictions, references, gen_tok_counts = [], [], []
    try:
        for i in tqdm(range(0, len(shard), batch_size), desc=desc, unit="batch",
                      leave=False, dynamic_ncols=True, disable=(rank != 0)):
            batch = shard[i:i + batch_size]
            prompts = [DIALOGUE_PREFIX + ex["dialogue"][:max_dialogue_chars] + SUMMARY_PREFIX
                       for ex in batch]
            enc = tokenizer(prompts, return_tensors="pt", padding=True,
                            truncation=True, max_length=gen_max_length).to(device)
            with autocast_ctx(device):
                gen_ids = model.generate(**enc, max_new_tokens=max_new_tokens,
                                         do_sample=False, num_beams=num_beams,
                                         pad_token_id=tokenizer.pad_token_id)
            gen_only = gen_ids[:, enc["input_ids"].shape[1]:]
            for row in gen_only:
                gen_tok_counts.append(int((row != tokenizer.pad_token_id).sum().item()))
            predictions.extend(tokenizer.batch_decode(gen_only, skip_special_tokens=True))
            references.extend(ex["summary"] for ex in batch)
    finally:
        tokenizer.padding_side = prev_padding_side

    # Gather (index, pred, ref, ntok) so rank 0 can restore the ORIGINAL example
    # order -- the shard stride examples[rank::world_size] interleaves ranks, and
    # the per-example dumps of the two arms must be aligned index-for-index for a
    # PAIRED bootstrap to be valid.
    my_indices = list(range(rank, len(examples), world_size))
    payload = list(zip(my_indices, predictions, references, gen_tok_counts))
    gathered = [None] * world_size
    dist.all_gather_object(gathered, payload)

    holder = [None]
    if rank == 0:
        flat = [item for shard_items in gathered for item in shard_items]
        flat.sort(key=lambda t: t[0])
        _, preds, refs, ntoks = zip(*flat)
        rouge = hf_evaluate.load("rouge")
        agg = _rouge_scores_to_floats(rouge.compute(predictions=list(preds), references=list(refs)))
        per_ex = rouge.compute(predictions=list(preds), references=list(refs),
                               use_aggregator=False)["rougeL"]
        holder[0] = (agg, [float(x) for x in per_ex], float(np.mean(ntoks)))
    dist.broadcast_object_list(holder, src=0)
    return holder[0]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", type=str, choices=["pruned", "dense"], required=True,
                    help="The ONLY thing that differs between the two arms: whether physical "
                         "surgery runs before LoRA wrapping.")
    ap.add_argument("--pruner_ckpt", type=str, default=None,
                    help="Required for --arm pruned. Path to a pruner.pt from "
                         "train_pruner_llama2_7b_samsum.py.")
    ap.add_argument("--lora_target", type=str, choices=["mlp", "mlp_attn"], default="mlp_attn")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-4,
                    help="BASE lr. Actual optimizer lr = --lr * world_size (linear scaling).")
    ap.add_argument("--lr_schedule", type=str, choices=["cosine", "linear", "constant"], default="cosine")
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_accum_steps", type=int, default=4)
    ap.add_argument("--gradient_checkpointing", action="store_true", default=True)
    ap.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    ap.add_argument("--max_steps", type=int, default=345,
                    help="OPTIMIZER steps. 345 = 3.00 epochs over SAMSum's 14,731-example train "
                         "split at global effective batch 128 (115.1 steps/epoch). NOT "
                         "CNN/DailyMail's 1122 -- see module docstring for why neither its rule "
                         "nor its step count transfers.")
    ap.add_argument("--eval_every", type=int, default=115,
                    help="Default = one epoch, so val CE is logged at each epoch boundary.")
    ap.add_argument("--batch_size", type=int, default=4, help="PER-RANK micro-batch size.")
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--gen_batch_size", type=int, default=4)
    ap.add_argument("--rouge_eval_examples", type=int, default=819,
                    help="Default = the ENTIRE SAMSum test split (819), not a prefix sample -- "
                         "see module docstring (F26 sampling-bias lesson).")
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--max_dialogue_chars", type=int, default=1600)
    ap.add_argument("--max_summary_tokens", type=int, default=96)
    ap.add_argument("--append_eos", action="store_true", default=False,
                    help="Append EOS to the supervised target so LoRA learns to STOP. Defaults "
                         "OFF to stay faithful to the CNN/DailyMail convention. See module "
                         "docstring's KNOWN METHODOLOGICAL FLAW section -- this is a real "
                         "decision, turn it on deliberately and for BOTH arms or neither.")
    ap.add_argument("--gap_eval_examples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hf_token", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default=OUT_ROOT)
    ap.add_argument("--sanity_check", action="store_true",
                    help="Surgery (if pruned) + pct cross-check + ONE LoRA step, then exit.")
    args = ap.parse_args()

    if args.arm == "pruned" and not args.pruner_ckpt:
        sys.exit("ERROR: --arm pruned requires --pruner_ckpt.")

    rank, world_size, local_rank, device = setup_distributed()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    base_lr = args.lr
    args.lr = base_lr * world_size
    effective_batch = args.batch_size * args.grad_accum_steps * world_size
    log(rank, f"ARM={args.arm} | world_size={world_size} | base_lr={base_lr:.2e} -> "
              f"scaled_lr={args.lr:.2e} | global effective batch={effective_batch}")

    log(rank, "Loading Llama-2-7B ...")
    model = rank0_first_call(rank, load_llama2_7b, device, hf_token=args.hf_token)
    log(rank, "Loading SAMSum ...")
    tokenizer, _unused, train_examples, val_examples, test_examples = rank0_first_call(
        rank, get_loaders, args.batch_size, hf_token=args.hf_token)
    log(rank, f"Data: train={len(train_examples):,} val={len(val_examples):,} test={len(test_examples):,}")

    examples_seen = args.max_steps * effective_batch
    log(rank, f"Train-split coverage: {args.max_steps} steps x {effective_batch} ex/step = "
              f"{examples_seen:,} examples = {examples_seen / len(train_examples):.2f} epochs "
              f"over the {len(train_examples):,}-example train split")
    if args.append_eos:
        log(rank, "  --append_eos IS ON: EOS appended to supervised targets (both arms must match).")

    target_modules = (["gate_proj", "up_proj", "down_proj"] if args.lora_target == "mlp"
                      else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    # ── surgery (pruned arm only) ──
    ckpt, per_layer_kept, actual_pct_pruned, lam = None, None, 0.0, None
    if args.arm == "pruned":
        log(rank, f"\nLoading pruner checkpoint: {args.pruner_ckpt}")
        gates, ckpt = load_pruner_final_gates(model, args.pruner_ckpt, device)
        lam = ckpt.get("lambda")
        ckpt_pct = 100 * (1 - sum(g.mean().item() for g in gates) / len(gates))
        log(rank, f"  reconstructed gate: {ckpt_pct:.2f}% pruned "
                  f"(lambda={lam}, converged={ckpt.get('converged')})")
        per_layer_kept = physically_prune_model(model, gates)
        actual_pct_pruned = 100 * (1 - sum(per_layer_kept) / (N_LAYERS * N_INTER))
        diff = abs(actual_pct_pruned - ckpt_pct)
        log(rank, f"  surgery result: {actual_pct_pruned:.4f}% pruned (diff vs gate "
                  f"reconstruct {diff:.4f}pp) | min kept/layer={min(per_layer_kept)}")
        if diff > 1e-2:
            log(rank, "  WARNING: surgery/gate mismatch -- check keep_idx indexing.")
        del gates
        torch.cuda.empty_cache()
    n_params = sum(p.numel() for p in model.parameters())
    log(rank, f"  model params ({args.arm} arm): {n_params:,}")

    ce_kw = dict(batch_size=args.eval_batch_size, max_dialogue_chars=args.max_dialogue_chars,
                 max_summary_tokens=args.max_summary_tokens, append_eos=args.append_eos)
    rouge_kw = dict(batch_size=args.gen_batch_size, max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams, max_dialogue_chars=args.max_dialogue_chars)
    rouge_sample = sample_examples(test_examples, args.rouge_eval_examples)

    if args.sanity_check:
        log(rank, "\n" + "=" * 70 + f"\nSANITY CHECK -- {args.arm} arm, 1 LoRA step\n" + "=" * 70)
        peft_model = wrap_with_lora_ddp(model, args, target_modules)
        if rank == 0:
            peft_model.print_trainable_parameters()
        ddp_model = DDP(peft_model, device_ids=[local_rank], output_device=local_rank)
        tparams = [p for p in peft_model.parameters() if p.requires_grad]
        opt, sched = build_optimizer_and_scheduler(tparams, args)
        ddp_model.train(); opt.zero_grad()
        sampler = DistributedSampler(train_examples, num_replicas=world_size, rank=rank,
                                     shuffle=True, seed=args.seed)
        batch = [train_examples[i] for i in list(sampler)[:args.batch_size]]
        loss = lora_forward_backward(ddp_model, tokenizer, batch, device, args.max_dialogue_chars,
                                     args.max_summary_tokens, args.grad_accum_steps, args.append_eos)
        torch.nn.utils.clip_grad_norm_(tparams, max_norm=1.0)
        opt.step(); sched.step()
        log(rank, f"  1 LoRA micro-step loss: {loss:.4f} (finite, no crash -> pipeline OK)")
        dist.barrier(); dist.destroy_process_group()
        return

    # ── pre-LoRA eval. For the DENSE arm this IS the dense zero-shot baseline,
    #    measured live at this run's own sample size (never read from a pruner
    #    checkpoint's cache -- different sample size, different statistic). ──
    log(rank, f"\nEvaluating {args.arm} model BEFORE LoRA ...")
    model.eval()
    pre_ce = evaluate_ce_ddp(model, test_examples, device, tokenizer, rank, world_size,
                             desc="pre-LoRA test CE", **ce_kw)
    pre_ppl = float(np.exp(pre_ce))
    pre_rouge, pre_per_ex, pre_gen_tok = evaluate_rouge_ddp(
        model, rouge_sample, device, tokenizer, rank, world_size, desc="pre-LoRA ROUGE", **rouge_kw)
    log(rank, f"  pre-LoRA ppl {pre_ppl:.3f} | R-L {pre_rouge['rougeL']*100:.2f}% | "
              f"mean gen tokens {pre_gen_tok:.1f}/{args.max_new_tokens}")
    if args.arm == "pruned" and ckpt is not None:
        log(rank, f"  (gated-eval checkpoint said ppl {ckpt['test_ppl_pruned']:.3f}; ROUGE not "
                  f"comparable -- checkpoint cached at 300 examples, this run uses "
                  f"{len(rouge_sample)})")

    # ── LoRA fine-tune ──
    log(rank, f"\nWrapping with LoRA (target={target_modules}, r={args.lora_r}) ...")
    peft_model = wrap_with_lora_ddp(model, args, target_modules)
    if rank == 0:
        peft_model.print_trainable_parameters()
    ddp_model = DDP(peft_model, device_ids=[local_rank], output_device=local_rank)
    tparams = [p for p in peft_model.parameters() if p.requires_grad]
    opt, sched = build_optimizer_and_scheduler(tparams, args)

    run_dir = os.path.join(args.out_dir, f"{args.arm}_lora" + (f"_lambda_{lam}" if lam is not None else ""))
    if rank == 0:
        os.makedirs(run_dir, exist_ok=True)

    gap_val_sample = sample_examples(val_examples, args.gap_eval_examples)
    sampler = DistributedSampler(train_examples, num_replicas=world_size, rank=rank,
                                 shuffle=True, seed=args.seed)
    train_loader = DataLoader(train_examples, batch_size=args.batch_size, sampler=sampler,
                              collate_fn=lambda b: b)
    train_loss_hist, val_points = [], []
    epoch = 0
    sampler.set_epoch(epoch)
    loader_iter = iter(train_loader)
    ddp_model.train()
    t0 = time.time()
    from tqdm import tqdm
    pbar = tqdm(total=args.max_steps, desc=f"LoRA {args.arm}", unit="step",
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
            is_last = (micro_idx == args.grad_accum_steps - 1)
            sync_ctx = contextlib.nullcontext() if is_last else ddp_model.no_sync()
            with sync_ctx:
                micro_losses.append(lora_forward_backward(
                    ddp_model, tokenizer, batch, device, args.max_dialogue_chars,
                    args.max_summary_tokens, args.grad_accum_steps, args.append_eos))
        torch.nn.utils.clip_grad_norm_(tparams, max_norm=1.0)
        opt.step(); sched.step()
        loss = sum(micro_losses) / len(micro_losses)
        train_loss_hist.append(loss)
        pbar.set_postfix(loss=f"{loss:.3f}", lr=f"{sched.get_last_lr()[0]:.2e}", refresh=False)
        pbar.update(1)
        if opt_step % args.eval_every == 0 or opt_step == args.max_steps:
            ddp_model.eval()
            val_ce = evaluate_ce_ddp(peft_model, gap_val_sample, device, tokenizer, rank,
                                     world_size, desc=f"val CE @ {opt_step}", **ce_kw)
            val_points.append((opt_step, val_ce))
            if rank == 0:
                tqdm.write(f"  step {opt_step:>5} | train loss {loss:.4f} | val CE {val_ce:.4f}")
            ddp_model.train()
    pbar.close()
    total_time = time.time() - t0

    # ── post-LoRA eval ──
    log(rank, f"\nFinal evaluation ({args.arm} + LoRA) ...")
    ddp_model.eval()
    post_ce = evaluate_ce_ddp(peft_model, test_examples, device, tokenizer, rank, world_size,
                              desc="post-LoRA test CE", **ce_kw)
    post_ppl = float(np.exp(post_ce))
    post_rouge, post_per_ex, post_gen_tok = evaluate_rouge_ddp(
        peft_model, rouge_sample, device, tokenizer, rank, world_size,
        desc="post-LoRA ROUGE", **rouge_kw)
    log(rank, f"  -> ppl {pre_ppl:.3f} -> {post_ppl:.3f} | R-L "
              f"{pre_rouge['rougeL']*100:.2f}% -> {post_rouge['rougeL']*100:.2f}% | "
              f"mean gen tokens {pre_gen_tok:.1f} -> {post_gen_tok:.1f}")

    if rank == 0:
        plot_finetune_run(train_loss_hist, val_points, os.path.join(run_dir, "plot.png"),
                          title=(f"Llama-2-7B — SAMSum — {args.arm} arm"
                                 + (f" (λ={lam}, {actual_pct_pruned:.1f}% pruned)" if lam is not None else "")
                                 + f" + LoRA (DDP x{world_size})"))
        lines = [
            f"Llama-2-7B — SAMSum — {args.arm.upper()} arm + LoRA (DDP x{world_size})",
            (f"pruner ckpt : {args.pruner_ckpt} (lambda={lam})" if args.arm == "pruned"
             else "pruner ckpt : n/a (dense control -- no surgery)"),
            f"lora: target={args.lora_target} r={args.lora_r} alpha={args.lora_alpha} "
            f"dropout={args.lora_dropout} base_lr={base_lr} scaled_lr={args.lr} "
            f"schedule={args.lr_schedule} warmup={args.warmup_ratio} wd={args.weight_decay}",
            f"steps={args.max_steps} grad_accum={args.grad_accum_steps} world_size={world_size} "
            f"eff_batch={effective_batch} append_eos={args.append_eos} time={total_time:.1f}s",
            f"coverage: {examples_seen:,} examples = {examples_seen/len(train_examples):.2f} epochs",
            "-" * 60,
            f"% FFN pruned : {actual_pct_pruned:.2f}%" if args.arm == "pruned" else "% FFN pruned : 0 (dense)",
            f"model params : {n_params:,}",
            "-" * 60,
            f"samsum test ({len(test_examples)} ex, full split, target-only CE):",
            f"  pre-LoRA  ppl : {pre_ppl:.3f}",
            f"  post-LoRA ppl : {post_ppl:.3f}",
            "-" * 60,
            f"ROUGE via generation ({len(rouge_sample)} ex = FULL test split, greedy):",
            f"  pre-LoRA  r1/r2/rL/rLsum : {pre_rouge['rouge1']*100:.2f}/{pre_rouge['rouge2']*100:.2f}/"
            f"{pre_rouge['rougeL']*100:.2f}/{pre_rouge['rougeLsum']*100:.2f}   "
            f"(mean gen tok {pre_gen_tok:.1f}/{args.max_new_tokens})",
            f"  post-LoRA r1/r2/rL/rLsum : {post_rouge['rouge1']*100:.2f}/{post_rouge['rouge2']*100:.2f}/"
            f"{post_rouge['rougeL']*100:.2f}/{post_rouge['rougeLsum']*100:.2f}   "
            f"(mean gen tok {post_gen_tok:.1f}/{args.max_new_tokens})",
        ]
        with open(os.path.join(run_dir, "summary.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        print("\n" + "\n".join(lines), flush=True)

        peft_model.save_pretrained(run_dir)
        with open(os.path.join(run_dir, "per_example_rougeL.json"), "w") as f:
            json.dump(post_per_ex, f)
        with open(os.path.join(run_dir, "meta.json"), "w") as f:
            json.dump({
                "arm": args.arm,
                "dataset": "samsum",
                "pruner_ckpt": args.pruner_ckpt, "lambda": lam,
                "per_layer_kept": per_layer_kept, "pct_pruned": actual_pct_pruned,
                "pre_lora_ppl": pre_ppl, "post_lora_ppl": post_ppl,
                "pre_lora_rouge": pre_rouge, "post_lora_rouge": post_rouge,
                "pre_lora_mean_gen_tokens": pre_gen_tok,
                "post_lora_mean_gen_tokens": post_gen_tok,
                "append_eos": args.append_eos,
                "optimizer_steps": args.max_steps, "grad_accum_steps": args.grad_accum_steps,
                "world_size": world_size, "global_effective_batch": effective_batch,
                "train_split_size": len(train_examples), "examples_seen": examples_seen,
                "train_coverage_epochs": examples_seen / len(train_examples),
                "rouge_eval_examples": len(rouge_sample),
                "test_split_size": len(test_examples), "seed": args.seed,
                "n_params": n_params, "n_params_pruned": n_params,
                "total_time": total_time, "lora_target": args.lora_target, "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha, "lora_dropout": args.lora_dropout,
                "base_lr": base_lr, "scaled_lr": args.lr, "lr_schedule": args.lr_schedule,
                "warmup_ratio": args.warmup_ratio, "weight_decay": args.weight_decay,
                "gradient_checkpointing": args.gradient_checkpointing,
            }, f, indent=2)
        print(f"\n[saved] {run_dir}/", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
