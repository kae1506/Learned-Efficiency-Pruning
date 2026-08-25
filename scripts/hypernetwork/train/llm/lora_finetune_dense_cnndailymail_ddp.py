"""
Data-parallel (DDP) variant of lora_finetune_dense_cnndailymail.py -- LoRA
fine-tuning of the ORIGINAL (unpruned) Llama-2-7B on CNN/DailyMail,
replicated across all GPUs on the node instead of using one.

WHY THIS SCRIPT EXISTS: prune_and_lora_cnndailymail_ddp.py already runs the
pruned arm of the B5 comparison at DDP scale, but its dense counterpart
(lora_finetune_dense_cnndailymail.py) is SINGLE-GPU only. Running the two
arms at different world_size means different global effective batch,
different scaled LR, and -- at a fixed --max_steps -- a world_size-fold
difference in how many CNN/DailyMail training examples each arm ever sees.
That is not an apples-to-apples control, and B5's load-bearing open
question (ideas.md: "does a plain dense fine-tune on the same data match
or beat the pruned model?") cannot be answered by comparing arms trained
on different amounts of data. This script is the dense arm at matched
world_size, so the ONLY variable between the two runs is whether physical
pruning happened first.

MATCHED EXACTLY to prune_and_lora_cnndailymail_ddp.py: every LoRA
hyperparameter, the linear-LR-scaling rule, no_sync() gradient
accumulation, non-reentrant gradient checkpointing, the sharded-then-
reduced eval (CE via SUM-all-reduce of NLL+tokens; ROUGE via
all_gather_object of raw pairs then a single rank-0 rouge.compute()), the
rank0-downloads-first barrier, and the rank-0-only artifact writes. Those
helpers are IMPORTED from that script rather than reimplemented -- one
definition, so the two arms cannot silently drift apart. Don't change one
script's defaults without mirroring the change in the other, or the
comparison stops being apples-to-apples.

DIFFERENCES FROM THE PRUNED ARM (all and only the pruning-specific parts):
  - No --pruner_ckpt, no gate reconstruction, no physical surgery, and no
    surgery-vs-checkpoint correctness cross-check (nothing to cross-check
    -- the weights are untouched).
  - The dense baseline is MEASURED LIVE here (pre-LoRA eval on the full
    test split + the ROUGE sample) rather than read from a pruner
    checkpoint's cached `test_ppl_orig`/`rouge_orig`. This matters once
    --rouge_eval_examples is raised above the 300 the pruning run cached:
    a cached 300-example ROUGE number is not comparable to a 3000-example
    one, but a live measurement on the same enlarged sample is. Because
    sample_examples() is a deterministic prefix slice (examples[:n], no
    RNG), both arms evaluate ROUGE on the byte-identical example set at
    equal --rouge_eval_examples.

*** UNTESTED at full scale. *** Same caveat as the sibling scripts. Run
--sanity_check on a SMALL --nproc_per_node (e.g. 2) before scaling to all 8.

LAUNCH (torchrun, not `python3` directly -- setup_distributed() exits
immediately if it doesn't see torchrun's env vars):
    torchrun --standalone --nproc_per_node=8 lora_finetune_dense_cnndailymail_ddp.py \\
        --hf_token <token> --max_steps 2244 \\
        --out_dir experiments/latest/llama2_7b_cnndailymail_lora_dense_ddp
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
from train_pruner_llama2_7b_cnndailymail import (
    load_llama2_7b, get_loaders, sample_examples,
)
from prune_and_lora_cnndailymail import (
    lora_forward_backward, build_optimizer_and_scheduler, plot_finetune_run,
)
from prune_and_lora_cnndailymail_ddp import (
    setup_distributed, rank0_first_call, log, wrap_with_lora_ddp,
    evaluate_ce_ddp, evaluate_rouge_ddp,
)

OUT_ROOT = "experiments/latest/llama2_7b_cnndailymail_lora_dense_ddp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora_target", type=str, choices=["mlp", "mlp_attn"], default="mlp_attn")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-4,
                    help="BASE lr (per effective-batch-16), same convention as the pruned arm. "
                         "Actual optimizer lr = --lr * world_size (linear scaling rule).")
    ap.add_argument("--lr_schedule", type=str, choices=["cosine", "linear", "constant"], default="cosine")
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_accum_steps", type=int, default=4,
                    help="PER-RANK micro-batches per optimizer step. Global effective batch = "
                         "batch_size * grad_accum_steps * world_size.")
    ap.add_argument("--gradient_checkpointing", action="store_true", default=True)
    ap.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    ap.add_argument("--max_steps", type=int, default=500,
                    help="OPTIMIZER steps. MUST match the pruned arm's --max_steps for the "
                         "comparison to hold examples-seen constant across the two arms.")
    ap.add_argument("--eval_every", type=int, default=125)
    ap.add_argument("--batch_size", type=int, default=4, help="PER-RANK micro-batch size.")
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--gen_batch_size", type=int, default=4)
    ap.add_argument("--rouge_eval_examples", type=int, default=300,
                    help="Prefix slice of the test split scored by generation. MUST match the "
                         "pruned arm's value -- sample_examples() is a deterministic prefix "
                         "slice, so equal values mean a byte-identical eval set.")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--max_article_chars", type=int, default=4000)
    ap.add_argument("--max_summary_tokens", type=int, default=128)
    ap.add_argument("--gap_eval_examples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hf_token", type=str, required=True,
                    help="HF token with ACCEPTED access to meta-llama/Llama-2-7b-hf.")
    ap.add_argument("--out_dir", type=str, default=OUT_ROOT)
    ap.add_argument("--skip_pre_lora_eval", action="store_true",
                    help="Skip the live dense (zero-shot) eval and reuse a previously measured "
                         "one. Off by default -- the live number IS the dense baseline this "
                         "whole comparison is against.")
    ap.add_argument("--sanity_check", action="store_true",
                    help="Wrap with LoRA+DDP, run ONE training step, then exit. NOT optional "
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

    log(rank, "Loading Llama-2-7B DENSE (GATE-LICENSED -- requires --hf_token) ...")
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

    target_modules = (["gate_proj", "up_proj", "down_proj"] if args.lora_target == "mlp"
                      else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    n_params_dense = sum(p.numel() for p in model.parameters())
    log(rank, f"Dense model params (per replica): {n_params_dense:,}")

    if args.sanity_check:
        if rank == 0:
            print("\n" + "=" * 70, flush=True)
            print("SANITY CHECK -- dense + LoRA + DDP, 1 training step", flush=True)
            print("=" * 70, flush=True)
        peft_model = wrap_with_lora_ddp(model, args, target_modules)
        if rank == 0:
            peft_model.print_trainable_parameters()
        ddp_model = DDP(peft_model, device_ids=[local_rank], output_device=local_rank)
        trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
        opt, sched = build_optimizer_and_scheduler(trainable_params, args)
        ddp_model.train()
        opt.zero_grad()
        sampler = DistributedSampler(train_examples, num_replicas=world_size, rank=rank,
                                     shuffle=True, seed=args.seed)
        my_indices = list(sampler)[:args.batch_size]
        batch = [train_examples[i] for i in my_indices]
        t_step = time.time()
        loss = lora_forward_backward(ddp_model, tokenizer, batch, device, args.max_article_chars,
                                     args.max_summary_tokens, args.grad_accum_steps)
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        opt.step(); sched.step()
        torch.cuda.synchronize()
        log(rank, f"  rank {rank} 1 LoRA micro-step loss : {loss:.4f} "
                  f"({time.time() - t_step:.2f}s incl. warmup -- finite, no crash -> pipeline OK)")
        dist.barrier()
        if rank == 0:
            print("=" * 70, flush=True)
        dist.destroy_process_group()
        return

    ce_kw = dict(batch_size=args.eval_batch_size, max_article_chars=args.max_article_chars,
                 max_summary_tokens=args.max_summary_tokens)
    rouge_kw = dict(batch_size=args.gen_batch_size, max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams, max_article_chars=args.max_article_chars)
    rouge_sample = sample_examples(test_examples, args.rouge_eval_examples)   # deterministic prefix, identical per rank AND per arm

    # ── 1) dense baseline, MEASURED LIVE (this is the number the pruned arm has to beat) ──
    pre_ppl, pre_rouge = None, None
    if not args.skip_pre_lora_eval:
        log(rank, "\nEvaluating DENSE model BEFORE any LoRA training (the zero-shot dense baseline) ...")
        model.eval()
        pre_test_ce = evaluate_ce_ddp(model, test_examples, device, tokenizer, rank, world_size,
                                      desc="pre-LoRA test CE", **ce_kw)
        pre_ppl = float(np.exp(pre_test_ce))
        pre_rouge = evaluate_rouge_ddp(model, rouge_sample, device, tokenizer, rank, world_size,
                                       desc="pre-LoRA ROUGE", **rouge_kw)
        log(rank, f"  dense ppl : {pre_ppl:.3f} | dense R-L : {pre_rouge['rougeL']*100:.2f}% "
                  f"(rougeLsum {pre_rouge['rougeLsum']*100:.2f}%)")

    # ── 2) LoRA fine-tune, DDP-wrapped ──
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

    run_dir = args.out_dir
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
    pbar = tqdm(total=args.max_steps, desc="Dense-LoRA fine-tune (optimizer steps)", unit="step",
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

    # ── 3) final eval POST-LoRA ──
    log(rank, "\nFinal evaluation (dense + LoRA fine-tuned) ...")
    ddp_model.eval()
    post_test_ce = evaluate_ce_ddp(peft_model, test_examples, device, tokenizer, rank, world_size,
                                   desc="post-LoRA test CE", **ce_kw)
    post_ppl = float(np.exp(post_test_ce))
    post_rouge = evaluate_rouge_ddp(peft_model, rouge_sample, device, tokenizer, rank, world_size,
                                    desc="post-LoRA ROUGE", **rouge_kw)

    if pre_ppl is not None:
        log(rank, f"  -> dense ppl {pre_ppl:.3f} -> post-LoRA ppl {post_ppl:.3f}")
        log(rank, f"  -> dense R-L {pre_rouge['rougeL']*100:.2f}% -> post-LoRA R-L "
                  f"{post_rouge['rougeL']*100:.2f}%")
    else:
        log(rank, f"  -> post-LoRA ppl {post_ppl:.3f} | post-LoRA R-L {post_rouge['rougeL']*100:.2f}%")

    if rank == 0:
        plot_finetune_run(train_loss_hist, val_points, os.path.join(run_dir, "plot.png"),
                          title=(f"Llama-2-7B — DENSE (unpruned) + LoRA (DDP x{world_size}) — "
                                 f"CNN/DailyMail — {args.max_steps} steps"))

        pre_ppl_s = f"{pre_ppl:.3f}" if pre_ppl is not None else "not measured (--skip_pre_lora_eval)"
        pre_rouge_s = (f"{pre_rouge['rouge1']*100:.2f}/{pre_rouge['rouge2']*100:.2f}/"
                       f"{pre_rouge['rougeL']*100:.2f}/{pre_rouge['rougeLsum']*100:.2f}"
                       if pre_rouge is not None else "not measured (--skip_pre_lora_eval)")
        lines = [
            f"Llama-2-7B — DENSE (no pruning) + LoRA (DDP x{world_size}) — CNN/DailyMail",
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
            f"model params (dense, no pruning) : {n_params_dense:,}",
            "-" * 60,
            f"cnn/dailymail test set ({len(test_examples)} examples, full split, target-only CE):",
            f"  pre-LoRA (dense, zero-shot)   ppl : {pre_ppl_s}",
            f"  post-LoRA (dense + LoRA)      ppl : {post_ppl:.3f}",
            "-" * 60,
            f"ROUGE via generation ({len(rouge_sample)}-example prefix sample of test, "
            f"greedy={args.num_beams == 1}, num_beams={args.num_beams}):",
            f"  pre-LoRA (dense, zero-shot)  rouge1/rouge2/rougeL/rougeLsum : {pre_rouge_s}",
            f"  post-LoRA (dense + LoRA)     rouge1/rouge2/rougeL/rougeLsum : "
            f"{post_rouge['rouge1']*100:.2f}/{post_rouge['rouge2']*100:.2f}/"
            f"{post_rouge['rougeL']*100:.2f}/{post_rouge['rougeLsum']*100:.2f}",
        ]
        with open(os.path.join(run_dir, "summary.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        print("\n" + "\n".join(lines), flush=True)

        peft_model.save_pretrained(run_dir)   # peft adapter weights only
        with open(os.path.join(run_dir, "meta.json"), "w") as f:
            json.dump({
                "arm": "dense",
                "n_params_dense": n_params_dense,
                "pre_lora_ppl": pre_ppl, "post_lora_ppl": post_ppl,
                "pre_lora_rouge": pre_rouge, "post_lora_rouge": post_rouge,
                "optimizer_steps": args.max_steps, "grad_accum_steps": args.grad_accum_steps,
                "world_size": world_size, "global_effective_batch": effective_batch,
                "train_split_size": len(train_examples), "examples_seen": examples_seen,
                "train_coverage_pct": 100 * examples_seen / len(train_examples),
                "rouge_eval_examples": len(rouge_sample),
                "test_split_size": len(test_examples),
                "total_time": total_time, "lora_target": args.lora_target, "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha, "lora_dropout": args.lora_dropout,
                "base_lr": base_lr, "scaled_lr": args.lr, "lr_schedule": args.lr_schedule,
                "warmup_ratio": args.warmup_ratio, "weight_decay": args.weight_decay,
                "gradient_checkpointing": args.gradient_checkpointing, "seed": args.seed,
            }, f, indent=2)
        print(f"\n[saved] {run_dir}/ (adapter weights + summary.txt + meta.json + plot.png)", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
