"""
Dense-LoRA fine-tuning of the ORIGINAL (unpruned) Llama-2-7B on CNN/DailyMail
-- the standard post-training recipe, no pruning anywhere in this script.

WHY THIS SCRIPT EXISTS: the B5 dense-fine-tune-control direction from
diary/ideas.md, finally standalone. train_pruner_llama2_7b_cnndailymail.py
prunes via a CE-delta objective; prune_and_lora_cnndailymail.py physically
prunes THEN LoRA-fine-tunes the smaller model. Neither of those answers
the actual question this script is for: if you just take the dense
pretrained model and LoRA-fine-tune it on CNN/DailyMail the normal way --
no pruning step anywhere -- how good does it get? That's the number
"pruning on pre-training" has to beat for pruning to have added anything
on top of standard post-training, rather than post-training alone
accounting for the whole quality gain.

MATCHED EXACTLY to prune_and_lora_cnndailymail.py's defaults (LoRA
target_modules/r/alpha/dropout/lr, LR schedule + warmup, gradient
accumulation, gradient checkpointing, weight decay, max_steps, eval_every,
batch sizes, prompt template, max_article_chars/max_summary_tokens, ROUGE
sample size/num_beams) so the ONLY variable between "dense + LoRA" (this
script) and "physically-pruned + LoRA" (the sibling script) is whether
pruning happened first -- everything else in the comparison is held
constant on purpose. Don't change one script's defaults without mirroring
the change in the other, or the comparison stops being apples-to-apples.

LoRA METHODOLOGY -- same as prune_and_lora_cnndailymail.py (revised
2026-08-19 to match common Llama-2-7B LoRA fine-tuning practice --
Alpaca-LoRA/QLoRA-style recipes -- rather than an ad hoc flat-LR/no-
accumulation setup): r=16/alpha=32 (Alpaca-LoRA's own config), all-linear
target_modules (QLoRA's recommendation), cosine schedule with
warmup_ratio=0.03, grad_accum_steps=4 (effective batch 16), gradient
checkpointing on, weight_decay=0.0. See the sibling script's module
docstring for the full reasoning -- not re-derived here.

*** UNTESTED. *** No local hardware at this scale. Verified: py_compile
only. Requires `pip install peft` -- see
requirements_llama2_cnndailymail_lora_runpod.txt.

PREREQUISITE -- meta-llama/Llama-2-7b-hf is GATE-LICENSED. Pass a token
via --hf_token (CLI arg, not an environment variable -- same convention as
every sibling script in this directory).

Reuses load_llama2_7b / get_loaders / sample_examples / build_training_batch
/ evaluate_ce / evaluate_rouge / autocast_ctx / _smooth from
train_pruner_llama2_7b_cnndailymail.py, and lora_forward_backward /
wrap_with_lora / build_optimizer_and_scheduler / plot_finetune_run from
prune_and_lora_cnndailymail.py (all already fully generic -- no
pruning-specific logic in any of them) -- no logic duplicated.
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_pruner_llama2_7b_cnndailymail import (
    load_llama2_7b, get_loaders, sample_examples, evaluate_ce, evaluate_rouge,
)
from prune_and_lora_cnndailymail import (
    lora_forward_backward, wrap_with_lora, build_optimizer_and_scheduler, plot_finetune_run,
)

OUT_ROOT = "/workspace/results/llama2_7b_cnndailymail_lora_dense"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora_target", type=str, choices=["mlp", "mlp_attn"], default="mlp_attn",
                    help="Matches prune_and_lora_cnndailymail.py's confirmed default "
                         "(2026-08-19) -- mlp_attn = gate/up/down_proj + q/k/v/o_proj.")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_schedule", type=str, choices=["cosine", "linear", "constant"], default="cosine")
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_accum_steps", type=int, default=4,
                    help="Matches prune_and_lora_cnndailymail.py -- batch_size=4 x "
                         "grad_accum_steps=4 = effective batch 16.")
    ap.add_argument("--gradient_checkpointing", action="store_true", default=True)
    ap.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    ap.add_argument("--max_steps", type=int, default=500,
                    help="OPTIMIZER steps -- matches prune_and_lora_cnndailymail.py's "
                         "post-accumulation default (500*4*4=8000 examples processed).")
    ap.add_argument("--eval_every", type=int, default=125)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--gen_batch_size", type=int, default=4)
    ap.add_argument("--rouge_eval_examples", type=int, default=300)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--max_article_chars", type=int, default=4000)
    ap.add_argument("--max_summary_tokens", type=int, default=128)
    ap.add_argument("--gap_eval_examples", type=int, default=200,
                    help="Val examples sampled for the periodic CE checkpoint.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--hf_token", type=str, default=None,
                    help="HF token with ACCEPTED access to meta-llama/Llama-2-7b-hf. "
                         "CLI arg, not read from HF_TOKEN env var.")
    ap.add_argument("--out_dir", type=str, default=OUT_ROOT)
    ap.add_argument("--sanity_check", action="store_true",
                    help="Wrap with LoRA, run ONE training step, then exit. "
                         "NOT optional before a real run.")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.hf_token is None:
        print("WARNING: --hf_token not set. meta-llama/Llama-2-7b-hf is gate-licensed -- "
              "this will fail unless you're using a cached local copy or a token with "
              "accepted license access is otherwise configured.", flush=True)

    print("Loading Llama-2-7B (dense, GATE-LICENSED -- requires --hf_token) ...", flush=True)
    model = load_llama2_7b(device, hf_token=args.hf_token)

    print("Loading CNN/DailyMail ...", flush=True)
    tokenizer, train_loader, train_examples, val_examples, test_examples = get_loaders(
        args.batch_size, hf_token=args.hf_token)
    print(f"Data: train={len(train_examples):,} val={len(val_examples):,} test={len(test_examples):,}",
          flush=True)

    target_modules = (["gate_proj", "up_proj", "down_proj"] if args.lora_target == "mlp"
                      else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    if args.sanity_check:
        print("\n" + "=" * 70, flush=True)
        print("SANITY CHECK -- wrap dense model with LoRA, 1 training step", flush=True)
        print("=" * 70, flush=True)
        model = wrap_with_lora(model, args, target_modules)
        model.print_trainable_parameters()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        opt, sched = build_optimizer_and_scheduler(trainable_params, args)
        model.train()
        opt.zero_grad()
        batch = sample_examples(train_examples, args.batch_size)
        loss = lora_forward_backward(model, tokenizer, batch, device, args.max_article_chars,
                                     args.max_summary_tokens, args.grad_accum_steps)
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        opt.step(); sched.step()
        print(f"  1 LoRA micro-step loss : {loss:.4f} (finite, no crash -> pipeline OK)")
        print("=" * 70, flush=True)
        return

    n_params_dense = sum(p.numel() for p in model.parameters())
    print(f"\nDense model params: {n_params_dense:,}", flush=True)

    # ── eval BEFORE any fine-tuning (the true dense/zero-shot baseline) ──
    print("\nEvaluating dense model BEFORE LoRA fine-tuning ...", flush=True)
    model.eval()
    ce_kw = dict(batch_size=args.eval_batch_size, max_article_chars=args.max_article_chars,
                max_summary_tokens=args.max_summary_tokens)
    pre_test_ce = evaluate_ce(model, test_examples, device, tokenizer, gates=None,
                              desc="pre-LoRA test CE", **ce_kw)
    pre_ppl = float(np.exp(pre_test_ce))
    rouge_sample = sample_examples(test_examples, args.rouge_eval_examples)
    rouge_kw = dict(batch_size=args.gen_batch_size, max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams, max_article_chars=args.max_article_chars)
    pre_rouge = evaluate_rouge(model, rouge_sample, device, tokenizer, gates=None,
                               desc="pre-LoRA ROUGE", **rouge_kw)
    print(f"  dense ppl : {pre_ppl:.3f} | dense R-L : {pre_rouge['rougeL']*100:.2f}% "
          f"(rougeLsum : {pre_rouge['rougeLsum']*100:.2f}%)", flush=True)

    # ── wrap with LoRA, fine-tune ──
    print(f"\nWrapping with LoRA (target_modules={target_modules}, r={args.lora_r}, "
          f"alpha={args.lora_alpha}, dropout={args.lora_dropout}, "
          f"gradient_checkpointing={args.gradient_checkpointing}) ...", flush=True)
    model = wrap_with_lora(model, args, target_modules)
    model.print_trainable_parameters()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt, sched = build_optimizer_and_scheduler(trainable_params, args)
    effective_batch = args.batch_size * args.grad_accum_steps
    print(f"  optimizer steps={args.max_steps} | grad_accum_steps={args.grad_accum_steps} | "
          f"effective batch={effective_batch} | lr_schedule={args.lr_schedule} "
          f"(warmup_ratio={args.warmup_ratio}) | weight_decay={args.weight_decay}", flush=True)

    run_dir = args.out_dir
    os.makedirs(run_dir, exist_ok=True)

    gap_val_sample = sample_examples(val_examples, args.gap_eval_examples)
    train_loss_hist, val_points = [], []
    loader_iter = iter(train_loader)
    model.train()
    t0 = time.time()
    from tqdm import tqdm
    pbar = tqdm(total=args.max_steps, desc="Dense-LoRA fine-tune (optimizer steps)", unit="step", dynamic_ncols=True)
    for opt_step in range(1, args.max_steps + 1):
        opt.zero_grad()
        micro_losses = []
        for _ in range(args.grad_accum_steps):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader); batch = next(loader_iter)
            micro_losses.append(lora_forward_backward(model, tokenizer, batch, device,
                                                       args.max_article_chars, args.max_summary_tokens,
                                                       args.grad_accum_steps))
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        opt.step(); sched.step()
        loss = sum(micro_losses) / len(micro_losses)
        train_loss_hist.append(loss)
        pbar.set_postfix(loss=f"{loss:.3f}", lr=f"{sched.get_last_lr()[0]:.2e}", refresh=False)
        pbar.update(1)

        if opt_step % args.eval_every == 0 or opt_step == args.max_steps:
            model.eval()
            val_ce = evaluate_ce(model, gap_val_sample, device, tokenizer, gates=None,
                                 desc=f"val CE @ step {opt_step}", **ce_kw)
            val_points.append((opt_step, val_ce))
            tqdm.write(f"  step {opt_step:>6} | train loss {loss:.4f} | val CE {val_ce:.4f} | "
                      f"lr {sched.get_last_lr()[0]:.2e}")
            model.train()
    pbar.close()
    total_time = time.time() - t0

    # ── final eval AFTER LoRA ──
    print("\nFinal evaluation (dense + LoRA fine-tuned) ...", flush=True)
    model.eval()
    post_test_ce = evaluate_ce(model, test_examples, device, tokenizer, gates=None,
                               desc="post-LoRA test CE", **ce_kw)
    post_ppl = float(np.exp(post_test_ce))
    post_rouge = evaluate_rouge(model, rouge_sample, device, tokenizer, gates=None,
                                desc="post-LoRA ROUGE", **rouge_kw)

    print(f"  → dense ppl {pre_ppl:.3f} → post-LoRA ppl {post_ppl:.3f}", flush=True)
    print(f"  → dense R-L {pre_rouge['rougeL']*100:.2f}% → post-LoRA R-L "
          f"{post_rouge['rougeL']*100:.2f}%  (rougeLsum {pre_rouge['rougeLsum']*100:.2f}% → "
          f"{post_rouge['rougeLsum']*100:.2f}%)", flush=True)

    plot_finetune_run(train_loss_hist, val_points, os.path.join(run_dir, "plot.png"),
                      title="Llama-2-7B — DENSE (unpruned) + LoRA — CNN/DailyMail")

    lines = [
        "Llama-2-7B — Dense-LoRA (no pruning) — CNN/DailyMail",
        f"lora: target={args.lora_target} r={args.lora_r} alpha={args.lora_alpha} "
        f"dropout={args.lora_dropout} lr={args.lr} schedule={args.lr_schedule} "
        f"warmup_ratio={args.warmup_ratio} weight_decay={args.weight_decay}",
        f"optimizer steps : {args.max_steps} | grad_accum_steps={args.grad_accum_steps} | "
        f"effective_batch={effective_batch} | gradient_checkpointing={args.gradient_checkpointing} | "
        f"time: {total_time:.1f}s",
        "-" * 60,
        f"model params (dense, no pruning) : {n_params_dense:,}",
        "-" * 60,
        f"cnn/dailymail test set ({len(test_examples)} examples, full split, target-only CE):",
        f"  pre-LoRA (dense)  ppl : {pre_ppl:.3f}",
        f"  post-LoRA         ppl : {post_ppl:.3f}",
        "-" * 60,
        f"ROUGE via generation ({len(rouge_sample)}-example sample of test, "
        f"greedy={args.num_beams == 1}, num_beams={args.num_beams}):",
        f"  pre-LoRA (dense)  rouge1/rouge2/rougeL/rougeLsum : "
        f"{pre_rouge['rouge1']*100:.2f}/{pre_rouge['rouge2']*100:.2f}/"
        f"{pre_rouge['rougeL']*100:.2f}/{pre_rouge['rougeLsum']*100:.2f}",
        f"  post-LoRA         rouge1/rouge2/rougeL/rougeLsum : "
        f"{post_rouge['rouge1']*100:.2f}/{post_rouge['rouge2']*100:.2f}/"
        f"{post_rouge['rougeL']*100:.2f}/{post_rouge['rougeLsum']*100:.2f}",
    ]
    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines), flush=True)

    model.save_pretrained(run_dir)   # peft adapter weights only, a few MB
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump({
            "n_params_dense": n_params_dense,
            "pre_lora_ppl": pre_ppl, "post_lora_ppl": post_ppl,
            "pre_lora_rouge": pre_rouge, "post_lora_rouge": post_rouge,
            "optimizer_steps": args.max_steps, "grad_accum_steps": args.grad_accum_steps,
            "effective_batch": effective_batch, "total_time": total_time,
            "lora_target": args.lora_target, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout, "lr": args.lr, "lr_schedule": args.lr_schedule,
            "warmup_ratio": args.warmup_ratio, "weight_decay": args.weight_decay,
            "gradient_checkpointing": args.gradient_checkpointing,
        }, f, indent=2)
    print(f"\n[saved] {run_dir}/ (adapter weights + summary.txt + meta.json + plot.png)", flush=True)


if __name__ == "__main__":
    main()
