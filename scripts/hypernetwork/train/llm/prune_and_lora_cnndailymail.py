"""
Physical structural pruning + Dense-LoRA fine-tuning on CNN/DailyMail.

Different pipeline from train_pruner_llama2_7b_cnndailymail.py's own
CE-delta pruning objective: THAT script trains a hypernetwork (Pruner) to
choose which FFN neurons to gate -- a multiplicative mask applied at
inference via a forward hook, base weights physically untouched. THIS
script takes an ALREADY-TRAINED pruner checkpoint from that script, reads
off its FINAL gates to decide which neurons survive, then PHYSICALLY
REMOVES the pruned rows/columns from gate_proj/up_proj/down_proj -- no
Pruner network, no gate hooks, nothing gated anywhere past that point,
just a genuinely smaller dense Llama-2-7B. It then runs a standard LoRA
fine-tune (peft) of THAT physically-pruned model on CNN/DailyMail, same
target-only-CE objective/prompt template as the sibling script, so the
final numbers are directly comparable to it.

WHY: tests how much of the pruning-training run's reported quality gain
(F23 -- pruned ROUGE-L consistently beats dense ROUGE-L across all 4 swept
lambdas, see diary/crisp-findings.md) survives / grows / shrinks once the
smaller architecture gets genuine gradient-based fine-tuning, vs. relying
entirely on the pruning run's own CE-delta objective. This is the B5
dense-fine-tune-control direction from diary/ideas.md, run on top of
PHYSICAL pruning rather than the gated model. See the conversation this
script was built from.

PIPELINE: pretrain (external, frozen) -> prune (physical surgery, reading
an already-trained pruner.pt's FINAL gates -- no pruner training happens
in this script) -> LoRA fine-tune (peft, MLP+attention target modules,
confirmed with the user 2026-08-19) -> evaluate (CE/ppl + ROUGE, full test
split / bounded ROUGE sample, identical convention to the sibling script).

BUILT-IN CORRECTNESS CHECK: after surgery, BEFORE any LoRA training, this
script evaluates the physically-pruned model (gates=None -- there IS no
gate anymore, the weights are just smaller) and compares that against the
ORIGINAL pruning run's own "pruned" numbers (test_ppl_pruned / rouge_pruned,
cached in the checkpoint from when that gate was applied via the runtime
hook). Physical removal and runtime masking of the SAME neurons should
produce numerically identical forward passes (same matmuls, same result,
different implementation) -- if pre-LoRA eval here diverges meaningfully
from the checkpoint's own cached numbers, that means the surgery has a
bug, and this is flagged loudly rather than silently proceeding to LoRA.

*** UNTESTED. *** No local hardware at this scale. Verified: py_compile
only. Requires `pip install peft` (not installed in this repo's venv,
checked before writing this -- same situation as evaluate/rouge_score was
for the sibling script, lm-eval for the downstream-eval script).

PREREQUISITE -- meta-llama/Llama-2-7b-hf is GATE-LICENSED. Pass a token
via --hf_token (CLI arg, not an environment variable -- same convention as
train_pruner_llama2_7b_cnndailymail.py).

INPUT -- a pruner.pt checkpoint from train_pruner_llama2_7b_cnndailymail.py
(--pruner_ckpt, e.g. experiments/latest/llama2_7b_cnndailymail/lambda_0.1/
pruner.pt). Its FINAL gates (STE forward, hard-thresholded, deterministic
given the same frozen dense weights -- not an approximation, see
eval_downstream_llama2_7b.py's load_pruner_and_gates() docstring for why)
decide which of the 11008 FFN neurons per layer survive. Once the
keep-mask is read off, the Pruner network is discarded -- it plays no
further role.

SURGERY, per layer, per this project's row-concat convention
(get_mlp_weights concatenates gate_proj+up_proj rows, one row = one SwiGLU
intermediate neuron -- see train_pruner_llama2_7b_wikitext2.py):
    keep_idx = gate.nonzero()                          # this layer's survivors
    gate_proj.weight <- gate_proj.weight[keep_idx, :]   # [kept, HIDDEN]
    up_proj.weight   <- up_proj.weight[keep_idx, :]     # [kept, HIDDEN]
    down_proj.weight <- down_proj.weight[:, keep_idx]   # [HIDDEN, kept]
Per-layer kept-count varies (this project's pruning is NOT uniform-width
across layers) -- fine, each layer's 3 Linear modules are swapped
independently and only need to agree with EACH OTHER, not with a shared
config.intermediate_size (HF's LlamaMLP.forward uses the submodules
directly, never re-reads config after init, so mismatched per-layer widths
across the model are not a problem).

CHOICES MADE (confirmed with the user, not silent defaults) -- LoRA
target_modules = MLP (gate/up/down_proj) + attention (q/k/v/o_proj), not
MLP-only -- the stronger/more standard full-capacity fine-tune, giving the
pruned model its best real shot at recovering/improving quality (2026-08-19).

LoRA METHODOLOGY -- matches common Llama-2-7B LoRA fine-tuning practice
(Alpaca-LoRA / QLoRA-style recipes), not an ad hoc setup (revised
2026-08-19 after the first version of this script used a flat LR, no
accumulation, no checkpointing -- none of which match how this is actually
done in the literature/community):
  - r=16, alpha=32 (the 2x-rank convention), dropout=0.05, lr=2e-4 --
    literally Alpaca-LoRA's own published config for Llama fine-tuning,
    not arbitrary numbers.
  - target_modules = ALL linear layers (attention + MLP) -- QLoRA's
    explicit finding/recommendation (Dettmers et al. 2023): LoRA on all
    linear layers matches full fine-tuning quality far more closely than
    attention-only, which the original LoRA paper's minimal q/v-only
    setup does not.
  - cosine LR schedule with warmup_ratio=0.03 (--lr_schedule, --warmup_ratio)
    -- standard for LLM SFT/LoRA recipes (Alpaca, Vicuna, QLoRA all use
    warmup + cosine decay, not a flat LR held constant throughout).
  - gradient accumulation, --grad_accum_steps=4 at --batch_size=4 ->
    effective batch 16 -- typical LoRA fine-tuning recipes use an
    effective batch in the 16-128 range; a raw micro-batch of 4 with no
    accumulation (the first version's setup) is well below that.
  - --max_steps now counts OPTIMIZER steps (each consuming grad_accum_steps
    micro-batches), default lowered 2000 -> 500 so the TOTAL compute
    budget is held constant vs. the first version: 500 x 4 x 4 = 8,000
    examples processed either way, just reorganized into a larger
    effective batch with fewer, larger updates instead of many small ones.
  - --gradient_checkpointing (default on) -- standard for LoRA fine-tuning
    a 7B model, trades some compute for the memory headroom real LoRA
    recipes rely on to run a non-trivial batch size at all. Requires
    model.enable_input_require_grads() before wrapping with LoRA (peft's
    documented gotcha: with the base frozen + checkpointing on, nothing in
    the chain requires grad until the LoRA layers unless this is set).
  - --weight_decay=0.0 -- HF Trainer's own default, and what most LoRA
    recipes leave it at (tiny adapter param count, little regularization
    benefit expected).

CHOICES NOT explicitly re-confirmed, flagged here instead (same convention
as every sibling script -- state the value AND why, let the user redirect):
  - eval_every=500 (optimizer steps) -- periodic val CE checkpoint, cheap
    relative to a training step, gives a training curve without
    materially slowing the run.
  - batch_size=4, eval_batch_size=8, gen_batch_size=4, max_article_chars=4000,
    max_summary_tokens=128, rouge_eval_examples=300, num_beams=1 --
    identical to the sibling script's defaults, for direct comparability
    of the final numbers.
  - Only LoRA adapter weights + surgery/eval metadata are saved (peft's
    save_pretrained, a few MB), NOT a merged full checkpoint -- matches
    this project's existing convention of saving small artifacts, not
    14GB model dumps.

Reuses Pruner / load_llama2_7b / get_mlp_weights / get_loaders /
sample_examples / build_training_batch / evaluate_ce / evaluate_rouge /
autocast_ctx / _smooth / N_LAYERS / N_INTER / LAYER_SHAPE verbatim from
train_pruner_llama2_7b_cnndailymail.py via direct import -- no logic
duplicated, same pattern as eval_downstream_llama2_7b.py importing from
train_pruner_llama2_7b_wikitext2.py.
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_pruner_llama2_7b_cnndailymail import (
    Pruner, load_llama2_7b, get_mlp_weights, get_loaders, sample_examples,
    build_training_batch, evaluate_ce, evaluate_rouge, autocast_ctx, _smooth,
    N_LAYERS, N_INTER, LAYER_SHAPE,
)

OUT_ROOT = "/workspace/results/llama2_7b_cnndailymail_lora"


# ─────────────────────────────────────────────────────────────────────────────
# Load pruner checkpoint -> final gates (deterministic, see module docstring).
# ─────────────────────────────────────────────────────────────────────────────

def load_pruner_final_gates(model, pruner_ckpt_path, device):
    ckpt = torch.load(pruner_ckpt_path, map_location=device, weights_only=False)
    layer_shapes = [LAYER_SHAPE] * N_LAYERS
    pruner = Pruner(layer_shapes, embed_dim=ckpt["embed_dim"], lstm_hidden=ckpt["lstm_hidden"]).to(device)
    pruner.load_state_dict(ckpt["pruner_state_dict"])
    pruner.eval()
    with torch.no_grad():
        gates = pruner(get_mlp_weights(model))
    del pruner
    return gates, ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Physical surgery -- mutates model in place.
# ─────────────────────────────────────────────────────────────────────────────

def physically_prune_model(model, gates):
    per_layer_kept = []
    for i, layer in enumerate(model.model.layers):
        keep_idx = gates[i].nonzero(as_tuple=True)[0]
        k = keep_idx.numel()
        if k == 0:
            raise RuntimeError(
                f"layer {i}: gate keeps 0/{gates[i].numel()} neurons -- a fully-dead MLP "
                f"layer is numerically degenerate (down_proj with 0 in-features always "
                f"outputs zero). This checkpoint's lambda is too aggressive for physical "
                f"surgery; investigate before proceeding.")
        mlp = layer.mlp
        w_device, w_dtype = mlp.gate_proj.weight.device, mlp.gate_proj.weight.dtype

        new_gate = nn.Linear(mlp.gate_proj.in_features, k, bias=False, device=w_device, dtype=w_dtype)
        new_gate.weight.data = mlp.gate_proj.weight.data[keep_idx, :].clone().contiguous()
        new_up = nn.Linear(mlp.up_proj.in_features, k, bias=False, device=w_device, dtype=w_dtype)
        new_up.weight.data = mlp.up_proj.weight.data[keep_idx, :].clone().contiguous()
        new_down = nn.Linear(k, mlp.down_proj.out_features, bias=False, device=w_device, dtype=w_dtype)
        new_down.weight.data = mlp.down_proj.weight.data[:, keep_idx].clone().contiguous()

        mlp.gate_proj, mlp.up_proj, mlp.down_proj = new_gate, new_up, new_down
        per_layer_kept.append(k)

    if next(model.parameters()).is_cuda:
        torch.cuda.empty_cache()
    return per_layer_kept


# ─────────────────────────────────────────────────────────────────────────────
# LoRA forward+backward for ONE micro-batch -- plain target-only CE, no
# orig/pruned delta (there is no "orig" anymore at this point, the model
# itself IS the pruned one). Loss is pre-scaled by 1/grad_accum_steps so
# accumulated .backward() calls sum to the correct mean-over-effective-batch
# gradient; does NOT call optimizer.zero_grad()/step() -- accumulation is a
# loop-level concern, handled by the caller (see main()'s training loop).
# ─────────────────────────────────────────────────────────────────────────────

def lora_forward_backward(model, tokenizer, batch_examples, device,
                          max_article_chars, max_summary_tokens, grad_accum_steps):
    input_ids, attn, labels = build_training_batch(batch_examples, tokenizer,
                                                    max_article_chars, max_summary_tokens)
    input_ids, attn, labels = input_ids.to(device), attn.to(device), labels.to(device)
    with autocast_ctx(device):
        loss = model(input_ids, attention_mask=attn, labels=labels).loss
    (loss / grad_accum_steps).backward()
    return loss.item()


def wrap_with_lora(model, args, target_modules):
    """Gradient checkpointing must be enabled, and enable_input_require_grads()
    called, BEFORE wrapping with LoRA -- see module docstring's peft gotcha
    note. Returns the peft-wrapped model."""
    from peft import LoraConfig, get_peft_model, TaskType
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                          target_modules=target_modules, task_type=TaskType.CAUSAL_LM, bias="none")
    return get_peft_model(model, lora_cfg)


def build_optimizer_and_scheduler(trainable_params, args):
    from transformers import (get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup,
                              get_constant_schedule_with_warmup)
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    warmup_steps = max(1, int(args.warmup_ratio * args.max_steps))
    if args.lr_schedule == "cosine":
        sched = get_cosine_schedule_with_warmup(opt, warmup_steps, args.max_steps)
    elif args.lr_schedule == "linear":
        sched = get_linear_schedule_with_warmup(opt, warmup_steps, args.max_steps)
    else:
        sched = get_constant_schedule_with_warmup(opt, warmup_steps)
    return opt, sched


def plot_finetune_run(train_loss, val_points, save_path, title):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(title, fontsize=11, fontweight="bold")

    steps = range(1, len(train_loss) + 1)
    axes[0].plot(steps, train_loss, alpha=0.15, color="steelblue")
    axes[0].plot(steps, _smooth(train_loss), color="steelblue", lw=2)
    axes[0].set_title("LoRA train loss (CE, summary tokens only)")
    axes[0].set_xlabel("step"); axes[0].grid(alpha=0.3)

    if val_points:
        vx, vy = zip(*val_points)
        axes[1].plot(vx, vy, color="tomato", lw=2, marker="o")
    axes[1].set_title("val CE (periodic checkpoint)")
    axes[1].set_xlabel("step"); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pruner_ckpt", type=str, required=True,
                    help="Path to a pruner.pt from train_pruner_llama2_7b_cnndailymail.py "
                         "(e.g. experiments/latest/llama2_7b_cnndailymail/lambda_0.1/pruner.pt).")
    ap.add_argument("--lora_target", type=str, choices=["mlp", "mlp_attn"], default="mlp_attn",
                    help="mlp_attn (confirmed default, 2026-08-19) = gate/up/down_proj + "
                         "q/k/v/o_proj. mlp = gate/up/down_proj only (matches exactly where "
                         "pruning happened, cleaner but weaker-capacity comparison).")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_schedule", type=str, choices=["cosine", "linear", "constant"], default="cosine",
                    help="Cosine-with-warmup is standard for LLM LoRA/SFT recipes "
                         "(Alpaca, Vicuna, QLoRA) -- see module docstring.")
    ap.add_argument("--warmup_ratio", type=float, default=0.03,
                    help="Fraction of --max_steps spent warming up. 0.03 matches "
                         "Alpaca-LoRA/QLoRA-style recipes.")
    ap.add_argument("--weight_decay", type=float, default=0.0,
                    help="HF Trainer's own default; most LoRA recipes leave this at 0.")
    ap.add_argument("--grad_accum_steps", type=int, default=4,
                    help="Micro-batches accumulated per optimizer step. batch_size=4 x "
                         "grad_accum_steps=4 = effective batch 16, in the typical LoRA "
                         "fine-tuning range (see module docstring).")
    ap.add_argument("--gradient_checkpointing", action="store_true", default=True,
                    help="On by default -- standard for LoRA fine-tuning a 7B model.")
    ap.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    ap.add_argument("--max_steps", type=int, default=500,
                    help="OPTIMIZER steps (not micro-batches) -- see module docstring for "
                         "why this is 500 here vs. the pre-accumulation version's 2000: "
                         "same total examples processed (500*4*4 = 8000 = 2000*4), just "
                         "reorganized into a larger effective batch.")
    ap.add_argument("--eval_every", type=int, default=125,
                    help="Optimizer steps -- 125*4(grad_accum)=500 micro-batches between "
                         "checkpoints, same cadence in examples-processed terms as the "
                         "pre-accumulation version's eval_every=500.")
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
                    help="Run surgery, verify pct_pruned matches the checkpoint's own "
                         "record, wrap with LoRA, run ONE training step, then exit. "
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

    print("Loading Llama-2-7B (GATE-LICENSED -- requires --hf_token with accepted access) ...", flush=True)
    model = load_llama2_7b(device, hf_token=args.hf_token)

    print(f"Loading CNN/DailyMail ...", flush=True)
    tokenizer, train_loader, train_examples, val_examples, test_examples = get_loaders(
        args.batch_size, hf_token=args.hf_token)
    print(f"Data: train={len(train_examples):,} val={len(val_examples):,} test={len(test_examples):,}",
          flush=True)

    print(f"\nLoading pruner checkpoint + reconstructing final gates: {args.pruner_ckpt}", flush=True)
    gates, ckpt = load_pruner_final_gates(model, args.pruner_ckpt, device)
    ckpt_pct_pruned = 100 * (1 - sum(g.mean().item() for g in gates) / len(gates))
    print(f"  reconstructed gate: {ckpt_pct_pruned:.2f}% FFN neurons pruned "
          f"(lambda={ckpt.get('lambda')}, converged={ckpt.get('converged')})", flush=True)

    target_modules = (["gate_proj", "up_proj", "down_proj"] if args.lora_target == "mlp"
                      else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    if args.sanity_check:
        print("\n" + "=" * 70, flush=True)
        print("SANITY CHECK -- surgery + pct_pruned cross-check + 1 LoRA step", flush=True)
        print("=" * 70, flush=True)
        per_layer_kept = physically_prune_model(model, gates)
        actual_pct = 100 * (1 - sum(per_layer_kept) / (N_LAYERS * N_INTER))
        diff = abs(actual_pct - ckpt_pct_pruned)
        print(f"  surgery result   : {actual_pct:.4f}% pruned")
        print(f"  gate reconstruct : {ckpt_pct_pruned:.4f}% pruned")
        print(f"  diff             : {diff:.4f} pp {'PASS' if diff < 1e-3 else 'FAIL -- mismatch, investigate surgery indexing'}")

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

    # ── 1) dense baseline -- reuse cached numbers from the original pruning run
    #      (already computed there, no need to spend GPU time recomputing) ──
    dense_test_ppl = ckpt["test_ppl_orig"]
    dense_rouge = ckpt["rouge_orig"]
    print(f"\nDense baseline (cached from pruning-run checkpoint): "
          f"ppl={dense_test_ppl:.3f} rougeL={dense_rouge['rougeL']*100:.2f}%", flush=True)

    # ── 2) physical surgery ──
    print(f"\nPhysically pruning model ({ckpt_pct_pruned:.2f}% target) ...", flush=True)
    per_layer_kept = physically_prune_model(model, gates)
    actual_pct_pruned = 100 * (1 - sum(per_layer_kept) / (N_LAYERS * N_INTER))
    diff = abs(actual_pct_pruned - ckpt_pct_pruned)
    print(f"  surgery result: {actual_pct_pruned:.4f}% pruned (gate reconstruct said "
          f"{ckpt_pct_pruned:.4f}%, diff={diff:.4f}pp)", flush=True)
    if diff > 1e-2:
        print("  WARNING: surgery pct_pruned does not match the gate reconstruction closely -- "
              "double-check keep_idx indexing before trusting downstream numbers.", flush=True)

    n_params_pruned = sum(p.numel() for p in model.parameters())
    print(f"  model params after surgery: {n_params_pruned:,}", flush=True)

    # ── eval PRE-LoRA (surgery only) -- correctness check against the pruning
    #    run's own cached "pruned" numbers, see module docstring ──
    print("\nEvaluating physically-pruned model BEFORE any LoRA training "
          "(correctness check vs. the pruning run's own gated-eval numbers) ...", flush=True)
    model.eval()
    ce_kw = dict(batch_size=args.eval_batch_size, max_article_chars=args.max_article_chars,
                max_summary_tokens=args.max_summary_tokens)
    pre_lora_test_ce = evaluate_ce(model, test_examples, device, tokenizer, gates=None,
                                   desc="pre-LoRA test CE", **ce_kw)
    pre_lora_ppl = float(np.exp(pre_lora_test_ce))
    rouge_sample = sample_examples(test_examples, args.rouge_eval_examples)
    rouge_kw = dict(batch_size=args.gen_batch_size, max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams, max_article_chars=args.max_article_chars)
    pre_lora_rouge = evaluate_rouge(model, rouge_sample, device, tokenizer, gates=None,
                                    desc="pre-LoRA ROUGE", **rouge_kw)
    ppl_check_diff = abs(pre_lora_ppl - ckpt["test_ppl_pruned"])
    rL_check_diff = abs(pre_lora_rouge["rougeL"] - ckpt["rouge_pruned"]["rougeL"]) * 100
    print(f"  pre-LoRA ppl   : {pre_lora_ppl:.3f}  (gated-eval checkpoint said "
          f"{ckpt['test_ppl_pruned']:.3f}, diff={ppl_check_diff:.4f})", flush=True)
    print(f"  pre-LoRA R-L   : {pre_lora_rouge['rougeL']*100:.2f}%  (gated-eval checkpoint said "
          f"{ckpt['rouge_pruned']['rougeL']*100:.2f}%, diff={rL_check_diff:.4f}pp)", flush=True)
    if ppl_check_diff > 0.05 or rL_check_diff > 1.0:
        print("  WARNING: physical-surgery eval diverges non-trivially from the original "
              "gated-eval numbers -- expected to be near-identical (same weights, same "
              "computation). Investigate before trusting the LoRA results below.", flush=True)

    # ── 3) LoRA fine-tune ──
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

    lambda_tag = f"lambda_{ckpt.get('lambda')}" if ckpt.get("lambda") is not None else "run"
    run_dir = os.path.join(args.out_dir, lambda_tag)
    os.makedirs(run_dir, exist_ok=True)

    gap_val_sample = sample_examples(val_examples, args.gap_eval_examples)
    train_loss_hist, val_points = [], []
    loader_iter = iter(train_loader)
    model.train()
    t0 = time.time()
    from tqdm import tqdm
    pbar = tqdm(total=args.max_steps, desc="LoRA fine-tune (optimizer steps)", unit="step", dynamic_ncols=True)
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

    # ── final eval POST-LoRA ──
    print("\nFinal evaluation (physically-pruned + LoRA fine-tuned) ...", flush=True)
    model.eval()
    post_test_ce = evaluate_ce(model, test_examples, device, tokenizer, gates=None,
                               desc="post-LoRA test CE", **ce_kw)
    post_ppl = float(np.exp(post_test_ce))
    post_rouge = evaluate_rouge(model, rouge_sample, device, tokenizer, gates=None,
                                desc="post-LoRA ROUGE", **rouge_kw)

    print(f"  → dense ppl {dense_test_ppl:.3f} | pre-LoRA (surgery only) ppl {pre_lora_ppl:.3f} | "
          f"post-LoRA ppl {post_ppl:.3f}", flush=True)
    print(f"  → dense R-L {dense_rouge['rougeL']*100:.2f}% | pre-LoRA R-L "
          f"{pre_lora_rouge['rougeL']*100:.2f}% | post-LoRA R-L {post_rouge['rougeL']*100:.2f}%", flush=True)

    plot_finetune_run(train_loss_hist, val_points, os.path.join(run_dir, "plot.png"),
                      title=(f"Llama-2-7B — physically-pruned ({actual_pct_pruned:.1f}%) + "
                            f"Dense-LoRA — CNN/DailyMail — {lambda_tag}"))

    lines = [
        f"Llama-2-7B — physical surgery + Dense-LoRA — CNN/DailyMail — {lambda_tag} "
        f"(from {args.pruner_ckpt})",
        f"lora: target={args.lora_target} r={args.lora_r} alpha={args.lora_alpha} "
        f"dropout={args.lora_dropout} lr={args.lr} schedule={args.lr_schedule} "
        f"warmup_ratio={args.warmup_ratio} weight_decay={args.weight_decay}",
        f"optimizer steps : {args.max_steps} | grad_accum_steps={args.grad_accum_steps} | "
        f"effective_batch={effective_batch} | gradient_checkpointing={args.gradient_checkpointing} | "
        f"time: {total_time:.1f}s",
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

    model.save_pretrained(run_dir)   # peft adapter weights only, a few MB
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump({
            "pruner_ckpt": args.pruner_ckpt, "lambda": ckpt.get("lambda"),
            "per_layer_kept": per_layer_kept, "pct_pruned": actual_pct_pruned,
            "dense_ppl": dense_test_ppl, "pre_lora_ppl": pre_lora_ppl, "post_lora_ppl": post_ppl,
            "dense_rouge": dense_rouge, "pre_lora_rouge": pre_lora_rouge, "post_lora_rouge": post_rouge,
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
