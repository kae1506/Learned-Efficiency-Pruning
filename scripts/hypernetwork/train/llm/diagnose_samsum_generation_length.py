"""
SAMSum generation-length diagnostic.

WHY THIS EXISTS: the 8-lambda SAMSum sweep (2026-08-26/27) produced a
perplexity curve that is perfectly monotonic in lambda (2.838 -> 3.538 as
sparsity goes 23% -> 80.5%, exactly as a real cost curve should) alongside a
ROUGE-L column that is NOT monotonic in anything (30.11 - 36.47 over that
same range, best at lambda=0.4, worst at lambda=0.8, lambda=1.6 nearly tying
lambda=0.1) and that has EVERY pruned point beating the dense baseline
(18.98) by roughly 2x -- including the point with 80.5% of FFN neurons
removed. "Pruned beats dense at 80% sparsity" is not plausible as a quality
claim, and one metric behaving like physics while the other behaves like
noise is the signature of the second one measuring something else.

HYPOTHESIS UNDER TEST (stated before running, per the project's
reason-before-you-run convention): a GENERATION-LENGTH artifact. SAMSum
reference summaries are very short (~110 chars / ~28 tokens). A
non-instruction-tuned base Llama-2-7B given a bare "Dialogue: ...\n\nSummary:"
prompt has no stopping cue and plausibly rambles to the max_new_tokens cap,
while a heavily-gated model emits shorter, more clipped output. ROUGE-L
F-measure penalises the long rambling generation on precision, so the pruned
model would score higher REGARDLESS of actual summary quality.

PREDICTION IF THE HYPOTHESIS IS RIGHT: dense generations sit at/near the
96-token cap; pruned generations are markedly shorter and closer to the
~28-token reference length; and the ROUGE gap tracks the length gap rather
than tracking sparsity.

PREDICTION IF IT IS WRONG: both arms generate comparable lengths, and the
ROUGE difference survives -- in which case the sweep's numbers mean what they
appear to mean and the non-monotonicity needs a different explanation.

This uses GATED eval (apply_gates), not physical surgery, because that is
exactly how the sweep itself measured ROUGE -- the point is to diagnose the
sweep's own numbers, not a different pipeline.

Also reports per-layer kept-neuron counts, since physically_prune_model()
raises on any layer that keeps 0 neurons and lambda=1.6 is aggressive enough
that this needs checking before any surgery-based follow-up.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "./huggingface")

from train_pruner_llama2_7b_samsum import (   # noqa: E402
    LAYER_SHAPE, N_LAYERS, N_INTER, DIALOGUE_PREFIX, SUMMARY_PREFIX,
    Pruner, load_llama2_7b, get_mlp_weights, apply_gates, autocast_ctx,
    get_loaders, sample_examples, _rouge_scores_to_floats,
)


def load_gates(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    pruner = Pruner([LAYER_SHAPE] * N_LAYERS, embed_dim=ckpt["embed_dim"],
                    lstm_hidden=ckpt["lstm_hidden"]).to(device)
    pruner.load_state_dict(ckpt["pruner_state_dict"])
    pruner.eval()
    with torch.no_grad():
        gates = pruner(get_mlp_weights(model))
    del pruner
    torch.cuda.empty_cache()
    return gates, ckpt


@torch.no_grad()
def generate_with_lengths(model, examples, device, tokenizer, gates, batch_size,
                          max_new_tokens, max_dialogue_chars):
    """Returns (decoded_texts, generated_token_counts). Token count is measured
    BEFORE decoding and counts only non-pad, non-EOS generated tokens -- i.e.
    what the model actually emitted, not the padded tensor width."""
    import contextlib
    gen_max_length = max_dialogue_chars // 2 + 64
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    texts, tok_counts = [], []
    ctx = apply_gates(model, gates) if gates is not None else contextlib.nullcontext()
    try:
        with ctx:
            for i in range(0, len(examples), batch_size):
                batch = examples[i:i + batch_size]
                prompts = [DIALOGUE_PREFIX + ex["dialogue"][:max_dialogue_chars] + SUMMARY_PREFIX
                           for ex in batch]
                enc = tokenizer(prompts, return_tensors="pt", padding=True,
                                truncation=True, max_length=gen_max_length).to(device)
                with autocast_ctx(device):
                    out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                         do_sample=False, num_beams=1,
                                         pad_token_id=tokenizer.pad_token_id)
                gen_only = out[:, enc["input_ids"].shape[1]:]
                for row in gen_only:
                    n = int((row != tokenizer.pad_token_id).sum().item())
                    tok_counts.append(n)
                texts.extend(tokenizer.batch_decode(gen_only, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = prev_side
    return texts, tok_counts


def describe(name, counts, cap):
    arr = np.array(counts)
    at_cap = (arr >= cap).mean() * 100
    print(f"  {name:<28} mean {arr.mean():6.1f} | median {np.median(arr):6.1f} | "
          f"p10 {np.percentile(arr,10):5.1f} | p90 {np.percentile(arr,90):5.1f} | "
          f"max {arr.max():4d} | at cap({cap}) {at_cap:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", type=str,
                    default="experiments/latest/llama2_7b_samsum")
    ap.add_argument("--lambdas", type=str, nargs="+", default=["1.6", "0.4"],
                    help="Which lambda checkpoints to diagnose. Defaults to the "
                         "most-pruned point and the best-ROUGE point.")
    ap.add_argument("--n_examples", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--max_dialogue_chars", type=int, default=1600)
    ap.add_argument("--n_show", type=int, default=3)
    ap.add_argument("--hf_token", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    import evaluate as hf_evaluate
    rouge = hf_evaluate.load("rouge")

    model = load_llama2_7b(device, hf_token=args.hf_token)
    tokenizer, _, _, _, test_examples = get_loaders(4, hf_token=args.hf_token)
    sample = sample_examples(test_examples, args.n_examples)
    refs = [ex["summary"] for ex in sample]
    ref_counts = [len(tokenizer(r)["input_ids"]) for r in refs]

    print("\n" + "=" * 100)
    print(f"SAMSum generation-length diagnostic — {len(sample)} test examples, "
          f"greedy, max_new_tokens={args.max_new_tokens}")
    print("=" * 100)
    print("\nGENERATED TOKEN COUNTS")
    describe("REFERENCE (gold summary)", ref_counts, args.max_new_tokens)

    dense_texts, dense_counts = generate_with_lengths(
        model, sample, device, tokenizer, None, args.batch_size,
        args.max_new_tokens, args.max_dialogue_chars)
    describe("dense (no gating)", dense_counts, args.max_new_tokens)
    dense_rouge = _rouge_scores_to_floats(rouge.compute(predictions=dense_texts, references=refs))

    results = {}
    for lam in args.lambdas:
        ckpt_path = os.path.join(args.sweep_dir, f"lambda_{lam}", "pruner.pt")
        gates, ckpt = load_gates(model, ckpt_path, device)
        pct = 100 * (1 - sum(g.mean().item() for g in gates) / len(gates))
        kept = [int(g.sum().item()) for g in gates]
        texts, counts = generate_with_lengths(
            model, sample, device, tokenizer, gates, args.batch_size,
            args.max_new_tokens, args.max_dialogue_chars)
        describe(f"pruned λ={lam} ({pct:.1f}% pruned)", counts, args.max_new_tokens)
        r = _rouge_scores_to_floats(rouge.compute(predictions=texts, references=refs))
        results[lam] = dict(texts=texts, counts=counts, rouge=r, pct=pct, kept=kept)
        del gates
        torch.cuda.empty_cache()

    print("\nROUGE ON THIS SAMPLE (n=%d)" % len(sample))
    print(f"  {'arm':<28} {'R-1':>7} {'R-2':>7} {'R-L':>7}   {'mean gen tok':>12}")
    print(f"  {'dense':<28} {dense_rouge['rouge1']*100:7.2f} {dense_rouge['rouge2']*100:7.2f} "
          f"{dense_rouge['rougeL']*100:7.2f}   {np.mean(dense_counts):12.1f}")
    for lam, d in results.items():
        print(f"  {'pruned λ=' + lam:<28} {d['rouge']['rouge1']*100:7.2f} "
              f"{d['rouge']['rouge2']*100:7.2f} {d['rouge']['rougeL']*100:7.2f}   "
              f"{np.mean(d['counts']):12.1f}")

    print("\nPER-LAYER KEPT NEURONS (surgery feasibility — physically_prune_model "
          "raises if any layer keeps 0)")
    for lam, d in results.items():
        k = d["kept"]
        print(f"  λ={lam}: min {min(k)} | max {max(k)} | mean {np.mean(k):.0f} "
              f"of {N_INTER}  -> surgery {'OK' if min(k) > 0 else 'WILL RAISE (dead layer)'}")

    print("\n" + "=" * 100)
    print(f"SAMPLE GENERATIONS (first {args.n_show})")
    print("=" * 100)
    for i in range(min(args.n_show, len(sample))):
        print(f"\n--- example {i} ---")
        print(f"[REFERENCE  {ref_counts[i]:>3} tok] {refs[i]!r}")
        print(f"[DENSE      {dense_counts[i]:>3} tok] {dense_texts[i][:400]!r}")
        for lam, d in results.items():
            print(f"[λ={lam:<5} {d['counts'][i]:>3} tok] {d['texts'][i][:400]!r}")


if __name__ == "__main__":
    main()
