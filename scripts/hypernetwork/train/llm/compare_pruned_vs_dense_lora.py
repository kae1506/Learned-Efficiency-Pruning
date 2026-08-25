"""
Final ROUGE-L comparison: physically-pruned + LoRA  vs.  dense + LoRA, on
CNN/DailyMail.

This is the readout for B5's load-bearing open question (diary/ideas.md):
does a plain dense LoRA fine-tune on the same data match or beat the
pruned model's quality? If it does, the pruning isn't the active
ingredient -- task adaptation is -- and B5 collapses into "fine-tuning,
dressed up." F23 flagged this as unresolved because no dense control
existed; the two DDP arms this script reads are that control.

INPUTS -- the meta.json written by each arm:
  --pruned_meta  from prune_and_lora_cnndailymail_ddp.py
  --dense_meta   from lora_finetune_dense_cnndailymail_ddp.py

WHAT THIS SCRIPT CHECKS BEFORE COMPARING (and refuses to print a headline
number if violated, unless --force): the two arms must be matched on the
knobs that would otherwise confound the ROUGE-L difference --
optimizer_steps, global_effective_batch, examples_seen, world_size,
rouge_eval_examples, and every LoRA hyperparameter. A ROUGE-L gap between
an arm that saw 64k examples and one that saw 32k is a data-budget
result, not a pruning result. Mismatches are reported explicitly rather
than folded into the table.

BOOTSTRAP CI: ROUGE-L here is a mean of per-example F-measures over an
N-example sample of the test split, so the arm-vs-arm difference carries
sampling error that a bare point-estimate table hides -- at N=300 the
standard error on ROUGE-L is roughly a point, which is the same order as
the differences actually observed between arms so far (18.85 -> 18.64 in
the first pruned run). This script reports a paired bootstrap CI on the
difference when --pruned_preds/--dense_preds per-example score dumps are
available, and otherwise says plainly that it cannot, rather than
implying the point estimates are separated.
"""
import os
import sys
import json
import argparse

METRICS = ("rouge1", "rouge2", "rougeL", "rougeLsum")

# Knobs that MUST match for the comparison to isolate "was it pruned first?".
MATCH_KEYS = (
    "optimizer_steps", "grad_accum_steps", "world_size", "global_effective_batch",
    "examples_seen", "rouge_eval_examples", "test_split_size",
    "lora_target", "lora_r", "lora_alpha", "lora_dropout",
    "base_lr", "scaled_lr", "lr_schedule", "warmup_ratio", "weight_decay",
    "gradient_checkpointing", "seed",
)


def load_meta(path, expected_arm):
    if not os.path.exists(path):
        sys.exit(f"ERROR: no meta.json at {path} -- has the {expected_arm} arm finished?")
    with open(path) as f:
        meta = json.load(f)
    arm = meta.get("arm")
    if arm is not None and arm != expected_arm:
        sys.exit(f"ERROR: {path} records arm={arm!r}, expected {expected_arm!r} -- "
                 f"--pruned_meta/--dense_meta look swapped.")
    return meta


def check_matched(pruned, dense):
    """Returns (mismatches, missing) -- mismatches is a list of (key, pruned_val, dense_val)."""
    mismatches, missing = [], []
    for k in MATCH_KEYS:
        if k not in pruned or k not in dense:
            missing.append(k)
            continue
        if pruned[k] != dense[k]:
            mismatches.append((k, pruned[k], dense[k]))
    return mismatches, missing


def pct(x):
    return None if x is None else 100 * x


def fmt(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def rouge_row(label, rouge):
    if rouge is None:
        return f"  {label:<34} " + " ".join(f"{'n/a':>9}" for _ in METRICS)
    return f"  {label:<34} " + " ".join(f"{pct(rouge[m]):>9.2f}" for m in METRICS)


def paired_bootstrap_ci(a_scores, b_scores, n_boot, seed, alpha=0.05):
    """Paired bootstrap over examples on (mean(a) - mean(b)). Paired because both arms
    are scored on the SAME example set (sample_examples is a deterministic prefix slice),
    so per-example difficulty cancels and the CI is tighter than an unpaired one."""
    import numpy as np
    a = np.asarray(a_scores, dtype=float)
    b = np.asarray(b_scores, dtype=float)
    if a.shape != b.shape:
        sys.exit(f"ERROR: per-example score dumps have different lengths "
                 f"({a.shape[0]} vs {b.shape[0]}) -- they must be the same example set.")
    n = a.shape[0]
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    # Two-sided bootstrap p-value for H0: mean difference = 0.
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(p, 1.0)), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pruned_meta", type=str, required=True,
                    help="meta.json from prune_and_lora_cnndailymail_ddp.py")
    ap.add_argument("--dense_meta", type=str, required=True,
                    help="meta.json from lora_finetune_dense_cnndailymail_ddp.py")
    ap.add_argument("--pruned_preds", type=str, default=None,
                    help="Optional JSON list of per-example rougeL scores for the pruned arm "
                         "(enables the paired bootstrap CI on the difference).")
    ap.add_argument("--dense_preds", type=str, default=None,
                    help="Optional JSON list of per-example rougeL scores for the dense arm.")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None,
                    help="Optional path to write this table to (also always printed).")
    ap.add_argument("--force", action="store_true",
                    help="Print the headline comparison even if the two arms are not matched. "
                         "The mismatch report is printed either way.")
    args = ap.parse_args()

    pruned = load_meta(args.pruned_meta, "pruned")
    dense = load_meta(args.dense_meta, "dense")

    L = []
    L.append("=" * 78)
    L.append("CNN/DailyMail — Llama-2-7B — pruned+LoRA vs dense+LoRA — final ROUGE-L")
    L.append("=" * 78)

    # ── matched-ness gate ──
    mismatches, missing = check_matched(pruned, dense)
    if missing:
        L.append(f"\nNOTE: {len(missing)} match-key(s) absent from one or both meta.json files "
                 f"(older run, written before these were recorded): {', '.join(missing)}. "
                 f"Cannot verify the arms agree on these.")
    if mismatches:
        L.append("\n*** ARMS NOT MATCHED — the ROUGE-L difference below is confounded. ***")
        L.append(f"  {'key':<26} {'pruned':>18} {'dense':>18}")
        for k, pv, dv in mismatches:
            L.append(f"  {k:<26} {str(pv):>18} {str(dv):>18}")
        if any(k in ("optimizer_steps", "examples_seen", "global_effective_batch") for k, _, _ in mismatches):
            L.append("  -> a training-budget mismatch: whichever arm saw more data is favoured, "
                     "independently of pruning.")
        if any(k == "rouge_eval_examples" for k, _, _ in mismatches):
            L.append("  -> the arms were scored on DIFFERENT-SIZED test samples; the ROUGE numbers "
                     "are not two estimates of the same statistic.")
    else:
        L.append("\nArms matched on all recorded knobs (steps, batch, data budget, LoRA config, "
                 "eval sample, seed) — the only difference is whether pruning happened first.")

    # ── setup ──
    L.append("\n" + "-" * 78)
    L.append("SETUP")
    L.append("-" * 78)
    L.append(f"  pruner ckpt          : {pruned.get('pruner_ckpt', 'n/a')} "
             f"(lambda={pruned.get('lambda', 'n/a')})")
    L.append(f"  FFN neurons pruned   : {fmt(pruned.get('pct_pruned'))}%")
    L.append(f"  params  pruned arm   : {pruned.get('n_params_pruned', 'n/a'):,}"
             if isinstance(pruned.get("n_params_pruned"), int) else
             f"  params  pruned arm   : {pruned.get('n_params_pruned', 'n/a')}")
    L.append(f"  params  dense  arm   : {dense.get('n_params_dense', 'n/a'):,}"
             if isinstance(dense.get("n_params_dense"), int) else
             f"  params  dense  arm   : {dense.get('n_params_dense', 'n/a')}")
    L.append(f"  optimizer steps      : {pruned.get('optimizer_steps')} "
             f"(global effective batch {pruned.get('global_effective_batch')}, "
             f"world_size {pruned.get('world_size')})")
    cov = pruned.get("train_coverage_pct")
    L.append(f"  train-split coverage : {pruned.get('examples_seen', 'n/a'):,} examples seen = "
             f"{fmt(cov, 1)}% of the {pruned.get('train_split_size', 0):,}-example train split"
             if isinstance(pruned.get("examples_seen"), int) else
             f"  train-split coverage : n/a (older run)")
    L.append(f"  ROUGE eval sample    : {pruned.get('rouge_eval_examples', 'n/a')} examples "
             f"(deterministic prefix of the {pruned.get('test_split_size', 'n/a')}-example test split)")
    L.append(f"  LoRA                 : target={pruned.get('lora_target')} r={pruned.get('lora_r')} "
             f"alpha={pruned.get('lora_alpha')} dropout={pruned.get('lora_dropout')} "
             f"base_lr={pruned.get('base_lr')} sched={pruned.get('lr_schedule')}")
    L.append(f"  wall time            : pruned {fmt(pruned.get('total_time'), 1)}s | "
             f"dense {fmt(dense.get('total_time'), 1)}s")

    # ── perplexity ──
    L.append("\n" + "-" * 78)
    L.append(f"PERPLEXITY (full {pruned.get('test_split_size', 'n/a')}-example test split, target-only CE)")
    L.append("-" * 78)
    L.append(f"  dense, zero-shot (live, dense arm)   : {fmt(dense.get('pre_lora_ppl'), 3)}")
    L.append(f"  dense + LoRA                         : {fmt(dense.get('post_lora_ppl'), 3)}")
    L.append(f"  pruned, no LoRA (surgery only)       : {fmt(pruned.get('pre_lora_ppl'), 3)}")
    L.append(f"  pruned + LoRA                        : {fmt(pruned.get('post_lora_ppl'), 3)}")

    # ── ROUGE table ──
    L.append("\n" + "-" * 78)
    L.append(f"ROUGE via greedy generation ({pruned.get('rouge_eval_examples', 'n/a')}-example sample)")
    L.append("-" * 78)
    L.append(f"  {'':<34} " + " ".join(f"{m:>9}" for m in METRICS))
    L.append(rouge_row("dense, zero-shot (live)", dense.get("pre_lora_rouge")))
    L.append(rouge_row("dense + LoRA", dense.get("post_lora_rouge")))
    L.append(rouge_row("pruned, no LoRA (surgery only)", pruned.get("pre_lora_rouge")))
    L.append(rouge_row("pruned + LoRA", pruned.get("post_lora_rouge")))
    if pruned.get("dense_rouge") is not None and pruned.get("dense_rouge_is_cached_at_300"):
        L.append(rouge_row("[legacy] dense cached @300", pruned.get("dense_rouge")))
        if pruned.get("rouge_eval_examples") not in (None, 300):
            L.append("    ^ measured on 300 examples by the pruning run, NOT this run's sample "
                     "size — shown for provenance only, do not compare it to the rows above.")

    # ── the headline ──
    L.append("\n" + "=" * 78)
    L.append("HEADLINE — final ROUGE-L, pruned+LoRA vs dense+LoRA")
    L.append("=" * 78)
    p_rl = pruned.get("post_lora_rouge", {}).get("rougeL") if pruned.get("post_lora_rouge") else None
    d_rl = dense.get("post_lora_rouge", {}).get("rougeL") if dense.get("post_lora_rouge") else None
    d_zs = dense.get("pre_lora_rouge", {}).get("rougeL") if dense.get("pre_lora_rouge") else None

    if mismatches and not args.force:
        L.append("  SUPPRESSED — arms are not matched (see above). Re-run the arms with matching "
                 "settings, or pass --force to print anyway.")
    elif p_rl is None or d_rl is None:
        L.append("  SUPPRESSED — one or both arms are missing post_lora_rouge.")
    else:
        delta = pct(p_rl) - pct(d_rl)
        L.append(f"  pruned + LoRA  R-L : {pct(p_rl):.2f}%")
        L.append(f"  dense  + LoRA  R-L : {pct(d_rl):.2f}%")
        L.append(f"  difference         : {delta:+.2f} pp "
                 f"({'pruned ahead' if delta > 0 else 'dense ahead' if delta < 0 else 'tie'})")
        if d_zs is not None:
            L.append(f"\n  For B5: the dense arm moved {pct(d_zs):.2f}% -> {pct(d_rl):.2f}% R-L "
                     f"({pct(d_rl) - pct(d_zs):+.2f} pp) on LoRA alone, with no pruning anywhere.")
            L.append(f"  Pruning is the active ingredient only to the extent the {delta:+.2f} pp "
                     f"gap above survives — a gap inside the eval sample's noise band does not "
                     f"establish it.")

        # ── paired bootstrap on the difference ──
        if args.pruned_preds and args.dense_preds:
            with open(args.pruned_preds) as f:
                a = json.load(f)
            with open(args.dense_preds) as f:
                b = json.load(f)
            mean_d, lo, hi, p, n = paired_bootstrap_ci(a, b, args.n_boot, args.seed)
            L.append(f"\n  Paired bootstrap over the {n} scored examples ({args.n_boot:,} resamples):")
            L.append(f"    mean R-L difference : {100*mean_d:+.2f} pp")
            L.append(f"    95% CI              : [{100*lo:+.2f}, {100*hi:+.2f}] pp")
            L.append(f"    two-sided p         : {p:.4f}")
            L.append(f"    -> {'CI excludes 0: the gap is separated at this sample size.' if lo > 0 or hi < 0 else 'CI straddles 0: the two arms are NOT separated at this sample size.'}")
        else:
            n_eval = pruned.get("rouge_eval_examples")
            L.append(f"\n  NO CONFIDENCE INTERVAL: per-example score dumps were not supplied "
                     f"(--pruned_preds/--dense_preds), so the {delta:+.2f} pp difference above is "
                     f"a bare point estimate over {n_eval if n_eval else 'N'} examples with no "
                     f"error bar. Do not read it as separated without one.")

    L.append("=" * 78)

    text = "\n".join(L)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
