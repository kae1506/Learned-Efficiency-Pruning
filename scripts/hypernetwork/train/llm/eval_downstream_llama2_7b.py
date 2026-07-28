"""
Zero-shot downstream task evaluation for pruned Llama-2-7B checkpoints from
train_pruner_llama2_7b_wikitext2.py, via lm-evaluation-harness.

WHY THIS SCRIPT EXISTS: perplexity alone is not what the comparable
literature (DISP-LLM, SlimLLM, LLM-Pruner, FLAP, GISP) reports as its
headline evidence -- all of them also report zero-shot accuracy on a
standard commonsense-reasoning task suite. This closes that gap (see the
"Venue gap analysis" section of diary/final-paper-direction.md, item 1).

NOT FINE-TUNING. Zero-shot task accuracy is scored the same way perplexity
is: log-likelihood of each answer option under the frozen (and, for pruned
runs, gated) model, argmax, no gradient updates, no task-specific training
of any kind. This is mechanically identical in spirit to the ppl eval
already used throughout this project -- lm-evaluation-harness just
aggregates it as accuracy instead of average NLL. See the conversation this
script was built from for the full "does downstream eval require
fine-tuning" discussion (answer: no).

PIPELINE, matching this project's own scope constraint (no fine-tuning of
the base model, ever) AND DISP-LLM's own PRIMARY reported numbers (their
`W Update? = (cross)` row, not their separate LoRA-finetuned appendix
variant): pretrain (external) -> prune (frozen weights, done already by
train_pruner_llama2_7b_wikitext2.py) -> evaluate zero-shot. No fine-tuning
step anywhere in this script.

TASK SUITE -- PIQA, HellaSwag, WinoGrande, ARC-easy, ARC-challenge, BoolQ,
OpenBookQA. Matches DISP-LLM Table 1/3's exact suite (minus dataset-specific
naming differences) so results are directly comparable to the Llama-2-7B
numbers already cited in past-work.md / the paper draft, and to SlimLLM's
Table 2 (LLaMA2-7B) suite.

*** UNTESTED. *** No local 7B hardware, and lm-eval is not installed in
this repo's venv (checked before writing this). Verified: py_compile only.
Requires `pip install lm-eval` (the current EleutherAI package name; API
used here -- lm_eval.simple_evaluate + lm_eval.models.huggingface.HFLM --
has been stable across recent lm-eval releases, but pin a version before
trusting results long-term).

Reuses Pruner / load_llama2_7b / get_mlp_weights / apply_gates verbatim
from train_pruner_llama2_7b_wikitext2.py (same directory) via direct import
-- no logic duplicated, same pattern as train_pruner_opt125m_converge.py
importing from train_pruner_opt125m.py.
"""
import os
import sys
import json
import argparse
import contextlib

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_pruner_llama2_7b_wikitext2 import (
    Pruner, load_llama2_7b, get_mlp_weights, apply_gates,
    N_LAYERS, N_INTER, LAYER_SHAPE, LLAMA2_REPO,
)

os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")

DEFAULT_TASKS = ["piqa", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "boolq", "openbookqa"]
DEFAULT_SWEEP_DIR = "/workspace/results/llama2_7b_wikitext2_sweep"
DEFAULT_LAMBDAS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]  # excludes 1.4 -- never converged, see F6


def load_pruner_and_gates(model, ckpt_path, device):
    """Reconstruct the exact same gates the training run's final eval used --
    deterministic given the same frozen weights (STE forward is a hard
    threshold, no sampling), so this is not an approximation of the
    checkpoint's mask, it IS the checkpoint's mask.

    weights_only=False is required and intentional, not an oversight: our
    checkpoint dict mixes tensors with plain Python objects (converged: bool,
    lr_state: str, per_layer_kept: list, ...) by design (see
    train_pruner_llama2_7b_wikitext2.py's torch.save call). Recent PyTorch
    versions default torch.load to weights_only=True, which rejects exactly
    this shape of file. Safe here specifically because this is our own
    checkpoint, created by our own training run, not an untrusted download."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    layer_shapes = [LAYER_SHAPE] * N_LAYERS
    pruner = Pruner(layer_shapes, embed_dim=ckpt["embed_dim"], lstm_hidden=ckpt["lstm_hidden"]).to(device)
    pruner.load_state_dict(ckpt["pruner_state_dict"])
    pruner.eval()
    with torch.no_grad():
        gates = pruner(get_mlp_weights(model))
    return gates, ckpt


@contextlib.contextmanager
def _null_ctx():
    yield


def run_eval(model, tokenizer, device, tasks, num_fewshot, batch_size, gates=None, limit=None):
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    ctx = apply_gates(model, gates) if gates is not None else _null_ctx()
    with ctx:
        lm = HFLM(pretrained=model, tokenizer=tokenizer, device=str(device), batch_size=batch_size)
        results = simple_evaluate(model=lm, tasks=tasks, num_fewshot=num_fewshot,
                                  bootstrap_iters=0, limit=limit)
    return results


def summarize(results, tasks):
    """Pull the primary accuracy metric per task -- acc_norm where the task
    provides it (matches DISP-LLM/SlimLLM's own reporting convention: they
    use acc_norm for HellaSwag/ARC/OBQA/PIQA, acc for BoolQ/WinoGrande),
    plus an unweighted average across tasks. Raises loudly on a missing
    metric key rather than silently returning NaN -- a wrong task name or an
    lm_eval results-schema mismatch should fail the sanity check immediately,
    not surface as garbage numbers after a multi-hour run."""
    per_task = {}
    for t in tasks:
        r = results["results"].get(t)
        if r is None:
            raise KeyError(f"task {t!r} missing from results['results']; got keys "
                           f"{list(results['results'].keys())}. Wrong task name, or "
                           f"lm_eval's results schema differs from what this script assumes.")
        metric = "acc_norm,none" if "acc_norm,none" in r else "acc,none"
        if metric not in r:
            raise KeyError(f"task {t!r}: neither 'acc_norm,none' nor 'acc,none' in {list(r.keys())}. "
                           f"lm_eval's metric-key naming may have changed -- update summarize().")
        per_task[t] = r[metric] * 100
    avg = sum(per_task.values()) / len(per_task)
    return per_task, avg


def write_summary(all_summaries, tasks, out_dir):
    """Called after every completed run, not just at the end -- if the sweep
    dies partway through, whatever's on disk here is always current, not
    stale from before the crash."""
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)

    sep = "-" * 100
    header = f"{'run':>12} | {'%pruned':>8} | " + " | ".join(f"{t[:10]:>10}" for t in tasks) + f" | {'avg':>6}"
    lines = ["Llama-2-7B zero-shot downstream accuracy, WikiText-2-pruned checkpoints", sep, header, sep]
    for name, s in all_summaries.items():
        if "error" in s:
            lines.append(f"{name:>12} | FAILED: {s['error']}")
            continue
        pct = f"{s.get('pct_pruned', 0.0):>7.2f}%" if "pct_pruned" in s else f"{'--':>8}"
        row = " | ".join(f"{s['per_task'][t]:>10.2f}" for t in tasks)
        lines.append(f"{name:>12} | {pct} | {row} | {s['average']:>6.2f}")
    summary_str = "\n".join(lines)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(summary_str + "\n")
    return summary_str


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", type=str, default=DEFAULT_SWEEP_DIR,
                    help="Directory containing lambda_X/pruner.pt checkpoints "
                         "from train_pruner_llama2_7b_wikitext2.py.")
    ap.add_argument("--lambdas", type=float, nargs="+", default=DEFAULT_LAMBDAS,
                    help="Which lambda_X checkpoints to evaluate. Default excludes "
                         "1.4 -- that run never converged (F6), not a real checkpoint "
                         "to report downstream numbers for.")
    ap.add_argument("--tasks", type=str, nargs="+", default=DEFAULT_TASKS)
    ap.add_argument("--num_fewshot", type=int, default=0, help="0 = zero-shot, matching the literature.")
    ap.add_argument("--batch_size", type=str, default="auto")
    ap.add_argument("--skip_dense", action="store_true",
                    help="Skip the dense (gates=None) baseline eval -- only useful if "
                         "you already have it from a previous run of this script.")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_dir", type=str, default="/workspace/results/llama2_7b_downstream")
    ap.add_argument("--sanity_check", action="store_true",
                    help="Run one task, limit=5 examples, dense model only, then exit. "
                         "Validates the full pipeline (model load, HFLM accepting a "
                         "pre-loaded model instance, lm_eval task names, results-dict "
                         "schema) in well under a minute before committing to the full "
                         "multi-hour sweep -- several of those are untested assumptions "
                         "about lm_eval's API, see this script's module docstring. "
                         "NOT optional before a real run, same convention as the "
                         "training script's own --sanity_check.")
    args = ap.parse_args()

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if os.environ.get("HF_TOKEN") is None:
        print("WARNING: HF_TOKEN not set -- meta-llama/Llama-2-7b-hf is gate-licensed, "
              "see train_pruner_llama2_7b_wikitext2.py's docstring.", flush=True)

    print("Loading Llama-2-7B ...", flush=True)
    model = load_llama2_7b(device)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(LLAMA2_REPO)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.sanity_check:
        print("\n" + "=" * 70, flush=True)
        print(f"SANITY CHECK -- task={args.tasks[0]!r}, limit=5, dense model only", flush=True)
        print("=" * 70, flush=True)
        try:
            test_results = run_eval(model, tokenizer, device, [args.tasks[0]], 0,
                                    args.batch_size, gates=None, limit=5)
            per_task, avg = summarize(test_results, [args.tasks[0]])
            print(f"  result: {per_task}", flush=True)
            print("  PASS -- pipeline works end to end (model load, HFLM wrapper, task "
                  "registry, results parsing all OK). Safe to run the full sweep.", flush=True)
        except Exception as e:
            print(f"  FAIL -- {type(e).__name__}: {e}", flush=True)
            print("  DO NOT run the full sweep until this is root-caused -- see the "
                  "'untested lm_eval API' caveats in this script's module docstring for "
                  "where to start looking.", flush=True)
            sys.exit(1)
        print("=" * 70, flush=True)
        return

    all_summaries = {}

    if not args.skip_dense:
        print(f"\n{'='*70}\nDense (unpruned) baseline -- {args.tasks}\n{'='*70}", flush=True)
        try:
            dense_results = run_eval(model, tokenizer, device, args.tasks, args.num_fewshot,
                                     args.batch_size, gates=None)
            per_task, avg = summarize(dense_results, args.tasks)
            print(f"  per-task: {per_task}", flush=True)
            print(f"  average : {avg:.2f}%", flush=True)
            with open(os.path.join(args.out_dir, "dense_full.json"), "w") as f:
                json.dump(dense_results["results"], f, indent=2, default=str)
            all_summaries["dense"] = {"per_task": per_task, "average": avg}
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
            print("  Continuing to pruned checkpoints -- dense can be backfilled later "
                  "with --skip_dense removed on a re-run.", flush=True)
            all_summaries["dense"] = {"error": f"{type(e).__name__}: {e}"}
            if device.type == "cuda":
                torch.cuda.empty_cache()
        write_summary(all_summaries, args.tasks, args.out_dir)

    for lam in args.lambdas:
        ckpt_path = os.path.join(args.sweep_dir, f"lambda_{lam}", "pruner.pt")
        if not os.path.exists(ckpt_path):
            print(f"  [skip] lambda={lam}: no checkpoint at {ckpt_path}", flush=True)
            continue
        print(f"\n{'='*70}\nlambda={lam} -- {ckpt_path}\n{'='*70}", flush=True)
        try:
            gates, ckpt = load_pruner_and_gates(model, ckpt_path, device)
            pct_pruned = 100 * (1 - sum(g.mean().item() for g in gates) / len(gates))
            print(f"  reconstructed mask: {pct_pruned:.2f}% FFN neurons pruned "
                  f"(checkpoint's own record: converged={ckpt['converged']}, "
                  f"pruned_ppl={ckpt['pruned_ppl']:.3f})", flush=True)

            results = run_eval(model, tokenizer, device, args.tasks, args.num_fewshot,
                               args.batch_size, gates=gates)
            per_task, avg = summarize(results, args.tasks)
            print(f"  per-task: {per_task}", flush=True)
            print(f"  average : {avg:.2f}%", flush=True)

            with open(os.path.join(args.out_dir, f"lambda_{lam}_full.json"), "w") as f:
                json.dump(results["results"], f, indent=2, default=str)
            all_summaries[f"lambda_{lam}"] = {
                "pct_pruned": pct_pruned, "per_task": per_task, "average": avg,
                "pruned_ppl": ckpt["pruned_ppl"], "converged": ckpt["converged"],
            }
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
            print(f"  Continuing to the next lambda -- this one is recorded as failed, "
                  f"not silently dropped.", flush=True)
            all_summaries[f"lambda_{lam}"] = {"error": f"{type(e).__name__}: {e}"}
            if device.type == "cuda":
                torch.cuda.empty_cache()

        write_summary(all_summaries, args.tasks, args.out_dir)

    summary_str = write_summary(all_summaries, args.tasks, args.out_dir)
    print("\n" + summary_str, flush=True)
    print(f"\nResults -> {args.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
