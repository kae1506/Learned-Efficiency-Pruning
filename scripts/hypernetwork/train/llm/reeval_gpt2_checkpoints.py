"""
Re-evaluate already-trained GPT-2 pruner checkpoints under a different eval
protocol, WITHOUT retraining.

Built for one specific question: does F16's "pruning improves ppl" result
(diary/crisp-findings.md) survive the eval-protocol fix in
train_pruner_gpt2.py (non-overlapping blocks -> sliding window), or was it an
artifact of the old eval? Training decided which neurons get pruned (the
--lambdas / per_layer_kept numbers) and is completely unaffected by this --
only the reported ppl for a GIVEN already-trained pruner can change.

Loads each pruner.pt under a results dir (e.g. gpt2_results/v1/lambda_*/
seed_*/pruner.pt), reconstructs the Pruner, gets its final gates, and
re-evaluates orig_ppl/pruned_ppl with the CURRENT (sliding-window) evaluate()
imported from train_pruner_gpt2.py -- so there's one copy of that logic, not
a forked duplicate that can drift out of sync.

Run from the same directory as train_pruner_gpt2.py:
    python reeval_gpt2_checkpoints.py \
        --results_dir /path/to/experiments/latest/gpt2_results/v1 \
        --device cuda   # or mps / cpu

Writes <results_dir>/reeval.csv (old-protocol vs new-protocol ppl side by
side per checkpoint) and prints the same as a table.
"""

import argparse
import csv
import glob
import os

import numpy as np
import torch

from train_pruner_gpt2 import (
    Pruner, load_gpt2, get_loaders, get_mlp_weights, evaluate,
    N_LAYERS, N_INTER, LAYER_SHAPE,
)


def find_checkpoints(results_dir: str) -> list[str]:
    multi_seed = glob.glob(os.path.join(results_dir, "lambda_*", "seed_*", "pruner.pt"))
    single_seed = glob.glob(os.path.join(results_dir, "lambda_*", "pruner.pt"))
    return multi_seed + single_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir",     type=str, required=True,
                    help="e.g. experiments/latest/gpt2_results/v1")
    ap.add_argument("--device",          type=str, default="cuda")
    ap.add_argument("--seq_len",         type=int, default=512,
                    help="Only used to build the (unused) train_loader inside "
                         "get_loaders(); doesn't affect eval.")
    ap.add_argument("--batch_size",      type=int, default=8)
    ap.add_argument("--eval_max_length", type=int, default=1024)
    ap.add_argument("--eval_stride",     type=int, default=512)
    ap.add_argument("--out_csv",         type=str, default=None,
                    help="default: <results_dir>/reeval.csv")
    args = ap.parse_args()

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    ckpt_paths = find_checkpoints(args.results_dir)
    if not ckpt_paths:
        raise SystemExit(f"No pruner.pt found under {args.results_dir}")
    print(f"Found {len(ckpt_paths)} checkpoints under {args.results_dir}")

    print("Loading GPT-2 small ...", flush=True)
    model = load_gpt2(device)

    print("Loading WikiText-2 (sliding-window test set) ...", flush=True)
    _, test_ids = get_loaders(args.seq_len, args.batch_size)

    # Same frozen model, same eval protocol -> orig_ppl is identical for
    # every checkpoint. Compute it once instead of once per checkpoint.
    print("Evaluating unpruned baseline under the NEW protocol ...", flush=True)
    orig_ce_new = evaluate(model, test_ids, device, gates=None, desc="orig (new protocol)",
                           max_length=args.eval_max_length, stride=args.eval_stride)
    orig_ppl_new = float(np.exp(orig_ce_new))
    print(f"  orig_ppl, new protocol: {orig_ppl_new:.3f}\n")

    rows = []
    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        lam, seed = ckpt["lambda"], ckpt["seed"]

        layer_shapes = [LAYER_SHAPE] * N_LAYERS
        pruner = Pruner(layer_shapes, embed_dim=ckpt["embed_dim"],
                        lstm_hidden=ckpt["lstm_hidden"]).to(device)
        pruner.load_state_dict(ckpt["pruner_state_dict"])
        pruner.eval()

        with torch.no_grad():
            gates = pruner(get_mlp_weights(model))
        per_layer_kept = [int(g.sum().item()) for g in gates]
        pct_pruned = 100.0 * (1 - sum(per_layer_kept) / (N_LAYERS * N_INTER))

        pruned_ce_new = evaluate(model, test_ids, device, gates=gates,
                                 desc=f"λ={lam} seed={seed} (new protocol)",
                                 max_length=args.eval_max_length, stride=args.eval_stride)
        pruned_ppl_new = float(np.exp(pruned_ce_new))

        old_orig_ppl   = ckpt.get("orig_ppl")
        old_pruned_ppl = ckpt.get("pruned_ppl")
        old_delta = (old_pruned_ppl - old_orig_ppl) if old_orig_ppl is not None else None
        new_delta = pruned_ppl_new - orig_ppl_new

        # efficiency = pct_pruned / exp(ΔCE) -- see train_pruner_gpt2.py's
        # plot_efficiency() docstring for why (replaces the old ppl-based
        # max(drop, 0.5) clamp, which goes flat/uninformative whenever
        # pruning improves ppl). Computed directly from CE, not round-tripped
        # through ppl, to avoid an unnecessary log/exp precision loss.
        delta_ce_new = pruned_ce_new - orig_ce_new
        efficiency_new = pct_pruned / np.exp(delta_ce_new)

        print(f"  λ={lam:<6} seed={seed} | pruned {pct_pruned:5.2f}% | "
              f"OLD proto: {old_orig_ppl:.3f} -> {old_pruned_ppl:.3f} ({old_delta:+.3f}) | "
              f"NEW proto: {orig_ppl_new:.3f} -> {pruned_ppl_new:.3f} ({new_delta:+.3f}) | "
              f"eff {efficiency_new:.2f}",
              flush=True)

        rows.append({
            "lambda": lam, "seed": seed, "pct_pruned": pct_pruned,
            "orig_ppl_old_protocol":   old_orig_ppl,
            "pruned_ppl_old_protocol": old_pruned_ppl,
            "delta_ppl_old_protocol":  old_delta,
            "orig_ppl_new_protocol":   orig_ppl_new,
            "pruned_ppl_new_protocol": pruned_ppl_new,
            "delta_ce_new_protocol":   delta_ce_new,
            "efficiency_new_protocol": efficiency_new,
            "delta_ppl_new_protocol":  new_delta,
        })

    rows.sort(key=lambda r: (r["lambda"], r["seed"]))

    out_csv = args.out_csv or os.path.join(args.results_dir, "reeval.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
