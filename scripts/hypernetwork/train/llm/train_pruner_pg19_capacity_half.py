"""
Pruner-capacity scaling ablation on pg19 (B7b / H1, diary/ideas.md) --
the HALF-capacity counterpart to train_pruner_pg19_capacity_scaling.py
(the 2x-ish ablation, tmp4). Same convergence-based stopping + train/test
CE gap diagnostic, same 4-lambda grid, same everything -- only the pruner
is smaller this time, not bigger.

Not standalone -- imports train_one_converge, MODEL_SPECS etc. straight
from train_pruner_pg19_converge.py, same pattern as the 2x-capacity script.

WHY: the 2x-capacity ablation (tmp4) did NOT cleanly resolve whether
OPT-125M/GPT-2's high-lambda underperformance is a capacity bottleneck or
an inherently harder landscape -- results were genuinely mixed (3/8 points
better, 3/8 worse, 2/8 flat), and the most informative single result was a
NEGATIVE one: opt125m lambda=0.3 (a "should stay stable" control point)
converged in FEWER steps at WORSE ppl with the bigger pruner, and its
gap-diagnostic trajectory showed a still-rising, non-noisy pct_pruned
trend that the fixed 250-step convergence window mistook for a plateau --
because the bigger/more expressive pruner moves through weight space
faster per step than the window was tuned for. That result reframed the
open question: it's not just "does more capacity help," it's "does the
FIXED convergence window even mean the same thing at different pruner
sizes." A single higher-capacity datapoint can't distinguish "capacity
helps up to a point" from "capacity monotonically helps but our window
increasingly mismeasures it" from "capacity doesn't matter, that one
result was noise." A THIRD point in the other direction is the cheapest
next piece of evidence: if a half-size pruner shows the same kind of
still-rising-trend-mistaken-for-plateau failure MORE often (smaller/less
expressive pruner, if anything, might move SLOWER per step -- opposite
prediction from what caused tmp4's opt125m lambda=0.3 case, so a clean
test of whether that mechanism is capacity-monotonic), that's evidence
the window-mismatch story generalizes; if half-size instead shows uniform
clean degradation with no window-mismatch artifacts, that's more
consistent with capacity mattering in the way H1 originally proposed.

PRUNER SIZE -- embed_dim=32, lstm_hidden=64, i.e. HALVE EACH DIMENSION,
not "find the config with exactly half the total params" (deliberately
the same asymmetry-accepting approach as tmp4's actual run, which used
embed_dim=128/lstm_hidden=256 -- naive-halve-each-dimension, not
exact-fraction-of-params). Verified against the real Pruner class:
base (64,128) = 2,037,517 params; half-each-dim (32,64) = 953,229 params,
ratio = 0.4678 (NOT exactly 0.5 -- same row-encoder-linear vs.
BiLSTM-quadratic scaling asymmetry noted in the 2x script's docstring,
just in the other direction: halving lstm_hidden shrinks the quadratic
LSTM term by 4x while halving embed_dim only shrinks the linear
row-encoder term by ~2x, so the LSTM's share of the total shrinks
disproportionately and the blended ratio lands above the naive 0.25x
"halve both, expect quarter" guess but below a clean 0.5x).

LAMBDA GRID -- same 4 points as the 2x-capacity script and already
present in train_pruner_pg19_converge.py's 9-point base sweep (0.05, 0.3,
0.8, 1.6), for direct three-way comparability across base / 2x / half
capacity at identical operating points, same models, same seed.

Usage:
    python train_pruner_pg19_capacity_half.py [--models gpt2 opt125m]
        [--lambdas 0.05 0.3 0.8 1.6] [--seeds 0] [--max_steps 18000]
        [--stop_pod]
"""
import csv
import os
import sys
import argparse

import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_pruner_pg19_converge import (
    MODEL_SPECS, GAP_CSV_COLUMNS, train_one_converge, get_pg19_loaders,
    plot_model_comparison, stop_pod,
)

OUT_ROOT = "/workspace/results/pg19_capacity_half_sweep"

# Verified via direct Pruner instantiation (see module docstring):
# base (64,128) = 2,037,517 params; half-each-dim (32,64) = 953,229 params,
# ratio = 0.4678 -- NOT exactly 0.5x, halving each dimension on purpose
# (mirrors tmp4's actual naive-halve-not-exact-fraction approach).
PRUNER_HALF_EMBED_DIM = 32
PRUNER_HALF_LSTM_HIDDEN = 64

# Same 4 lambdas as the 2x-capacity script, all already in
# train_pruner_pg19_converge.py's 9-point default grid -- see module
# docstring for why these four (base/2x/half three-way comparability).
DEFAULT_LAMBDAS = [0.05, 0.3, 0.8, 1.6]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gpt2", "opt125m"],
                    choices=["gpt2", "opt125m"])
    ap.add_argument("--lambdas", type=float, nargs="+", default=DEFAULT_LAMBDAS,
                    help=f"Same grid for both models. Default: {DEFAULT_LAMBDAS} "
                         "-- matches both the base sweep and the 2x-capacity "
                         "ablation for direct three-way comparability.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--check_every", type=int, default=50)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--rel_tol", type=float, default=0.05)
    ap.add_argument("--abs_tol", type=float, default=0.01)
    ap.add_argument("--burn_in", type=int, default=500)
    ap.add_argument("--max_steps", type=int, default=18000,
                    help="Safety cap, not a target -- same reasoning as "
                         "train_pruner_opt125m_converge.py.")
    ap.add_argument("--gap_eval_every", type=int, default=200,
                    help="Must be a multiple of --check_every.")
    ap.add_argument("--gap_eval_tokens", type=int, default=50_000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--n_train_tokens", type=int, default=None,
                    help="Default (None) sizes to max_steps*batch_size*seq_len -- "
                         "same reasoning as train_pruner_pg19_converge.py.")
    ap.add_argument("--n_test_tokens", type=int, default=245_000)
    ap.add_argument("--embed_dim", type=int, default=PRUNER_HALF_EMBED_DIM,
                    help=f"Default {PRUNER_HALF_EMBED_DIM} -- half-each-dimension "
                         "pruner (see module docstring), NOT the base size. "
                         "Override to run some other capacity point instead.")
    ap.add_argument("--lstm_hidden", type=int, default=PRUNER_HALF_LSTM_HIDDEN,
                    help=f"Default {PRUNER_HALF_LSTM_HIDDEN} -- pairs with "
                         "--embed_dim's default (ratio=0.4678x base params, "
                         "not exactly 0.5x -- see module docstring).")
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--log_every", type=int, default=250)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_dir", type=str, default=OUT_ROOT)
    ap.add_argument("--stop_pod", action="store_true")
    args = ap.parse_args()
    out_root = args.out_dir

    if args.n_train_tokens is None:
        args.n_train_tokens = args.max_steps * args.batch_size * args.seq_len
        print(f"--n_train_tokens not set, defaulting to max_steps*batch_size*seq_len "
              f"= {args.n_train_tokens:,} tokens")

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device} | models={args.models} | λs={args.lambdas} | "
          f"pruner embed_dim={args.embed_dim} lstm_hidden={args.lstm_hidden} "
          f"(half-capacity ablation) | convergence-based (max_steps={args.max_steps})")
    print(f"Output: {out_root}\n")

    os.makedirs(out_root, exist_ok=True)
    all_results = []

    for model_name in args.models:
        spec = MODEL_SPECS[model_name]
        print(f"\n{'='*70}\nLoading {spec['display_name']} ...\n{'='*70}", flush=True)
        model = spec["load_fn"](device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{spec['display_name']} loaded — {n_params:,} params, frozen.", flush=True)

        tokenizer = spec["tokenizer_fn"]()
        print(f"Streaming pg19 for {spec['display_name']}'s tokenizer "
              f"(train={args.n_train_tokens:,} test={args.n_test_tokens:,} tokens) ...", flush=True)
        train_loader, test_ids = get_pg19_loaders(
            tokenizer, args.seq_len, args.batch_size,
            args.n_train_tokens, args.n_test_tokens, model_name,
        )
        train_ids_flat = train_loader.dataset.tensors[0].reshape(-1)
        print(f"Data ready: train_blocks={len(train_loader.dataset)} "
              f"test_tokens={test_ids.size(0):,} "
              f"(eval: max_length={spec['eval_max_length']} stride={spec['eval_stride']})", flush=True)

        total_runs = len(args.lambdas) * len(args.seeds)
        run_num = 0
        for lam in args.lambdas:
            for seed in args.seeds:
                run_num += 1
                tqdm.write(f"\n{'='*70}\n[{model_name}] Run {run_num}/{total_runs} "
                          f"(half-capacity pruner)\n{'='*70}")
                run_dir = (os.path.join(out_root, model_name, f"lambda_{lam}", f"seed_{seed}")
                           if len(args.seeds) > 1
                           else os.path.join(out_root, model_name, f"lambda_{lam}"))
                res = train_one_converge(model_name, model, train_loader, train_ids_flat,
                                         test_ids, lam, seed, args, device, run_dir)
                all_results.append(res)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    plot_model_comparison(all_results, os.path.join(out_root, "gpt2_vs_opt125m_pg19_half.png"))

    combined_path = os.path.join(out_root, "gap_diagnostic_all.csv")
    with open(combined_path, "w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=GAP_CSV_COLUMNS)
        writer.writeheader()
        for model_name in args.models:
            for lam in args.lambdas:
                for seed in args.seeds:
                    run_dir = (os.path.join(out_root, model_name, f"lambda_{lam}", f"seed_{seed}")
                               if len(args.seeds) > 1
                               else os.path.join(out_root, model_name, f"lambda_{lam}"))
                    run_csv = os.path.join(run_dir, "gap_diagnostic.csv")
                    if not os.path.exists(run_csv):
                        continue
                    with open(run_csv, newline="") as in_f:
                        for row in csv.DictReader(in_f):
                            writer.writerow(row)
    print(f"Combined gap diagnostic -> {combined_path}")

    sep = "-" * 100
    rows = [f"pg19 HALF-CAPACITY sweep — models={args.models} | seeds={args.seeds} | "
           f"embed_dim={args.embed_dim} lstm_hidden={args.lstm_hidden} | "
           f"max_steps={args.max_steps} | device={device}", sep,
           f"{'model':>10} {'lambda':>7} {'seed':>5} | {'steps':>7} {'conv?':>6} | "
           f"{'% pruned':>9} | {'orig ppl':>9} | {'pruned ppl':>10} | {'ppl rise':>9}", sep]
    for r in all_results:
        rows.append(f"{r['model']:>10} {r['lambda']:>7} {r['seed']:>5} | "
                    f"{r['steps_taken']:>7} {'YES' if r['converged'] else 'NO':>6} | "
                    f"{r['pct_pruned']:>8.2f}% | {r['orig_ppl']:>9.3f} | "
                    f"{r['pruned_ppl']:>10.3f} | {r['pruned_ppl']-r['orig_ppl']:>+9.3f}")
    summary_str = "\n".join(rows)
    with open(os.path.join(out_root, "summary.txt"), "w") as f:
        f.write(summary_str + "\n")
    print("\n" + summary_str)
    print(f"\nResults → {out_root}/  -- compare point-for-point against "
          f"tmp3/pg19_converge_sweep (base pruner) and tmp4/pg19_capacity_2x_sweep "
          f"(2.26x pruner), same 4 lambdas present in both")

    if args.stop_pod:
        stop_pod()


if __name__ == "__main__":
    main()
