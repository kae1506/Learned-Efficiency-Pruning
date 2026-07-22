"""
Pruner-capacity scaling ablation on pg19 (B7b / H1, diary/ideas.md) --
GPT-2 small vs OPT-125M, same convergence-based stopping + train/test CE
gap diagnostic as train_pruner_pg19_converge.py, but with the pruner's
total parameter count EXACTLY DOUBLED.

Not standalone -- imports train_one_converge (the single-run training loop,
convergence check, and gap diagnostic all already wired together) plus the
model dispatch straight from train_pruner_pg19_converge.py. Nothing about
the training/convergence/diagnostic logic changes here -- only the pruner
size and the lambda grid tested.

WHY: F20/B7(b) raised an open question the 9-point pg19 sweep (tmp3, see
diary chat log) reinforced rather than answered -- OPT-125M's high-lambda
tail (0.8, 1.6) converges (by the block-mean criterion) to a WORSE result
than the old fixed-18750-step budget achieved, on BOTH WikiText-2 and pg19
now. Two live, undistinguished hypotheses for why: (a) the pruner's fixed
capacity (embed_dim=64, lstm_hidden=128, ~2M params, unchanged since
project inception) is the actual bottleneck at hard/high-lambda operating
points -- a bigger pruner might find the same or a better mask BEFORE the
block-mean check declares convergence, i.e. genuinely faster AND better;
(b) it's an inherently harder/slower optimization landscape at high lambda
regardless of pruner size, and the current convergence tolerance is simply
too loose there -- a capacity increase wouldn't fix it, only a stricter/
longer confirmation window would (see the tmp3 discussion's proposed next
step). This script is the controlled test for (a) -- everything else
(architecture, training regime, convergence criterion, tolerances) held
fixed, only pruner capacity changes.

PRUNER SIZE -- embed_dim=144, lstm_hidden=216, NOT the naive "double both
numbers" (embed_dim=128, lstm_hidden=256). The row-encoder scales roughly
linearly with embed_dim (per-layer: d_in*embed_dim + embed_dim + embed_dim
+ 1, d_in=768 fixed) but the BiLSTM scales roughly QUADRATICALLY with
lstm_hidden (4 gates x 2 directions x (hidden^2 + hidden^2 + 2*hidden)) --
doubling both independently gives 2.257x total params, not 2x (verified
against the real Pruner class: base=2,037,517 -> naive-double=4,599,309).
embed_dim=144/lstm_hidden=216 gives 4,075,069 params, ratio=2.0000 (off by
35 params / 0.001%) -- verified the same way, not just computed by hand.

LAMBDA GRID -- 4 points, all already present in train_pruner_pg19_converge
.py's default 9-point grid (tmp3 results already have base-size-pruner
data at all four, so this ablation's results are directly point-for-point
comparable against tmp3 without needing anything else re-run):
  0.05, 0.3  -- "known-good" control points where the base-size pruner
               already converges cleanly and matches-or-beats the old
               fixed-step baseline (see tmp3 discussion) -- included to
               confirm doubling capacity doesn't quietly hurt what
               already works, not just to chase the failure case.
  0.8,  1.6  -- the two specifically-flagged high-lambda points where the
               base-size pruner's convergence-based result underperformed
               the old fixed-budget baseline on both WikiText-2 and pg19 --
               the actual points this ablation exists to test.
Same grid for both models, same reasoning as train_pruner_pg19_converge.py
(mirrors that script's own choice, not independently re-derived per model).

Usage:
    python train_pruner_pg19_capacity_scaling.py [--models gpt2 opt125m]
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

OUT_ROOT = "/workspace/results/pg19_capacity_2x_sweep"

# Verified via direct Pruner instantiation (see module docstring), not
# just computed by hand: base (64,128) = 2,037,517 params;
# (144,216) = 4,075,069 params; ratio = 2.0000.
PRUNER_2X_EMBED_DIM = 144
PRUNER_2X_LSTM_HIDDEN = 216

# Same 4 lambdas for both models, all already in train_pruner_pg19_converge
# .py's 9-point default grid -- see module docstring for why these four.
DEFAULT_LAMBDAS = [0.05, 0.3, 0.8, 1.6]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gpt2", "opt125m"],
                    choices=["gpt2", "opt125m"])
    ap.add_argument("--lambdas", type=float, nargs="+", default=DEFAULT_LAMBDAS,
                    help=f"Same grid for both models. Default: {DEFAULT_LAMBDAS} "
                         "-- all four already tested at base pruner size in "
                         "train_pruner_pg19_converge.py's default sweep, so "
                         "results here are directly comparable point-for-point.")
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
    ap.add_argument("--embed_dim", type=int, default=PRUNER_2X_EMBED_DIM,
                    help=f"Default {PRUNER_2X_EMBED_DIM} -- the 2x-capacity "
                         "pruner (see module docstring for the exact-2x "
                         "derivation), NOT the base size. Override to run "
                         "some other capacity point instead.")
    ap.add_argument("--lstm_hidden", type=int, default=PRUNER_2X_LSTM_HIDDEN,
                    help=f"Default {PRUNER_2X_LSTM_HIDDEN} -- pairs with "
                         "--embed_dim's default for exactly 2x base params.")
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

    pruner_params_2x = None  # printed once loaded, informational
    print(f"Device: {device} | models={args.models} | λs={args.lambdas} | "
          f"pruner embed_dim={args.embed_dim} lstm_hidden={args.lstm_hidden} "
          f"(2x-capacity ablation) | convergence-based (max_steps={args.max_steps})")
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
                          f"(2x-capacity pruner)\n{'='*70}")
                run_dir = (os.path.join(out_root, model_name, f"lambda_{lam}", f"seed_{seed}")
                           if len(args.seeds) > 1
                           else os.path.join(out_root, model_name, f"lambda_{lam}"))
                res = train_one_converge(model_name, model, train_loader, train_ids_flat,
                                         test_ids, lam, seed, args, device, run_dir)
                all_results.append(res)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    plot_model_comparison(all_results, os.path.join(out_root, "gpt2_vs_opt125m_pg19_2x.png"))

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
    rows = [f"pg19 2x-CAPACITY sweep — models={args.models} | seeds={args.seeds} | "
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
          f"tmp3/pg19_converge_sweep (base pruner size, same 4 lambdas present there)")

    if args.stop_pod:
        stop_pod()


if __name__ == "__main__":
    main()
