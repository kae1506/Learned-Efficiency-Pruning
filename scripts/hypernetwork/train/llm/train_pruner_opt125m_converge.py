"""
OPT-125M MLP pruner — dense 9-λ sweep, 1 seed, CONVERGENCE-BASED stopping
instead of a fixed step count. Answers B7(a) (ideas.md): does steps-needed
actually vary by λ the way F20 suggests, and if we let each run stop when
IT decides it's done, what does the resulting (λ, steps_to_converge) curve
look like?

Not standalone -- imports the model/data/eval plumbing from
train_pruner_opt125m.py (same directory), same pattern as
train_pruner_pg19_sweep.py.

CONVERGENCE CRITERION (see diary chat log for the derivation, and for the
v1 bug this version fixes):
  Checkpoint every `check_every` (50) steps. At checkpoint step t, look at
  the trailing window of the last `window` (5) checkpoints (t, t-50, t-100,
  t-150, t-200 -- 200 steps of history). For EVERY layer independently
  (not an aggregate/mean -- F20's actual failure mode was a SINGLE layer
  still drifting while the other 11 had long settled; averaging would
  dilute exactly the signal we're trying to catch), require every reading
  in that window to be within tol of the most recent reading:

      tol = max(rel_tol * |mean_g_l(t)|, abs_tol)   # rel_tol=0.05, abs_tol=0.01
      |mean_g_l(t_i) - mean_g_l(t)| <= tol   for all t_i in window, all layers l

  mean_g_l(t) = the BLOCK MEAN of history["per_layer_keep"][l] over the
  check_every raw steps ending at t (i.e. steps [t-check_every, t)), NOT
  the single raw value at step t.

  v1 of this script checked the raw per-step value directly, reasoning that
  gates = pruner_phi(frozen_weights) don't depend on the current minibatch
  so there was nothing to smooth. That reasoning has a hole: while gates
  at a FIXED phi are indeed batch-independent, phi_t itself is a noisy
  trajectory -- Adam gets a fresh stochastic-minibatch gradient every
  step, forever, with no LR decay -- so the raw per-step readout inherits
  a NON-DECAYING noise floor through phi's own update noise. A real run
  (dense-9-lambda sweep, all of lambda in [0.1, 0.2, 0.25, 0.3, 0.4])
  confirmed this empirically: every plot.png showed all 12 layers visibly
  flat by ~2000-2500 steps, yet none satisfied the raw-signal check before
  hitting the 18000-step cap. The plotted "flat" line is ALREADY the
  100-step-smoothed version (plot_one_run's per-layer panel only ever
  plots _smooth(per), never the raw series) -- the raw signal the v1
  checker actually read was noisier than what the plot showed, and
  requiring all 12 layers x 5 checkpoints (60 simultaneous comparisons)
  to independently avoid that non-decaying noise was, in practice, close
  to impossible to satisfy no matter how long training ran. Block-averaging
  each checkpoint reading over the preceding check_every raw steps (same
  idea as the plot's own smoothing, just applied to what's actually
  checked) filters that noise floor out directly, rather than requiring a
  lucky quiet patch across 60 conditions at once.

  abs_tol exists because rel_tol alone is unstable near g_l -> 0 (a heavily
  pruned layer jittering 0.02 -> 0.021 is a "5% violation" that means
  nothing) -- use whichever tolerance is looser.

  burn_in (500 steps): the row-encoder's output bias is initialized to
  +2.0 (gates start near-fully-open by design), so the first several
  hundred steps can look artificially flat before the logit has even
  moved off init -- convergence checks are disabled before this floor to
  avoid a false-positive "converged" at step ~0.

  max_steps (30000, a safety valve, NOT a soft default like the old
  script's 18750): F20 already showed OPT's extreme lambdas (0.01, 1.8)
  were STILL drifting at 18750. If a run hits max_steps without
  satisfying the convergence check, it stops anyway and is flagged in the
  summary as NOT CONVERGED -- a genuinely-hard-to-converge lambda should
  show up as a labeled data point, not hang forever or be silently
  truncated at whatever step count happened to be hardcoded.
"""

import os
import sys
import time
import csv
import argparse

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_pruner_opt125m import (
    N_LAYERS, N_INTER, LAYER_SHAPE, OUT_ROOT,
    Pruner, load_opt125m, get_mlp_weights, get_loaders, evaluate,
    pruner_step, plot_one_run, plot_multiseed_comparison, plot_efficiency,
    stop_pod,
)


# ─────────────────────────────────────────────────────────────────────────────
# Convergence check
# ─────────────────────────────────────────────────────────────────────────────

def _block_mean(layer_hist, cp, check_every):
    """Mean of the check_every raw per-step readings ending at (1-indexed)
    step cp -- i.e. steps [cp - check_every, cp). Filters the per-step Adam
    noise the raw signal alone doesn't (see module docstring)."""
    lo = max(0, cp - check_every)
    return sum(layer_hist[lo:cp]) / (cp - lo)


def check_converged(history, step, check_every, window, rel_tol, abs_tol, burn_in):
    """
    Returns True iff every layer's last `window` checkpoint BLOCK MEANS
    (each averaged over the check_every raw steps ending at that checkpoint)
    all sit within tolerance of the most recent block mean.
    `step` is 1-indexed count of steps completed so far (len(history["loss"])).
    """
    if step < burn_in:
        return False
    if step < window * check_every:
        return False
    if step % check_every != 0:
        return False

    checkpoint_steps = [step - i * check_every for i in range(window)]  # newest first
    per_layer_keep = history["per_layer_keep"]  # list[N_LAYERS] of list[step]

    for layer_hist in per_layer_keep:
        block_means = [_block_mean(layer_hist, cp, check_every) for cp in checkpoint_steps]
        ref_val = block_means[0]   # most recent checkpoint's block mean
        tol = max(rel_tol * abs(ref_val), abs_tol)
        for val in block_means:
            if abs(val - ref_val) > tol:
                return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Train-vs-test CE gap diagnostic — periodic checkpoints DURING training,
# not just a single post-hoc check. Tests the overtraining hypothesis: does
# train_delta (CE improvement on train) keep growing past the point the
# DISCRETE mask has locked, while test_delta plateaus or degrades -- which
# would mean the gap between them is being driven by continued fitting of
# train-specific structure after the mask itself stopped changing, not by
# genuine redundancy discovery.
#
# Uses a REDUCED, fixed-size token slice (gap_eval_tokens, default 50k, vs.
# the full ~287k-token test set) for BOTH train and test at each periodic
# checkpoint -- these are for tracking the TREND across checkpoints inside
# one run, not the final reported numbers (the end-of-run eval still uses
# the full test set, unchanged). Deliberate cost/resolution tradeoff, not
# a default to blindly trust for final ppl figures.
# ─────────────────────────────────────────────────────────────────────────────

def sample_tokens(ids_flat, n_tokens):
    """Deterministic front-slice, matching improvement_result_check's
    matched-size-sample convention (apples-to-apples eval cost across splits)."""
    n = min(n_tokens, ids_flat.size(0))
    return ids_flat[:n]


def gap_diagnostic_checkpoint(pruner, model, train_sample, test_sample, device, args):
    """One periodic measurement: recompute gates fresh from the pruner's
    CURRENT parameters, then CE on both a train sample and a test sample
    at that exact mask. Returns a dict of the numbers worth logging."""
    pruner.eval()
    with torch.no_grad():
        gates = pruner(get_mlp_weights(model))
    per_layer_keep = [g.mean().item() for g in gates]
    avg_gate = float(np.mean(per_layer_keep))

    train_orig_ce   = evaluate(model, train_sample, device, gates=None,
                               desc="gap: train orig",
                               max_length=args.eval_max_length, stride=args.eval_stride)
    train_pruned_ce = evaluate(model, train_sample, device, gates=gates,
                               desc="gap: train pruned",
                               max_length=args.eval_max_length, stride=args.eval_stride)
    test_orig_ce    = evaluate(model, test_sample, device, gates=None,
                               desc="gap: test orig",
                               max_length=args.eval_max_length, stride=args.eval_stride)
    test_pruned_ce  = evaluate(model, test_sample, device, gates=gates,
                               desc="gap: test pruned",
                               max_length=args.eval_max_length, stride=args.eval_stride)
    pruner.train()

    train_delta = train_orig_ce - train_pruned_ce   # positive = improvement
    test_delta  = test_orig_ce - test_pruned_ce
    return {
        "avg_gate": avg_gate,
        "pct_pruned": (1 - avg_gate) * 100,
        "per_layer_keep": per_layer_keep,
        "train_orig_ce": train_orig_ce, "train_pruned_ce": train_pruned_ce,
        "train_delta": train_delta,
        "test_orig_ce": test_orig_ce, "test_pruned_ce": test_pruned_ce,
        "test_delta": test_delta,
        "gap": train_delta - test_delta,   # >0 means train improving more than test -- the signature we're hunting
    }


GAP_CSV_COLUMNS = [
    "lambda", "seed", "step",
    "avg_gate", "pct_pruned", "delta_pct_pruned", "max_layer_delta_pct",
    "would_be_converged",
    "train_orig_ce", "train_pruned_ce", "train_delta",
    "test_orig_ce", "test_pruned_ce", "test_delta",
    "gap",
]


# ─────────────────────────────────────────────────────────────────────────────
# Per-(λ, seed) training loop with convergence-based stopping
# ─────────────────────────────────────────────────────────────────────────────

def train_one_converge(lam, seed, model, train_loader, train_ids, test_ids, args, device, run_dir):
    torch.manual_seed(seed); np.random.seed(seed)

    layer_shapes = [LAYER_SHAPE] * N_LAYERS
    pruner = Pruner(layer_shapes, embed_dim=args.embed_dim,
                    lstm_hidden=args.lstm_hidden).to(device)
    opt = torch.optim.Adam(pruner.parameters(), lr=args.lr)

    tag = f"λ={lam} seed={seed}"
    print(f"\n── {tag} ── pruner params: "
          f"{sum(p.numel() for p in pruner.parameters()):,} "
          f"(convergence-based stopping, max_steps={args.max_steps})", flush=True)

    history = {
        "loss": [], "ce_orig": [], "ce_pruned": [], "avg_gate": [],
        "per_layer_keep": [[] for _ in range(N_LAYERS)],
    }

    # Train-vs-test gap diagnostic: fixed reduced-size samples reused at every
    # checkpoint (same slice each time -> only the pruner's gates change
    # between checkpoints, isolating that as the one variable).
    gap_train_sample = sample_tokens(train_ids, args.gap_eval_tokens)
    gap_test_sample  = sample_tokens(test_ids, args.gap_eval_tokens)
    gap_csv_path = os.path.join(run_dir, "gap_diagnostic.csv")
    os.makedirs(run_dir, exist_ok=True)
    gap_csv_file = open(gap_csv_path, "w", newline="")
    gap_writer = csv.DictWriter(gap_csv_file, fieldnames=GAP_CSV_COLUMNS)
    gap_writer.writeheader()
    prev_pct_pruned = None
    prev_per_layer_pct = None

    t0 = time.time()
    step = 0
    converged = False
    loader_iter = iter(train_loader)
    pbar = tqdm(total=args.max_steps, desc=tag, unit="step", dynamic_ncols=True)

    while step < args.max_steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        ids = batch["input_ids"].to(device)
        m = pruner_step(pruner, model, opt, ids, lam, device)

        history["loss"].append(m["loss"])
        history["ce_orig"].append(m["ce_orig"])
        history["ce_pruned"].append(m["ce_pruned"])
        history["avg_gate"].append(m["avg_gate"])
        for i, k in enumerate(m["per_layer_keep"]):
            history["per_layer_keep"][i].append(k)

        step += 1
        avg_pruned = (1 - m["avg_gate"]) * 100
        pbar.set_postfix(loss=f"{m['loss']:+.3f}", pruned=f"{avg_pruned:.1f}%", refresh=False)
        pbar.update(1)
        if step % args.log_every == 0:
            tqdm.write(f"  [{tag}] step {step:>6} | loss {m['loss']:+.3f} | "
                       f"CE orig {m['ce_orig']:.3f} pruned {m['ce_pruned']:.3f} | "
                       f"avg pruned {avg_pruned:5.1f}%")

        would_converge = False
        if step % args.check_every == 0:
            would_converge = check_converged(history, step, args.check_every, args.window,
                                             args.rel_tol, args.abs_tol, args.burn_in)
            if would_converge:
                converged = True

        if step % args.gap_eval_every == 0:
            g = gap_diagnostic_checkpoint(pruner, model, gap_train_sample, gap_test_sample,
                                          device, args)
            delta_pct_pruned = (g["pct_pruned"] - prev_pct_pruned) if prev_pct_pruned is not None else 0.0
            if prev_per_layer_pct is not None:
                cur_per_layer_pct = [(1 - k) * 100 for k in g["per_layer_keep"]]
                max_layer_delta = max(abs(c - p) for c, p in zip(cur_per_layer_pct, prev_per_layer_pct))
            else:
                cur_per_layer_pct = [(1 - k) * 100 for k in g["per_layer_keep"]]
                max_layer_delta = 0.0
            gap_writer.writerow({
                "lambda": lam, "seed": seed, "step": step,
                "avg_gate": g["avg_gate"], "pct_pruned": g["pct_pruned"],
                "delta_pct_pruned": delta_pct_pruned, "max_layer_delta_pct": max_layer_delta,
                "would_be_converged": would_converge,
                "train_orig_ce": g["train_orig_ce"], "train_pruned_ce": g["train_pruned_ce"],
                "train_delta": g["train_delta"],
                "test_orig_ce": g["test_orig_ce"], "test_pruned_ce": g["test_pruned_ce"],
                "test_delta": g["test_delta"], "gap": g["gap"],
            })
            gap_csv_file.flush()
            prev_pct_pruned = g["pct_pruned"]
            prev_per_layer_pct = cur_per_layer_pct
            tqdm.write(f"  [{tag}] gap-check step {step:>6} | pct_pruned {g['pct_pruned']:5.2f}% "
                       f"(Δ{delta_pct_pruned:+.3f}pp, max-layer-Δ{max_layer_delta:+.3f}pp) | "
                       f"train_delta {g['train_delta']:+.4f} test_delta {g['test_delta']:+.4f} "
                       f"gap {g['gap']:+.4f} | conv={would_converge}")

        if converged:
            tqdm.write(f"  [{tag}] CONVERGED at step {step} "
                       f"(all {N_LAYERS} layers flat within tol over last "
                       f"{args.window * args.check_every} steps)")
            break

    pbar.close()
    total_time = time.time() - t0
    if not converged:
        print(f"  [{tag}] NOT CONVERGED — hit max_steps={args.max_steps} safety cap. "
              f"This is itself a result (genuinely slow/non-converging λ), not a failure.",
              flush=True)

    # Final gap-diagnostic row, even if it doesn't land on a gap_eval_every
    # boundary, so the CSV always has a data point matching the run's actual
    # stopping step (whether by convergence or the max_steps cap).
    if step % args.gap_eval_every != 0:
        g = gap_diagnostic_checkpoint(pruner, model, gap_train_sample, gap_test_sample,
                                      device, args)
        delta_pct_pruned = (g["pct_pruned"] - prev_pct_pruned) if prev_pct_pruned is not None else 0.0
        cur_per_layer_pct = [(1 - k) * 100 for k in g["per_layer_keep"]]
        max_layer_delta = (max(abs(c - p) for c, p in zip(cur_per_layer_pct, prev_per_layer_pct))
                           if prev_per_layer_pct is not None else 0.0)
        gap_writer.writerow({
            "lambda": lam, "seed": seed, "step": step,
            "avg_gate": g["avg_gate"], "pct_pruned": g["pct_pruned"],
            "delta_pct_pruned": delta_pct_pruned, "max_layer_delta_pct": max_layer_delta,
            "would_be_converged": converged,
            "train_orig_ce": g["train_orig_ce"], "train_pruned_ce": g["train_pruned_ce"],
            "train_delta": g["train_delta"],
            "test_orig_ce": g["test_orig_ce"], "test_pruned_ce": g["test_pruned_ce"],
            "test_delta": g["test_delta"], "gap": g["gap"],
        })
    gap_csv_file.close()

    pruner.eval()
    with torch.no_grad():
        final_gates = pruner(get_mlp_weights(model))
    per_layer_kept = [int(g.sum().item()) for g in final_gates]

    orig_ce   = evaluate(model, test_ids, device, gates=None,
                        desc=f"[{tag}] eval orig",
                        max_length=args.eval_max_length, stride=args.eval_stride)
    pruned_ce = evaluate(model, test_ids, device, gates=final_gates,
                        desc=f"[{tag}] eval pruned",
                        max_length=args.eval_max_length, stride=args.eval_stride)
    orig_ppl   = float(np.exp(orig_ce))
    pruned_ppl = float(np.exp(pruned_ce))

    final_gate = history["avg_gate"][-1]
    pct_pruned = (1 - final_gate) * 100
    print(f"  → [{tag}] {'converged' if converged else 'CAPPED'} at step {step} "
          f"({total_time:.0f}s) | final keep {final_gate:.3f} pruned {pct_pruned:.2f}% | "
          f"orig ppl {orig_ppl:.3f} → pruned ppl {pruned_ppl:.3f}", flush=True)

    plot_one_run(
        history, os.path.join(run_dir, "plot.png"),
        title=(f"OPT-125M MLP — λ={lam} seed={seed} — "
               f"{'converged' if converged else 'CAPPED'} @ step {step} — "
               f"{pct_pruned:.1f}% pruned, ppl {pruned_ppl:.2f}"),
    )

    os.makedirs(run_dir, exist_ok=True)
    lines = [
        f"OPT-125M MLP pruner — λ={lam}, seed={seed} — CONVERGENCE-BASED STOPPING",
        f"layers : 12 FFN blocks, 3072 intermediate neurons each",
        f"steps taken       : {step}",
        f"converged         : {converged} (max_steps cap = {args.max_steps})",
        f"convergence check : window={args.window} x check_every={args.check_every} "
        f"({args.window * args.check_every} steps flat) | rel_tol={args.rel_tol} "
        f"abs_tol={args.abs_tol} | burn_in={args.burn_in}",
        f"time              : {total_time:.1f}s",
        "-" * 60,
        f"final avg keep gate          : {final_gate:.4f}",
        f"final % FFN neurons pruned   : {pct_pruned:.2f}%",
        f"per-block neurons kept       : {per_layer_kept}",
        "-" * 60,
        f"FULL test set (WikiText-2):",
        f"  original  ppl              : {orig_ppl:.3f}",
        f"  pruned    ppl              : {pruned_ppl:.3f}",
        f"  ppl increase               : {pruned_ppl - orig_ppl:+.3f}",
    ]
    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    torch.save({
        "pruner_state_dict": pruner.state_dict(),
        "lambda": lam, "seed": seed,
        "embed_dim": args.embed_dim, "lstm_hidden": args.lstm_hidden,
        "per_layer_kept": per_layer_kept,
        "orig_ppl": orig_ppl, "pruned_ppl": pruned_ppl,
        "steps_taken": step, "converged": converged,
    }, os.path.join(run_dir, "pruner.pt"))
    print(f"  [saved] {run_dir}/  (plot.png, summary.txt, pruner.pt)", flush=True)

    return {
        "lambda": lam, "seed": seed, "per_layer_kept": per_layer_kept,
        "pct_pruned": pct_pruned, "orig_ppl": orig_ppl, "pruned_ppl": pruned_ppl,
        "total_time": total_time, "steps_taken": step, "converged": converged,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.8, 1.6],
                    help="Dense 9-point grid, extra resolution around the "
                         "0.2-0.4 crossover region (F21).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--check_every", type=int, default=50)
    ap.add_argument("--window", type=int, default=5,
                    help="Number of trailing checkpoints that must all be flat.")
    ap.add_argument("--rel_tol", type=float, default=0.05)
    ap.add_argument("--abs_tol", type=float, default=0.01)
    ap.add_argument("--burn_in", type=int, default=500)
    ap.add_argument("--max_steps", type=int, default=18000,
                    help="Safety cap, not a target -- F20 showed some lambdas "
                         "still drifting at 18750. Hitting this is a labeled "
                         "NOT-CONVERGED result, not a silent truncation.")
    ap.add_argument("--gap_eval_every", type=int, default=200,
                    help="Train-vs-test CE gap diagnostic checkpoint interval. "
                         "Must be a multiple of --check_every for the logged "
                         "would_be_converged flag to align with an actual "
                         "convergence check at that step.")
    ap.add_argument("--gap_eval_tokens", type=int, default=50_000,
                    help="Fixed reduced token count for BOTH train and test "
                         "samples at each gap-diagnostic checkpoint (vs. the "
                         "full ~287k-token test set used for the final "
                         "end-of-run eval) -- keeps per-checkpoint eval cost "
                         "roughly constant. Trend data, not final numbers.")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--embed_dim", type=int, default=64)
    ap.add_argument("--lstm_hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--log_every", type=int, default=250)
    ap.add_argument("--eval_max_length", type=int, default=2048)
    ap.add_argument("--eval_stride", type=int, default=1024)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_dir", type=str,
                    default=os.path.join(os.path.dirname(OUT_ROOT), "opt125m_converge_sweep"))
    ap.add_argument("--stop_pod", action="store_true")
    args = ap.parse_args()

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device} | λs={args.lambdas} | seeds={args.seeds} | "
          f"convergence-based (max_steps={args.max_steps})")

    print("Loading OPT-125M ...", flush=True)
    model = load_opt125m(device)
    print("Loading WikiText-2 ...", flush=True)
    train_loader, train_ids, test_ids = get_loaders(args.seq_len, args.batch_size)

    os.makedirs(args.out_dir, exist_ok=True)
    all_results = []
    total_runs = len(args.lambdas) * len(args.seeds)
    run_num = 0

    for lam in args.lambdas:
        for seed in args.seeds:
            run_num += 1
            tqdm.write(f"\n{'='*70}\nRun {run_num}/{total_runs}\n{'='*70}")
            run_dir = (os.path.join(args.out_dir, f"lambda_{lam}", f"seed_{seed}")
                       if len(args.seeds) > 1
                       else os.path.join(args.out_dir, f"lambda_{lam}"))
            res = train_one_converge(lam, seed, model, train_loader, train_ids, test_ids,
                                     args, device, run_dir)
            all_results.append(res)

    # Concatenate every run's gap_diagnostic.csv into one sweep-level file --
    # this is the actual artifact for answering "where does test_delta fall
    # behind train_delta relative to lambda / steps / mask convergence."
    combined_path = os.path.join(args.out_dir, "gap_diagnostic_all.csv")
    with open(combined_path, "w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=GAP_CSV_COLUMNS)
        writer.writeheader()
        for lam in args.lambdas:
            for seed in args.seeds:
                run_dir = (os.path.join(args.out_dir, f"lambda_{lam}", f"seed_{seed}")
                           if len(args.seeds) > 1
                           else os.path.join(args.out_dir, f"lambda_{lam}"))
                run_csv = os.path.join(run_dir, "gap_diagnostic.csv")
                if not os.path.exists(run_csv):
                    continue
                with open(run_csv, newline="") as in_f:
                    for row in csv.DictReader(in_f):
                        writer.writerow(row)
    print(f"Combined gap diagnostic -> {combined_path}")

    # Aggregate: the actual point of this sweep is the (lambda, steps_taken) curve.
    header = f"OPT-125M convergence-based sweep | seeds={args.seeds} | device={device}"
    sep = "-" * 100
    col = (f"{'lambda':>7} {'seed':>5} | {'steps':>7} {'conv?':>6} | {'% pruned':>9} | "
           f"{'orig ppl':>9} | {'pruned ppl':>10} | {'ppl rise':>9}")
    rows = [header, sep, col, sep]
    for r in all_results:
        rows.append(
            f"{r['lambda']:>7} {r['seed']:>5} | {r['steps_taken']:>7} "
            f"{'YES' if r['converged'] else 'NO':>6} | "
            f"{r['pct_pruned']:>8.2f}% | {r['orig_ppl']:>9.3f} | "
            f"{r['pruned_ppl']:>10.3f} | {r['pruned_ppl'] - r['orig_ppl']:>+9.3f}"
        )
    summary_str = "\n".join(rows)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(summary_str + "\n")
    print("\n" + summary_str)
    print(f"\nResults → {args.out_dir}/")

    per_lambda_stats = [{
        "lambda": r["lambda"], "pct_pruned_mean": r["pct_pruned"], "pct_pruned_std": 0.0,
        "pruned_ppl_mean": r["pruned_ppl"], "pruned_ppl_std": 0.0,
        "orig_ppl": r["orig_ppl"], "runs": [r],
    } for r in all_results]
    plot_multiseed_comparison(per_lambda_stats, os.path.join(args.out_dir, "comparison.png"))
    plot_efficiency(per_lambda_stats, os.path.join(args.out_dir, "efficiency.png"))

    if args.stop_pod:
        stop_pod()


if __name__ == "__main__":
    main()
