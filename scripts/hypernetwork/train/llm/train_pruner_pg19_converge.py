"""
GPT-2 small vs OPT-125M on pg19 -- CONVERGENCE-BASED stopping + train/test
CE gap diagnostic, same concept as train_pruner_opt125m_converge.py, ported
to the two-model pg19 comparison (B6, diary/ideas.md).

Not standalone -- imports the per-model dispatch (load/get_mlp_weights/
apply_gates/tokenizer, MODEL_SPECS, pg19 streaming loaders, pruner_step,
plotting) from train_pruner_pg19_sweep.py rather than re-deriving it a
second time, and Pruner/evaluate/autocast_ctx from train_pruner_gpt2.py
(same import chain train_pruner_pg19_sweep.py already uses). Only the
convergence check, the gap diagnostic, and the training loop that wires
them together are new here.

WHY: train_pruner_pg19_sweep.py trains every (model, lambda, seed) for a
fixed --steps (default 18750) -- the same inherited, never-re-derived
budget problem F20/B7 found on WikiText-2. This script replaces that with
the same block-mean convergence-based stopping validated on OPT-125M/
WikiText-2 (see diary chat log, tmp2/opt125m_converge_sweep -- 20.4x fewer
total steps, similar-or-better results for lambda <= 0.4, though high
lambda (0.8, 1.6) needed closer scrutiny there -- worth watching for the
same pattern here), plus the periodic train-vs-test CE gap diagnostic to
directly check the overtraining hypothesis on the pg19 comparison too, not
just WikiText-2.

CONVERGENCE CRITERION: identical to train_pruner_opt125m_converge.py --
see that file's docstring for the full derivation, including the v1 bug
(checking the RAW per-step per-layer keep-fraction, which inherits a
non-decaying noise floor from Adam's per-step stochastic-minibatch
updates) and the v2 fix (block-mean each checkpoint's reading over the
preceding check_every raw steps before comparing). Not re-derived here,
just reused -- it's model-agnostic (operates purely on
history["per_layer_keep"]).

LAMBDA GRIDS: both models default to the SAME 9-point grid,
[0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.8, 1.6] -- directly copied from
the OPT-125M/WikiText-2 convergence sweep that motivated this script
(tmp2/opt125m_converge_sweep), not independently re-derived per model.
This is a deliberate simplification, not a claim that lambda is
comparable in effect between the two models (it isn't -- see
train_pruner_pg19_sweep.py's docstring, OPT prunes more aggressively than
GPT-2 at matched nominal lambda on WikiText-2) -- override --gpt2_lambdas
/ --opt125m_lambdas independently once this sweep's own results suggest
better model-specific ranges, the same way train_pruner_pg19_sweep.py's
original grids were sized to each model's predicted interesting region.

Usage:
    python train_pruner_pg19_converge.py [--models gpt2 opt125m]
        [--gpt2_lambdas ...] [--opt125m_lambdas ...] [--seeds 0]
        [--max_steps 18000] [--stop_pod]
"""
import csv
import os
import sys
import time
import argparse

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_pruner_gpt2 import evaluate
from train_pruner_pg19_sweep import (
    MODEL_SPECS, N_LAYERS, N_INTER, LAYER_SHAPE, PG19_REPO,
    get_pg19_loaders, pruner_step, plot_one_run, plot_model_comparison,
    stop_pod, Pruner,
)

OUT_ROOT = "/workspace/results/pg19_converge_sweep"

# Same 9-point grid for both models by default -- see module docstring.
DEFAULT_LAMBDAS = [0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.8, 1.6]


# ─────────────────────────────────────────────────────────────────────────────
# Convergence check -- identical logic to train_pruner_opt125m_converge.py's
# v2 (block-mean) fix. Model-agnostic: only reads history["per_layer_keep"].
# ─────────────────────────────────────────────────────────────────────────────

def _block_mean(layer_hist, cp, check_every):
    lo = max(0, cp - check_every)
    return sum(layer_hist[lo:cp]) / (cp - lo)


def check_converged(history, step, check_every, window, rel_tol, abs_tol, burn_in):
    """Returns True iff every layer's last `window` checkpoint BLOCK MEANS
    (each averaged over the check_every raw steps ending at that checkpoint)
    all sit within tolerance of the most recent block mean."""
    if step < burn_in:
        return False
    if step < window * check_every:
        return False
    if step % check_every != 0:
        return False

    checkpoint_steps = [step - i * check_every for i in range(window)]
    for layer_hist in history["per_layer_keep"]:
        block_means = [_block_mean(layer_hist, cp, check_every) for cp in checkpoint_steps]
        ref_val = block_means[0]
        tol = max(rel_tol * abs(ref_val), abs_tol)
        for val in block_means:
            if abs(val - ref_val) > tol:
                return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Train-vs-test CE gap diagnostic -- generalized to take per-model
# get_mlp_weights/apply_gates/eval window (via `spec`) instead of the
# OPT-125M-only hardcoding in the WikiText-2 version.
# ─────────────────────────────────────────────────────────────────────────────

def sample_tokens(ids_flat, n_tokens):
    n = min(n_tokens, ids_flat.size(0))
    return ids_flat[:n]


def gap_diagnostic_checkpoint(pruner, model, train_sample, test_sample, device, spec, args):
    pruner.eval()
    with torch.no_grad():
        gates = pruner(spec["get_mlp_weights"](model))
    per_layer_keep = [g.mean().item() for g in gates]
    avg_gate = float(np.mean(per_layer_keep))

    kw = dict(max_length=spec["eval_max_length"], stride=spec["eval_stride"],
             apply_gates_fn=spec["apply_gates"])
    train_orig_ce   = evaluate(model, train_sample, device, gates=None,   desc="gap: train orig",   **kw)
    train_pruned_ce = evaluate(model, train_sample, device, gates=gates, desc="gap: train pruned", **kw)
    test_orig_ce    = evaluate(model, test_sample,  device, gates=None,   desc="gap: test orig",    **kw)
    test_pruned_ce  = evaluate(model, test_sample,  device, gates=gates, desc="gap: test pruned",  **kw)
    pruner.train()

    train_delta = train_orig_ce - train_pruned_ce
    test_delta  = test_orig_ce - test_pruned_ce
    return {
        "avg_gate": avg_gate, "pct_pruned": (1 - avg_gate) * 100,
        "per_layer_keep": per_layer_keep,
        "train_orig_ce": train_orig_ce, "train_pruned_ce": train_pruned_ce, "train_delta": train_delta,
        "test_orig_ce": test_orig_ce, "test_pruned_ce": test_pruned_ce, "test_delta": test_delta,
        "gap": train_delta - test_delta,
    }


GAP_CSV_COLUMNS = [
    "model", "lambda", "seed", "step",
    "avg_gate", "pct_pruned", "delta_pct_pruned", "max_layer_delta_pct",
    "would_be_converged",
    "train_orig_ce", "train_pruned_ce", "train_delta",
    "test_orig_ce", "test_pruned_ce", "test_delta",
    "gap",
]


# ─────────────────────────────────────────────────────────────────────────────
# Per-(model, λ, seed) training loop -- convergence-based stopping + gap
# diagnostic, pg19 data, per-model dispatch via MODEL_SPECS.
# ─────────────────────────────────────────────────────────────────────────────

def train_one_converge(model_name, model, train_loader, train_ids, test_ids,
                       lam, seed, args, device, run_dir):
    torch.manual_seed(seed); np.random.seed(seed)
    spec = MODEL_SPECS[model_name]

    layer_shapes = [LAYER_SHAPE] * N_LAYERS
    pruner = Pruner(layer_shapes, embed_dim=args.embed_dim,
                    lstm_hidden=args.lstm_hidden).to(device)
    opt = torch.optim.Adam(pruner.parameters(), lr=args.lr)

    tag = f"{model_name} λ={lam} seed={seed}"
    print(f"\n── {tag} ── pruner params: "
          f"{sum(p.numel() for p in pruner.parameters()):,} "
          f"(convergence-based stopping, max_steps={args.max_steps})", flush=True)

    history = {
        "loss": [], "ce_orig": [], "ce_pruned": [], "avg_gate": [],
        "per_layer_keep": [[] for _ in range(N_LAYERS)],
    }

    gap_train_sample = sample_tokens(train_ids, args.gap_eval_tokens)
    gap_test_sample  = sample_tokens(test_ids, args.gap_eval_tokens)
    os.makedirs(run_dir, exist_ok=True)
    gap_csv_path = os.path.join(run_dir, "gap_diagnostic.csv")
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

        ids = batch[0].to(device)   # TensorDataset -> (tensor,) tuple
        m = pruner_step(pruner, model, opt, ids, lam, device,
                        spec["get_mlp_weights"], spec["apply_gates"])

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
                                          device, spec, args)
            delta_pct_pruned = (g["pct_pruned"] - prev_pct_pruned) if prev_pct_pruned is not None else 0.0
            cur_per_layer_pct = [(1 - k) * 100 for k in g["per_layer_keep"]]
            max_layer_delta = (max(abs(c - p) for c, p in zip(cur_per_layer_pct, prev_per_layer_pct))
                               if prev_per_layer_pct is not None else 0.0)
            gap_writer.writerow({
                "model": model_name, "lambda": lam, "seed": seed, "step": step,
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
        print(f"  [{tag}] NOT CONVERGED — hit max_steps={args.max_steps} safety cap.",
              flush=True)

    if step % args.gap_eval_every != 0:
        g = gap_diagnostic_checkpoint(pruner, model, gap_train_sample, gap_test_sample,
                                      device, spec, args)
        delta_pct_pruned = (g["pct_pruned"] - prev_pct_pruned) if prev_pct_pruned is not None else 0.0
        cur_per_layer_pct = [(1 - k) * 100 for k in g["per_layer_keep"]]
        max_layer_delta = (max(abs(c - p) for c, p in zip(cur_per_layer_pct, prev_per_layer_pct))
                           if prev_per_layer_pct is not None else 0.0)
        gap_writer.writerow({
            "model": model_name, "lambda": lam, "seed": seed, "step": step,
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
        final_gates = pruner(spec["get_mlp_weights"](model))
    per_layer_kept = [int(g.sum().item()) for g in final_gates]

    eval_kw = dict(max_length=spec["eval_max_length"], stride=spec["eval_stride"],
                  apply_gates_fn=spec["apply_gates"])
    orig_ce   = evaluate(model, test_ids, device, gates=None,        desc=f"[{tag}] eval orig",   **eval_kw)
    pruned_ce = evaluate(model, test_ids, device, gates=final_gates, desc=f"[{tag}] eval pruned", **eval_kw)
    orig_ppl, pruned_ppl = float(np.exp(orig_ce)), float(np.exp(pruned_ce))

    final_gate = history["avg_gate"][-1]
    pct_pruned = (1 - final_gate) * 100
    print(f"  → [{tag}] {'converged' if converged else 'CAPPED'} at step {step} "
          f"({total_time:.0f}s) | final keep {final_gate:.3f} pruned {pct_pruned:.2f}% | "
          f"orig ppl {orig_ppl:.3f} → pruned ppl {pruned_ppl:.3f}", flush=True)

    plot_one_run(history, os.path.join(run_dir, "plot.png"),
                title=(f"{spec['display_name']} MLP pruner (pg19) — λ={lam} seed={seed} — "
                      f"{'converged' if converged else 'CAPPED'} @ step {step} — "
                      f"{pct_pruned:.1f}% pruned, ppl {pruned_ppl:.2f}"))

    lines = [
        f"{spec['display_name']} MLP pruner (pg19) — λ={lam}, seed={seed} — CONVERGENCE-BASED STOPPING",
        f"layers : {N_LAYERS} MLP blocks, {N_INTER} intermediate neurons each",
        f"steps taken       : {step}",
        f"converged         : {converged} (max_steps cap = {args.max_steps})",
        f"convergence check : window={args.window} x check_every={args.check_every} "
        f"({args.window * args.check_every} steps flat, block-mean) | rel_tol={args.rel_tol} "
        f"abs_tol={args.abs_tol} | burn_in={args.burn_in}",
        f"time              : {total_time:.1f}s",
        "-" * 60,
        f"final avg keep gate          : {final_gate:.4f}",
        f"final % MLP neurons pruned   : {pct_pruned:.2f}%",
        f"per-block neurons kept       : {per_layer_kept}",
        "-" * 60,
        f"pg19 test set:",
        f"  original  ppl              : {orig_ppl:.3f}",
        f"  pruned    ppl              : {pruned_ppl:.3f}",
        f"  ppl increase               : {pruned_ppl - orig_ppl:+.3f}",
    ]
    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    torch.save({
        "pruner_state_dict": pruner.state_dict(), "model": model_name,
        "lambda": lam, "seed": seed, "embed_dim": args.embed_dim,
        "lstm_hidden": args.lstm_hidden, "per_layer_kept": per_layer_kept,
        "orig_ppl": orig_ppl, "pruned_ppl": pruned_ppl,
        "steps_taken": step, "converged": converged,
    }, os.path.join(run_dir, "pruner.pt"))
    print(f"  [saved] {run_dir}/", flush=True)

    return {
        "model": model_name, "lambda": lam, "seed": seed,
        "per_layer_kept": per_layer_kept, "pct_pruned": pct_pruned,
        "orig_ppl": orig_ppl, "pruned_ppl": pruned_ppl, "total_time": total_time,
        "steps_taken": step, "converged": converged,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gpt2", "opt125m"],
                    choices=["gpt2", "opt125m"])
    ap.add_argument("--gpt2_lambdas", type=float, nargs="+", default=None,
                    help=f"Default (same grid for both models): {DEFAULT_LAMBDAS}")
    ap.add_argument("--opt125m_lambdas", type=float, nargs="+", default=None,
                    help=f"Default (same grid for both models): {DEFAULT_LAMBDAS}")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--check_every", type=int, default=50)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--rel_tol", type=float, default=0.05)
    ap.add_argument("--abs_tol", type=float, default=0.01)
    ap.add_argument("--burn_in", type=int, default=500)
    ap.add_argument("--max_steps", type=int, default=18000,
                    help="Safety cap, not a target -- see train_pruner_opt125m_"
                         "converge.py's docstring for why this isn't a soft default.")
    ap.add_argument("--gap_eval_every", type=int, default=200,
                    help="Must be a multiple of --check_every.")
    ap.add_argument("--gap_eval_tokens", type=int, default=50_000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--n_train_tokens", type=int, default=None,
                    help="pg19 train tokens to stream+cache. Default (None) sizes "
                         "to max_steps*batch_size*seq_len -- enough for the worst "
                         "case (a run that never converges and hits the cap) -- "
                         "same 'one epoch, zero repeats' reasoning as "
                         "train_pruner_pg19_sweep.py, just anchored to max_steps "
                         "instead of a fixed --steps since there isn't one here.")
    ap.add_argument("--n_test_tokens", type=int, default=245_000,
                    help="Matches WikiText-2 test's token count.")
    ap.add_argument("--embed_dim", type=int, default=64)
    ap.add_argument("--lstm_hidden", type=int, default=128)
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
              f"= {args.n_train_tokens:,} tokens (worst-case sizing, see --help)")

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    lambdas_by_model = {
        "gpt2":    args.gpt2_lambdas    or DEFAULT_LAMBDAS,
        "opt125m": args.opt125m_lambdas or DEFAULT_LAMBDAS,
    }
    print(f"Device: {device} | models={args.models} | convergence-based "
          f"(max_steps={args.max_steps})")
    for m in args.models:
        print(f"  {m}: λs={lambdas_by_model[m]}")
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
        # Flat train stream for the gap diagnostic's train-sample eval (same
        # role as train_pruner_opt125m.py's train_ids) -- pull it from the
        # already-blocked loader's underlying dataset rather than re-streaming.
        train_ids_flat = train_loader.dataset.tensors[0].reshape(-1)
        print(f"Data ready: train_blocks={len(train_loader.dataset)} "
              f"test_tokens={test_ids.size(0):,} "
              f"(eval: max_length={spec['eval_max_length']} stride={spec['eval_stride']})", flush=True)

        lambdas = lambdas_by_model[model_name]
        total_runs = len(lambdas) * len(args.seeds)
        run_num = 0
        for lam in lambdas:
            for seed in args.seeds:
                run_num += 1
                tqdm.write(f"\n{'='*70}\n[{model_name}] Run {run_num}/{total_runs}\n{'='*70}")
                run_dir = (os.path.join(out_root, model_name, f"lambda_{lam}", f"seed_{seed}")
                           if len(args.seeds) > 1
                           else os.path.join(out_root, model_name, f"lambda_{lam}"))
                res = train_one_converge(model_name, model, train_loader, train_ids_flat,
                                         test_ids, lam, seed, args, device, run_dir)
                all_results.append(res)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    plot_model_comparison(all_results, os.path.join(out_root, "gpt2_vs_opt125m_pg19.png"))

    # Combined gap diagnostic across every (model, lambda, seed) run.
    combined_path = os.path.join(out_root, "gap_diagnostic_all.csv")
    with open(combined_path, "w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=GAP_CSV_COLUMNS)
        writer.writeheader()
        for model_name in args.models:
            for lam in lambdas_by_model[model_name]:
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
    rows = [f"pg19 convergence-based sweep — models={args.models} | seeds={args.seeds} | "
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
    print(f"\nResults → {out_root}/")

    if args.stop_pod:
        stop_pod()


if __name__ == "__main__":
    main()
