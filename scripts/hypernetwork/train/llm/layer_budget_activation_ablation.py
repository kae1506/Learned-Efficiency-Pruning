"""
Layer-budget vs. within-layer-selection decomposition ablation (follow-up
to F3/F4/F22, diary/crisp-findings.md).

QUESTION: F4 found pruner capacity (0.47x-2.26x base) has no meaningful
effect on %pruned, ppl, or convergence speed. Is that suspicious -- is the
pruner not really doing anything smart? Architecturally, the pruner's two
jobs are cleanly separable: context_bias (BiLSTM output) is a SINGLE scalar
added identically to every neuron in a layer, so it can only set that
layer's overall threshold/budget -- it cannot discriminate WITHIN a layer.
100% of within-layer neuron selection is therefore the row-encoder's job.
This ablation asks: does most of the trained pruner's advantage over a
magnitude-style heuristic (F22: 6.3x lower cost than an activation-magnitude
baseline at matched sparsity) come from the LEARNED PER-LAYER BUDGET (a
12-number decision) or from sophisticated WITHIN-LAYER selection (a genuine
~3072-way-per-layer decision)? If capacity doesn't matter (F4) because the
real "smartness" lives almost entirely in the low-dimensional budget
decision, replacing within-layer selection with a simple heuristic --
while keeping the trained pruner's OWN discovered per-layer budget exactly
-- should reproduce close to the trained pruner's actual ppl. If within-
layer selection is doing real necessary work, ppl should degrade toward
F22's activation-magnitude-baseline numbers instead.

METHOD:
  1. Load a trained pruner CHECKPOINT (not the pruner itself -- only
     `per_layer_kept`, the BiLSTM's discovered per-layer neuron COUNT, is
     needed, already saved in pruner.pt).
  2. Collect mean post-ReLU activation magnitude per neuron over a
     calibration pass (same methodology as scripts/baselines/
     activation_pruning_opt125m.py / F22 -- NOT weight magnitude, per
     explicit instruction, to stay consistent with the already-validated
     F22 baseline criterion rather than introduce a third, untested one).
  3. Within each layer, keep the per_layer_kept[l] HIGHEST-activation
     neurons (prune the lowest-activation ones) -- same per-layer COUNT as
     the real trained mask, different WITHIN-layer choice of which
     specific neurons.
  4. Evaluate pg19 test ppl with this budget-matched-but-activation-
     selected mask; compare against the checkpoint's own recorded
     orig_ppl/pruned_ppl.

CHECKPOINTS: tmp3/pg19_converge_sweep/{gpt2,opt125m}/lambda_0.3/pruner.pt
(base-size pruner -- embed_dim=64/lstm_hidden=128 -- convergence-based
sweep, mid-range lambda=0.3: real, non-trivial sparsity in both models
without landing in F3/F4's still-uncertain high-lambda region).

Not standalone -- imports model dispatch from train_pruner_pg19_sweep.py
and evaluate() from train_pruner_gpt2.py, same pattern as every other
pg19 script in this directory.
"""
import os
import sys
import argparse

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_pruner_gpt2 import evaluate, autocast_ctx
from train_pruner_pg19_sweep import MODEL_SPECS, N_LAYERS, N_INTER, get_pg19_loaders

CHECKPOINTS = {
    "gpt2":    "tmp3/pg19_converge_sweep/gpt2/lambda_0.3/pruner.pt",
    "opt125m": "tmp3/pg19_converge_sweep/opt125m/lambda_0.3/pruner.pt",
}

CALIB_BATCHES = 50
BATCH_SIZE = 8
SEQ_LEN = 512
N_TEST_TOKENS = 245_000


def get_hook_modules_gpt2(model):
    return [model.transformer.h[i].mlp.c_proj for i in range(N_LAYERS)]


def get_hook_modules_opt(model):
    return [model.model.decoder.layers[i].fc2 for i in range(N_LAYERS)]


HOOK_MODULES_FN = {"gpt2": get_hook_modules_gpt2, "opt125m": get_hook_modules_opt}


@torch.no_grad()
def collect_mean_activations(model, hook_modules, calib_loader, n_batches, device):
    """Mean post-ReLU/GELU activation per FFN neuron, hooked at the same
    point apply_gates() gates (post-activation, pre-down-projection) --
    identical methodology to activation_pruning_opt125m.py / F22, just
    generalized to either model via hook_modules."""
    sums = [torch.zeros(N_INTER, device=device) for _ in hook_modules]
    counts = [0] * len(hook_modules)

    hooks = []
    def make_hook(idx):
        def hook(module, args):
            x = args[0].reshape(-1, N_INTER)
            sums[idx] += x.sum(dim=0)
            counts[idx] += x.shape[0]
            return None
        return hook
    for i, m in enumerate(hook_modules):
        hooks.append(m.register_forward_pre_hook(make_hook(i)))

    loader_iter = iter(calib_loader)
    for _ in tqdm(range(n_batches), desc="calibration", unit="batch"):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(calib_loader)
            batch = next(loader_iter)
        ids = batch[0].to(device)
        with autocast_ctx(device):
            model(ids)

    for h in hooks:
        h.remove()
    return [s / c for s, c in zip(sums, counts)]


def build_budget_matched_gates(mean_acts, per_layer_kept, device):
    """Within each layer, keep the per_layer_kept[l] HIGHEST-activation
    neurons -- same per-layer COUNT as the real trained mask (the BiLSTM's
    discovered budget), different choice of WHICH neurons within it."""
    gates = []
    for acts, k in zip(mean_acts, per_layer_kept):
        topk_idx = torch.topk(acts, k).indices
        g = torch.zeros(N_INTER, device=device)
        g[topk_idx] = 1.0
        gates.append(g)
    return gates


def run_one_model(model_name, device, verbose=True):
    spec = MODEL_SPECS[model_name]
    ckpt_path = CHECKPOINTS[model_name]
    print(f"\n{'='*70}\n{spec['display_name']} — loading checkpoint {ckpt_path}\n{'='*70}", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    per_layer_kept = ckpt["per_layer_kept"]
    real_orig_ppl, real_pruned_ppl = ckpt["orig_ppl"], ckpt["pruned_ppl"]
    lam = ckpt["lambda"]
    print(f"  λ={lam}  per_layer_kept={per_layer_kept}  "
          f"(real trained pruner: orig_ppl={real_orig_ppl:.3f} pruned_ppl={real_pruned_ppl:.3f})")

    model = spec["load_fn"](device)
    tokenizer = spec["tokenizer_fn"]()

    n_calib_tokens = CALIB_BATCHES * BATCH_SIZE * SEQ_LEN
    print(f"Streaming pg19 for calibration ({n_calib_tokens:,} tokens) + test "
          f"({N_TEST_TOKENS:,} tokens) ...", flush=True)
    calib_loader, test_ids = get_pg19_loaders(
        tokenizer, SEQ_LEN, BATCH_SIZE, n_calib_tokens, N_TEST_TOKENS, model_name,
    )

    print(f"Collecting mean activations over {CALIB_BATCHES} calibration batches ...", flush=True)
    hook_modules = HOOK_MODULES_FN[model_name](model)
    mean_acts = collect_mean_activations(model, hook_modules, calib_loader, CALIB_BATCHES, device)
    for i, ma in enumerate(mean_acts):
        print(f"  L{i}: min={ma.min():.4f} mean={ma.mean():.4f} max={ma.max():.4f}")

    gates = build_budget_matched_gates(mean_acts, per_layer_kept, device)
    actual_kept = [int(g.sum().item()) for g in gates]
    assert actual_kept == per_layer_kept, f"budget mismatch: {actual_kept} vs {per_layer_kept}"
    print(f"Budget-matched gates built, per-layer kept confirmed == checkpoint's per_layer_kept.")

    print("Evaluating budget-matched activation-selected mask on pg19 test ...", flush=True)
    decomposed_ce = evaluate(model, test_ids, device, gates=gates, desc=f"[{model_name}] decomposed",
                             max_length=spec["eval_max_length"], stride=spec["eval_stride"],
                             apply_gates_fn=spec["apply_gates"])
    decomposed_ppl = float(np.exp(decomposed_ce))

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    result = dict(
        model=model_name, lam=lam, per_layer_kept=per_layer_kept,
        orig_ppl=real_orig_ppl, real_pruned_ppl=real_pruned_ppl,
        decomposed_ppl=decomposed_ppl,
    )
    if verbose:
        real_rise = real_pruned_ppl - real_orig_ppl
        decomp_rise = decomposed_ppl - real_orig_ppl
        gap_to_real = decomposed_ppl - real_pruned_ppl
        pct_of_way_to_baseline = None
        print(f"\n{'-'*70}")
        print(f"{spec['display_name']} @ λ={lam}, budget-matched to real pruner's per-layer counts:")
        print(f"  orig ppl (unpruned)                : {real_orig_ppl:.3f}")
        print(f"  REAL trained pruner  pruned ppl     : {real_pruned_ppl:.3f}  (rise {real_rise:+.3f})")
        print(f"  DECOMPOSED (act-mag within budget)  : {decomposed_ppl:.3f}  (rise {decomp_rise:+.3f})")
        print(f"  gap (decomposed - real)             : {gap_to_real:+.3f}")
        print(f"{'-'*70}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gpt2", "opt125m"], choices=["gpt2", "opt125m"])
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    results = [run_one_model(m, device) for m in args.models]

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"{'model':>10} {'lam':>5} | {'orig':>8} {'real_pruned':>12} {'decomposed':>11} | {'gap':>8}")
    for r in results:
        gap = r["decomposed_ppl"] - r["real_pruned_ppl"]
        print(f"{r['model']:>10} {r['lam']:>5} | {r['orig_ppl']:>8.3f} {r['real_pruned_ppl']:>12.3f} "
              f"{r['decomposed_ppl']:>11.3f} | {gap:>+8.3f}")


if __name__ == "__main__":
    main()
