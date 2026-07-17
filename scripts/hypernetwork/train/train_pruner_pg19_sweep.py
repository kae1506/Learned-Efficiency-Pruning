"""
GPT-2 small vs OPT-125M — one combined λ-sweep on pg19 (B6, diary/ideas.md).

Not standalone (unlike train_pruner_gpt2.py / train_pruner_opt125m.py) —
imports the architecture-agnostic pieces (Pruner, evaluate, autocast_ctx)
from train_pruner_gpt2.py rather than duplicating them a third time, since
this runs against a full repo clone, not a single wget-pulled file.

WHY THIS SWEEP: F18 (GPT-2 on WikiText-2) shows pruning hurts monotonically;
F19 (OPT-125M on WikiText-2) shows it helps in-domain at every λ tested.
Hypothesis: the gap tracks which model is zero-shot WORSE-calibrated to the
target domain, not which model it is. zero_shot_ce_check.py confirmed the
predicted flip on pg19 (GPT-2 CE=3.326 > OPT CE=2.910 -- opposite of their
WikiText-2 order, GPT-2 CE=3.218 < OPT CE=3.893). This sweep tests whether
that flip in zero-shot calibration also flips which model's pruning curve
shows the F19-style in-domain improvement.

λ grids differ per model on purpose -- NOT because λ is comparable between
them (it isn't, see diary/crisp-findings.md: OPT prunes ~1.5-2x more
aggressively than GPT-2 at the same nominal λ on WikiText-2), but because
each grid is sized to where that model is predicted to show the interesting
region: OPT-125M is predicted to behave like GPT-2-on-WikiText-2 (F18,
monotonic hurt, no low-λ free lunch), GPT-2 is predicted to behave like
OPT-on-WikiText-2 (F19, real improvement, extending further than 1.8 was
enough to find OPT's own knee on WikiText-2). Compare the two models'
results via their %pruned-vs-ΔCE curves, never via matched-λ rows.

DATA -- pg19 (`emozilla/pg19` parquet mirror; the canonical `pg19`/
`deepmind/pg19` repos use a legacy dataset-loading-script format the
current `datasets` library rejects -- verified, not assumed). Both train
and test are STREAMED and CAPPED at a token budget, not eagerly tokenized
in full -- pg19's train split is a large long-document corpus (unlike
WikiText-2's few MB), and the training loop already cycles via
StopIteration on a smaller-than-full-epoch dataset (same as every prior
sweep in this project), so there's no need for the full split up front.
Token concatenation uses itertools.chain, not sum(lists, []) -- see the
train_pruner_opt125m.py fix, sum() is O(k*N) and looks like a hang on a
corpus with many documents.

Usage:
    python train_pruner_pg19_sweep.py [--models gpt2 opt125m]
        [--gpt2_lambdas ...] [--opt125m_lambdas ...] [--seeds 0 1]
        [--steps 18750] [--batch_size 8] [--timing_probe] [--stop_pod]

If HF Hub requests are being throttled/slow, set HF_TOKEN in the pod
environment -- picked up automatically by `datasets`/`huggingface_hub`,
no code change needed here.
"""
import contextlib
import itertools
import os
import shutil
import signal
import subprocess
import sys
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_pruner_gpt2 import Pruner, evaluate, autocast_ctx

os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")

OUT_ROOT = "/workspace/results/pg19_sweep"

# Both models verified architecturally compute-equivalent earlier in this
# project (12 layers, hidden=768, 12 heads, FFN=3072) -- shared constants.
N_LAYERS    = 12
N_INTER     = 3072
EMBED_DIM   = 768
LAYER_SHAPE = (N_INTER, EMBED_DIM)

PG19_REPO = "emozilla/pg19"   # parquet mirror of the official pg19 train/test splits


# ─────────────────────────────────────────────────────────────────────────────
# Per-model dispatch -- everything that actually differs between GPT-2 and
# OPT-125M lives here, explicitly, rather than being inferred/hidden.
# ─────────────────────────────────────────────────────────────────────────────

def load_gpt2(device):
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_opt125m(device):
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(
        "facebook/opt-125m", use_safetensors=True, torch_dtype=torch.float32
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def get_mlp_weights_gpt2(model):
    """GPT-2's Conv1D stores [in, out] -- needs .T for [out_nodes, in_features]."""
    return [model.transformer.h[i].mlp.c_fc.weight.T.detach() for i in range(N_LAYERS)]


def get_mlp_weights_opt(model):
    """OPT's fc1 is a plain nn.Linear, already [out_features, in_features] -- no .T."""
    return [model.model.decoder.layers[i].fc1.weight.detach() for i in range(N_LAYERS)]


@contextlib.contextmanager
def apply_gates_gpt2(model, gates):
    hooks = []
    for block, gate in zip(model.transformer.h, gates):
        def make_hook(g):
            def hook(module, args):
                return (args[0] * g.view(1, 1, -1),)
            return hook
        hooks.append(block.mlp.c_proj.register_forward_pre_hook(make_hook(gate)))
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


@contextlib.contextmanager
def apply_gates_opt(model, gates):
    hooks = []
    for block, gate in zip(model.model.decoder.layers, gates):
        def make_hook(g):
            def hook(module, args):
                return (args[0] * g.view(1, 1, -1),)
            return hook
        hooks.append(block.fc2.register_forward_pre_hook(make_hook(gate)))
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


def get_tokenizer_gpt2():
    from transformers import GPT2TokenizerFast
    return GPT2TokenizerFast.from_pretrained("gpt2")


def get_tokenizer_opt():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("facebook/opt-125m")


MODEL_SPECS = {
    "gpt2": dict(
        load_fn=load_gpt2, get_mlp_weights=get_mlp_weights_gpt2,
        apply_gates=apply_gates_gpt2, tokenizer_fn=get_tokenizer_gpt2,
        eval_max_length=1024, eval_stride=512,
        default_lambdas=[0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4],
        display_name="GPT-2 small",
    ),
    "opt125m": dict(
        load_fn=load_opt125m, get_mlp_weights=get_mlp_weights_opt,
        apply_gates=apply_gates_opt, tokenizer_fn=get_tokenizer_opt,
        eval_max_length=2048, eval_stride=1024,
        default_lambdas=[0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2],
        display_name="OPT-125M",
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# pg19 data -- streamed, capped, itertools.chain (not sum(lists, [])).
# ─────────────────────────────────────────────────────────────────────────────

def get_pg19_ids(tokenizer, split: str, n_tokens: int, desc: str) -> torch.Tensor:
    from datasets import load_dataset
    ds = load_dataset(PG19_REPO, split=split, streaming=True)
    ids = []
    for example in tqdm(ds, desc=desc, unit="book"):
        ids.extend(tokenizer(example["text"])["input_ids"])
        if len(ids) >= n_tokens:
            break
    if len(ids) < n_tokens:
        raise RuntimeError(f"pg19 {split} stream exhausted at {len(ids)} tokens, "
                           f"wanted {n_tokens}.")
    return torch.tensor(ids[:n_tokens], dtype=torch.long)


def get_pg19_loaders(tokenizer, seq_len: int, batch_size: int,
                     n_train_tokens: int, n_test_tokens: int, tag: str):
    """
    Train: streamed+capped, then chunked into non-overlapping seq_len blocks
    (same convention as WikiText-2's get_loaders() in the sibling scripts).
    The training loop cycles via StopIteration if this is smaller than a
    full epoch's worth of steps -- same as every prior sweep, WikiText-2
    itself is far smaller than 18,750 steps' worth of unique tokens too.
    Test: flat token stream, walked by evaluate()'s sliding window.
    """
    from torch.utils.data import DataLoader, TensorDataset

    train_ids = get_pg19_ids(tokenizer, "train", n_train_tokens, f"[{tag}] streaming pg19 train")
    total = (train_ids.size(0) // seq_len) * seq_len
    blocks = train_ids[:total].view(-1, seq_len)
    train_loader = DataLoader(TensorDataset(blocks), batch_size=batch_size, shuffle=True)

    test_ids = get_pg19_ids(tokenizer, "test", n_test_tokens, f"[{tag}] streaming pg19 test")

    return train_loader, test_ids


# ─────────────────────────────────────────────────────────────────────────────
# Single pruner training step -- model-agnostic given the right
# get_mlp_weights/apply_gates functions passed in.
# ─────────────────────────────────────────────────────────────────────────────

def pruner_step(pruner, model, optimizer, input_ids, sparsity_weight, device,
                get_mlp_weights, apply_gates):
    optimizer.zero_grad()

    weights = get_mlp_weights(model)
    gates   = pruner(weights)

    with torch.no_grad(), autocast_ctx(device):
        ce_orig = model(input_ids, labels=input_ids).loss.item()

    with apply_gates(model, gates), autocast_ctx(device):
        ce_pruned = model(input_ids, labels=input_ids).loss

    sparsity_loss = sum(g.mean() for g in gates) / len(gates)
    loss = (ce_pruned - ce_orig) + sparsity_weight * sparsity_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(pruner.parameters(), max_norm=1.0)
    optimizer.step()

    per_layer_keep = [g.mean().item() for g in gates]
    avg_gate = sum(per_layer_keep) / len(per_layer_keep)
    return {
        "loss": loss.item(), "ce_orig": ce_orig, "ce_pruned": ce_pruned.item(),
        "avg_gate": avg_gate, "per_layer_keep": per_layer_keep,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting (per-run 3-panel, identical style to the sibling scripts)
# ─────────────────────────────────────────────────────────────────────────────

def _smooth(values, window=100):
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo:i + 1]) / (i - lo + 1))
    return out


def plot_one_run(history, save_path, title):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    steps = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(title, fontsize=11, fontweight="bold")

    axes[0].plot(steps, history["loss"], alpha=0.15, color="steelblue")
    axes[0].plot(steps, _smooth(history["loss"]), color="steelblue", lw=2)
    axes[0].axhline(0, color="gray", ls="--", lw=0.8)
    axes[0].set_title("Pruner loss"); axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss"); axes[0].grid(alpha=0.3)

    axes[1].plot(steps, _smooth(history["ce_orig"]),   color="steelblue", lw=2, label="orig")
    axes[1].plot(steps, _smooth(history["ce_pruned"]), color="tomato",    lw=2, label="pruned")
    axes[1].set_title("CE loss (nats)"); axes[1].set_xlabel("step")
    axes[1].set_ylabel("CE"); axes[1].grid(alpha=0.3); axes[1].legend()

    cmap = plt.cm.tab20(np.linspace(0, 1, N_LAYERS))
    for i in range(N_LAYERS):
        per = [(1 - k) * 100 for k in history["per_layer_keep"][i]]
        axes[2].plot(steps, _smooth(per), color=cmap[i], lw=1.5, label=f"L{i}")
    axes[2].set_title("per-layer % pruned"); axes[2].set_xlabel("step")
    axes[2].set_ylabel("% pruned"); axes[2].set_ylim(0, 100)
    axes[2].grid(alpha=0.3); axes[2].legend(ncol=3, fontsize=7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(all_results, save_path):
    """% pruned vs ΔCE, one line per model -- the ONLY valid cross-model
    comparison (never matched-λ rows, see module docstring)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    colors = {"gpt2": "steelblue", "opt125m": "tomato"}
    ax.axhline(0, color="gray", ls=":", lw=1)
    for model_name, color in colors.items():
        rows = [r for r in all_results if r["model"] == model_name]
        if not rows:
            continue
        lambdas = sorted(set(r["lambda"] for r in rows))
        pts = []
        for lam in lambdas:
            rs = [r for r in rows if r["lambda"] == lam]
            pct_pruned = float(np.mean([r["pct_pruned"] for r in rs]))
            delta_ce = float(np.mean([np.log(r["pruned_ppl"] / r["orig_ppl"]) for r in rs]))
            pts.append((pct_pruned, delta_ce, lam))
        pts.sort(key=lambda p: p[0])
        xs, ys, lams = zip(*pts)
        ax.plot(xs, ys, "o-", color=color, lw=1.8, markersize=7,
               label=MODEL_SPECS[model_name]["display_name"])
        for x, y, lam in zip(xs, ys, lams):
            ax.annotate(f"λ={lam}", (x, y), xytext=(6, 6),
                       textcoords="offset points", fontsize=7)
    ax.set_xlabel("% pruned")
    ax.set_ylabel("ΔCE (nats) — negative = improvement over dense model")
    ax.set_title("GPT-2 small vs OPT-125M — pg19 pruning sweep (B6)", fontweight="bold")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Summary txt
# ─────────────────────────────────────────────────────────────────────────────

def write_run_summary(path, model_name, lam, seed, history, per_layer_kept,
                      orig_ppl, pruned_ppl, total_time):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    final_gate = history["avg_gate"][-1]
    pct_pruned = (1 - final_gate) * 100
    lines = [
        f"{MODEL_SPECS[model_name]['display_name']} MLP pruner (pg19) — λ={lam}, seed={seed}",
        f"layers : {N_LAYERS} MLP blocks, {N_INTER} intermediate neurons each",
        f"steps  : {len(history['loss'])}",
        f"time   : {total_time:.1f}s",
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
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Per-(model, λ, seed) training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one(model_name, model, train_loader, test_ids, lam, seed, args, device, run_dir):
    torch.manual_seed(seed); np.random.seed(seed)
    spec = MODEL_SPECS[model_name]

    layer_shapes = [LAYER_SHAPE] * N_LAYERS
    pruner = Pruner(layer_shapes, embed_dim=args.embed_dim,
                    lstm_hidden=args.lstm_hidden).to(device)
    opt = torch.optim.Adam(pruner.parameters(), lr=args.lr)

    tag = f"{model_name} λ={lam} seed={seed}"
    print(f"\n── {tag} ── pruner params: "
          f"{sum(p.numel() for p in pruner.parameters()):,}", flush=True)

    history = {
        "loss": [], "ce_orig": [], "ce_pruned": [], "avg_gate": [],
        "per_layer_keep": [[] for _ in range(N_LAYERS)],
    }

    t0 = time.time()
    step = 0
    loader_iter = iter(train_loader)
    pbar = tqdm(total=args.steps, desc=tag, unit="step", dynamic_ncols=True)

    while step < args.steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        ids = batch[0].to(device)   # TensorDataset -> (tensor,) tuple, not a dict
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
            tqdm.write(f"  [{tag}] step {step:>5}/{args.steps} | loss {m['loss']:+.3f} | "
                       f"CE orig {m['ce_orig']:.3f} pruned {m['ce_pruned']:.3f} | "
                       f"avg pruned {avg_pruned:5.1f}%")

        if args.timing_probe and step == 50:
            elapsed = time.time() - t0
            t_per_step = elapsed / 50
            projected = t_per_step * args.steps
            pbar.close()
            print(f"\n  TIMING PROBE [{tag}]: {t_per_step*1000:.0f}ms/step → "
                  f"full run ({args.steps} steps) ≈ {projected/60:.1f} min", flush=True)
            return None

    pbar.close()
    total_time = time.time() - t0

    pruner.eval()
    with torch.no_grad():
        final_gates = pruner(spec["get_mlp_weights"](model))
    per_layer_kept = [int(g.sum().item()) for g in final_gates]

    orig_ce = evaluate(model, test_ids, device, gates=None, desc=f"[{tag}] eval orig",
                       max_length=spec["eval_max_length"], stride=spec["eval_stride"])
    pruned_ce = evaluate(model, test_ids, device, gates=final_gates, desc=f"[{tag}] eval pruned",
                         max_length=spec["eval_max_length"], stride=spec["eval_stride"])
    orig_ppl, pruned_ppl = float(np.exp(orig_ce)), float(np.exp(pruned_ce))

    final_gate = history["avg_gate"][-1]
    pct_pruned = (1 - final_gate) * 100
    print(f"  → [{tag}] pruned {pct_pruned:.2f}% | orig ppl {orig_ppl:.3f} → "
          f"pruned ppl {pruned_ppl:.3f} | {total_time:.0f}s", flush=True)

    plot_one_run(history, os.path.join(run_dir, "plot.png"),
                title=f"{spec['display_name']} MLP pruner (pg19) — λ={lam} seed={seed} — "
                      f"{pct_pruned:.1f}% pruned, ppl {pruned_ppl:.2f}")
    write_run_summary(os.path.join(run_dir, "summary.txt"), model_name, lam, seed,
                      history, per_layer_kept, orig_ppl, pruned_ppl, total_time)
    torch.save({
        "pruner_state_dict": pruner.state_dict(), "model": model_name,
        "lambda": lam, "seed": seed, "embed_dim": args.embed_dim,
        "lstm_hidden": args.lstm_hidden, "per_layer_kept": per_layer_kept,
        "orig_ppl": orig_ppl, "pruned_ppl": pruned_ppl,
    }, os.path.join(run_dir, "pruner.pt"))
    print(f"  [saved] {run_dir}/", flush=True)

    return {
        "model": model_name, "lambda": lam, "seed": seed,
        "per_layer_kept": per_layer_kept, "pct_pruned": pct_pruned,
        "orig_ppl": orig_ppl, "pruned_ppl": pruned_ppl, "total_time": total_time,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stop the pod (identical two-tier mechanism to the sibling scripts)
# ─────────────────────────────────────────────────────────────────────────────

def stop_pod():
    print("\nStopping pod (compute billing off, /workspace preserved) in 10s...", flush=True)
    time.sleep(10)
    os.sync()
    pod_id, api_key = os.environ.get("RUNPOD_POD_ID"), os.environ.get("RUNPOD_API_KEY")
    if pod_id and api_key and shutil.which("runpodctl"):
        subprocess.run(["runpodctl", "config", "--apiKey", api_key], check=False)
        result = subprocess.run(["runpodctl", "stop", "pod", pod_id], check=False)
        if result.returncode == 0:
            print("  Stopped via runpodctl.", flush=True)
            return
        print("  runpodctl stop failed, falling back to kill PID 1.", flush=True)
    else:
        print("  No RUNPOD_API_KEY/runpodctl available, using kill PID 1 fallback.", flush=True)
    os.kill(1, signal.SIGTERM)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gpt2", "opt125m"],
                    choices=["gpt2", "opt125m"])
    ap.add_argument("--gpt2_lambdas", type=float, nargs="+", default=None,
                    help=f"Default: {MODEL_SPECS['gpt2']['default_lambdas']}")
    ap.add_argument("--opt125m_lambdas", type=float, nargs="+", default=None,
                    help=f"Default: {MODEL_SPECS['opt125m']['default_lambdas']}")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=18750)
    ap.add_argument("--batch_size", type=int, default=8,
                    help="Kept at 8 to match every prior WikiText-2 sweep "
                         "(GPT-2 and OPT-125M scripts both default to this) -- "
                         "changing it changes optimization dynamics AND what "
                         "a given λ produces in terms of %%pruned, see the "
                         "chat discussion. Printed at startup either way.")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--n_train_tokens", type=int, default=2_400_000,
                    help="pg19 train tokens to stream+cache, roughly matching "
                         "WikiText-2's own train-set scale. The training loop "
                         "cycles via StopIteration if this is smaller than a "
                         "full epoch's worth of steps, same as every prior sweep.")
    ap.add_argument("--n_test_tokens", type=int, default=245_000,
                    help="Matches WikiText-2 test's token count for comparable "
                         "statistical power in the eval CE.")
    ap.add_argument("--embed_dim", type=int, default=64)
    ap.add_argument("--lstm_hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--log_every", type=int, default=250)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_dir", type=str, default=OUT_ROOT)
    ap.add_argument("--timing_probe", action="store_true",
                    help="Run 50 steps per model (first λ, seed 0), print "
                         "per-step time, then exit. No full sweep.")
    ap.add_argument("--stop_pod", action="store_true")
    args = ap.parse_args()
    out_root = args.out_dir

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    lambdas_by_model = {
        "gpt2":    args.gpt2_lambdas    or MODEL_SPECS["gpt2"]["default_lambdas"],
        "opt125m": args.opt125m_lambdas or MODEL_SPECS["opt125m"]["default_lambdas"],
    }
    print(f"Device: {device} | models={args.models} | batch_size={args.batch_size} "
          f"(unchanged from WikiText-2 sweeps unless overridden) | steps={args.steps}")
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
        print(f"Data ready: train_blocks={len(train_loader.dataset)} "
              f"test_tokens={test_ids.size(0):,} "
              f"(eval: max_length={spec['eval_max_length']} stride={spec['eval_stride']})", flush=True)

        if args.timing_probe:
            lam0 = lambdas_by_model[model_name][0]
            print(f"\n── TIMING PROBE [{model_name}] (50 steps, λ={lam0} seed=0) ──", flush=True)
            run_dir = os.path.join(out_root, model_name, "timing_probe")
            train_one(model_name, model, train_loader, test_ids, lam0, 0, args, device, run_dir)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        lambdas = lambdas_by_model[model_name]
        total_runs = len(lambdas) * len(args.seeds)
        run_num = 0
        for lam in lambdas:
            for seed in args.seeds:
                run_num += 1
                tqdm.write(f"\n{'='*70}\n[{model_name}] Run {run_num}/{total_runs}\n{'='*70}")
                run_dir = os.path.join(out_root, model_name, f"lambda_{lam}", f"seed_{seed}")
                res = train_one(model_name, model, train_loader, test_ids,
                               lam, seed, args, device, run_dir)
                if res is not None:
                    all_results.append(res)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.timing_probe:
        return

    plot_model_comparison(all_results, os.path.join(out_root, "gpt2_vs_opt125m_pg19.png"))

    sep = "-" * 90
    rows = [f"pg19 sweep — models={args.models} | seeds={args.seeds} | "
           f"steps={args.steps} | batch_size={args.batch_size} | device={device}", sep,
           f"{'model':>10} {'lambda':>7} {'seed':>5} | {'% pruned':>9} | "
           f"{'orig ppl':>9} | {'pruned ppl':>10} | {'ppl rise':>9}", sep]
    for r in all_results:
        rows.append(f"{r['model']:>10} {r['lambda']:>7} {r['seed']:>5} | "
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
