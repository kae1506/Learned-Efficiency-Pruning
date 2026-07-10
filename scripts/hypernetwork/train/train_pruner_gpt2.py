"""
GPT-2 small MLP pruner — λ × seed sweep. Standalone file, no repo deps.

Prunes the 3072 intermediate neurons in each of GPT-2 small's 12 MLP blocks.
Frozen GPT-2 weights; only the BiLSTM pruner (inlined below) trains.

RunPod workflow (manual pod, web terminal, single file):
  1. Create a pod in the RunPod UI dashboard (Volume Disk only, no network
     volume needed — see storage notes below).
  2. Open the web terminal.
  3. Pull this file directly, e.g.:
       wget -O train_pruner_gpt2.py <raw-github-url-of-this-file>
  4. pip install transformers datasets matplotlib numpy tqdm
  5. python train_pruner_gpt2.py [--lambdas ...] [--seeds ...] [--stop_pod]

Outputs land under OUT_ROOT (default /workspace/..., i.e. the Volume Disk):
  lambda_<λ>/[seed_<s>/]plot.png       3-panel: pruner loss / LM loss / per-layer %
  lambda_<λ>/[seed_<s>/]summary.txt    config + final eval perplexity
  lambda_<λ>/[seed_<s>/]pruner.pt      pruner checkpoint (state_dict + config)
  comparison.png, efficiency.png, summary.txt   aggregate, written after full sweep

Each (λ, seed) run's outputs are written immediately after that run finishes —
not batched at the end — so a mid-sweep interruption still leaves completed
runs on disk.

PERFORMANCE — bf16 autocast on CUDA:
  The frozen GPT-2 forward passes (the dominant cost — see autocast_ctx) run
  under torch.autocast(dtype=torch.bfloat16) when on CUDA, engaging Tensor
  Cores instead of running plain fp32 matmuls. This roughly doubles achievable
  throughput on Ada/Blackwell cards (4090/5090) at no extra cost — measure
  with --timing_probe before assuming a number; GPT-2 small at this batch/seq
  size may be bandwidth- or overhead-bound rather than purely compute-bound.
  The pruner itself stays fp32 (negligible compute, keeps the STE gate
  threshold simple). No-op on CPU/MPS.

STORAGE — Volume Disk only, no network volume:
  - GPT-2 weights (~500MB) + WikiText-2 (~1MB) are cached under HF_HOME, which
    this script points at /root/.cache (Container Disk, ephemeral) rather than
    /workspace (Volume Disk). They don't count against your billed persistent
    storage; the cost is a ~500MB re-download if the pod is later recreated.
  - Per-run outputs are small: plot.png (~200-500KB) + summary.txt (~1KB) +
    pruner.pt (~8MB at default embed_dim=64/lstm_hidden=128, fp32 state_dict
    only — no optimizer state saved). A default 6λ × 2-seed sweep (12 runs) is
    roughly 100-110MB total under /workspace.
  - Nothing else touches the Volume Disk: no intermediate/step-level history
    arrays are written to disk, only the final plot + summary + checkpoint.

STOPPING THE POD (--stop_pod):
  Two-tier, in order:
   1. Documented path: `runpodctl stop pod $RUNPOD_POD_ID`, using RUNPOD_API_KEY
      if you set one as a pod environment variable when creating the pod in
      the UI. This is RunPod's actual supported stop mechanism — guaranteed
      to behave like clicking "Stop": compute billing stops, Volume Disk
      (/workspace) is preserved. RUNPOD_POD_ID is auto-injected into every
      pod; RUNPOD_API_KEY is not — you opt in by adding it yourself.
   2. Fallback if no API key/runpodctl available: kill PID 1 inside the
      container, which ends the container — RunPod reports this as the pod
      exiting, same billing/storage effect as (1) in practice, but it's a
      container-exit side effect rather than a documented API contract, so
      it's not guaranteed. Confirmed against RunPod's ToS: nothing prohibits
      ending your own pod's process (the "undue burden on system resources"
      clause targets abuse of shared infra, not ending your own compute
      early). Still, watch the dashboard the first time to confirm it flips
      to "Stopped" rather than restarting.
  NEVER use "Terminate" without downloading /workspace first — unlike Stop,
  Terminate deletes the Volume Disk along with the pod.
"""

import contextlib
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

# Keep HF model/dataset cache off the Volume Disk (see STORAGE note above).
os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

OUT_ROOT = "/workspace/results/gpt2_lambda_sweep"

N_LAYERS    = 12
N_INTER     = 3072   # intermediate (c_fc output) neurons per MLP block
EMBED_DIM   = 768    # GPT-2 small hidden size
LAYER_SHAPE = (N_INTER, EMBED_DIM)   # [out_nodes, in_features] per layer


# ─────────────────────────────────────────────────────────────────────────────
# Pruner — inlined from src/pruners/bilstm.py (+ binary_ste from mlp.py) so
# this file has zero local-repo imports and can be pulled/run standalone.
# ─────────────────────────────────────────────────────────────────────────────

def binary_ste(logits: torch.Tensor) -> torch.Tensor:
    """
    Straight-Through Estimator for hard binary gates.
    Forward:  hard 0/1 threshold at 0.5
    Backward: gradient flows through sigmoid (non-zero everywhere)
    """
    soft = torch.sigmoid(logits)
    hard = (soft > 0.5).float()
    return hard - soft.detach() + soft


class Pruner(nn.Module):
    """
    Hybrid RowEncoder + Bidirectional LSTM pruner.

    Per-node path: each row of a layer's weight matrix (incoming weights to
    one node) maps to a scalar logit via a shared 2-layer MLP.

    Cross-layer context path: a BiLSTM over per-layer embeddings lets the
    context bias for layer i be informed by all layers in both directions,
    so the pruner can reason globally (e.g. "layer 2 has many high-norm
    nodes, so layer 1 can be pruned harder without losing capacity").

    tanh bounds the context bias to (-1, 1): it can modulate but never
    override the per-node logit (initialised to +2.0), which prevents
    runaway gate collapse early in training.
    """

    def __init__(
        self,
        layer_shapes: list[tuple[int, int]],
        embed_dim: int   = 64,
        lstm_hidden: int = 64,
    ):
        super().__init__()

        self.row_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, 1),
            )
            for _, in_features in layer_shapes
        ])
        for enc in self.row_encoders:
            nn.init.constant_(enc[-1].bias, 2.0)

        self.layer_projectors = nn.ModuleList([
            nn.Linear(in_features, lstm_hidden)
            for _, in_features in layer_shapes
        ])

        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )

        self.context_norm = nn.LayerNorm(lstm_hidden * 2)
        self.context_head = nn.Linear(lstm_hidden * 2, 1)
        nn.init.zeros_(self.context_head.weight)
        nn.init.zeros_(self.context_head.bias)

    def _node_scores(self, weight_matrices: list[torch.Tensor]) -> list[torch.Tensor]:
        node_logits = [
            enc(W).squeeze(-1)
            for enc, W in zip(self.row_encoders, weight_matrices)
        ]

        layer_embeds = [
            F.relu(proj(W.mean(dim=0)))
            for proj, W in zip(self.layer_projectors, weight_matrices)
        ]

        seq = torch.stack(layer_embeds, dim=0).unsqueeze(0)
        lstm_out, _ = self.lstm(seq)

        context_biases = torch.tanh(
            self.context_head(self.context_norm(lstm_out.squeeze(0))).squeeze(-1)
        )

        return [logits + ctx for logits, ctx in zip(node_logits, context_biases)]

    def forward(self, weight_matrices: list[torch.Tensor]) -> list[torch.Tensor]:
        return [binary_ste(s) for s in self._node_scores(weight_matrices)]

    @torch.no_grad()
    def scores(self, weight_matrices: list[torch.Tensor]) -> list[torch.Tensor]:
        return self._node_scores(weight_matrices)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_gpt2(device):
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def get_mlp_weights(model) -> list[torch.Tensor]:
    """Detached c_fc weights transposed to [out_nodes, in_features] = [3072, 768]."""
    return [
        model.transformer.h[i].mlp.c_fc.weight.T.detach()
        for i in range(N_LAYERS)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Mixed precision — bf16 autocast around the frozen GPT-2 forward passes
# (the dominant cost). bf16 shares fp32's exponent range so there's no
# underflow/loss-scaling concern the way there is with fp16; GPT-2 is frozen
# throughout, so this only affects forward-pass activations, not any trained
# weights. The pruner itself (small matrices, STE threshold) stays in fp32 —
# it's a negligible fraction of compute and the binary gate threshold is the
# one place precision is worth keeping simple. CPU/MPS (local dev) skip
# autocast entirely — this only engages Tensor Cores on CUDA.
# ─────────────────────────────────────────────────────────────────────────────

def autocast_ctx(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


# ─────────────────────────────────────────────────────────────────────────────
# Masked forward via pre-hooks on c_proj (post-GELU interception)
# ─────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def apply_gates(model, gates):
    """Multiply the 3072-dim post-GELU activation by gate before c_proj."""
    hooks = []
    for block, gate in zip(model.transformer.h, gates):
        def make_hook(g):
            def hook(module, args):
                x = args[0]
                return (x * g.view(1, 1, -1),)
            return hook
        hooks.append(block.mlp.c_proj.register_forward_pre_hook(make_hook(gate)))
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Data — WikiText-2 (no API key needed, fully public)
# ─────────────────────────────────────────────────────────────────────────────

def get_loaders(seq_len: int, batch_size: int, num_workers: int = 2):
    """
    Train: non-overlapping fixed-length blocks — standard for training,
    unaffected by the eval-protocol choice below.
    Test: returned as ONE flat token stream (not chunked/batched) — evaluate()
    walks it with a sliding window so eval isn't penalized by short-context
    block boundaries (see evaluate() docstring).
    """
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast
    from torch.utils.data import DataLoader

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    def tokenize(examples):
        return tokenizer(examples["text"])

    def group(examples):
        ids = sum(examples["input_ids"], [])
        total = (len(ids) // seq_len) * seq_len
        blocks = [ids[i:i + seq_len] for i in range(0, total, seq_len)]
        return {"input_ids": blocks}

    tokenized = raw.map(tokenize, batched=True, remove_columns=["text"])

    train_blocked = tokenized["train"].map(group, batched=True,
                                           remove_columns=tokenized["train"].column_names)
    train_blocked.set_format(type="torch", columns=["input_ids"])
    train_loader = DataLoader(train_blocked, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)

    test_ids = torch.tensor(sum(tokenized["test"]["input_ids"], []), dtype=torch.long)

    return train_loader, test_ids


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation — sliding-window perplexity on the full test stream
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, test_ids, device, gates=None, desc="eval",
            max_length: int = 1024, stride: int = 512) -> float:
    """
    Returns cross-entropy loss (nats). Set gates=None for unpruned model.

    Standard sliding-window protocol (GPT-2 paper, SparseGPT, Wanda), not
    non-overlapping blocks: walk a max_length-token window over the full test
    stream in stride-token steps; at each step only the NEW (non-overlapping)
    `stride` tokens are scored (via -100 label-masking on the rest), so no
    token is double-counted and every scored token past the first window has
    close to full max_length context. Non-overlapping block eval inflates ppl
    because every block's leading tokens see artificially short context.
    max_length=1024 is GPT-2 small's actual context window (independent of
    the training seq_len, which can stay shorter for cost reasons).
    """
    total_len = test_ids.size(0)
    total_nll = total_tokens = 0
    prev_end = 0
    positions = list(range(0, total_len, stride))
    for begin in tqdm(positions, desc=desc, unit="window", leave=False, dynamic_ncols=True):
        end = min(begin + max_length, total_len)
        trg_len = end - prev_end   # tokens newly covered since the last window
        ids = test_ids[begin:end].unsqueeze(0).to(device)
        labels = ids.clone()
        labels[:, :-trg_len] = -100   # mask tokens already scored by the previous window

        with autocast_ctx(device):
            if gates is None:
                loss = model(ids, labels=labels).loss
            else:
                with apply_gates(model, gates):
                    loss = model(ids, labels=labels).loss

        n_tok = (labels[:, 1:] != -100).sum().item()
        total_nll    += loss.item() * n_tok
        total_tokens += n_tok
        prev_end = end
        if end == total_len:
            break
    return total_nll / total_tokens   # mean CE in nats


# ─────────────────────────────────────────────────────────────────────────────
# Single pruner training step
# ─────────────────────────────────────────────────────────────────────────────

def pruner_step(pruner, model, optimizer, input_ids, sparsity_weight, device):
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
        "loss":          loss.item(),
        "ce_orig":       ce_orig,
        "ce_pruned":     ce_pruned.item(),
        "avg_gate":      avg_gate,
        "per_layer_keep": per_layer_keep,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
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

    # loss
    axes[0].plot(steps, history["loss"], alpha=0.15, color="steelblue")
    axes[0].plot(steps, _smooth(history["loss"]), color="steelblue", lw=2)
    axes[0].axhline(0, color="gray", ls="--", lw=0.8)
    axes[0].set_title("Pruner loss"); axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss"); axes[0].grid(alpha=0.3)

    # CE orig vs pruned
    axes[1].plot(steps, _smooth(history["ce_orig"]),   color="steelblue", lw=2, label="orig")
    axes[1].plot(steps, _smooth(history["ce_pruned"]), color="tomato",    lw=2, label="pruned")
    axes[1].set_title("CE loss (nats)"); axes[1].set_xlabel("step")
    axes[1].set_ylabel("CE"); axes[1].grid(alpha=0.3); axes[1].legend()

    # per-layer % pruned (12 layers, use colormap)
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


def plot_multiseed_comparison(per_lambda_stats, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    lambdas  = [s["lambda"] for s in per_lambda_stats]
    pp_mean  = [s["pct_pruned_mean"] for s in per_lambda_stats]
    pp_std   = [s["pct_pruned_std"]  for s in per_lambda_stats]
    ppl_mean = [s["pruned_ppl_mean"] for s in per_lambda_stats]
    ppl_std  = [s["pruned_ppl_std"]  for s in per_lambda_stats]
    orig_ppl = per_lambda_stats[0]["orig_ppl"]
    ax.errorbar(pp_mean, ppl_mean, xerr=pp_std, yerr=ppl_std,
                fmt="o", color="tomato", markersize=10, capsize=4, lw=1.5, zorder=3)
    for lam, x, y in zip(lambdas, pp_mean, ppl_mean):
        ax.annotate(f"λ={lam}", (x, y), xytext=(8, 4),
                    textcoords="offset points", fontsize=10)
    ax.axhline(orig_ppl, color="steelblue", ls="--", lw=1.2,
               label=f"unpruned ppl = {orig_ppl:.2f}")
    ax.set_xlabel("% MLP intermediate neurons pruned (avg over 12 blocks)")
    ax.set_ylabel("pruned test perplexity")
    ax.set_title("GPT-2 small MLP — multi-seed Pareto (mean ± stdev)", fontweight="bold")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_efficiency(per_lambda_stats, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    lambdas = [s["lambda"] for s in per_lambda_stats]
    orig_ppl = per_lambda_stats[0]["orig_ppl"]
    eff = []
    for s in per_lambda_stats:
        ppl_drop = s["pruned_ppl_mean"] - orig_ppl
        eff.append(s["pct_pruned_mean"] / max(ppl_drop, 0.5))
    ax.plot(lambdas, eff, "o-", color="darkorange", markersize=10, lw=2)
    for lam, e in zip(lambdas, eff):
        ax.annotate(f"{e:.1f}", (lam, e), xytext=(6, 4),
                    textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("λ (log scale)")
    ax.set_ylabel("efficiency  =  (% pruned) / max(ppl_drop, 0.5)")
    ax.set_title("GPT-2 small MLP — pruning efficiency vs λ", fontweight="bold")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Summary txt
# ─────────────────────────────────────────────────────────────────────────────

def write_run_summary(path, lam, seed, history, per_layer_kept,
                      orig_ppl, pruned_ppl, total_time):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    final_gate = history["avg_gate"][-1]
    pct_pruned = (1 - final_gate) * 100
    lines = [
        f"GPT-2 small MLP pruner — λ={lam}, seed={seed}",
        f"layers : 12 MLP blocks, 3072 intermediate neurons each",
        f"steps  : {len(history['loss'])}",
        f"time   : {total_time:.1f}s",
        "-" * 60,
        f"final avg keep gate          : {final_gate:.4f}",
        f"final % MLP neurons pruned   : {pct_pruned:.2f}%",
        f"per-block neurons kept       : {per_layer_kept}",
        "-" * 60,
        f"FULL test set (WikiText-2):",
        f"  original  ppl              : {orig_ppl:.3f}",
        f"  pruned    ppl              : {pruned_ppl:.3f}",
        f"  ppl increase               : {pruned_ppl - orig_ppl:+.3f}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Per-(λ, seed) training loop — saves plot.png, summary.txt, pruner.pt for
# THIS run before returning, so results are on disk incrementally rather
# than held in memory / written only at the end of the whole sweep.
# ─────────────────────────────────────────────────────────────────────────────

def train_one(lam, seed, model, train_loader, test_ids, args, device, run_dir):
    torch.manual_seed(seed); np.random.seed(seed)

    layer_shapes = [LAYER_SHAPE] * N_LAYERS
    pruner = Pruner(layer_shapes, embed_dim=args.embed_dim,
                    lstm_hidden=args.lstm_hidden).to(device)
    opt = torch.optim.Adam(pruner.parameters(), lr=args.lr)

    tag = f"λ={lam} seed={seed}"
    print(f"\n── {tag} ── pruner params: "
          f"{sum(p.numel() for p in pruner.parameters()):,}", flush=True)

    history = {
        "loss":          [],
        "ce_orig":       [],
        "ce_pruned":     [],
        "avg_gate":      [],
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
            tqdm.write(f"  [{tag}] step {step:>5}/{args.steps} | "
                       f"loss {m['loss']:+.3f} | "
                       f"CE orig {m['ce_orig']:.3f} pruned {m['ce_pruned']:.3f} | "
                       f"avg pruned {avg_pruned:5.1f}%")

        if args.timing_probe and step == 50:
            elapsed = time.time() - t0
            t_per_step = elapsed / 50
            projected = t_per_step * args.steps
            pbar.close()
            print(f"\n  TIMING PROBE: {t_per_step*1000:.0f}ms/step → "
                  f"full run ({args.steps} steps) ≈ {projected/60:.1f} min", flush=True)
            return None

    pbar.close()

    total_time = time.time() - t0

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
    print(f"  → [{tag}] final keep {final_gate:.3f}  pruned {pct_pruned:.2f}% | "
          f"per-block kept {per_layer_kept} | "
          f"orig ppl {orig_ppl:.3f} → pruned ppl {pruned_ppl:.3f} | "
          f"{total_time:.0f}s", flush=True)

    plot_one_run(
        history,
        os.path.join(run_dir, "plot.png"),
        title=(f"GPT-2 small MLP pruner — λ={lam} seed={seed} — "
               f"{pct_pruned:.1f}% pruned, ppl {pruned_ppl:.2f}"),
    )
    write_run_summary(
        os.path.join(run_dir, "summary.txt"),
        lam, seed, history, per_layer_kept, orig_ppl, pruned_ppl, total_time,
    )
    torch.save({
        "pruner_state_dict": pruner.state_dict(),
        "lambda":            lam,
        "seed":              seed,
        "embed_dim":         args.embed_dim,
        "lstm_hidden":       args.lstm_hidden,
        "per_layer_kept":    per_layer_kept,
        "orig_ppl":          orig_ppl,
        "pruned_ppl":        pruned_ppl,
    }, os.path.join(run_dir, "pruner.pt"))
    print(f"  [saved] {run_dir}/  (plot.png, summary.txt, pruner.pt)", flush=True)

    # history is intentionally not returned — it's already on disk in plot.png
    # /summary.txt, and dropping it here keeps memory flat across the sweep
    # instead of accumulating every step of every run.
    return {
        "lambda":         lam,
        "seed":           seed,
        "per_layer_kept": per_layer_kept,
        "pct_pruned":     pct_pruned,
        "orig_ppl":       orig_ppl,
        "pruned_ppl":     pruned_ppl,
        "total_time":     total_time,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stop the pod (see STOPPING THE POD note in the module docstring)
# ─────────────────────────────────────────────────────────────────────────────

def stop_pod():
    print("\nStopping pod (compute billing off, /workspace preserved) in 10s...",
          flush=True)
    time.sleep(10)
    os.sync()  # flush any buffered writes to the Volume Disk before halting

    pod_id  = os.environ.get("RUNPOD_POD_ID")
    api_key = os.environ.get("RUNPOD_API_KEY")
    if pod_id and api_key and shutil.which("runpodctl"):
        subprocess.run(["runpodctl", "config", "--apiKey", api_key], check=False)
        result = subprocess.run(["runpodctl", "stop", "pod", pod_id], check=False)
        if result.returncode == 0:
            print("  Stopped via runpodctl (documented path).", flush=True)
            return
        print("  runpodctl stop failed, falling back to kill PID 1.", flush=True)
    else:
        print("  No RUNPOD_API_KEY/runpodctl available, using kill PID 1 fallback.",
              flush=True)

    os.kill(1, signal.SIGTERM)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas",      type=float, nargs="+",
                    default=[0.01, 0.02, 0.05, 0.10, 0.20, 0.40])
    ap.add_argument("--seeds",        type=int,   nargs="+", default=[0, 1])
    ap.add_argument("--steps",        type=int,   default=18750)
    ap.add_argument("--batch_size",   type=int,   default=8)
    ap.add_argument("--seq_len",      type=int,   default=512)
    ap.add_argument("--embed_dim",    type=int,   default=64)
    ap.add_argument("--lstm_hidden",  type=int,   default=128)
    ap.add_argument("--lr",           type=float, default=0.001)
    ap.add_argument("--log_every",    type=int,   default=250)
    ap.add_argument("--eval_max_length", type=int, default=1024,
                    help="Sliding-window eval context length (GPT-2 small's "
                         "actual n_positions). Independent of --seq_len.")
    ap.add_argument("--eval_stride",  type=int,   default=512,
                    help="Sliding-window eval stride (512 = 50%% overlap, "
                         "the standard tradeoff point).")
    ap.add_argument("--device",       type=str,   default="cuda")
    ap.add_argument("--out_dir",      type=str,   default=OUT_ROOT)
    ap.add_argument("--timing_probe", action="store_true",
                    help="Run 50 steps, print per-step time, then exit.")
    ap.add_argument("--stop_pod",     action="store_true",
                    help="Stop the RunPod pod after the sweep: tries "
                         "`runpodctl stop pod` (if RUNPOD_API_KEY is set), "
                         "else falls back to kill PID 1 (see module "
                         "docstring). Volume Disk is preserved either way.")
    args = ap.parse_args()
    out_root = args.out_dir

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device} | λs={args.lambdas} | seeds={args.seeds} | steps={args.steps}")
    print(f"Output: {out_root}")

    print("Loading GPT-2 small (downloads ~500MB on first run) ...", flush=True)
    model = load_gpt2(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GPT-2 small loaded — {n_params:,} params, frozen.", flush=True)

    print("Loading WikiText-2 (downloads on first run) ...", flush=True)
    train_loader, test_ids = get_loaders(args.seq_len, args.batch_size)
    print(f"Data: seq_len={args.seq_len} batch={args.batch_size} "
          f"train_batches={len(train_loader)} test_tokens={test_ids.size(0):,} "
          f"(eval: max_length={args.eval_max_length} stride={args.eval_stride})", flush=True)

    if args.timing_probe:
        print("\n── TIMING PROBE (50 steps, λ=0.05 seed=0) ──", flush=True)
        run_dir = os.path.join(out_root, "timing_probe")
        train_one(0.05, 0, model, train_loader, test_ids, args, device, run_dir)
        return

    os.makedirs(out_root, exist_ok=True)
    all_results = []
    total_runs = len(args.lambdas) * len(args.seeds)
    run_num = 0

    for lam in args.lambdas:
        for seed in args.seeds:
            run_num += 1
            tqdm.write(f"\n{'='*70}\nRun {run_num}/{total_runs}\n{'='*70}")
            run_dir = (os.path.join(out_root, f"lambda_{lam}", f"seed_{seed}")
                       if len(args.seeds) > 1
                       else os.path.join(out_root, f"lambda_{lam}"))
            res = train_one(lam, seed, model, train_loader, test_ids,
                            args, device, run_dir)
            if res is None:
                return
            all_results.append(res)

    # Aggregate per-λ across seeds
    per_lambda_stats = []
    for lam in args.lambdas:
        runs = [r for r in all_results if r["lambda"] == lam]
        pcts = [r["pct_pruned"]  for r in runs]
        ppls = [r["pruned_ppl"]  for r in runs]
        per_lambda_stats.append({
            "lambda":          lam,
            "pct_pruned_mean": float(np.mean(pcts)),
            "pct_pruned_std":  float(np.std(pcts)),
            "pruned_ppl_mean": float(np.mean(ppls)),
            "pruned_ppl_std":  float(np.std(ppls)),
            "orig_ppl":        runs[0]["orig_ppl"],
            "runs":            runs,
        })

    plot_multiseed_comparison(per_lambda_stats, os.path.join(out_root, "comparison.png"))
    plot_efficiency(per_lambda_stats, os.path.join(out_root, "efficiency.png"))

    header = (f"GPT-2 small MLP pruner — λ sweep | "
              f"seeds={args.seeds} | steps={args.steps} | device={device}")
    sep = "-" * 90
    col = (f"{'lambda':>7} {'seed':>5} | {'% pruned':>9} | "
           f"{'orig ppl':>9} | {'pruned ppl':>10} | {'ppl rise':>9} | {'per-block kept (12)':>20}")
    rows = [header, sep, col, sep]
    for s in per_lambda_stats:
        for r in s["runs"]:
            rows.append(
                f"{r['lambda']:>7} {r['seed']:>5} | "
                f"{r['pct_pruned']:>8.2f}% | "
                f"{r['orig_ppl']:>9.3f} | {r['pruned_ppl']:>10.3f} | "
                f"{r['pruned_ppl'] - r['orig_ppl']:>+9.3f} | "
                f"{r['per_layer_kept']}"
            )
        if len(args.seeds) > 1:
            rows.append(
                f"{s['lambda']:>7} {'MEAN':>5} | "
                f"{s['pct_pruned_mean']:>6.2f}±{s['pct_pruned_std']:>4.2f}% | "
                f"{s['orig_ppl']:>9.3f} | "
                f"{s['pruned_ppl_mean']:>7.3f}±{s['pruned_ppl_std']:>5.3f} | "
                f"{'':>9} | -"
            )
            rows.append(sep)

    summary_str = "\n".join(rows)
    with open(os.path.join(out_root, "summary.txt"), "w") as f:
        f.write(summary_str + "\n")
    print("\n" + summary_str)
    print(f"\nResults → {out_root}/")

    if args.stop_pod:
        stop_pod()


if __name__ == "__main__":
    main()
