"""
Llama-3-8B MLP pruner -- convergence-based stopping (block-mean, validated
on OPT-125M/GPT-2, see train_pruner_opt125m_converge.py) + train/test CE
gap diagnostic + plateau-triggered cosine LR decay (B9, diary/ideas.md,
NOT previously built or tested anywhere -- this is its first
implementation).

*** UNTESTED. *** Every other script in this project got at least a local
compile-check, a unit test of new logic against synthetic data, and often
a full local run before being trusted. None of that was possible here --
no local GPU has 8B-model-scale memory, and meta-llama/Meta-Llama-3-8B is
gate-licensed (needs an accepted-license HF_TOKEN, not just present but
actually granted access). What WAS verified: py_compile, and the LR-decay
state machine's logic in isolation against synthetic history (mirrors how
check_converged was verified before its first real run). Everything
touching the actual model (get_mlp_weights_llama's concat+upcast,
apply_gates_llama's hook, the dtype boundary) is reasoned through but
UNVERIFIED against a real forward pass. Run --sanity_check (identity-gate
no-op) before trusting any real training run, same convention as
train_pruner_opt125m.py -- this is not optional here, it's the only real
verification this code will get before commit.

ARCHITECTURE PORTING (see diary chat log for the full derivation):

  Llama's FFN is SwiGLU: down_proj(SiLU(gate_proj(x)) * up_proj(x)) --
  THREE weight matrices, not the two (fc1/fc2) every other script in this
  project was built around. Two separate decisions, not equally settled:

  1. WHERE to gate -- unambiguous. down_proj's INPUT is exactly the
     14336-dim per-neuron vector, the direct SwiGLU analogue of GPT-2's
     c_proj / OPT's fc2. Hook down_proj's forward_pre_hook, same pattern
     as every sibling script.

  2. WHAT the row-encoder sees per neuron -- a real design choice, not
     obvious. A SwiGLU neuron at index i is defined by TWO incoming
     vectors (gate_proj[i,:] and up_proj[i,:]), not one. Chosen here:
     CONCATENATE them (dim 2*4096=8192) -- the natural generalization of
     "the row-encoder sees this neuron's incoming weights" when there are
     two incoming pathways instead of one. (Alternatives considered:
     gate_proj alone -- simpler, matches the single-row convention
     exactly, throws away up_proj's information; or a 3-way concat
     including down_proj's outgoing column too -- most complete,
     untested whether the extra dimension helps. Concatenation of
     gate_proj+up_proj chosen as the defensible middle ground.)

  The Pruner class itself needed ZERO changes -- it's already
  parameterized by layer_shapes=(N_INTER, in_features), so passing
  in_features=8192 instead of 768 is the only difference. This is the
  concrete confirmation of the "base-model-agnostic architecture" claim
  in diary/final-paper-direction.md.

DTYPE / MEMORY (see diary chat log): frozen weights in bf16 (torch_dtype=
torch.bfloat16), NOT fp32 like the OPT-125M/GPT-2 scripts -- 8B params in
fp32 is 32GB just for weights, tight-to-risky on an 80GB H100 once
activation memory for two forward+backward-ish passes is added. This does
NOT change compute precision (torch.autocast(bf16) already wraps every
frozen-model forward pass in every sibling script regardless of storage
dtype -- switching storage to bf16 is purely a memory move). The ONLY
dtype-sensitive boundary is get_mlp_weights_llama's output: the row-
encoder's own nn.Linear layers stay fp32 (small model, no reason to
compromise its own Adam training precision), so the concatenated
gate_proj+up_proj SLICE (not the whole model) is explicitly .float()'d
before being handed to the pruner -- same root-cause fix as the original
OPT-125M dtype-mismatch bug, just scoped to a small slice instead of
forcing full-model fp32. apply_gates_llama's hook casts the gate to the
ACTIVATION's dtype (g.to(x.dtype)) rather than relying on type-promotion
rules -- informed directly by the MPS shape-mismatch bug found in
activation_pruning_opt125m.py, where an unverified assumption about a
hook's tensor shape/dtype silently corrupted a real run.

DATA: WikiText-2, same "join lines into one string, tokenize once"
convention as train_pruner_opt125m.py -- Llama's tokenizer also defaults
add_bos_token=True, so per-line tokenization would reproduce F21's exact
bug class (spurious BOS-token document-boundary scattering) on a new
model. Chosen as the default dataset (most validated in this project) --
flagging this as a choice, not a given; swap to pg19 if that's preferred,
same loader shape.

EVAL WINDOW: max_length=2048, stride=1024 -- NOT Llama-3's full native
8192-token context (the "use full native context" convention every prior
script followed). Deliberately smaller: attention cost scales
quadratically with context length, and 8192 was judged too large a
compute-cost jump to pick silently without confirmation. Flagging this
explicitly rather than defaulting to the established convention's letter
over its actual intent (affordable eval).

PREREQUISITE: meta-llama/Meta-Llama-3-8B is gate-licensed on HF Hub --
HF_TOKEN must be set to a token with ACCEPTED access to that repo's
license, not just any token.

CONVERGENCE CRITERION: identical block-mean check_converged to
train_pruner_opt125m_converge.py -- see that file's docstring for the
full derivation (checkpoint every check_every steps, compare BLOCK MEANS
over a trailing window, per-layer not aggregate, abs_tol floor, burn_in).
Not re-derived here, just reused verbatim.

LR DECAY (B9, diary/ideas.md -- first implementation): the block-mean fix
filters noise by averaging over it; it doesn't shrink the noise itself,
since lr stays flat for the whole run. First time check_converged returns
True (state "pre_decay" -> "decaying"), do NOT stop -- cosine-decay lr
from lr_0 to lr_min over the next lr_decay_window steps (default = window
* check_every = 250, so the reconfirmation check's trailing window lands
almost entirely INSIDE the decay period by construction -- deliberate
alignment, not a coincidence). At the end of that window, the SAME
step's regular check_converged call (this step is automatically a
multiple of check_every since lr_decay_window is) doubles as the
reconfirmation:
  - passes -> true convergence (state "decaying" -> stop). The original
    signal was real, just noisy; decay gave a clean confirmation.
  - fails -> state "decaying" -> "post_decay": hold lr at lr_min (do NOT
    re-decay from scratch -- avoids thrashing) and keep running the
    normal check loop, now at reduced noise, until it passes for real.
"""
import csv
import math
import os
import sys
import time
import contextlib
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")

OUT_ROOT = "/workspace/results/llama3_8b_converge_sweep"

N_LAYERS      = 32
N_INTER       = 14336   # SwiGLU intermediate size
HIDDEN        = 4096
ROW_INPUT_DIM = 2 * HIDDEN   # gate_proj row concat up_proj row
LAYER_SHAPE   = (N_INTER, ROW_INPUT_DIM)

LLAMA_REPO = "meta-llama/Meta-Llama-3-8B"


# ─────────────────────────────────────────────────────────────────────────────
# Pruner -- verbatim copy, zero changes needed (see module docstring).
# ─────────────────────────────────────────────────────────────────────────────

def binary_ste(logits: torch.Tensor) -> torch.Tensor:
    soft = torch.sigmoid(logits)
    hard = (soft > 0.5).float()
    return hard - soft.detach() + soft


class Pruner(nn.Module):
    def __init__(self, layer_shapes, embed_dim=64, lstm_hidden=128):
        super().__init__()
        self.row_encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(in_features, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1))
            for _, in_features in layer_shapes
        ])
        for enc in self.row_encoders:
            nn.init.constant_(enc[-1].bias, 2.0)

        self.layer_projectors = nn.ModuleList([
            nn.Linear(in_features, lstm_hidden) for _, in_features in layer_shapes
        ])
        self.lstm = nn.LSTM(input_size=lstm_hidden, hidden_size=lstm_hidden,
                            batch_first=True, bidirectional=True)
        self.context_norm = nn.LayerNorm(lstm_hidden * 2)
        self.context_head = nn.Linear(lstm_hidden * 2, 1)
        nn.init.zeros_(self.context_head.weight)
        nn.init.zeros_(self.context_head.bias)

    def _node_scores(self, weight_matrices):
        node_logits = [enc(W).squeeze(-1) for enc, W in zip(self.row_encoders, weight_matrices)]
        layer_embeds = [F.relu(proj(W.mean(dim=0))) for proj, W in zip(self.layer_projectors, weight_matrices)]
        seq = torch.stack(layer_embeds, dim=0).unsqueeze(0)
        lstm_out, _ = self.lstm(seq)
        context_biases = torch.tanh(self.context_head(self.context_norm(lstm_out.squeeze(0))).squeeze(-1))
        return [logits + ctx for logits, ctx in zip(node_logits, context_biases)]

    def forward(self, weight_matrices):
        return [binary_ste(s) for s in self._node_scores(weight_matrices)]

    @torch.no_grad()
    def scores(self, weight_matrices):
        return self._node_scores(weight_matrices)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading / SwiGLU dispatch
# ─────────────────────────────────────────────────────────────────────────────

def load_llama8b(device):
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(
        LLAMA_REPO, use_safetensors=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def get_mlp_weights(model):
    """Concatenate gate_proj + up_proj rows per neuron (dim 2*4096=8192),
    upcast ONLY this slice to fp32 -- see module docstring. NOT the whole
    8B-param model, just the ~14336x8192-per-layer piece the row-encoder
    actually consumes."""
    return [
        torch.cat([
            model.model.layers[i].mlp.gate_proj.weight,
            model.model.layers[i].mlp.up_proj.weight,
        ], dim=1).float().detach()
        for i in range(N_LAYERS)
    ]


@contextlib.contextmanager
def apply_gates(model, gates):
    """Hook down_proj's input (post-SiLU*up_proj, pre-down-projection) --
    the SwiGLU analogue of GPT-2's c_proj / OPT's fc2. Rank-agnostic view
    shape (matches the fix already needed once for activation_pruning_
    opt125m.py -- don't assume 3D) and explicit dtype cast on the gate
    (g.to(x.dtype), not relying on type-promotion -- see module docstring)."""
    hooks = []
    for block, gate in zip(model.model.layers, gates):
        def make_hook(g):
            def hook(module, args):
                x = args[0]
                view_shape = (1,) * (x.dim() - 1) + (-1,)
                return (x * g.to(x.dtype).view(*view_shape),)
            return hook
        hooks.append(block.mlp.down_proj.register_forward_pre_hook(make_hook(gate)))
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


def autocast_ctx(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


# ─────────────────────────────────────────────────────────────────────────────
# Data -- WikiText-2, join-then-tokenize-once (see module docstring; same
# fix as train_pruner_opt125m.py's get_loaders, Llama's tokenizer also
# defaults add_bos_token=True).
# ─────────────────────────────────────────────────────────────────────────────

def get_loaders(seq_len: int, batch_size: int, num_workers: int = 2):
    from datasets import load_dataset, Dataset
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    tokenizer = AutoTokenizer.from_pretrained(LLAMA_REPO)
    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    def tokenize_split(split):
        return tokenizer("\n\n".join(raw[split]["text"]))["input_ids"]

    test_ids  = torch.tensor(tokenize_split("test"),  dtype=torch.long)
    train_ids = torch.tensor(tokenize_split("train"), dtype=torch.long)

    total = (train_ids.size(0) // seq_len) * seq_len
    train_blocked = Dataset.from_dict({"input_ids": train_ids[:total].view(-1, seq_len).tolist()})
    train_blocked.set_format(type="torch", columns=["input_ids"])
    train_loader = DataLoader(train_blocked, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, train_ids, test_ids


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation -- identical sliding-window protocol to every sibling script.
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, test_ids, device, gates=None, desc="eval",
            max_length: int = 2048, stride: int = 1024) -> float:
    total_len = test_ids.size(0)
    total_nll = total_tokens = 0
    prev_end = 0
    positions = list(range(0, total_len, stride))
    for begin in tqdm(positions, desc=desc, unit="window", leave=False, dynamic_ncols=True):
        end = min(begin + max_length, total_len)
        trg_len = end - prev_end
        ids = test_ids[begin:end].unsqueeze(0).to(device)
        labels = ids.clone()
        labels[:, :-trg_len] = -100
        with autocast_ctx(device):
            if gates is None:
                loss = model(ids, labels=labels).loss
            else:
                with apply_gates(model, gates):
                    loss = model(ids, labels=labels).loss
        n_tok = (labels[:, 1:] != -100).sum().item()
        total_nll += loss.item() * n_tok
        total_tokens += n_tok
        prev_end = end
        if end == total_len:
            break
    return total_nll / total_tokens


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check -- identical to train_pruner_opt125m.py's, NOT optional here
# (see module docstring -- this is the only verification this code gets
# before it's trusted on real compute).
# ─────────────────────────────────────────────────────────────────────────────

def sanity_check(model, test_ids, device, args):
    print("\n" + "=" * 70)
    print("SANITY CHECK — identity-gate control (gates=None vs all-ones)")
    print("=" * 70, flush=True)
    ce_none = evaluate(model, test_ids, device, gates=None, desc="gates=None",
                       max_length=args.eval_max_length, stride=args.eval_stride)
    ones_gates = [torch.ones(N_INTER, device=device) for _ in range(N_LAYERS)]
    ce_ones = evaluate(model, test_ids, device, gates=ones_gates, desc="gates=all-ones",
                       max_length=args.eval_max_length, stride=args.eval_stride)
    diff = ce_ones - ce_none
    print(f"  CE (gates=None)      : {ce_none:.6f}")
    print(f"  CE (gates=all-ones)  : {ce_ones:.6f}")
    print(f"  diff                 : {diff:+.6f}")
    if abs(diff) < 1e-3:
        print("  PASS — all-ones gate is a numerical no-op. Hook/dtype path is clean.")
    else:
        print("  FAIL — bug in apply_gates or the dtype/shape handling. DO NOT TRUST any "
              "pruned-CE number from this code until this is root-caused.")
    print("=" * 70, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Convergence check -- verbatim from train_pruner_opt125m_converge.py (v2,
# block-mean). Model-agnostic, not re-derived.
# ─────────────────────────────────────────────────────────────────────────────

def _block_mean(layer_hist, cp, check_every):
    lo = max(0, cp - check_every)
    return sum(layer_hist[lo:cp]) / (cp - lo)


def check_converged(history, step, check_every, window, rel_tol, abs_tol, burn_in):
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
# LR decay -- B9, first implementation. Plateau-triggered cosine.
# ─────────────────────────────────────────────────────────────────────────────

def cosine_lr(s, lr_decay_window, lr_0, lr_min):
    s = min(s, lr_decay_window)
    return lr_min + 0.5 * (lr_0 - lr_min) * (1 + math.cos(math.pi * s / lr_decay_window))


def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g["lr"] = lr


# ─────────────────────────────────────────────────────────────────────────────
# Gap diagnostic -- same shape as train_pruner_opt125m_converge.py's.
# ─────────────────────────────────────────────────────────────────────────────

def sample_tokens(ids_flat, n_tokens):
    return ids_flat[:min(n_tokens, ids_flat.size(0))]


def gap_diagnostic_checkpoint(pruner, model, train_sample, test_sample, device, args):
    pruner.eval()
    with torch.no_grad():
        gates = pruner(get_mlp_weights(model))
    per_layer_keep = [g.mean().item() for g in gates]
    avg_gate = float(np.mean(per_layer_keep))

    kw = dict(max_length=args.eval_max_length, stride=args.eval_stride)
    train_orig_ce   = evaluate(model, train_sample, device, gates=None,   desc="gap: train orig",   **kw)
    train_pruned_ce = evaluate(model, train_sample, device, gates=gates, desc="gap: train pruned", **kw)
    test_orig_ce    = evaluate(model, test_sample,  device, gates=None,   desc="gap: test orig",    **kw)
    test_pruned_ce  = evaluate(model, test_sample,  device, gates=gates, desc="gap: test pruned",  **kw)
    pruner.train()

    train_delta = train_orig_ce - train_pruned_ce
    test_delta  = test_orig_ce - test_pruned_ce
    return {
        "avg_gate": avg_gate, "pct_pruned": (1 - avg_gate) * 100, "per_layer_keep": per_layer_keep,
        "train_orig_ce": train_orig_ce, "train_pruned_ce": train_pruned_ce, "train_delta": train_delta,
        "test_orig_ce": test_orig_ce, "test_pruned_ce": test_pruned_ce, "test_delta": test_delta,
        "gap": train_delta - test_delta,
    }


GAP_CSV_COLUMNS = [
    "lambda", "seed", "step", "lr", "lr_state",
    "avg_gate", "pct_pruned", "delta_pct_pruned", "max_layer_delta_pct",
    "would_be_converged",
    "train_orig_ce", "train_pruned_ce", "train_delta",
    "test_orig_ce", "test_pruned_ce", "test_delta", "gap",
]


# ─────────────────────────────────────────────────────────────────────────────
# Single pruner training step
# ─────────────────────────────────────────────────────────────────────────────

def pruner_step(pruner, model, optimizer, input_ids, sparsity_weight, device):
    optimizer.zero_grad()
    weights = get_mlp_weights(model)
    gates = pruner(weights)
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
    return {"loss": loss.item(), "ce_orig": ce_orig, "ce_pruned": ce_pruned.item(),
            "avg_gate": avg_gate, "per_layer_keep": per_layer_keep}


def _smooth(values, window=100):
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo:i + 1]) / (i - lo + 1))
    return out


def plot_one_run(history, save_path, title):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    steps = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle(title, fontsize=11, fontweight="bold")

    axes[0].plot(steps, history["loss"], alpha=0.15, color="steelblue")
    axes[0].plot(steps, _smooth(history["loss"]), color="steelblue", lw=2)
    axes[0].axhline(0, color="gray", ls="--", lw=0.8)
    axes[0].set_title("Pruner loss"); axes[0].set_xlabel("step"); axes[0].grid(alpha=0.3)

    axes[1].plot(steps, _smooth(history["ce_orig"]),   color="steelblue", lw=2, label="orig")
    axes[1].plot(steps, _smooth(history["ce_pruned"]), color="tomato",    lw=2, label="pruned")
    axes[1].set_title("CE loss (nats)"); axes[1].set_xlabel("step"); axes[1].grid(alpha=0.3); axes[1].legend()

    cmap = plt.cm.tab20(np.linspace(0, 1, min(N_LAYERS, 20)))
    for i in range(N_LAYERS):
        per = [(1 - k) * 100 for k in history["per_layer_keep"][i]]
        axes[2].plot(steps, _smooth(per), color=cmap[i % 20], lw=1.0)
    axes[2].set_title("per-layer % pruned (32 layers)"); axes[2].set_xlabel("step")
    axes[2].set_ylim(0, 100); axes[2].grid(alpha=0.3)

    axes[3].plot(steps, history["lr"], color="darkorange", lw=1.5)
    axes[3].set_title("learning rate"); axes[3].set_xlabel("step"); axes[3].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Per-(λ, seed) training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_converge(lam, seed, model, train_loader, train_ids, test_ids, args, device, run_dir):
    torch.manual_seed(seed); np.random.seed(seed)

    layer_shapes = [LAYER_SHAPE] * N_LAYERS
    pruner = Pruner(layer_shapes, embed_dim=args.embed_dim, lstm_hidden=args.lstm_hidden).to(device)
    opt = torch.optim.Adam(pruner.parameters(), lr=args.lr)

    tag = f"λ={lam} seed={seed}"
    print(f"\n── {tag} ── pruner params: {sum(p.numel() for p in pruner.parameters()):,} "
          f"(convergence-based, max_steps={args.max_steps}, LR-decay window={args.lr_decay_window})",
          flush=True)

    history = {"loss": [], "ce_orig": [], "ce_pruned": [], "avg_gate": [], "lr": [],
               "per_layer_keep": [[] for _ in range(N_LAYERS)]}

    gap_train_sample = sample_tokens(train_ids, args.gap_eval_tokens)
    gap_test_sample  = sample_tokens(test_ids, args.gap_eval_tokens)
    os.makedirs(run_dir, exist_ok=True)
    gap_csv_file = open(os.path.join(run_dir, "gap_diagnostic.csv"), "w", newline="")
    gap_writer = csv.DictWriter(gap_csv_file, fieldnames=GAP_CSV_COLUMNS)
    gap_writer.writeheader()
    prev_pct_pruned = None
    prev_per_layer_pct = None

    lr_state = "pre_decay"       # pre_decay -> decaying -> post_decay
    decay_start_step = None

    t0 = time.time()
    step = 0
    converged = False
    loader_iter = iter(train_loader)
    pbar = tqdm(total=args.max_steps, desc=tag, unit="step", dynamic_ncols=True)

    while step < args.max_steps:
        if lr_state == "decaying":
            s = step - decay_start_step
            cur_lr = cosine_lr(s, args.lr_decay_window, args.lr, args.lr_min)
            set_lr(opt, cur_lr)
        elif lr_state == "post_decay":
            cur_lr = args.lr_min
            set_lr(opt, cur_lr)
        else:
            cur_lr = args.lr

        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)
        ids = batch["input_ids"].to(device)
        m = pruner_step(pruner, model, opt, ids, lam, device)

        history["loss"].append(m["loss"]); history["ce_orig"].append(m["ce_orig"])
        history["ce_pruned"].append(m["ce_pruned"]); history["avg_gate"].append(m["avg_gate"])
        history["lr"].append(cur_lr)
        for i, k in enumerate(m["per_layer_keep"]):
            history["per_layer_keep"][i].append(k)

        step += 1
        avg_pruned = (1 - m["avg_gate"]) * 100
        pbar.set_postfix(loss=f"{m['loss']:+.3f}", pruned=f"{avg_pruned:.1f}%",
                         lr=f"{cur_lr:.2e}", state=lr_state, refresh=False)
        pbar.update(1)
        if step % args.log_every == 0:
            tqdm.write(f"  [{tag}] step {step:>6} | loss {m['loss']:+.3f} | "
                       f"pruned {avg_pruned:5.1f}% | lr {cur_lr:.2e} | state={lr_state}")

        would_converge = False
        if step % args.check_every == 0:
            would_converge = check_converged(history, step, args.check_every, args.window,
                                             args.rel_tol, args.abs_tol, args.burn_in)

        if lr_state == "pre_decay":
            if would_converge:
                lr_state = "decaying"
                decay_start_step = step
                tqdm.write(f"  [{tag}] raw convergence signal at step {step} — "
                          f"starting {args.lr_decay_window}-step cosine LR decay "
                          f"({args.lr:.2e} -> {args.lr_min:.2e}) before trusting it")
        elif lr_state == "decaying":
            if step - decay_start_step >= args.lr_decay_window:
                if would_converge:
                    converged = True
                    tqdm.write(f"  [{tag}] CONFIRMED converged at step {step} "
                              f"(post-decay reconfirmation passed)")
                else:
                    lr_state = "post_decay"
                    tqdm.write(f"  [{tag}] decay reconfirmation FAILED at step {step} — "
                              f"original signal was noise masking real movement. "
                              f"Holding lr={args.lr_min:.2e}, continuing.")
        elif lr_state == "post_decay":
            if would_converge:
                converged = True
                tqdm.write(f"  [{tag}] CONVERGED at step {step} (post-decay, lr={args.lr_min:.2e})")

        if step % args.gap_eval_every == 0:
            g = gap_diagnostic_checkpoint(pruner, model, gap_train_sample, gap_test_sample, device, args)
            delta_pct_pruned = (g["pct_pruned"] - prev_pct_pruned) if prev_pct_pruned is not None else 0.0
            cur_per_layer_pct = [(1 - k) * 100 for k in g["per_layer_keep"]]
            max_layer_delta = (max(abs(c - p) for c, p in zip(cur_per_layer_pct, prev_per_layer_pct))
                               if prev_per_layer_pct is not None else 0.0)
            gap_writer.writerow({
                "lambda": lam, "seed": seed, "step": step, "lr": cur_lr, "lr_state": lr_state,
                "avg_gate": g["avg_gate"], "pct_pruned": g["pct_pruned"],
                "delta_pct_pruned": delta_pct_pruned, "max_layer_delta_pct": max_layer_delta,
                "would_be_converged": would_converge,
                "train_orig_ce": g["train_orig_ce"], "train_pruned_ce": g["train_pruned_ce"],
                "train_delta": g["train_delta"], "test_orig_ce": g["test_orig_ce"],
                "test_pruned_ce": g["test_pruned_ce"], "test_delta": g["test_delta"], "gap": g["gap"],
            })
            gap_csv_file.flush()
            prev_pct_pruned = g["pct_pruned"]; prev_per_layer_pct = cur_per_layer_pct

        if converged:
            break

    pbar.close()
    total_time = time.time() - t0
    if not converged:
        print(f"  [{tag}] NOT CONVERGED — hit max_steps={args.max_steps} safety cap.", flush=True)
    gap_csv_file.close()

    pruner.eval()
    with torch.no_grad():
        final_gates = pruner(get_mlp_weights(model))
    per_layer_kept = [int(g.sum().item()) for g in final_gates]

    orig_ce   = evaluate(model, test_ids, device, gates=None,        desc=f"[{tag}] eval orig",
                        max_length=args.eval_max_length, stride=args.eval_stride)
    pruned_ce = evaluate(model, test_ids, device, gates=final_gates, desc=f"[{tag}] eval pruned",
                        max_length=args.eval_max_length, stride=args.eval_stride)
    orig_ppl, pruned_ppl = float(np.exp(orig_ce)), float(np.exp(pruned_ce))

    final_gate = history["avg_gate"][-1]
    pct_pruned = (1 - final_gate) * 100
    print(f"  → [{tag}] {'converged' if converged else 'CAPPED'} at step {step} ({total_time:.0f}s) | "
          f"final keep {final_gate:.3f} pruned {pct_pruned:.2f}% | "
          f"orig ppl {orig_ppl:.3f} → pruned ppl {pruned_ppl:.3f}", flush=True)

    plot_one_run(history, os.path.join(run_dir, "plot.png"),
                title=(f"Llama-3-8B MLP — λ={lam} seed={seed} — "
                      f"{'converged' if converged else 'CAPPED'} @ step {step} — "
                      f"{pct_pruned:.1f}% pruned, ppl {pruned_ppl:.2f}"))

    lines = [
        f"Llama-3-8B MLP pruner — λ={lam}, seed={seed} — CONVERGENCE-BASED + LR-DECAY",
        f"layers : {N_LAYERS} MLP blocks, {N_INTER} intermediate neurons each",
        f"steps taken       : {step}",
        f"converged         : {converged} (max_steps cap = {args.max_steps})",
        f"convergence check : window={args.window} x check_every={args.check_every} "
        f"(block-mean) | rel_tol={args.rel_tol} abs_tol={args.abs_tol} | burn_in={args.burn_in}",
        f"LR decay          : window={args.lr_decay_window} | {args.lr:.2e} -> {args.lr_min:.2e} | "
        f"final state={lr_state}",
        f"time              : {total_time:.1f}s",
        "-" * 60,
        f"final avg keep gate          : {final_gate:.4f}",
        f"final % FFN neurons pruned   : {pct_pruned:.2f}%",
        "-" * 60,
        f"WikiText-2 test set:",
        f"  original  ppl              : {orig_ppl:.3f}",
        f"  pruned    ppl              : {pruned_ppl:.3f}",
        f"  ppl increase               : {pruned_ppl - orig_ppl:+.3f}",
    ]
    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    torch.save({
        "pruner_state_dict": pruner.state_dict(), "lambda": lam, "seed": seed,
        "embed_dim": args.embed_dim, "lstm_hidden": args.lstm_hidden,
        "per_layer_kept": per_layer_kept, "orig_ppl": orig_ppl, "pruned_ppl": pruned_ppl,
        "steps_taken": step, "converged": converged, "lr_state": lr_state,
    }, os.path.join(run_dir, "pruner.pt"))
    print(f"  [saved] {run_dir}/", flush=True)

    return {"lambda": lam, "seed": seed, "per_layer_kept": per_layer_kept, "pct_pruned": pct_pruned,
            "orig_ppl": orig_ppl, "pruned_ppl": pruned_ppl, "total_time": total_time,
            "steps_taken": step, "converged": converged}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.8, 1.6],
                    help="Inherited from the OPT-125M convergence sweep -- UNVALIDATED "
                         "at this scale, no established lambda-vs-model-size law (H2/H3 "
                         "are still open). Starting point, not a claim it's right here.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--check_every", type=int, default=50)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--rel_tol", type=float, default=0.05)
    ap.add_argument("--abs_tol", type=float, default=0.01)
    ap.add_argument("--burn_in", type=int, default=500)
    ap.add_argument("--max_steps", type=int, default=18000,
                    help="Safety cap, inherited default -- UNVALIDATED at this scale.")
    ap.add_argument("--lr_decay_window", type=int, default=250,
                    help="B9 default: window*check_every, so the reconfirmation's "
                         "trailing window lands almost entirely inside the decay period.")
    ap.add_argument("--lr_min", type=float, default=None,
                    help="Default (None) = lr/10.")
    ap.add_argument("--gap_eval_every", type=int, default=200)
    ap.add_argument("--gap_eval_tokens", type=int, default=50_000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--embed_dim", type=int, default=64)
    ap.add_argument("--lstm_hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--log_every", type=int, default=250)
    ap.add_argument("--eval_max_length", type=int, default=2048,
                    help="NOT Llama-3's full native 8192 context -- deliberately "
                         "smaller, flagged as a choice, see module docstring.")
    ap.add_argument("--eval_stride", type=int, default=1024)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_dir", type=str, default=OUT_ROOT)
    ap.add_argument("--sanity_check", action="store_true",
                    help="Run the identity-gate no-op check and exit. NOT optional "
                         "before a real run here -- see module docstring.")
    ap.add_argument("--timing_probe", action="store_true",
                    help="Run 50 steps, print per-step time, then exit.")
    args = ap.parse_args()
    if args.lr_min is None:
        args.lr_min = args.lr / 10

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device} | λs={args.lambdas} | seeds={args.seeds} | "
          f"max_steps={args.max_steps} | LR decay window={args.lr_decay_window} "
          f"({args.lr:.2e} -> {args.lr_min:.2e})")

    print("Loading Llama-3-8B (requires HF_TOKEN with accepted license access) ...", flush=True)
    model = load_llama8b(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Llama-3-8B loaded — {n_params:,} params, frozen.", flush=True)

    print("Loading WikiText-2 ...", flush=True)
    train_loader, train_ids, test_ids = get_loaders(args.seq_len, args.batch_size)
    print(f"Data: train_tokens={train_ids.size(0):,} test_tokens={test_ids.size(0):,}", flush=True)

    if args.sanity_check:
        sanity_check(model, test_ids, device, args)
        return

    if args.timing_probe:
        print("\n── TIMING PROBE (λ=0.05, seed=0) ──", flush=True)
        run_dir = os.path.join(args.out_dir, "timing_probe")
        layer_shapes = [LAYER_SHAPE] * N_LAYERS
        pruner = Pruner(layer_shapes, embed_dim=args.embed_dim, lstm_hidden=args.lstm_hidden).to(device)
        opt = torch.optim.Adam(pruner.parameters(), lr=args.lr)
        loader_iter = iter(train_loader)
        t0 = time.time()
        for i in range(50):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader); batch = next(loader_iter)
            pruner_step(pruner, model, opt, batch["input_ids"].to(device), 0.05, device)
        elapsed = time.time() - t0
        print(f"TIMING PROBE: {elapsed/50*1000:.0f}ms/step -> "
              f"{elapsed/50:.3f}s/step, {50/elapsed:.2f} steps/s", flush=True)
        return

    os.makedirs(args.out_dir, exist_ok=True)
    all_results = []
    total_runs = len(args.lambdas) * len(args.seeds)
    run_num = 0
    for lam in args.lambdas:
        for seed in args.seeds:
            run_num += 1
            tqdm.write(f"\n{'='*70}\nRun {run_num}/{total_runs}\n{'='*70}")
            run_dir = (os.path.join(args.out_dir, f"lambda_{lam}", f"seed_{seed}")
                       if len(args.seeds) > 1 else os.path.join(args.out_dir, f"lambda_{lam}"))
            res = train_one_converge(lam, seed, model, train_loader, train_ids, test_ids,
                                     args, device, run_dir)
            all_results.append(res)

    combined_path = os.path.join(args.out_dir, "gap_diagnostic_all.csv")
    with open(combined_path, "w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=GAP_CSV_COLUMNS)
        writer.writeheader()
        for lam in args.lambdas:
            for seed in args.seeds:
                run_dir = (os.path.join(args.out_dir, f"lambda_{lam}", f"seed_{seed}")
                           if len(args.seeds) > 1 else os.path.join(args.out_dir, f"lambda_{lam}"))
                run_csv = os.path.join(run_dir, "gap_diagnostic.csv")
                if not os.path.exists(run_csv):
                    continue
                with open(run_csv, newline="") as in_f:
                    for row in csv.DictReader(in_f):
                        writer.writerow(row)
    print(f"Combined gap diagnostic -> {combined_path}")

    sep = "-" * 100
    rows = [f"Llama-3-8B convergence+LR-decay sweep | seeds={args.seeds} | "
           f"max_steps={args.max_steps} | device={device}", sep,
           f"{'lambda':>7} {'seed':>5} | {'steps':>7} {'conv?':>6} | {'% pruned':>9} | "
           f"{'orig ppl':>9} | {'pruned ppl':>10} | {'ppl rise':>9}", sep]
    for r in all_results:
        rows.append(f"{r['lambda']:>7} {r['seed']:>5} | {r['steps_taken']:>7} "
                    f"{'YES' if r['converged'] else 'NO':>6} | {r['pct_pruned']:>8.2f}% | "
                    f"{r['orig_ppl']:>9.3f} | {r['pruned_ppl']:>10.3f} | "
                    f"{r['pruned_ppl']-r['orig_ppl']:>+9.3f}")
    summary_str = "\n".join(rows)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(summary_str + "\n")
    print("\n" + summary_str)
    print(f"\nResults → {args.out_dir}/")


if __name__ == "__main__":
    main()
