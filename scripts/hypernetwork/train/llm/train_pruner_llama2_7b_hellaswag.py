"""
Llama-2-7B MLP pruner -- HellaSwag, ACCURACY-DELTA objective (not a CE/
perplexity proxy). Same Pruner / SwiGLU gate hook / block-mean convergence /
B9 LR decay as train_pruner_llama2_7b_wikitext2.py, but the training signal
itself is different -- see "OBJECTIVE" below. This supersedes an earlier
draft of this file that trained on HellaSwag gold-ending text as an ordinary
CLM corpus (CE-loss proxy); that measured language-modeling quality on
HellaSwag-flavored text, not HellaSwag's own accuracy metric, per user
correction.

*** UNTESTED. *** No local hardware at this scale. Verified: py_compile only.

PREREQUISITE -- meta-llama/Llama-2-7b-hf is GATE-LICENSED. HF_TOKEN must be
set to a token with ACCEPTED access. See train_pruner_llama2_7b_wikitext2.py.

DATASET -- Rowan/hellaswag (parquet mirror, no trust_remote_code). TRAIN
split for pruner training, VALIDATION split (HellaSwag's own labeled
held-out set) for eval -- HellaSwag's public TEST split ships with labels
stripped (label == "" for every row) and is unusable for either role.

OBJECTIVE, mathematically -- for context c and 4 endings e_1..e_4 with gold
index g, model M's length-normalized log-likelihood score for ending k
(this IS HellaSwag's own "acc_norm" metric, exactly as lm-eval-harness / the
sibling eval_downstream_llama2_7b.py script compute it):

    ŝ_k(M) = ( Σ_{t in e_k} log P_M(token_t | c, e_k,<t) ) / len_chars(e_k)

prediction = argmax_k ŝ_k(M); accuracy = 1[argmax == g]. That argmax is a
non-differentiable step function of the gates -- can't drive STE gradients
directly. Surrogate: treat ŝ = (ŝ_1..ŝ_4) as 4-way classification logits and
use categorical cross-entropy against gold g (the standard differentiable
proxy for classification accuracy, same relationship any classifier's CE
loss has to its own accuracy metric):

    L_acc(M) = -log( exp(ŝ_g) / Σ_k exp(ŝ_k) )   =  F.cross_entropy(ŝ, g)

Training loss, mirroring the exact (pruned - orig) structure of every
sibling script (dense term under no_grad, exactly like ce_orig there):

    L_acc_dense  = L_acc(dense model)     # no_grad, frozen weights
    L_acc_pruned = L_acc(gated model)     # grad flows through STE gates
    loss = (L_acc_pruned - L_acc_dense) + λ * mean(gate)

Consequence: perplexity is no longer a meaningful quantity for this run, so
final-checkpoint reporting and the gap diagnostic report ACCURACY (%, true
argmax metric, not the surrogate) on train-sample / validation instead of
ppl -- there is no coherent way to keep reporting ppl once the training
objective no longer targets it.

len_chars() normalization, the 4-way-softmax-CE surrogate form, and the
per-step batch unit (examples, not token blocks) were explicit user
decisions, not defaults picked silently -- see the conversation this file
was built from.

BATCHING -- each HellaSwag example now costs 4 endings x 2 (dense + gated)
forward passes, vs. 1 CLM sequence in the WikiText-2 sibling. --batch_size
defaults to 4 examples/step (16 sequences for the gated forward, 16 for the
dense forward) -- a deliberately smaller default than the WikiText-2
sibling's batch_size=8 single-sequence setup, given the ~2x-4x higher
per-step forward-pass count this implies.

Tokenizer continuation-offset method (ctx_ids vs. full_ids diff to locate
continuation tokens) is the same imperfect-but-standard approach
lm-eval-harness itself uses -- tokenizer merges can shift a token or two at
the ctx/ending boundary; not something this script invents or can trivially
fix, flagging so it isn't mistaken for a bug.

max_steps default is 8000 -- explicit user instruction for this script.
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

OUT_ROOT = "/workspace/results/llama2_7b_hellaswag_sweep"

N_LAYERS      = 32
N_INTER       = 11008   # SwiGLU intermediate size -- Llama-2-7B specific, NOT 14336
HIDDEN        = 4096
ROW_INPUT_DIM = 2 * HIDDEN   # gate_proj row concat up_proj row
LAYER_SHAPE   = (N_INTER, ROW_INPUT_DIM)
NUM_CHOICES   = 4       # HellaSwag: always 4 candidate endings per example

LLAMA2_REPO = "meta-llama/Llama-2-7b-hf"
HELLASWAG_REPO = "Rowan/hellaswag"   # parquet mirror, no trust_remote_code needed


# ─────────────────────────────────────────────────────────────────────────────
# Pruner -- verbatim, see train_pruner_llama2_7b_wikitext2.py's docstring.
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
# Model loading / SwiGLU dispatch -- verbatim.
# ─────────────────────────────────────────────────────────────────────────────

def load_llama2_7b(device):
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(
        LLAMA2_REPO, use_safetensors=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def get_mlp_weights(model):
    """Concatenate gate_proj + up_proj rows per neuron (dim 8192), upcast
    ONLY this slice to fp32 -- see train_pruner_llama3_8b.py's docstring."""
    return [
        torch.cat([
            model.model.layers[i].mlp.gate_proj.weight,
            model.model.layers[i].mlp.up_proj.weight,
        ], dim=1).float().detach()
        for i in range(N_LAYERS)
    ]


@contextlib.contextmanager
def apply_gates(model, gates):
    """Hook down_proj's input -- see train_pruner_llama3_8b.py's docstring."""
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
# Data -- HellaSwag raw (ctx, endings, label) rows, no CLM chunking.
# ─────────────────────────────────────────────────────────────────────────────

def get_loaders(batch_size: int, num_workers: int = 0):
    """
    Returns (tokenizer, train_loader, train_examples, val_examples).
    train_examples / val_examples are plain lists of
    {"ctx": str, "endings": [str]*4, "label": int} rows -- HellaSwag's
    `train` and `validation` splits respectively (both carry real labels;
    `test` does not and is not used anywhere in this script).
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    tokenizer = AutoTokenizer.from_pretrained(LLAMA2_REPO)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset(HELLASWAG_REPO)

    def to_examples(split):
        return [
            {"ctx": ex["ctx"], "endings": ex["endings"], "label": int(ex["label"])}
            for ex in raw[split]
            if ex["label"] != "" and ex["label"] is not None
        ]

    train_examples = to_examples("train")
    val_examples = to_examples("validation")

    train_loader = DataLoader(train_examples, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=lambda b: b)
    return tokenizer, train_loader, train_examples, val_examples


def sample_examples(examples, n):
    return examples[:min(n, len(examples))]


# ─────────────────────────────────────────────────────────────────────────────
# Scoring -- length-normalized (char length) per-ending log-likelihood.
# ─────────────────────────────────────────────────────────────────────────────

def build_sequences(examples, tokenizer):
    """For each example, tokenize ctx alone (to find the continuation-token
    offset) and ctx+" "+ending for each of the 4 endings. Returns a flat
    list of (full_ids, ctx_len, char_len_of_ending) -- length
    len(examples)*NUM_CHOICES, endings ordered within each example -- plus
    the list of gold labels (one per example)."""
    sequences = []
    labels = []
    for ex in examples:
        ctx_ids = tokenizer(ex["ctx"])["input_ids"]
        for ending in ex["endings"]:
            full_ids = tokenizer(ex["ctx"] + " " + ending)["input_ids"]
            sequences.append((full_ids, len(ctx_ids), len(ending)))
        labels.append(ex["label"])
    return sequences, labels


def score_sequences(model, sequences, device, pad_id):
    """Batched forward pass (right-padded) -> length-normalized summed
    log-likelihood of the continuation span of each sequence. Differentiable
    w.r.t. the model's current parameters/gates (no no_grad here -- caller
    decides ambient grad mode)."""
    max_len = max(len(ids) for ids, _, _ in sequences)
    B = len(sequences)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((B, max_len), dtype=torch.long)
    for i, (ids, _, _) in enumerate(sequences):
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attn[i, :len(ids)] = 1
    input_ids = input_ids.to(device)
    attn = attn.to(device)

    with autocast_ctx(device):
        logits = model(input_ids=input_ids, attention_mask=attn).logits
    logprobs = F.log_softmax(logits.float(), dim=-1)
    target = input_ids[:, 1:]
    gathered = logprobs[:, :-1, :].gather(-1, target.unsqueeze(-1)).squeeze(-1)   # (B, max_len-1)

    scores = []
    for i, (ids, ctx_len, char_len) in enumerate(sequences):
        L = len(ids)
        cont_logprob_sum = gathered[i, ctx_len - 1:L - 1].sum()
        scores.append(cont_logprob_sum / char_len)
    return torch.stack(scores)


def score_choices(model, examples, tokenizer, device):
    """Returns (scores: (len(examples), NUM_CHOICES), labels: (len(examples),))."""
    sequences, labels = build_sequences(examples, tokenizer)
    flat = score_sequences(model, sequences, device, tokenizer.pad_token_id)
    scores = flat.view(len(examples), NUM_CHOICES)
    label_t = torch.tensor(labels, device=device, dtype=torch.long)
    return scores, label_t


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation -- true accuracy (argmax metric) + the surrogate CE, both dense
# and gated. Used for the gap diagnostic and final-checkpoint reporting.
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_accuracy(model, examples, device, tokenizer, gates=None, desc="eval", batch_size=8):
    ctx = apply_gates(model, gates) if gates is not None else contextlib.nullcontext()
    correct = total = 0
    total_l_acc = 0.0
    with ctx:
        for i in tqdm(range(0, len(examples), batch_size), desc=desc, unit="batch",
                      leave=False, dynamic_ncols=True):
            batch = examples[i:i + batch_size]
            s, label_t = score_choices(model, batch, tokenizer, device)
            l_acc = F.cross_entropy(s, label_t)
            total_l_acc += l_acc.item() * len(batch)
            correct += (s.argmax(-1) == label_t).sum().item()
            total += len(batch)
    return {"l_acc": total_l_acc / total, "accuracy": 100.0 * correct / total}


def sanity_check(model, examples, device, tokenizer, args):
    print("\n" + "=" * 70)
    print("SANITY CHECK — identity-gate control (gates=None vs all-ones)")
    print("=" * 70, flush=True)
    res_none = evaluate_accuracy(model, examples, device, tokenizer, gates=None,
                                 desc="gates=None", batch_size=args.eval_batch_size)
    ones_gates = [torch.ones(N_INTER, device=device) for _ in range(N_LAYERS)]
    res_ones = evaluate_accuracy(model, examples, device, tokenizer, gates=ones_gates,
                                 desc="gates=all-ones", batch_size=args.eval_batch_size)
    diff = res_ones["l_acc"] - res_none["l_acc"]
    print(f"  L_acc (gates=None)      : {res_none['l_acc']:.6f}  | accuracy {res_none['accuracy']:.2f}%")
    print(f"  L_acc (gates=all-ones)  : {res_ones['l_acc']:.6f}  | accuracy {res_ones['accuracy']:.2f}%")
    print(f"  diff                    : {diff:+.6f}")
    if abs(diff) < 1e-3:
        print("  PASS — all-ones gate is a numerical no-op. Hook/dtype path is clean.")
    else:
        print("  FAIL — bug in apply_gates or the dtype/shape handling. DO NOT TRUST any "
              "pruned-accuracy number from this code until this is root-caused.")
    print("=" * 70, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Convergence check -- verbatim, see train_pruner_opt125m_converge.py.
# Operates on per_layer_keep gate statistics, unaffected by the loss change.
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
# LR decay -- B9, verbatim from train_pruner_llama3_8b.py.
# ─────────────────────────────────────────────────────────────────────────────

def cosine_lr(s, lr_decay_window, lr_0, lr_min):
    s = min(s, lr_decay_window)
    return lr_min + 0.5 * (lr_0 - lr_min) * (1 + math.cos(math.pi * s / lr_decay_window))


def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g["lr"] = lr


# ─────────────────────────────────────────────────────────────────────────────
# Gap diagnostic -- accuracy-based (train vs held-out), same shape as every
# converge script's CE-based version, just with accuracy% in place of CE.
# ─────────────────────────────────────────────────────────────────────────────

def gap_diagnostic_checkpoint(pruner, model, tokenizer, train_sample, test_sample, device, args):
    pruner.eval()
    with torch.no_grad():
        gates = pruner(get_mlp_weights(model))
    per_layer_keep = [g.mean().item() for g in gates]
    avg_gate = float(np.mean(per_layer_keep))

    kw = dict(batch_size=args.eval_batch_size)
    train_orig   = evaluate_accuracy(model, train_sample, device, tokenizer, gates=None,  desc="gap: train orig",   **kw)
    train_pruned = evaluate_accuracy(model, train_sample, device, tokenizer, gates=gates, desc="gap: train pruned", **kw)
    test_orig    = evaluate_accuracy(model, test_sample,  device, tokenizer, gates=None,  desc="gap: test orig",    **kw)
    test_pruned  = evaluate_accuracy(model, test_sample,  device, tokenizer, gates=gates, desc="gap: test pruned",  **kw)
    pruner.train()

    # delta = pruned - orig: NEGATIVE means pruning hurt accuracy (the
    # expected/typical case) -- opposite sign convention from the CE-based
    # sibling scripts' (orig - pruned), chosen because "negative = worse" is
    # more directly readable for an accuracy metric.
    train_delta = train_pruned["accuracy"] - train_orig["accuracy"]
    test_delta  = test_pruned["accuracy"] - test_orig["accuracy"]
    return {
        "avg_gate": avg_gate, "pct_pruned": (1 - avg_gate) * 100, "per_layer_keep": per_layer_keep,
        "train_orig_acc": train_orig["accuracy"], "train_pruned_acc": train_pruned["accuracy"], "train_delta": train_delta,
        "test_orig_acc": test_orig["accuracy"], "test_pruned_acc": test_pruned["accuracy"], "test_delta": test_delta,
        "gap": train_delta - test_delta,
    }


GAP_CSV_COLUMNS = [
    "lambda", "seed", "step", "lr", "lr_state",
    "avg_gate", "pct_pruned", "delta_pct_pruned", "max_layer_delta_pct",
    "would_be_converged",
    "train_orig_acc", "train_pruned_acc", "train_delta",
    "test_orig_acc", "test_pruned_acc", "test_delta", "gap",
]


# ─────────────────────────────────────────────────────────────────────────────
# Single pruner training step -- accuracy-delta objective (see module
# docstring for the L_acc derivation).
# ─────────────────────────────────────────────────────────────────────────────

def pruner_step(pruner, model, tokenizer, optimizer, batch_examples, sparsity_weight, device):
    optimizer.zero_grad()
    weights = get_mlp_weights(model)
    gates = pruner(weights)

    with torch.no_grad():
        s_dense, label_t = score_choices(model, batch_examples, tokenizer, device)
        l_acc_dense = F.cross_entropy(s_dense, label_t).item()

    with apply_gates(model, gates):
        s_pruned, _ = score_choices(model, batch_examples, tokenizer, device)
    l_acc_pruned = F.cross_entropy(s_pruned, label_t)

    sparsity_loss = sum(g.mean() for g in gates) / len(gates)
    loss = (l_acc_pruned - l_acc_dense) + sparsity_weight * sparsity_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(pruner.parameters(), max_norm=1.0)
    optimizer.step()

    per_layer_keep = [g.mean().item() for g in gates]
    avg_gate = sum(per_layer_keep) / len(per_layer_keep)
    with torch.no_grad():
        train_acc_pruned = (s_pruned.argmax(-1) == label_t).float().mean().item() * 100
        train_acc_orig = (s_dense.argmax(-1) == label_t).float().mean().item() * 100
    return {"loss": loss.item(), "l_acc_orig": l_acc_dense, "l_acc_pruned": l_acc_pruned.item(),
            "train_acc_orig": train_acc_orig, "train_acc_pruned": train_acc_pruned,
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

    axes[1].plot(steps, _smooth(history["l_acc_orig"]),   color="steelblue", lw=2, label="orig")
    axes[1].plot(steps, _smooth(history["l_acc_pruned"]), color="tomato",    lw=2, label="pruned")
    axes[1].set_title("Accuracy-CE surrogate L_acc (nats)"); axes[1].set_xlabel("step")
    axes[1].grid(alpha=0.3); axes[1].legend()

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
# Per-(λ, seed) training loop -- same convergence/LR-decay control flow as
# every sibling script; only the per-step metrics and final reporting
# (accuracy in place of CE/ppl) differ.
# ─────────────────────────────────────────────────────────────────────────────

def train_one_converge(lam, seed, model, tokenizer, train_loader, train_examples, val_examples,
                       args, device, run_dir):
    torch.manual_seed(seed); np.random.seed(seed)

    layer_shapes = [LAYER_SHAPE] * N_LAYERS
    pruner = Pruner(layer_shapes, embed_dim=args.embed_dim, lstm_hidden=args.lstm_hidden).to(device)
    opt = torch.optim.Adam(pruner.parameters(), lr=args.lr)

    tag = f"λ={lam} seed={seed}"
    print(f"\n── {tag} ── pruner params: {sum(p.numel() for p in pruner.parameters()):,} "
          f"(convergence-based, max_steps={args.max_steps}, LR-decay window={args.lr_decay_window})",
          flush=True)

    history = {"loss": [], "l_acc_orig": [], "l_acc_pruned": [], "avg_gate": [], "lr": [],
               "per_layer_keep": [[] for _ in range(N_LAYERS)]}

    gap_train_sample = sample_examples(train_examples, args.gap_eval_examples)
    gap_test_sample  = sample_examples(val_examples, args.gap_eval_examples)
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
        m = pruner_step(pruner, model, tokenizer, opt, batch, lam, device)

        history["loss"].append(m["loss"]); history["l_acc_orig"].append(m["l_acc_orig"])
        history["l_acc_pruned"].append(m["l_acc_pruned"]); history["avg_gate"].append(m["avg_gate"])
        history["lr"].append(cur_lr)
        for i, k in enumerate(m["per_layer_keep"]):
            history["per_layer_keep"][i].append(k)

        step += 1
        avg_pruned = (1 - m["avg_gate"]) * 100
        pbar.set_postfix(loss=f"{m['loss']:+.3f}", pruned=f"{avg_pruned:.1f}%",
                         acc=f"{m['train_acc_pruned']:.0f}%", lr=f"{cur_lr:.2e}",
                         state=lr_state, refresh=False)
        pbar.update(1)
        if step % args.log_every == 0:
            tqdm.write(f"  [{tag}] step {step:>6} | loss {m['loss']:+.3f} | "
                       f"pruned {avg_pruned:5.1f}% | train_acc(orig/pruned) "
                       f"{m['train_acc_orig']:.0f}%/{m['train_acc_pruned']:.0f}% | "
                       f"lr {cur_lr:.2e} | state={lr_state}")

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
            g = gap_diagnostic_checkpoint(pruner, model, tokenizer, gap_train_sample, gap_test_sample, device, args)
            delta_pct_pruned = (g["pct_pruned"] - prev_pct_pruned) if prev_pct_pruned is not None else 0.0
            cur_per_layer_pct = [(1 - k) * 100 for k in g["per_layer_keep"]]
            max_layer_delta = (max(abs(c - p) for c, p in zip(cur_per_layer_pct, prev_per_layer_pct))
                               if prev_per_layer_pct is not None else 0.0)
            gap_writer.writerow({
                "lambda": lam, "seed": seed, "step": step, "lr": cur_lr, "lr_state": lr_state,
                "avg_gate": g["avg_gate"], "pct_pruned": g["pct_pruned"],
                "delta_pct_pruned": delta_pct_pruned, "max_layer_delta_pct": max_layer_delta,
                "would_be_converged": would_converge,
                "train_orig_acc": g["train_orig_acc"], "train_pruned_acc": g["train_pruned_acc"],
                "train_delta": g["train_delta"], "test_orig_acc": g["test_orig_acc"],
                "test_pruned_acc": g["test_pruned_acc"], "test_delta": g["test_delta"], "gap": g["gap"],
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

    final_eval_set = (val_examples if args.final_eval_limit is None
                      else sample_examples(val_examples, args.final_eval_limit))
    orig_res   = evaluate_accuracy(model, final_eval_set, device, tokenizer, gates=None,
                                   desc=f"[{tag}] eval orig", batch_size=args.eval_batch_size)
    pruned_res = evaluate_accuracy(model, final_eval_set, device, tokenizer, gates=final_gates,
                                   desc=f"[{tag}] eval pruned", batch_size=args.eval_batch_size)
    orig_acc, pruned_acc = orig_res["accuracy"], pruned_res["accuracy"]

    final_gate = history["avg_gate"][-1]
    pct_pruned = (1 - final_gate) * 100
    print(f"  → [{tag}] {'converged' if converged else 'CAPPED'} at step {step} ({total_time:.0f}s) | "
          f"final keep {final_gate:.3f} pruned {pct_pruned:.2f}% | "
          f"orig acc {orig_acc:.2f}% → pruned acc {pruned_acc:.2f}%", flush=True)

    plot_one_run(history, os.path.join(run_dir, "plot.png"),
                title=(f"Llama-2-7B MLP — HellaSwag (accuracy-delta) — λ={lam} seed={seed} — "
                      f"{'converged' if converged else 'CAPPED'} @ step {step} — "
                      f"{pct_pruned:.1f}% pruned, acc {pruned_acc:.2f}%"))

    lines = [
        f"Llama-2-7B MLP pruner — HellaSwag (accuracy-delta objective, char-length-normalized "
        f"acc_norm scoring) — λ={lam}, seed={seed} — CONVERGENCE-BASED + LR-DECAY",
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
        f"hellaswag validation set ({len(final_eval_set)} examples, true argmax accuracy, "
        f"not the surrogate):",
        f"  original  accuracy         : {orig_acc:.3f}%",
        f"  pruned    accuracy         : {pruned_acc:.3f}%",
        f"  accuracy drop              : {orig_acc - pruned_acc:+.3f}pp",
    ]
    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    torch.save({
        "pruner_state_dict": pruner.state_dict(), "lambda": lam, "seed": seed,
        "embed_dim": args.embed_dim, "lstm_hidden": args.lstm_hidden,
        "per_layer_kept": per_layer_kept, "orig_acc": orig_acc, "pruned_acc": pruned_acc,
        "steps_taken": step, "converged": converged, "lr_state": lr_state,
    }, os.path.join(run_dir, "pruner.pt"))
    print(f"  [saved] {run_dir}/", flush=True)

    return {"lambda": lam, "seed": seed, "per_layer_kept": per_layer_kept, "pct_pruned": pct_pruned,
            "orig_acc": orig_acc, "pruned_acc": pruned_acc, "total_time": total_time,
            "steps_taken": step, "converged": converged}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.8, 1.6],
                    help="Inherited from the OPT-125M convergence sweep -- UNVALIDATED "
                         "at this scale/dataset/objective, no established lambda law.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--check_every", type=int, default=50)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--rel_tol", type=float, default=0.05)
    ap.add_argument("--abs_tol", type=float, default=0.01)
    ap.add_argument("--burn_in", type=int, default=500)
    ap.add_argument("--max_steps", type=int, default=8000,
                    help="Safety cap. 8000 per explicit instruction for this script.")
    ap.add_argument("--lr_decay_window", type=int, default=250,
                    help="B9 default: window*check_every.")
    ap.add_argument("--lr_min", type=float, default=None, help="Default (None) = lr/10.")
    ap.add_argument("--gap_eval_every", type=int, default=200)
    ap.add_argument("--gap_eval_examples", type=int, default=200,
                    help="Number of HellaSwag examples sampled per side (train/held-out) "
                         "for the gap diagnostic -- unit is examples, not tokens, unlike "
                         "the CE-based sibling scripts.")
    ap.add_argument("--batch_size", type=int, default=4,
                    help="Examples per training step (each expands to NUM_CHOICES=4 "
                         "sequences x 2 forward passes [dense+gated] -- smaller default "
                         "than the CLM sibling scripts' batch_size=8 given this cost.")
    ap.add_argument("--eval_batch_size", type=int, default=8,
                    help="Examples per no_grad eval batch (gap diagnostic + final eval) -- "
                         "can be larger than --batch_size since no gradients are held.")
    ap.add_argument("--final_eval_limit", type=int, default=None,
                    help="Cap on validation examples used for the final per-run accuracy "
                         "report. Default (None) = full validation split (~10k examples), "
                         "matching the 'test set used whole, untruncated' convention from "
                         "the WikiText-2 sibling. Set lower for a quick smoke test.")
    ap.add_argument("--embed_dim", type=int, default=64)
    ap.add_argument("--lstm_hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--log_every", type=int, default=250)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_dir", type=str, default=OUT_ROOT)
    ap.add_argument("--sanity_check", action="store_true",
                    help="Run the identity-gate no-op check (on a slice of validation) "
                         "and exit. NOT optional before a real run here.")
    ap.add_argument("--sanity_check_examples", type=int, default=50)
    ap.add_argument("--timing_probe", action="store_true")
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

    if os.environ.get("HF_TOKEN") is None:
        print("WARNING: HF_TOKEN not set. meta-llama/Llama-2-7b-hf is gate-licensed -- "
              "this will fail unless you're using a cached local copy or a token with "
              "accepted license access is otherwise configured.", flush=True)

    print("Loading Llama-2-7B (GATE-LICENSED -- requires HF_TOKEN with accepted access) ...", flush=True)
    model = load_llama2_7b(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Llama-2-7B loaded — {n_params:,} params, frozen.", flush=True)

    print(f"Loading HellaSwag ({HELLASWAG_REPO}) -- train for training, validation for eval ...",
          flush=True)
    tokenizer, train_loader, train_examples, val_examples = get_loaders(args.batch_size)
    print(f"Data: train_examples={len(train_examples):,} val_examples={len(val_examples):,}", flush=True)

    if args.sanity_check:
        sanity_check(model, sample_examples(val_examples, args.sanity_check_examples), device, tokenizer, args)
        return

    if args.timing_probe:
        print("\n── TIMING PROBE (λ=0.05, seed=0) ──", flush=True)
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
            pruner_step(pruner, model, tokenizer, opt, batch, 0.05, device)
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
            res = train_one_converge(lam, seed, model, tokenizer, train_loader, train_examples,
                                     val_examples, args, device, run_dir)
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
    rows = [f"Llama-2-7B convergence+LR-decay sweep | HellaSwag (accuracy-delta) | "
           f"seeds={args.seeds} | max_steps={args.max_steps} | device={device}", sep,
           f"{'lambda':>7} {'seed':>5} | {'steps':>7} {'conv?':>6} | {'% pruned':>9} | "
           f"{'orig acc':>9} | {'pruned acc':>10} | {'acc drop':>9}", sep]
    for r in all_results:
        rows.append(f"{r['lambda']:>7} {r['seed']:>5} | {r['steps_taken']:>7} "
                    f"{'YES' if r['converged'] else 'NO':>6} | {r['pct_pruned']:>8.2f}% | "
                    f"{r['orig_acc']:>8.2f}% | {r['pruned_acc']:>9.2f}% | "
                    f"{r['orig_acc']-r['pruned_acc']:>+8.2f}pp")
    summary_str = "\n".join(rows)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(summary_str + "\n")
    print("\n" + summary_str)
    print(f"\nResults → {args.out_dir}/")


if __name__ == "__main__":
    main()
