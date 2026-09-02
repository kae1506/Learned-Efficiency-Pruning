"""
Llama-2-7B MLP pruner -- SAMSum dialogue summarization. Direct port of
train_pruner_llama2_7b_cnndailymail.py -- same Pruner / SwiGLU gate hook /
block-mean convergence / B9 LR decay / CE-DELTA objective, only the dataset
plumbing and length-related defaults change. See that file's docstring for
the full derivation of everything not called out below.

VERIFIED LIVE (2026-08-26, on the 8xA6000 box) -- unlike the CNN/DailyMail
script's original "*** UNTESTED ***" state, this one has actually been
checked against real hardware before the first real run:
  - dataset repo/fields confirmed by a live `load_dataset` call (see below)
  - `--timing_probe` has been run for real against the live Llama-2-7B
    checkpoint + this dataset (see the timing probe's own log for the
    measured s/step)
NOT yet run: a real training loop past the timing probe, the sanity check,
or any (lambda, seed) sweep -- treat those the same as any fresh script.

DATASET -- bare `samsum` is loading-script-based (404s on datasets>=4.x,
same failure mode as bare `cnn_dailymail` -- see that script's docstring).
`knkarthick/samsum` is the parquet mirror, verified live: fields are
`id`/`dialogue`/`summary`, splits train=14,731 / validation=818 / test=819
(the original SAMSum sizes -- Gliwa et al. 2019). THREE-WAY split usage,
identical discipline to the CNN/DailyMail script:
  - `train`      -> pruner training (CE-delta objective below)
  - `validation` -> gap diagnostic (cheap, per-checkpoint, CE-only)
  - `test`       -> final report ONLY (full-split CE/ppl, plus bounded-
    sample ROUGE-via-generation)

OBJECTIVE -- target-only teacher-forced CE, identical shape to the
CNN/DailyMail script (translation-style: dialogue is CONTEXT/no loss, gold
summary is TARGET/loss only):

    ctx_text  = "Dialogue: " + dialogue[:MAX_DIALOGUE_CHARS] + "\n\nSummary: "
    full_text = ctx_text + summary
    labels    = full_ids with positions < len(ctx_ids) set to -100

    ce_orig   = CE(dense_model(ids), labels)     # no_grad, frozen weights
    ce_pruned = CE(gated_model(ids), labels)
    loss = (ce_pruned - ce_orig) + λ * mean(gate)

WHY SAMSum, vs re-running CNN/DailyMail again -- picked specifically to be
a DIFFERENT kind of task from the existing CNN/DailyMail case study (F23/
F25/B5 in the diary), not a re-run of the same one: conversational, multi-
speaker dialogue input rather than news prose, and a ~20x smaller train
split (14,731 vs 287,113), both of which stress different parts of the
pipeline (short-sequence throughput, low-data-regime step budgeting) than
CNN/DailyMail did.

CHOICES MADE HERE, NOT re-derived from CNN/DailyMail's defaults (flagged,
not silent -- same convention as every sibling script):
  - MAX_DIALOGUE_CHARS=1600 (~400 tokens), NOT 4000 like CNN/DailyMail's
    MAX_ARTICLE_CHARS. Measured live on the real train split: dialogue
    length is mean 502 / p95 1272 / max 5474 chars -- 1600 chars covers
    the large majority of dialogues UNTRUNCATED (unlike CNN/DailyMail,
    where 4000 chars deliberately truncates most articles for cost). A
    generous cap given how cheap this dataset's inputs are, not a
    cost-driven truncation like the CNN/DailyMail sibling's.
  - MAX_SUMMARY_TOKENS=96, NOT 128. Measured live: summary length mean 110
    / p95 235 / max 300 chars (~75 tokens at 4 chars/token) -- 96 is a
    tighter safety cap with margin, matching the same non-binding-safety-
    net role CNN/DailyMail's 128 plays there, just resized to this
    dataset's much shorter summaries.
  - max_steps LEFT AT 12000, CNN/DailyMail's own value -- explicitly NOT
    re-derived for this dataset. Flag: 12000 steps * batch_size=4 = 48,000
    examples seen = ~3.3 epochs over SAMSum's 14,731-example train split,
    vs. CNN/DailyMail's same 12000-step cap being ~0.17 epochs over its
    287,113-example split. Very different data-repetition regime -- worth
    a real decision before a full sweep, not just the timing probe this
    script was first used for.
  - DIALOGUE_PREFIX="Dialogue: " (not "Article: ") -- cosmetic, keeps the
    prompt template honest about what the context actually is.
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

os.environ.setdefault("HF_HOME", "./huggingface")

OUT_ROOT = "/workspace/results/llama2_7b_samsum_sweep"

N_LAYERS      = 32
N_INTER       = 11008   # SwiGLU intermediate size -- Llama-2-7B specific, NOT 14336
HIDDEN        = 4096
ROW_INPUT_DIM = 2 * HIDDEN   # gate_proj row concat up_proj row
LAYER_SHAPE   = (N_INTER, ROW_INPUT_DIM)

LLAMA2_REPO = "meta-llama/Llama-2-7b-hf"
SAMSUM_REPO = "knkarthick/samsum"   # bare "samsum" 404s on datasets>=4.x (loading-script-based
# dataset repos were dropped), same failure mode as cnn_dailymail -- verified live against this
# repo on the 8xA6000 box (2026-08-26): parquet mirror, fields id/dialogue/summary, splits
# train=14731/validation=818/test=819, matching the original Gliwa et al. 2019 SAMSum sizes.
DIALOGUE_PREFIX = "Dialogue: "
SUMMARY_PREFIX  = "\n\nSummary: "


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

def load_llama2_7b(device, hf_token=None):
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(
        LLAMA2_REPO, use_safetensors=True, torch_dtype=torch.bfloat16, token=hf_token
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
# Data -- SAMSum raw (dialogue, summary) rows, three-way split.
# ─────────────────────────────────────────────────────────────────────────────

def get_loaders(batch_size: int, hf_token=None, num_workers: int = 0):
    """Returns (tokenizer, train_loader, train_examples, val_examples, test_examples).
    Each *_examples is a plain list of {"dialogue": str, "summary": str}."""
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    tokenizer = AutoTokenizer.from_pretrained(LLAMA2_REPO, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset(SAMSUM_REPO)

    def to_examples(split):
        return [{"dialogue": ex["dialogue"], "summary": ex["summary"]} for ex in raw[split]]

    train_examples = to_examples("train")
    val_examples = to_examples("validation")
    test_examples = to_examples("test")

    train_loader = DataLoader(train_examples, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=lambda b: b)
    return tokenizer, train_loader, train_examples, val_examples, test_examples


def sample_examples(examples, n):
    return examples[:min(n, len(examples))]


# ─────────────────────────────────────────────────────────────────────────────
# Training-batch construction -- dialogue masked out, loss on summary only.
# ─────────────────────────────────────────────────────────────────────────────

def build_training_batch(examples, tokenizer, max_dialogue_chars, max_summary_tokens,
                         append_eos=False):
    """ctx_ids vs full_ids diff to locate the continuation span -- same F21-
    style trick as WikiText-2/HellaSwag/CNN-DailyMail (tokenizer merge
    boundary imprecision at the ctx/summary join is the same known, unfixed
    caveat as those scripts). Dialogue truncated to max_dialogue_chars
    BEFORE tokenizing (character-level, not exact token budget -- see
    module docstring; unlike the CNN/DailyMail sibling this rarely fires
    given how short SAMSum dialogues are).

    append_eos mirrors lora_samsum_ddp.py's build_batch fix verbatim: appended
    AFTER truncation, so the supervised target always ends on eos_token_id and
    the CE-delta objective can score whether pruning preserves the model's
    ability to predict its own stopping point -- see that script's module
    docstring, KNOWN METHODOLOGICAL FLAW section, for why this matters."""
    rows = []
    for ex in examples:
        dialogue = ex["dialogue"][:max_dialogue_chars]
        ctx_text = DIALOGUE_PREFIX + dialogue + SUMMARY_PREFIX
        full_text = ctx_text + ex["summary"]
        ctx_ids = tokenizer(ctx_text)["input_ids"]
        full_ids = tokenizer(full_text)["input_ids"]
        if len(full_ids) - len(ctx_ids) > max_summary_tokens:
            full_ids = full_ids[:len(ctx_ids) + max_summary_tokens]
        if append_eos:
            full_ids = full_ids + [tokenizer.eos_token_id]
        rows.append((full_ids, len(ctx_ids)))

    max_len = max(len(ids) for ids, _ in rows)
    B = len(rows)
    input_ids = torch.full((B, max_len), tokenizer.pad_token_id, dtype=torch.long)
    attn = torch.zeros((B, max_len), dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)
    for i, (ids, ctx_len) in enumerate(rows):
        L = len(ids)
        ids_t = torch.tensor(ids, dtype=torch.long)
        input_ids[i, :L] = ids_t
        attn[i, :L] = 1
        labels[i, ctx_len:L] = ids_t[ctx_len:]
    return input_ids, attn, labels


# ─────────────────────────────────────────────────────────────────────────────
# Cheap CE/perplexity evaluation -- same mechanism as training, used for the
# gap diagnostic and the (full-test-split) final ppl report.
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_ce(model, examples, device, tokenizer, gates=None, desc="eval",
                batch_size=8, max_dialogue_chars=1600, max_summary_tokens=96,
                append_eos=False):
    ctx = apply_gates(model, gates) if gates is not None else contextlib.nullcontext()
    total_nll = total_tokens = 0
    with ctx:
        for i in tqdm(range(0, len(examples), batch_size), desc=desc, unit="batch",
                      leave=False, dynamic_ncols=True):
            batch = examples[i:i + batch_size]
            input_ids, attn, labels = build_training_batch(batch, tokenizer, max_dialogue_chars,
                                                            max_summary_tokens, append_eos=append_eos)
            input_ids, attn, labels = input_ids.to(device), attn.to(device), labels.to(device)
            with autocast_ctx(device):
                loss = model(input_ids, attention_mask=attn, labels=labels).loss
            n_tok = (labels != -100).sum().item()
            total_nll += loss.item() * n_tok
            total_tokens += n_tok
    return total_nll / total_tokens


def sanity_check(model, examples, device, tokenizer, args):
    print("\n" + "=" * 70)
    print("SANITY CHECK — identity-gate control (gates=None vs all-ones)")
    print("=" * 70, flush=True)
    kw = dict(batch_size=args.eval_batch_size, max_dialogue_chars=args.max_dialogue_chars,
             max_summary_tokens=args.max_summary_tokens)
    ce_none = evaluate_ce(model, examples, device, tokenizer, gates=None, desc="gates=None", **kw)
    ones_gates = [torch.ones(N_INTER, device=device) for _ in range(N_LAYERS)]
    ce_ones = evaluate_ce(model, examples, device, tokenizer, gates=ones_gates, desc="gates=all-ones", **kw)
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
# ROUGE evaluation -- actual generation, the real summarization metric.
# Requires `pip install evaluate rouge_score`.
# ─────────────────────────────────────────────────────────────────────────────

def _rouge_scores_to_floats(scores):
    """Normalizes evaluate.load('rouge').compute()'s return into plain
    floats regardless of library version -- see the CNN/DailyMail sibling's
    docstring for the full version-skew explanation."""
    out = {}
    for k, v in scores.items():
        out[k] = float(v.mid.fmeasure) if hasattr(v, "mid") else float(v)
    return out


@torch.no_grad()
def evaluate_rouge(model, examples, device, tokenizer, gates=None, desc="rouge",
                   batch_size=4, max_new_tokens=96, num_beams=1, max_dialogue_chars=1600):
    import evaluate as hf_evaluate
    rouge = hf_evaluate.load("rouge")

    # Generous, self-scaling safety net, NOT an active constraint -- same
    # reasoning as the CNN/DailyMail sibling's gen_max_length, resized to
    # this dataset's much shorter dialogues.
    gen_max_length = max_dialogue_chars // 2 + 64

    prev_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"   # required for correct batched generation
    ctx = apply_gates(model, gates) if gates is not None else contextlib.nullcontext()
    predictions, references = [], []
    try:
        with ctx:
            for i in tqdm(range(0, len(examples), batch_size), desc=desc, unit="batch",
                          leave=False, dynamic_ncols=True):
                batch = examples[i:i + batch_size]
                prompts = [DIALOGUE_PREFIX + ex["dialogue"][:max_dialogue_chars] + SUMMARY_PREFIX
                          for ex in batch]
                enc = tokenizer(prompts, return_tensors="pt", padding=True,
                               truncation=True, max_length=gen_max_length).to(device)
                with autocast_ctx(device):
                    gen_ids = model.generate(**enc, max_new_tokens=max_new_tokens,
                                             do_sample=False, num_beams=num_beams,
                                             pad_token_id=tokenizer.pad_token_id)
                gen_only = gen_ids[:, enc["input_ids"].shape[1]:]
                predictions.extend(tokenizer.batch_decode(gen_only, skip_special_tokens=True))
                references.extend(ex["summary"] for ex in batch)
    finally:
        tokenizer.padding_side = prev_padding_side

    return _rouge_scores_to_floats(rouge.compute(predictions=predictions, references=references))


# ─────────────────────────────────────────────────────────────────────────────
# Convergence check -- verbatim, see train_pruner_opt125m_converge.py.
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
# Gap diagnostic -- CE-based (train vs validation), same shape/sign
# convention as train_pruner_llama2_7b_wikitext2.py.
# ─────────────────────────────────────────────────────────────────────────────

def gap_diagnostic_checkpoint(pruner, model, tokenizer, train_sample, val_sample, device, args):
    pruner.eval()
    with torch.no_grad():
        gates = pruner(get_mlp_weights(model))
    per_layer_keep = [g.mean().item() for g in gates]
    avg_gate = float(np.mean(per_layer_keep))

    kw = dict(batch_size=args.eval_batch_size, max_dialogue_chars=args.max_dialogue_chars,
             max_summary_tokens=args.max_summary_tokens, append_eos=args.append_eos)
    train_orig_ce   = evaluate_ce(model, train_sample, device, tokenizer, gates=None,   desc="gap: train orig",   **kw)
    train_pruned_ce = evaluate_ce(model, train_sample, device, tokenizer, gates=gates, desc="gap: train pruned", **kw)
    val_orig_ce     = evaluate_ce(model, val_sample,   device, tokenizer, gates=None,   desc="gap: val orig",     **kw)
    val_pruned_ce   = evaluate_ce(model, val_sample,   device, tokenizer, gates=gates, desc="gap: val pruned",   **kw)
    pruner.train()

    train_delta = train_orig_ce - train_pruned_ce
    val_delta   = val_orig_ce - val_pruned_ce
    return {
        "avg_gate": avg_gate, "pct_pruned": (1 - avg_gate) * 100, "per_layer_keep": per_layer_keep,
        "train_orig_ce": train_orig_ce, "train_pruned_ce": train_pruned_ce, "train_delta": train_delta,
        "val_orig_ce": val_orig_ce, "val_pruned_ce": val_pruned_ce, "val_delta": val_delta,
        "gap": train_delta - val_delta,
    }


GAP_CSV_COLUMNS = [
    "lambda", "seed", "step", "lr", "lr_state",
    "avg_gate", "pct_pruned", "delta_pct_pruned", "max_layer_delta_pct",
    "would_be_converged",
    "train_orig_ce", "train_pruned_ce", "train_delta",
    "val_orig_ce", "val_pruned_ce", "val_delta", "gap",
]


# ─────────────────────────────────────────────────────────────────────────────
# Single pruner training step -- CE-delta objective (target-only loss).
# ─────────────────────────────────────────────────────────────────────────────

def pruner_step(pruner, model, tokenizer, optimizer, batch_examples, sparsity_weight, device,
                max_dialogue_chars, max_summary_tokens, append_eos=False):
    optimizer.zero_grad()
    weights = get_mlp_weights(model)
    gates = pruner(weights)

    input_ids, attn, labels = build_training_batch(batch_examples, tokenizer, max_dialogue_chars,
                                                    max_summary_tokens, append_eos=append_eos)
    input_ids, attn, labels = input_ids.to(device), attn.to(device), labels.to(device)

    with torch.no_grad(), autocast_ctx(device):
        ce_orig = model(input_ids, attention_mask=attn, labels=labels).loss.item()
    with apply_gates(model, gates), autocast_ctx(device):
        ce_pruned = model(input_ids, attention_mask=attn, labels=labels).loss
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
    axes[1].set_title("CE loss (nats, summary tokens only)"); axes[1].set_xlabel("step")
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
# every sibling script.
# ─────────────────────────────────────────────────────────────────────────────

def train_one_converge(lam, seed, model, tokenizer, train_loader, train_examples, val_examples,
                       test_examples, args, device, run_dir):
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

    gap_train_sample = sample_examples(train_examples, args.gap_eval_examples)
    gap_val_sample   = sample_examples(val_examples, args.gap_eval_examples)
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
        m = pruner_step(pruner, model, tokenizer, opt, batch, lam, device,
                        args.max_dialogue_chars, args.max_summary_tokens,
                        append_eos=args.append_eos)

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
            g = gap_diagnostic_checkpoint(pruner, model, tokenizer, gap_train_sample, gap_val_sample, device, args)
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
                "train_delta": g["train_delta"], "val_orig_ce": g["val_orig_ce"],
                "val_pruned_ce": g["val_pruned_ce"], "val_delta": g["val_delta"], "gap": g["gap"],
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

    # Cheap CE/ppl -- full test split, matching the "test set used whole,
    # untruncated" convention from the WikiText-2 sibling.
    ce_kw = dict(batch_size=args.eval_batch_size, max_dialogue_chars=args.max_dialogue_chars,
                max_summary_tokens=args.max_summary_tokens, append_eos=args.append_eos)
    test_orig_ce   = evaluate_ce(model, test_examples, device, tokenizer, gates=None,
                                desc=f"[{tag}] test CE orig", **ce_kw)
    test_pruned_ce = evaluate_ce(model, test_examples, device, tokenizer, gates=final_gates,
                                desc=f"[{tag}] test CE pruned", **ce_kw)
    test_ppl_orig, test_ppl_pruned = float(np.exp(test_orig_ce)), float(np.exp(test_pruned_ce))

    # Expensive ROUGE-via-generation -- bounded sample only, see module docstring.
    rouge_sample = sample_examples(test_examples, args.rouge_eval_examples)
    rouge_kw = dict(batch_size=args.gen_batch_size, max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams, max_dialogue_chars=args.max_dialogue_chars)
    rouge_orig   = evaluate_rouge(model, rouge_sample, device, tokenizer, gates=None,
                                  desc=f"[{tag}] ROUGE orig", **rouge_kw)
    rouge_pruned = evaluate_rouge(model, rouge_sample, device, tokenizer, gates=final_gates,
                                  desc=f"[{tag}] ROUGE pruned", **rouge_kw)

    final_gate = history["avg_gate"][-1]
    pct_pruned = (1 - final_gate) * 100
    print(f"  → [{tag}] {'converged' if converged else 'CAPPED'} at step {step} ({total_time:.0f}s) | "
          f"final keep {final_gate:.3f} pruned {pct_pruned:.2f}% | "
          f"test ppl {test_ppl_orig:.3f} → {test_ppl_pruned:.3f} | "
          f"ROUGE-L {rouge_orig['rougeL']*100:.2f} → {rouge_pruned['rougeL']*100:.2f}", flush=True)

    plot_one_run(history, os.path.join(run_dir, "plot.png"),
                title=(f"Llama-2-7B MLP — SAMSum — λ={lam} seed={seed} — "
                      f"{'converged' if converged else 'CAPPED'} @ step {step} — "
                      f"{pct_pruned:.1f}% pruned, ROUGE-L {rouge_pruned['rougeL']*100:.2f}"))

    lines = [
        f"Llama-2-7B MLP pruner — SAMSum (target-only CE-delta objective) — "
        f"λ={lam}, seed={seed} — CONVERGENCE-BASED + LR-DECAY",
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
        f"samsum test set ({len(test_examples)} examples, full split, target-only CE):",
        f"  original  ppl              : {test_ppl_orig:.3f}",
        f"  pruned    ppl              : {test_ppl_pruned:.3f}",
        "-" * 60,
        f"ROUGE via generation ({len(rouge_sample)}-example sample of test, "
        f"greedy={args.num_beams == 1}, num_beams={args.num_beams}, max_new_tokens={args.max_new_tokens}):",
        f"  original  rouge1/rouge2/rougeL : "
        f"{rouge_orig['rouge1']*100:.2f}/{rouge_orig['rouge2']*100:.2f}/{rouge_orig['rougeL']*100:.2f}",
        f"  pruned    rouge1/rouge2/rougeL : "
        f"{rouge_pruned['rouge1']*100:.2f}/{rouge_pruned['rouge2']*100:.2f}/{rouge_pruned['rougeL']*100:.2f}",
    ]
    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    torch.save({
        "pruner_state_dict": pruner.state_dict(), "lambda": lam, "seed": seed,
        "embed_dim": args.embed_dim, "lstm_hidden": args.lstm_hidden,
        "per_layer_kept": per_layer_kept,
        "test_ppl_orig": test_ppl_orig, "test_ppl_pruned": test_ppl_pruned,
        "rouge_orig": rouge_orig, "rouge_pruned": rouge_pruned,
        "steps_taken": step, "converged": converged, "lr_state": lr_state,
    }, os.path.join(run_dir, "pruner.pt"))
    print(f"  [saved] {run_dir}/", flush=True)

    return {"lambda": lam, "seed": seed, "per_layer_kept": per_layer_kept, "pct_pruned": pct_pruned,
            "test_ppl_orig": test_ppl_orig, "test_ppl_pruned": test_ppl_pruned,
            "rouge_orig": rouge_orig, "rouge_pruned": rouge_pruned,
            "total_time": total_time, "steps_taken": step, "converged": converged}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.8, 1.6],
                    help="Inherited from the OPT-125M convergence sweep -- UNVALIDATED "
                         "at this scale/dataset.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--check_every", type=int, default=50)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--rel_tol", type=float, default=0.05)
    ap.add_argument("--abs_tol", type=float, default=0.01)
    ap.add_argument("--burn_in", type=int, default=500)
    ap.add_argument("--max_steps", type=int, default=12000,
                    help="Safety cap, INHERITED VERBATIM from the CNN/DailyMail sibling -- "
                         "NOT re-derived for SAMSum's ~20x-smaller train split. See module "
                         "docstring: this is ~3.3 epochs here vs. ~0.17 epochs there.")
    ap.add_argument("--lr_decay_window", type=int, default=250,
                    help="B9 default: window*check_every.")
    ap.add_argument("--lr_min", type=float, default=None, help="Default (None) = lr/10.")
    ap.add_argument("--gap_eval_every", type=int, default=200)
    ap.add_argument("--gap_eval_examples", type=int, default=200,
                    help="Examples sampled per side (train/val) for the gap diagnostic.")
    ap.add_argument("--batch_size", type=int, default=4,
                    help="(dialogue, summary) pairs per training step.")
    ap.add_argument("--eval_batch_size", type=int, default=8,
                    help="Batch size for the cheap CE/ppl eval (gap diagnostic + final ppl).")
    ap.add_argument("--gen_batch_size", type=int, default=4,
                    help="Batch size for ROUGE generation -- smaller than --eval_batch_size, "
                         "generation's KV cache is more memory-hungry than a scoring forward.")
    ap.add_argument("--rouge_eval_examples", type=int, default=300,
                    help="Bounded sample of the test split for the expensive ROUGE-via-"
                         "generation eval. NOTE: SAMSum's full test split is only 819 "
                         "examples, so 300 is ~37% of it -- a much larger fraction than the "
                         "CNN/DailyMail sibling's 300/11490 (~2.6%).")
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--num_beams", type=int, default=1,
                    help="Greedy (1) for speed. Literature commonly reports beam=4 for "
                         "summarization ROUGE -- flagged, not silently matched.")
    ap.add_argument("--max_dialogue_chars", type=int, default=1600,
                    help="Character-level truncation of the dialogue BEFORE tokenizing "
                         "(~400 tokens) -- covers the large majority of SAMSum dialogues "
                         "UNTRUNCATED (measured live: mean 502 / p95 1272 / max 5474 chars). "
                         "See module docstring.")
    ap.add_argument("--max_summary_tokens", type=int, default=96,
                    help="Safety cap on the target continuation length (measured live: "
                         "summary max is 300 chars / ~75 tokens).")
    ap.add_argument("--append_eos", action="store_true", default=False,
                    help="Append eos_token_id to the supervised target so the CE-delta "
                         "objective itself sees a stopping signal, not just downstream LoRA. "
                         "Mirrors lora_samsum_ddp.py's --append_eos verbatim. Defaults to "
                         "FALSE to stay faithful to the original 8-lambda sweep's convention "
                         "-- opt in per run, not silently changed.")
    ap.add_argument("--embed_dim", type=int, default=64)
    ap.add_argument("--lstm_hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--log_every", type=int, default=250)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--hf_token", type=str, default=None,
                    help="HF token with ACCEPTED access to meta-llama/Llama-2-7b-hf "
                         "(gate-licensed). CLI arg, not read from HF_TOKEN env var.")
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

    if args.hf_token is None:
        print("WARNING: --hf_token not set. meta-llama/Llama-2-7b-hf is gate-licensed -- "
              "this will fail unless you're using a cached local copy or a token with "
              "accepted license access is otherwise configured (e.g. `huggingface-cli "
              "login`).", flush=True)

    print("Loading Llama-2-7B (GATE-LICENSED -- requires --hf_token with accepted access) ...", flush=True)
    model = load_llama2_7b(device, hf_token=args.hf_token)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Llama-2-7B loaded — {n_params:,} params, frozen.", flush=True)

    print(f"Loading SAMSum ({SAMSUM_REPO}) ...", flush=True)
    tokenizer, train_loader, train_examples, val_examples, test_examples = get_loaders(
        args.batch_size, hf_token=args.hf_token)
    print(f"Data: train={len(train_examples):,} val={len(val_examples):,} test={len(test_examples):,}",
          flush=True)

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
            pruner_step(pruner, model, tokenizer, opt, batch, 0.05, device,
                       args.max_dialogue_chars, args.max_summary_tokens)
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
                                     val_examples, test_examples, args, device, run_dir)
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
    rows = [f"Llama-2-7B convergence+LR-decay sweep | SAMSum | seeds={args.seeds} | "
           f"max_steps={args.max_steps} | device={device}", sep,
           f"{'lambda':>7} {'seed':>5} | {'steps':>7} {'conv?':>6} | {'% pruned':>9} | "
           f"{'orig ppl':>9} | {'pruned ppl':>10} | {'orig R-L':>9} | {'pruned R-L':>10}", sep]
    for r in all_results:
        rows.append(f"{r['lambda']:>7} {r['seed']:>5} | {r['steps_taken']:>7} "
                    f"{'YES' if r['converged'] else 'NO':>6} | {r['pct_pruned']:>8.2f}% | "
                    f"{r['test_ppl_orig']:>9.3f} | {r['test_ppl_pruned']:>10.3f} | "
                    f"{r['rouge_orig']['rougeL']*100:>8.2f}% | {r['rouge_pruned']['rougeL']*100:>9.2f}%")
    summary_str = "\n".join(rows)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(summary_str + "\n")
    print("\n" + summary_str)
    print(f"\nResults → {args.out_dir}/")


if __name__ == "__main__":
    main()
