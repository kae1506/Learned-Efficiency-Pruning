"""
Out-of-domain generalization check — OPT-125M pruner sweep (v1+v2, up to 21
checkpoints across 11 λ values, 18.85%-65.67% pruned; see
reconciled_opt125m_sweep.csv / opt125m_pruned_vs_loss.png). Every
checkpoint's CE improvement over the dense model (pruned CE below original
CE, holding across the ENTIRE swept range) has so far only been measured on
WikiText-2 TEST -- the same domain the pruner trained on (WikiText-2
TRAIN). The --improvement_result flag already ruled out overfitting to the
exact training tokens (full train/test transfer within WikiText-2), but
train and test there are both Wikipedia-style text -- correlated, not
independent domains. That leaves two live explanations:
  1. genuinely general improvement -- would hold on any text
  2. Wikipedia-domain-specific calibration -- narrower, still real, but not
     "pruning with no capability loss" in the way the literature uses the term

This script is the decisive test: evaluate the SAME trained gates (no
retraining) on C4 (Common Crawl web text -- a genuinely different
register/domain from Wikipedia), and compare ΔCE there against ΔCE on
WikiText-2 for every checkpoint:
  - improvement survives on C4  -> (1), a real general property
  - improvement shrinks/reverses -> (2), Wikipedia-specific

Standalone (Pruner / evaluate / apply_gates copied verbatim from
train_pruner_opt125m.py, not imported) so this can be pulled to a GPU pod
independently of the training script, same convention as that script's own
"no repo deps" design. Needs the v1/v2 checkpoint directories present
locally -- each pruner.pt's sibling summary.txt supplies the WikiText-2
numbers already computed for it (deterministic and already verified
bit-identical across all 21 runs, orig_ppl=49.038 everywhere -- safe to
reuse rather than re-running that eval), so this only adds ONE new pass per
checkpoint (C4), not two.

pip install: same pins as train_pruner_opt125m.py --
    transformers==5.12.1 datasets==5.0.0 matplotlib==3.10.8 numpy==2.4.2 tqdm==4.68.3
(excludes torch deliberately -- use the pod's CUDA-matched build.)

Usage (defaults assume running from experiments/latest/opt125m_results/,
mirroring reconcile_opt125m_sweep.py's convention):
    python ood_eval_opt125m.py \
        [--v1_dir opt125m_lambda_sweep] [--v2_dir v2/opt125m_lambda_sweep_v2] \
        [--ood_tokens 245000] [--out_dir .]

Produces:
  ood_eval.csv                per-checkpoint: lambda, seed, pct_pruned,
                               wikitext_delta_ce, c4_delta_ce, transfer_ratio
  opt125m_ood_vs_wikitext.png % pruned vs ΔCE, two lines (WikiText-2, C4)
"""
import argparse
import contextlib
import csv
import glob
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


N_LAYERS    = 12
N_INTER     = 3072   # intermediate (fc1 output) neurons per FFN block
EMBED_DIM   = 768    # OPT-125M hidden size
LAYER_SHAPE = (N_INTER, EMBED_DIM)


# ─────────────────────────────────────────────────────────────────────────────
# Pruner — verbatim copy from train_pruner_opt125m.py (see that file for the
# architecture rationale). Needed here only to reconstruct trained gates
# from a checkpoint's state_dict, no training.
# ─────────────────────────────────────────────────────────────────────────────

def binary_ste(logits: torch.Tensor) -> torch.Tensor:
    soft = torch.sigmoid(logits)
    hard = (soft > 0.5).float()
    return hard - soft.detach() + soft


class Pruner(nn.Module):
    def __init__(self, layer_shapes, embed_dim: int = 64, lstm_hidden: int = 64):
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


def load_opt125m(device):
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(
        "facebook/opt-125m", use_safetensors=True, torch_dtype=torch.float32
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def get_mlp_weights(model):
    return [model.model.decoder.layers[i].fc1.weight.detach() for i in range(N_LAYERS)]


def autocast_ctx(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@contextlib.contextmanager
def apply_gates(model, gates):
    hooks = []
    for block, gate in zip(model.model.decoder.layers, gates):
        def make_hook(g):
            def hook(module, args):
                x = args[0]
                return (x * g.view(1, 1, -1),)
            return hook
        hooks.append(block.fc2.register_forward_pre_hook(make_hook(gate)))
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


@torch.no_grad()
def evaluate(model, ids_flat, device, gates=None, desc="eval",
            max_length: int = 2048, stride: int = 1024) -> float:
    """Identical protocol to train_pruner_opt125m.py's evaluate() -- sliding
    window, -100 label-masking on already-scored tokens. Domain-agnostic:
    works on WikiText-2 or C4 identically, since it just walks whatever
    flat token tensor it's given."""
    total_len = ids_flat.size(0)
    total_nll = total_tokens = 0
    prev_end = 0
    positions = list(range(0, total_len, stride))
    for begin in tqdm(positions, desc=desc, unit="window", leave=False, dynamic_ncols=True):
        end = min(begin + max_length, total_len)
        trg_len = end - prev_end
        ids = ids_flat[begin:end].unsqueeze(0).to(device)
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
# OOD data — C4 (Common Crawl web text), streamed so this doesn't pull the
# full ~300GB+ corpus, just enough documents to hit the token budget.
# validation split (not train): no training happens on this data at all, but
# validation is the cleaner default for an eval-only use.
# ─────────────────────────────────────────────────────────────────────────────

def get_ood_ids(n_tokens: int) -> torch.Tensor:
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    ids = []
    for example in tqdm(ds, desc="streaming C4", unit="doc"):
        ids.extend(tokenizer(example["text"])["input_ids"])
        if len(ids) >= n_tokens:
            break
    if len(ids) < n_tokens:
        raise RuntimeError(f"C4 validation stream exhausted at {len(ids)} tokens, "
                           f"wanted {n_tokens} -- lower --ood_tokens.")
    return torch.tensor(ids[:n_tokens], dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint discovery — reuse each run's own summary.txt for the WikiText-2
# numbers (already computed, deterministic, no need to re-run that eval).
# ─────────────────────────────────────────────────────────────────────────────

def find_checkpoints(sweep_dir, source_label):
    runs = []
    for ckpt_path in sorted(glob.glob(os.path.join(sweep_dir, "lambda_*", "seed_*", "pruner.pt"))):
        run_dir = os.path.dirname(ckpt_path)
        summary_path = os.path.join(run_dir, "summary.txt")
        text = open(summary_path).read()
        header = re.search(r"λ=([\d.]+),\s*seed=(\d+)", text)
        lam, seed = float(header.group(1)), int(header.group(2))
        pct_pruned = float(re.search(r"final % FFN neurons pruned\s*:\s*([\d.]+)%", text).group(1))
        orig_ppl   = float(re.search(r"original\s+ppl\s*:\s*([\d.]+)", text).group(1))
        pruned_ppl = float(re.search(r"pruned\s+ppl\s*:\s*([\d.]+)", text).group(1))
        runs.append({
            "ckpt_path": ckpt_path, "lambda": lam, "seed": seed, "source": source_label,
            "pct_pruned": pct_pruned, "wikitext_orig_ppl": orig_ppl, "wikitext_pruned_ppl": pruned_ppl,
        })
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1_dir", default="opt125m_lambda_sweep")
    ap.add_argument("--v2_dir", default="v2/opt125m_lambda_sweep_v2")
    ap.add_argument("--ood_tokens", type=int, default=245_000,
                    help="Roughly matches WikiText-2 test's token count, "
                         "for a comparison with similar statistical power.")
    ap.add_argument("--eval_max_length", type=int, default=2048)
    ap.add_argument("--eval_stride",     type=int, default=1024)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_dir", type=str, default=".")
    args = ap.parse_args()

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    runs = find_checkpoints(args.v1_dir, "v1") + find_checkpoints(args.v2_dir, "v2")
    if not runs:
        raise SystemExit(f"No pruner.pt found under {args.v1_dir} or {args.v2_dir}")
    print(f"Found {len(runs)} checkpoints "
          f"({sum(1 for r in runs if r['source']=='v1')} v1, "
          f"{sum(1 for r in runs if r['source']=='v2')} v2)")

    print("Loading OPT-125M ...", flush=True)
    model = load_opt125m(device)

    print(f"Streaming C4 (en, validation) for {args.ood_tokens:,} tokens ...", flush=True)
    ood_ids = get_ood_ids(args.ood_tokens)
    print(f"C4 sample: {ood_ids.size(0):,} tokens", flush=True)

    print("\nEvaluating unpruned model on C4 (shared baseline, computed once) ...", flush=True)
    c4_orig_ce = evaluate(model, ood_ids, device, gates=None, desc="C4 orig",
                          max_length=args.eval_max_length, stride=args.eval_stride)
    print(f"  C4 orig CE = {c4_orig_ce:.4f}  (ppl = {np.exp(c4_orig_ce):.3f})")

    for r in runs:
        r["wikitext_delta_ce"] = float(np.log(r["wikitext_pruned_ppl"] / r["wikitext_orig_ppl"]))

    print(f"\n{'='*70}\nEvaluating {len(runs)} checkpoints on C4\n{'='*70}")
    for i, r in enumerate(runs, 1):
        tag = f"λ={r['lambda']} seed={r['seed']} ({r['source']})"
        ckpt = torch.load(r["ckpt_path"], map_location=device, weights_only=False)
        layer_shapes = [LAYER_SHAPE] * N_LAYERS
        pruner = Pruner(layer_shapes, embed_dim=ckpt["embed_dim"], lstm_hidden=ckpt["lstm_hidden"]).to(device)
        pruner.load_state_dict(ckpt["pruner_state_dict"])
        pruner.eval()
        with torch.no_grad():
            gates = pruner(get_mlp_weights(model))

        c4_pruned_ce = evaluate(model, ood_ids, device, gates=gates, desc=f"[{i}/{len(runs)}] {tag}",
                                max_length=args.eval_max_length, stride=args.eval_stride)
        r["c4_orig_ce"]    = c4_orig_ce
        r["c4_pruned_ce"]  = c4_pruned_ce
        r["c4_delta_ce"]   = c4_pruned_ce - c4_orig_ce
        r["transfer_ratio"] = (r["c4_delta_ce"] / r["wikitext_delta_ce"]
                               if r["wikitext_delta_ce"] != 0 else float("nan"))
        print(f"  [{i}/{len(runs)}] {tag} | {r['pct_pruned']:.2f}% pruned | "
              f"wikitext ΔCE {r['wikitext_delta_ce']:+.4f} | c4 ΔCE {r['c4_delta_ce']:+.4f} | "
              f"transfer {r['transfer_ratio']*100:.1f}%", flush=True)

    # ---- per-lambda aggregate ----
    lambdas = sorted(set(r["lambda"] for r in runs))
    agg = []
    for lam in lambdas:
        rs = [r for r in runs if r["lambda"] == lam]
        agg.append({
            "lambda": lam, "source": rs[0]["source"], "n_seeds": len(rs),
            "pct_pruned_mean":      float(np.mean([r["pct_pruned"] for r in rs])),
            "wikitext_delta_ce_mean": float(np.mean([r["wikitext_delta_ce"] for r in rs])),
            "c4_delta_ce_mean":      float(np.mean([r["c4_delta_ce"] for r in rs])),
            "c4_delta_ce_std":       float(np.std([r["c4_delta_ce"] for r in rs])),
            "transfer_ratio_mean":   float(np.mean([r["transfer_ratio"] for r in rs])),
        })

    print(f"\n{'='*90}\n{'lambda':>7} {'src':>4} | {'%pruned':>8} | {'wikitext ΔCE':>13} | "
          f"{'c4 ΔCE':>9} | {'transfer':>9}\n{'-'*90}")
    for a in agg:
        print(f"{a['lambda']:>7} {a['source']:>4} | {a['pct_pruned_mean']:>7.2f}% | "
              f"{a['wikitext_delta_ce_mean']:>+13.4f} | {a['c4_delta_ce_mean']:>+9.4f} | "
              f"{a['transfer_ratio_mean']*100:>8.1f}%")
    print("=" * 90)

    overall_transfer = float(np.mean([r["transfer_ratio"] for r in runs]))
    print(f"\nOVERALL mean transfer ratio: {overall_transfer*100:.1f}%")
    if overall_transfer > 0.7:
        print("Improvement transfers strongly to out-of-domain data -- consistent with a "
              "genuinely general property of the pruned model, not Wikipedia-specific fitting.")
    elif overall_transfer > 0.3:
        print("Partial transfer -- some real signal, but a meaningful chunk of the WikiText-2 "
              "improvement doesn't carry over to C4.")
    else:
        print("Improvement is mostly Wikipedia-specific and doesn't transfer to C4 -- the mask "
              "is calibrated to Wikipedia-style text, not a generally better subnetwork.")

    # ---- csv ----
    out_csv = os.path.join(args.out_dir, "ood_eval.csv")
    fieldnames = ["lambda", "seed", "source", "pct_pruned", "wikitext_orig_ppl", "wikitext_pruned_ppl",
                 "wikitext_delta_ce", "c4_orig_ce", "c4_pruned_ce", "c4_delta_ce", "transfer_ratio"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in runs:
            w.writerow({k: r[k] for k in fieldnames})
    print(f"\nSaved: {out_csv}")

    # ---- plot: % pruned vs delta CE, wikitext vs c4 ----
    fig, ax = plt.subplots(figsize=(8, 6))
    full_agg_sorted = sorted(agg, key=lambda a: a["pct_pruned_mean"])
    x_all = [a["pct_pruned_mean"] for a in full_agg_sorted]
    wt_all = [a["wikitext_delta_ce_mean"] for a in full_agg_sorted]
    c4_all = [a["c4_delta_ce_mean"] for a in full_agg_sorted]
    c4_err = [a["c4_delta_ce_std"] for a in full_agg_sorted]

    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.plot(x_all, wt_all, "o-", color="steelblue", lw=1.5, markersize=7, label="WikiText-2 (train domain)")
    ax.errorbar(x_all, c4_all, yerr=c4_err, fmt="o-", color="tomato", lw=1.5, markersize=7,
               capsize=4, label="C4 (out-of-domain)")
    for a in full_agg_sorted:
        ax.annotate(f"λ={a['lambda']}", (a["pct_pruned_mean"], a["wikitext_delta_ce_mean"]),
                   xytext=(6, 6), textcoords="offset points", fontsize=7)
    ax.set_xlabel("% FFN neurons pruned")
    ax.set_ylabel("ΔCE (nats)  —  negative = improvement over dense model")
    ax.set_title("OPT-125M — improvement on WikiText-2 vs. out-of-domain (C4)", fontweight="bold")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    out_png = os.path.join(args.out_dir, "opt125m_ood_vs_wikitext.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
