"""
Diagnostic for the STE-top-K underperformance (curriculum run hit 41pp @ K=492
vs the λ floor's 2pp). Isolates the two suspects with ONE-SHOT top-K (no
curriculum), a healthy 1000-step budget:

  1. Curriculum / early-stop the culprit?  -> one-shot top-K @ K=492 should
     recover ~2pp if so.
  2. STE saturation (unbounded per-node scores -> sigmoid'(s)->0 -> frozen
     ranking)?  -> compare PLAIN STE (soft=σ(s)) vs CENTERED STE
     (soft=σ((s-thresh)/T)): centering puts the strongest gradient on the
     boundary neurons regardless of absolute score scale.

Grid: STE ∈ {plain, centered} × K ∈ {1024 (50% kept), 492 (24% kept)} × 2 seeds.
Reference: λ floor = 2pp @ K=492.

Run:  venv/bin/python scripts/hypernetwork/topk_diagnostic.py
"""

import os
import sys
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.append(".")
from src.model import MLP
from src.dataset import get_mnist_loaders
from src.pruners.bilstm_topk import TopKPruner
from src.prune_train import get_hidden_weights, masked_forward
from src.interpretability import evaluate_with_gates

CKPT        = "experiments/checkpoints/mnist_model.pt"
CONFIG_PATH = "configs/config.yaml"
OUT_DIR     = "experiments/latest/hypernetwork/topk_curriculum"
STEPS       = 1000
SAMPLES     = 64
LR          = 1e-3
SEEDS       = [0, 1]
KS          = [1024, 492]      # 50% kept, 24% kept (the λ-floor operating point)
TEMP        = 1.0


def set_seed(s):
    torch.manual_seed(s); np.random.seed(s); random.seed(s)


def topk_gates(scores, k, center):
    """Global top-K STE gates; centered variant divides by the threshold."""
    flat = torch.cat([s.reshape(-1) for s in scores])
    n = flat.numel(); k = max(1, min(int(k), n))
    thresh = torch.topk(flat, k).values.min()
    out = []
    for s in scores:
        soft = torch.sigmoid((s - thresh) / TEMP) if center else torch.sigmoid(s)
        hard = (s >= thresh).float()
        out.append(hard - soft.detach() + soft)
    return out


def train_oneshot(pruner, model, train_loader, k, center):
    opt = torch.optim.Adam(pruner.parameters(), lr=LR)
    it = iter(train_loader)
    pruner.train()
    for _ in range(STEPS):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader); x, y = next(it)
        x, y = x[:SAMPLES].to(DEV), y[:SAMPLES].to(DEV)
        opt.zero_grad()
        scores = pruner.node_scores(get_hidden_weights(model))
        gates = topk_gates(scores, k, center)
        with torch.no_grad():
            ce_orig = F.cross_entropy(model(x), y)
        ce_pruned = F.cross_entropy(masked_forward(model, gates, x), y)
        (ce_pruned - ce_orig).backward()
        torch.nn.utils.clip_grad_norm_(pruner.parameters(), 1.0)
        opt.step()


@torch.no_grad()
def eval_oneshot(pruner, model, test_loader, k, center, baseline):
    pruner.eval()
    gates = topk_gates(pruner.node_scores(get_hidden_weights(model)), k, center)
    surv = int(sum(int(g.sum().item()) for g in gates))
    acc = evaluate_with_gates(model, gates, test_loader, DEV)
    return (baseline - acc) * 100, surv


def main():
    global DEV
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_mnist_loaders(**cfg["data"])
    ck = torch.load(CKPT, map_location=DEV, weights_only=True)
    model = MLP(**ck["config"]).to(DEV); model.load_state_dict(ck["state_dict"]); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layer_shapes = [(w.shape[0], w.shape[1]) for w in get_hidden_weights(model)]
    full = [torch.ones(w.shape[0], device=DEV) for w in get_hidden_weights(model)]
    baseline = evaluate_with_gates(model, full, test_loader, DEV)

    print(f"Device {DEV} | one-shot top-K, {STEPS} steps | baseline {baseline*100:.2f}%")
    print(f"{'STE':>9} | {'K':>5} | {'%pruned':>8} | {'drop(pp) mean[seeds]':>26}")
    print("-" * 60)
    lines = [
        "ONE-SHOT TOP-K DIAGNOSTIC (no curriculum, 1000 steps)",
        f"baseline {baseline*100:.2f}%   |   ref: λ floor = 2pp @ K=492",
        f"{'STE':>9} | {'K':>5} | {'%pruned':>8} | drop(pp) mean [per-seed]",
        "-" * 60,
    ]
    for center in [False, True]:
        for k in KS:
            drops = []
            for s in SEEDS:
                set_seed(s)
                pr = TopKPruner(layer_shapes, use_layernorm=True).to(DEV)
                train_oneshot(pr, model, train_loader, k, center)
                d, surv = eval_oneshot(pr, model, test_loader, k, center, baseline)
                drops.append(d)
            tag = "centered" if center else "plain"
            pct = 100 * (1 - k / 2048)
            row = (f"{tag:>9} | {k:>5} | {pct:>7.1f}% | {np.mean(drops):>6.2f}  "
                   f"[{', '.join(f'{d:.2f}' for d in drops)}]")
            print(row); lines.append(row)
    lines.append("-" * 60)
    with open(os.path.join(OUT_DIR, "diagnostic.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved {OUT_DIR}/diagnostic.txt")


if __name__ == "__main__":
    main()
