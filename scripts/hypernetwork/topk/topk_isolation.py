"""
Isolation sweep: what broke v2 at the 50%-kept stage (88pp collapse)?

Fixed operating point K=1024 (50% kept), FRESH init (removes warm-start as a
variable), 1 seed, 500 steps. Toggle the two new v2 ingredients one at a time.
Anchor: one-shot centered, norm=none, T=1 gave 2.09pp in the diagnostic.

  A  none        T=1     sanity (should ~2pp)
  B  std         T=1     is the standardisation the culprit?
  C  std_detach  T=1     does a stabilised normalisation stay sane?
  D  none        T4->1   does the T-anneal alone misbehave?
  E  std_detach  T4->1   the combo intended for the full curriculum

Run:  venv/bin/python scripts/hypernetwork/topk_isolation.py
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

CKPT, CONFIG_PATH = "experiments/checkpoints/mnist_model.pt", "configs/config.yaml"
OUT_DIR = "experiments/latest/hypernetwork/topk_curriculum"
K, STEPS, SAMPLES, LR, SEED = 1024, 500, 64, 1e-3, 0

# (tag, node_norm, anneal_T)
RUNS = [
    ("A none      T=1  ", "none",       False),
    ("B std       T=1  ", "std",        False),
    ("C std_detach T=1 ", "std_detach", False),
    ("D none      T4->1", "none",       True),
    ("E std_detach T4->1","std_detach", True),
]


def set_seed(s):
    torch.manual_seed(s); np.random.seed(s); random.seed(s)


def train(pruner, model, train_loader, anneal):
    opt = torch.optim.Adam(pruner.parameters(), lr=LR)
    it = iter(train_loader); pruner.train()
    for step in range(STEPS):
        T = (4.0 + (1.0 - 4.0) * step / (STEPS - 1)) if anneal else 1.0
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader); x, y = next(it)
        x, y = x[:SAMPLES].to(DEV), y[:SAMPLES].to(DEV)
        opt.zero_grad()
        scores = pruner.node_scores(get_hidden_weights(model))
        gates = pruner.forward(get_hidden_weights(model), K, temp=T)  # centered
        with torch.no_grad():
            ce_orig = F.cross_entropy(model(x), y)
        ce_pruned = F.cross_entropy(masked_forward(model, gates, x), y)
        (ce_pruned - ce_orig).backward()
        torch.nn.utils.clip_grad_norm_(pruner.parameters(), 1.0)
        opt.step()


@torch.no_grad()
def evaluate(pruner, model, test_loader, baseline):
    pruner.eval()
    gates = pruner(get_hidden_weights(model), K)
    surv = int(sum(int(g.sum().item()) for g in gates))
    # how degenerate is the score distribution? (std of node scores per layer)
    sc = pruner.node_scores(get_hidden_weights(model))
    stds = [float(s.std()) for s in sc]
    acc = evaluate_with_gates(model, gates, test_loader, DEV)
    return (baseline - acc) * 100, surv, stds


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

    print(f"Device {DEV} | K={K} (50% kept) | {STEPS} steps | baseline {baseline*100:.2f}%\n")
    hdr = f"{'run':>19} | {'drop(pp)':>9} | {'survivors':>9} | {'node-score std (L1,L2)':>24}"
    print(hdr); print("-" * len(hdr))
    lines = [f"ISOLATION @ K={K} (50% kept), {STEPS} steps, baseline {baseline*100:.2f}%",
             "centered STE throughout; anchor (none,T=1) was 2.09pp one-shot", hdr, "-" * len(hdr)]
    for tag, norm, anneal in RUNS:
        set_seed(SEED)
        pr = TopKPruner(layer_shapes, use_layernorm=True, node_norm=norm).to(DEV)
        train(pr, model, train_loader, anneal)
        drop, surv, stds = evaluate(pr, model, test_loader, baseline)
        row = f"{tag:>19} | {drop:>8.2f} | {surv:>9} | ({stds[0]:.3f}, {stds[1]:.3f})"
        print(row); lines.append(row)
    print("-" * len(hdr))
    with open(os.path.join(OUT_DIR, "isolation.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved {OUT_DIR}/isolation.txt")


if __name__ == "__main__":
    main()
