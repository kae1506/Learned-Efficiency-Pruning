"""
SparseGPT vs LEP on cifar_big.pt (CIFAR_big, 10.4M) FC head {fc1,fc2,fc3}.

Collects calibration activations ONCE from the dense model, then sweeps SparseGPT
over a set of weight-sparsities (restoring dense weights between points), evaluates
test accuracy at each, and prints a comparison against the published LEP numbers.

Design decisions (all surfaced in chat, repeated here so they are not buried):
  D1 axis = FC-head WEIGHT sparsity vs test-acc drop. LEP is structured (neuron)
     pruning; we convert its per-layer neuron keeps to an equivalent weight
     sparsity (see LEP_* below). At equal weight-sparsity SparseGPT has the easier
     job (drops any weight; LEP drops whole neurons + gets dense speedup).
  D2 device = CPU (8192x8192 Cholesky; MPS linalg is incomplete).
  D3 n_calib = 10000 (>= fc1 input dim 8192 so H=2XXt is full rank; fc1 is at the
     rank boundary -> its compensation is the least over-determined part).
  D4 uniform per-layer sparsity, swept; LEP's allocation is non-uniform (flagged).
  D5 percdamp=0.01, blocksize=128 (paper defaults).

Run from project root:
    venv/bin/python past_work/sparsegpt/run_comparison.py
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import copy

import torch
import torchvision
from torchvision.transforms import v2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sparsegpt import SparseGPT
from models import load_checkpoint, CIFAR_MEAN, CIFAR_STD

CKPT      = "experiments/checkpoints/cifar_big.pt"
DATA_DIR  = "./data"
N_CALIB   = 10000
BLOCKSIZE = 128
PERCDAMP  = 0.01
SWEEP     = [0.50, 0.70, 0.80, 0.84, 0.90]
DEVICE    = torch.device("cpu")

# ── LEP baseline on CIFAR_big (F13 / Appendix D), 3 seeds, lambda=0.03 ──────────
LEP_BASE_ACC   = 87.39          # dense test acc reported for the LEP run
LEP_NEURON_PCT = 70.9           # % neurons pruned across fc1,fc2,fc3
LEP_DROP_PP    = 1.48           # test-acc drop (pp)
LEP_KEEPS      = {"fc1": (171, 1024, 8192),   # (kept_out, total_out, in_dim)
                  "fc2": (177, 512, 1024),
                  "fc3": (92,  256, 512)}


def lep_weight_sparsity():
    """Convert LEP's per-layer neuron keeps into an FC-head weight sparsity.
    A pruned hidden neuron removes its row here AND its column in the next layer,
    so fc2/fc3 columns shrink with the previous layer's keeps."""
    keeps = {n: k for n, (k, _, _) in LEP_KEEPS.items()}
    orig = kept = 0
    prev_in = {"fc1": LEP_KEEPS["fc1"][2],       # fc1 input dim (conv) not pruned
               "fc2": keeps["fc1"],              # fc2 in-cols = fc1 kept neurons
               "fc3": keeps["fc2"]}              # fc3 in-cols = fc2 kept neurons
    for n, (k, tot_out, in_dim) in LEP_KEEPS.items():
        orig += tot_out * in_dim
        kept += k * prev_in[n]
    return 1 - kept / orig, {n: 1 - (keeps[n] * prev_in[n]) / (LEP_KEEPS[n][1] * LEP_KEEPS[n][2])
                             for n in LEP_KEEPS}


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


@torch.no_grad()
def collect_inputs(model, layers, loader):
    store = {n: [] for n in layers}
    handles = []
    for n, l in layers.items():
        def hook(mod, inp, out, _n=n):
            store[_n].append(inp[0].detach().cpu())
        handles.append(l.register_forward_hook(hook))
    for x, _ in loader:
        model(x.to(DEVICE))
    for h in handles:
        h.remove()
    return store


def main():
    tf = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                     v2.Normalize(CIFAR_MEAN, CIFAR_STD)])
    train = torchvision.datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=tf)
    test  = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=tf)
    calib = torch.utils.data.DataLoader(torch.utils.data.Subset(train, list(range(N_CALIB))),
                                        batch_size=256, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=256, shuffle=False)

    model, prunable = load_checkpoint(CKPT, DEVICE)
    named = dict(model.named_modules())
    target = {n: named[n] for n in prunable}
    n_params = sum(p.numel() for p in model.parameters())

    lep_ws, lep_per_layer = lep_weight_sparsity()

    print(f"ckpt={CKPT}  params={n_params:,}  device={DEVICE}")
    print(f"prunable FC head: {prunable}")
    print(f"n_calib={N_CALIB}  blocksize={BLOCKSIZE}  percdamp={PERCDAMP}\n")
    print(f"LEP baseline (F13/App.D): base {LEP_BASE_ACC:.2f}%  ->  "
          f"{LEP_NEURON_PCT:.1f}% neurons pruned  @ -{LEP_DROP_PP:.2f}pp")
    print(f"LEP equiv FC-head WEIGHT sparsity = {lep_ws*100:.1f}%  "
          f"(per-layer: " + ", ".join(f"{n} {s*100:.0f}%" for n, s in lep_per_layer.items()) + ")\n")

    dense_acc = evaluate(model, test_loader)
    print(f"our dense test acc: {dense_acc*100:.2f}%  (LEP-paper dense {LEP_BASE_ACC:.2f}%)\n")

    print("collecting calibration activations (once) ...", flush=True)
    inputs = collect_inputs(model, target, calib)
    # snapshot dense FC weights so we can restore between sweep points
    dense_fc = {n: target[n].weight.data.clone() for n in target}

    rows = []
    for sp in SWEEP:
        for n in target:                              # restore dense weights
            target[n].weight.data.copy_(dense_fc[n])
        for n, layer in target.items():
            gpt = SparseGPT(layer)
            for x in inputs[n]:
                gpt.add_batch(x.to(DEVICE))
            gpt.fasterprune(sparsity=sp, blocksize=BLOCKSIZE, percdamp=PERCDAMP)
            gpt.free()
        # achieved overall weight sparsity over the FC head
        z = tot = 0
        for n in target:
            z += (target[n].weight == 0).sum().item(); tot += target[n].weight.numel()
        ach = z / tot
        acc = evaluate(model, test_loader)
        drop = (dense_acc - acc) * 100
        rows.append((sp, ach, acc * 100, drop))
        print(f"  sparsity={sp:.2f}  achieved={ach*100:5.2f}%  "
              f"acc={acc*100:5.2f}%  drop={drop:+.2f}pp", flush=True)

    # ── comparison table ──────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("SparseGPT (unstructured, one-shot, no retrain)  vs  LEP (structured)")
    print("=" * 66)
    print(f"{'method':<24}{'weight-sp':>10}{'test-acc':>10}{'drop pp':>10}")
    print("-" * 66)
    for sp, ach, acc, drop in rows:
        tag = "SparseGPT" + ("  <- iso-LEP" if abs(sp - 0.84) < 1e-6 else "")
        print(f"{tag:<24}{ach*100:>9.1f}%{acc:>9.2f}%{drop:>+9.2f}")
    print("-" * 66)
    print(f"{'LEP (F13, 3-seed)':<24}{lep_ws*100:>9.1f}%"
          f"{LEP_BASE_ACC-LEP_DROP_PP:>9.2f}%{-LEP_DROP_PP:>+9.2f}   "
          f"(={LEP_NEURON_PCT:.0f}% neurons)")
    print("=" * 66)


if __name__ == "__main__":
    main()
