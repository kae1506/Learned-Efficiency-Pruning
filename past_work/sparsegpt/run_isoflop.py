"""
ISO-FLOP experiment: structured OBC (structured sibling of SparseGPT) vs LEP on
cifar_big.pt FC head. Both remove whole neurons → identical dense architecture at
a given budget → identical FLOPs → this isolates the SELECTION CRITERION
(one-shot second-order layer reconstruction vs LEP's learned task-aware mask).

Mapping (a neuron of layer ℓ = an input column of layer ℓ+1):
    prune fc2 columns  -> keeps K fc1 neurons  -> zero dropped rows of fc1
    prune fc3 columns  -> keeps K fc2 neurons  -> zero dropped rows of fc2
    prune fc4 columns  -> keeps K fc3 neurons  -> zero dropped rows of fc3
fc4's rows (the 10 outputs) are never pruned.

Design/caveats (surfaced in chat, repeated so nothing is buried):
  - iso-FLOP == iso per-layer neuron count. LEP config (F13): fc1 171, fc2 177,
    fc3 92 kept. f=1.0 reproduces it exactly; other f draw the Pareto curve.
  - structured OBC (Frantar&Alistarh 2022), the STRUCTURED sibling of SparseGPT,
    NOT SparseGPT's own (unstructured) algorithm. Exact greedy (layers small).
  - independent-layer / one-shot: all Hessians from the DENSE model.
  - Hessian over post-ReLU inputs; ReLU nonlinearity re-clipping ignored (OBS).
  - LEP is 3-seed; structured OBC is deterministic (closed-form greedy), 1 run.

Run from project root:
    venv/bin/python past_work/sparsegpt/run_isoflop.py
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys

import torch
import torchvision
from torchvision.transforms import v2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from structured import StructuredOBC
from models import load_checkpoint, CIFAR_MEAN, CIFAR_STD

CKPT     = "experiments/checkpoints/cifar_big.pt"
DATA_DIR = "./data"
N_CALIB  = 10000
PERCDAMP = 0.01
DEVICE   = torch.device("cpu")
FACTORS  = [0.5, 0.75, 1.0, 1.5, 2.0]      # multiples of LEP's kept-neuron config

# downstream layer whose COLUMNS we prune -> (upstream layer, LEP kept count)
# pruning fcN's columns keeps K neurons of the upstream layer.
PLAN = {"fc2": ("fc1", 171), "fc3": ("fc2", 177), "fc4": ("fc3", 92)}

LEP_BASE_ACC = 87.39
LEP_DROP_PP  = 1.48
LEP_NEURONS  = 70.9


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


def fc_flops(named, fc_names):
    """dense-structured MACs of the FC head = Σ (nonzero rows)·(nonzero cols)."""
    total = 0
    for n in fc_names:
        w = named[n].weight.data
        kept_out = (w.abs().sum(1) > 0).sum().item()
        kept_in  = (w.abs().sum(0) > 0).sum().item()
        total += kept_out * kept_in
    return total


def main():
    tf = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                     v2.Normalize(CIFAR_MEAN, CIFAR_STD)])
    train = torchvision.datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=tf)
    test  = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=tf)
    calib = torch.utils.data.DataLoader(torch.utils.data.Subset(train, list(range(N_CALIB))),
                                        batch_size=256, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=256, shuffle=False)

    model, _ = load_checkpoint(CKPT, DEVICE)
    named = dict(model.named_modules())
    fc_all = ["fc1", "fc2", "fc3", "fc4"]
    downstream = {n: named[n] for n in PLAN}

    dense_flops = fc_flops(named, fc_all)
    dense_acc = evaluate(model, test_loader)
    print(f"ckpt={CKPT}  device={DEVICE}  n_calib={N_CALIB}  percdamp={PERCDAMP}")
    print(f"dense FC-head MACs = {dense_flops:,}   dense test acc = {dense_acc*100:.2f}%")
    print(f"LEP (F13, 3-seed): {LEP_NEURONS:.0f}% neurons  @ -{LEP_DROP_PP:.2f}pp "
          f"(base {LEP_BASE_ACC:.2f}%)\n")

    print("collecting calibration activations (fc2,fc3,fc4 inputs) ...", flush=True)
    inputs = collect_inputs(model, downstream, calib)
    dense_fc = {n: named[n].weight.data.clone() for n in fc_all}
    dense_bias = {n: named[n].bias.data.clone() for n in fc_all}

    rows = []
    for f in FACTORS:
        # restore dense weights
        for n in fc_all:
            named[n].weight.data.copy_(dense_fc[n])
            named[n].bias.data.copy_(dense_bias[n])
        keeps = {}
        for dname, (uname, base_keep) in PLAN.items():
            layer = named[dname]
            keep = max(1, min(layer.weight.shape[1], round(f * base_keep)))
            keeps[dname] = keep
            obc = StructuredOBC(layer)
            for x in inputs[dname]:
                obc.add_batch(x)
            alive = obc.prune(keep, percdamp=PERCDAMP)      # bool mask over columns
            obc.free()
            # a dropped column of `dname` = a dead neuron (row) of the upstream layer
            dead_rows = (~alive)
            named[uname].weight.data[dead_rows] = 0.0
            named[uname].bias.data[dead_rows] = 0.0

        flops = fc_flops(named, fc_all)
        acc = evaluate(model, test_loader)
        drop = (dense_acc - acc) * 100
        # neurons pruned across fc1,fc2,fc3 (for parity with LEP's 70.9% figure)
        kept_neu = keeps["fc2"] + keeps["fc3"] + keeps["fc4"]   # fc1,fc2,fc3 kept
        tot_neu = 1024 + 512 + 256
        neu_pruned = (1 - kept_neu / tot_neu) * 100
        rows.append((f, flops, flops / dense_flops, neu_pruned, acc * 100, drop))
        print(f"  f={f:<4}  keeps(fc1/fc2/fc3)={keeps['fc2']}/{keeps['fc3']}/{keeps['fc4']:<4}"
              f"  FLOP%={flops/dense_flops*100:5.1f}  neurons_pruned={neu_pruned:4.1f}%"
              f"  acc={acc*100:5.2f}%  drop={drop:+.2f}pp", flush=True)

    print("\n" + "=" * 72)
    print("ISO-FLOP: structured OBC (2nd-order recon)  vs  LEP (learned mask)")
    print("=" * 72)
    print(f"{'method':<26}{'FLOP%':>8}{'neurons_pruned':>16}{'test-acc':>10}{'drop pp':>9}")
    print("-" * 72)
    for f, fl, fr, neu, acc, drop in rows:
        tag = "structured-OBC" + ("  <- iso-LEP" if abs(f - 1.0) < 1e-9 else "")
        print(f"{tag:<26}{fr*100:>7.1f}%{neu:>15.1f}%{acc:>9.2f}%{drop:>+8.2f}")
    print("-" * 72)
    # LEP FLOP% == its kept-weight fraction (structured) == 16.0%
    iso = next(r for r in rows if abs(r[0] - 1.0) < 1e-9)
    print(f"{'LEP (F13, 3-seed)':<26}{iso[2]*100:>7.1f}%{LEP_NEURONS:>15.1f}%"
          f"{LEP_BASE_ACC-LEP_DROP_PP:>9.2f}%{-LEP_DROP_PP:>+8.2f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
