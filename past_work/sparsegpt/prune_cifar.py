"""
Driver: one-shot SparseGPT pruning of a CIFAR conv-net's FC head.

Pipeline (no retraining, no gradient descent on any mask):
  1. load a trained checkpoint (cifar_cnn.pt / cifar_mid.pt / cifar_big.pt);
  2. collect a small calibration set of activations (the true INPUT to each
     target Linear layer) via forward hooks over `n_calib` train images;
  3. for each target FC layer, accumulate H = 2 X Xᵀ and run SparseGPT;
  4. report test accuracy before vs after, and the achieved sparsity.

Layers are pruned in forward order so that when we prune fc1 the calibration
inputs to fc2 are already collected from the ORIGINAL model. (SparseGPT is a
per-layer method; collecting all activations up-front from the dense model is
the standard "independent layer" variant and is what we do here — simple and
matches the paper's single-pass calibration.)

Run from project root, e.g.:
    venv/bin/python past_work/sparsegpt/prune_cifar.py \
        --ckpt experiments/checkpoints/cifar_cnn.pt --sparsity 0.5
    venv/bin/python past_work/sparsegpt/prune_cifar.py \
        --ckpt experiments/checkpoints/cifar_big.pt --sparsity 0.7 --n-calib 1024
    venv/bin/python past_work/sparsegpt/prune_cifar.py \
        --ckpt experiments/checkpoints/cifar_cnn.pt --nm 2:4     # 2:4 structured
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import argparse

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2

# allow running as a script from anywhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sparsegpt import SparseGPT
from models import load_checkpoint, CIFAR_MEAN, CIFAR_STD


def pick_device(requested: str) -> torch.device:
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_loaders(data_dir: str, batch_size: int, n_calib: int):
    tf = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    train = torchvision.datasets.CIFAR10(root=data_dir, train=True,  download=True, transform=tf)
    test  = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=tf)
    # calibration = first n_calib training images (clean transform, no augmentation:
    # SparseGPT wants the network's true operating-point activations).
    calib = torch.utils.data.Subset(train, list(range(n_calib)))
    calib_loader = torch.utils.data.DataLoader(calib, batch_size=batch_size, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test,  batch_size=256, shuffle=False)
    return calib_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total


@torch.no_grad()
def collect_inputs(model, layers: dict, loader, device):
    """Forward the calibration set once, capturing the INPUT tensor to each target
    layer via hooks. Returns {name: list[Tensor]}."""
    store = {name: [] for name in layers}
    handles = []
    for name, layer in layers.items():
        def hook(mod, inp, out, _n=name):
            store[_n].append(inp[0].detach().to(device))
        handles.append(layer.register_forward_hook(hook))
    for x, _ in loader:
        model(x.to(device))
    for h in handles:
        h.remove()
    return store


def sparsity_of(model, layers) -> float:
    z = tot = 0
    for name in layers:
        w = dict(model.named_modules())[name].weight
        z += (w == 0).sum().item()
        tot += w.numel()
    return z / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="experiments/checkpoints/cifar_cnn.pt")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--sparsity", type=float, default=0.5,
                    help="unstructured target fraction of weights to zero")
    ap.add_argument("--nm", type=str, default="",
                    help="n:m semi-structured, e.g. '2:4' (overrides --sparsity)")
    ap.add_argument("--n-calib", type=int, default=512, help="calibration images")
    ap.add_argument("--blocksize", type=int, default=128)
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--layers", default="",
                    help="comma-separated layer names to prune; default = the FC head")
    args = ap.parse_args()

    prunen = prunem = 0
    if args.nm:
        prunen, prunem = (int(t) for t in args.nm.split(":"))

    device = pick_device(args.device)
    model, default_prunable = load_checkpoint(args.ckpt, device)
    prunable = args.layers.split(",") if args.layers else default_prunable

    named = dict(model.named_modules())
    target_layers = {n: named[n] for n in prunable}
    for n, l in target_layers.items():
        assert isinstance(l, nn.Linear), f"{n} is {type(l)}, only Linear supported by this driver"

    calib_loader, test_loader = get_loaders(args.data_dir, args.batch_size, args.n_calib)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"ckpt={args.ckpt}  device={device}  params={n_params:,}")
    print(f"prunable FC head: {prunable}")
    mode = f"{prunen}:{prunem} structured" if prunen else f"{args.sparsity:.0%} unstructured"
    print(f"target: {mode}   calib={args.n_calib} imgs   blocksize={args.blocksize}   "
          f"percdamp={args.percdamp}")

    acc0 = evaluate(model, test_loader, device)
    print(f"\ndense test acc: {acc0*100:.2f}%")

    # collect calibration inputs to every target layer from the dense model
    print("collecting calibration activations ...", flush=True)
    inputs = collect_inputs(model, target_layers, calib_loader, device)

    # prune each target layer independently
    print("pruning:")
    for name, layer in target_layers.items():
        gpt = SparseGPT(layer)
        for x in inputs[name]:
            gpt.add_batch(x)
        loss = gpt.fasterprune(sparsity=args.sparsity, prunen=prunen, prunem=prunem,
                               blocksize=args.blocksize, percdamp=args.percdamp)
        gpt.free()
        s = (layer.weight == 0).float().mean().item()
        print(f"  {name:>4}: shape={tuple(layer.weight.shape)}  "
              f"achieved_sparsity={s*100:5.2f}%  recon_loss={loss:.4f}")

    acc1 = evaluate(model, test_loader, device)
    overall = sparsity_of(model, prunable)
    print(f"\nFC-head weight sparsity: {overall*100:.2f}%")
    print(f"test acc: {acc0*100:.2f}%  ->  {acc1*100:.2f}%   "
          f"(drop {(acc0-acc1)*100:+.2f}pp)")


if __name__ == "__main__":
    main()
