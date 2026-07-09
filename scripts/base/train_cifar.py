"""
CIFAR-10 base-model trainer (LeNet-style CNN), MPS-enabled.

Architecture (canonical CIFAR LeNet; the single 2x2 max-pool is applied after
EACH conv, so the flattened feature size is 16*5*5 = 400):
    conv1  Conv2d(3, 6, 5)     32x32x3  -> 28x28x6
    pool   MaxPool2d(2, 2)     28x28x6  -> 14x14x6
    conv2  Conv2d(6, 16, 5)    14x14x6  -> 10x10x16
    pool   MaxPool2d(2, 2)     10x10x16 -> 5x5x16  (= 400 flattened)
    fc1    Linear(400, 128)
    fc2    Linear(128, 64)
    fc3    Linear(64, 10)
ReLU between layers; CrossEntropy loss; SGD(lr, momentum).

Run from project root:  venv/bin/python scripts/base/train_cifar.py
Saves: experiments/checkpoints/cifar_cnn.pt  (config + state_dict, MNIST-style)
       experiments/latest/base_model/cifar/{plot.png, summary.txt}
"""

import os
# Let any op without an MPS kernel silently fall back to CPU instead of crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import time

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import v2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(".")

CONFIG_PATH = "configs/cifar.yaml"
CKPT_PATH   = "experiments/checkpoints/cifar_cnn.pt"
OUT_DIR     = "experiments/latest/base_model/cifar"

# Standard CIFAR-10 channel mean/std (FLAGGED: normalisation helps optimisation).
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)


class CIFARNet(nn.Module):
    """LeNet-style CNN for CIFAR-10. Plain Conv2d/Linear so it is prunable later."""
    def __init__(self, output_dim: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16 * 5 * 5, 128)   # 400 -> 128
        self.fc2   = nn.Linear(128, 64)
        self.fc3   = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))      # -> 14x14x6
        x = self.pool(F.relu(self.conv2(x)))      # -> 5x5x16
        x = x.view(x.size(0), -1)                 # flatten -> 400
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CIFARNetBig(nn.Module):
    """Larger CNN (~10.4M params): 3 conv blocks (64/256/512) + 4 FC (1024/512/256/10).
    Pool after each conv (32->16->8->4) so the flatten is 512*4*4 = 8192. fc1 holds ~8.4M
    of the params. Plain Conv2d/Linear so it is prunable later."""
    def __init__(self, output_dim: int = 10):
        super().__init__()
        # conv -> BatchNorm -> ReLU -> pool. BN stabilises/accelerates conv training.
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1);   self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 256, 3, padding=1); self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, 3, padding=1);self.bn3 = nn.BatchNorm2d(512)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(512 * 4 * 4, 1024)   # 8192 -> 1024 (kept: the pruning target)
        self.fc2   = nn.Linear(1024, 512)
        self.fc3   = nn.Linear(512, 256)
        self.fc4   = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))    # -> 16x16x64
        x = self.pool(F.relu(self.bn2(self.conv2(x))))    # -> 8x8x256
        x = self.pool(F.relu(self.bn3(self.conv3(x))))    # -> 4x4x512
        x = x.view(x.size(0), -1)               # flatten -> 8192
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class CIFARNetMid(nn.Module):
    """Mid CIFAR conv-net (~1.22M params): scaled-down CIFAR_big with same structural
    pattern (3 conv blocks + BN + pool, then 4-FC head). Sits between LeNet (63K) and
    CIFAR_big (10.4M) on log scale — geometric mean is ~810K, this lands a bit above.

    Forward shapes:
        input  : 32x32x3
        conv1  : 32x32x32  → pool → 16x16x32
        conv2  : 16x16x64  → pool → 8x8x64
        conv3  : 8x8x128   → pool → 4x4x128
        flatten: 128*4*4 = 2048
        fc1 → fc2 → fc3 → fc4  :  2048 → 512 → 128 → 64 → 10
    """
    def __init__(self, output_dim: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3,   32,  3, padding=1); self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,  64,  3, padding=1); self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128,  3, padding=1); self.bn3 = nn.BatchNorm2d(128)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(128 * 4 * 4, 512)   # 2048 -> 512 (largest pruning target)
        self.fc2   = nn.Linear(512, 128)
        self.fc3   = nn.Linear(128, 64)
        self.fc4   = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))    # -> 16x16x32
        x = self.pool(F.relu(self.bn2(self.conv2(x))))    # -> 8x8x64
        x = self.pool(F.relu(self.bn3(self.conv3(x))))    # -> 4x4x128
        x = x.view(x.size(0), -1)                          # flatten -> 2048
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


def build_model(arch: str, output_dim: int) -> nn.Module:
    if arch == "big":
        return CIFARNetBig(output_dim)
    if arch == "mid":
        return CIFARNetMid(output_dim)
    return CIFARNet(output_dim)


def pick_device(requested: str) -> torch.device:
    """mps if requested+available, else cpu (with an honest message)."""
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_loaders(cfg):
    d = cfg["data"]
    augment = cfg["training"].get("augment", True)
    # Train transform: optional RandomCrop(pad 4) + HorizontalFlip — the standard
    # CIFAR augmentation that attacks the overfit. Test transform stays clean.
    train_ops = [v2.ToImage()]
    if augment:
        train_ops += [v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip()]
    train_ops += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(CIFAR_MEAN, CIFAR_STD)]
    train_tf = v2.Compose(train_ops)
    test_tf = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    train = torchvision.datasets.CIFAR10(root=d["data_dir"], train=True,  download=True, transform=train_tf)
    test  = torchvision.datasets.CIFAR10(root=d["data_dir"], train=False, download=True, transform=test_tf)
    tl = torch.utils.data.DataLoader(train, batch_size=d["batch_size"], shuffle=True,
                                     num_workers=d["num_workers"])
    vl = torch.utils.data.DataLoader(test,  batch_size=256, shuffle=False,
                                     num_workers=d["num_workers"])
    return tl, vl


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    arch = cfg["model"].get("arch", "lenet")
    out_dir = f"experiments/latest/base_model/cifar_{arch}"
    ckpt_path = f"experiments/checkpoints/cifar_{arch}.pt"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    device = pick_device(cfg.get("device", "mps"))
    train_loader, test_loader = get_loaders(cfg)

    model = build_model(arch, cfg["model"]["output_dim"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    criterion = nn.CrossEntropyLoss()
    wd = cfg["training"].get("weight_decay", 0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=wd)
    epochs = cfg["training"]["epochs"]
    # Cosine-anneal the LR from its initial value down to ~0 over the whole run.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    print(f"Device: {device} | params: {n_params:,} | batch {cfg['data']['batch_size']} | "
          f"epochs {epochs} | AdamW wd={wd} | cosine | aug={cfg['training'].get('augment', True)}",
          flush=True)

    hist = {"epoch": [], "train_loss": [], "test_acc": [], "secs": []}
    t_start = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running, nb = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item(); nb += 1
        train_loss = running / nb
        acc = evaluate(model, test_loader, device)
        dt = time.time() - t0
        hist["epoch"].append(ep); hist["train_loss"].append(train_loss)
        hist["test_acc"].append(acc); hist["secs"].append(dt)
        print(f"  epoch {ep:>2}/{epochs}: train_loss={train_loss:.3f} "
              f"test_acc={acc*100:5.2f}% lr={opt.param_groups[0]['lr']:.2e} [{dt:4.1f}s]", flush=True)
        sched.step()
    total = time.time() - t_start

    # ── checkpoint (MNIST-style: rebuildable config + weights) ────────────────
    torch.save({"config": {"output_dim": cfg["model"]["output_dim"], "arch": arch},
                "state_dict": model.state_dict()}, ckpt_path)

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.plot(hist["epoch"], hist["train_loss"], "o-", color="#c0392b")
    a1.set_xlabel("epoch"); a1.set_ylabel("train loss"); a1.set_title("Training loss")
    a1.grid(alpha=0.3)
    a2.plot(hist["epoch"], [a*100 for a in hist["test_acc"]], "o-", color="#27ae60")
    a2.set_xlabel("epoch"); a2.set_ylabel("test accuracy (%)"); a2.set_title("Test accuracy")
    a2.grid(alpha=0.3)
    fig.suptitle(f"CIFAR-10 {arch} ({n_params/1e6:.1f}M) — {device} — "
                 f"final {hist['test_acc'][-1]*100:.2f}% in {total:.0f}s", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{out_dir}/plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── summary ─────────────────────────────────────────────────────────────────
    lines = [
        "CIFAR-10 LeNet base-model training",
        f"device={device}  params={n_params:,}  batch={cfg['data']['batch_size']}  epochs={epochs}",
        f"optimizer=AdamW(lr={cfg['training']['lr']}, wd={wd}) + cosine  "
        f"aug={cfg['training'].get('augment', True)}  BN=yes  loss=CrossEntropy",
        "-" * 56,
        f"{'epoch':>5} | {'train_loss':>10} | {'test_acc':>8} | {'secs':>6}",
        "-" * 56,
    ] + [f"{e:>5} | {l:>10.3f} | {a*100:>7.2f}% | {s:>6.1f}"
         for e, l, a, s in zip(hist["epoch"], hist["train_loss"], hist["test_acc"], hist["secs"])] + [
        "-" * 56,
        f"final test acc : {hist['test_acc'][-1]*100:.2f}%",
        f"total time     : {total:.1f}s  (mean {np.mean(hist['secs']):.1f}s/epoch)",
        f"checkpoint     : {ckpt_path}",
    ]
    with open(f"{out_dir}/summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[-3:]))


if __name__ == "__main__":
    main()
