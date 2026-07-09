"""
CIFAR Mid base-model HYPERPARAMETER SWEEP.

Trains CIFARNetMid (~1.22M params, the in-between datapoint for the H2 / λ_opt
scaling study) under 4 (lr, wd) combinations, picks the highest test-acc config,
and saves it as the canonical experiments/checkpoints/cifar_mid.pt for the
downstream pruner λ sweep.

The sweep grid is intentionally small (4 configs) because CIFAR_big's default
(lr=0.001, wd=0.0005) already works well; we just bracket it to confirm.

Sweep grid:
  (lr=0.001,  wd=0.0005)   ← CIFAR_big default; baseline expected best
  (lr=0.002,  wd=0.0005)   ← higher lr for smaller model
  (lr=0.001,  wd=0.001)    ← more weight decay (stronger regularization)
  (lr=0.0005, wd=0.0005)   ← slower lr

Other training knobs are fixed: 40 epochs, batch 256, AdamW + cosine LR,
RandomCrop(pad=4) + HFlip augmentation, BN (already in CIFARNetMid).

Per-config artifacts go under experiments/latest/base_model/cifar_mid_sweep/<tag>/.
Best checkpoint copied to experiments/checkpoints/cifar_mid.pt.

Time estimate: ~10-15 min per config on MPS (smaller than CIFAR_big's 47 s/epoch
because conv channels are 2-4× narrower and fc1 is 16× smaller) → ~50 min total.

Run from project root:
  venv/bin/python scripts/base/train_cifar_mid_sweep.py
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import time
import shutil

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(".")
from scripts.base.train_cifar import (
    CIFARNetMid, CIFAR_MEAN, CIFAR_STD, pick_device, evaluate,
)


SWEEP_ROOT      = "experiments/latest/base_model/cifar_mid_sweep"
CANONICAL_CKPT  = "experiments/checkpoints/cifar_mid.pt"

SWEEP_CONFIGS = [
    # (tag,             lr,      wd)
    ("baseline",        0.001,   0.0005),
    ("highlr",          0.002,   0.0005),
    ("highwd",          0.001,   0.001),
    ("slowlr",          0.0005,  0.0005),
]

# Other knobs (fixed across all configs)
EPOCHS       = 40
BATCH_SIZE   = 256
NUM_WORKERS  = 2
DEVICE_PREF  = "mps"


def get_loaders(batch_size: int, num_workers: int):
    """CIFAR-10 train (with RandomCrop+HFlip aug) + test (clean) loaders."""
    train_tf = v2.Compose([
        v2.ToImage(),
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_tf = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    train = torchvision.datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_tf)
    test  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)
    tl = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    vl = torch.utils.data.DataLoader(test,  batch_size=256,        shuffle=False, num_workers=num_workers)
    return tl, vl


def train_one_config(tag: str, lr: float, wd: float, device: torch.device,
                     train_loader, test_loader) -> dict:
    """Train CIFARNetMid for EPOCHS with the given (lr, wd). Returns history dict."""
    out_dir = os.path.join(SWEEP_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "ckpt.pt")

    print(f"\n══════════════════════════════════════════════════")
    print(f" CIFAR Mid sweep — tag={tag}  lr={lr}  wd={wd}")
    print(f"══════════════════════════════════════════════════", flush=True)

    model = CIFARNetMid(output_dim=10).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    print(f"Device: {device} | params: {n_params:,} | batch {BATCH_SIZE} | "
          f"epochs {EPOCHS} | AdamW(lr={lr}, wd={wd}) | cosine | aug=True", flush=True)

    history = {"epoch": [], "train_loss": [], "test_acc": [], "secs": []}
    t_start = time.time()
    for ep in range(1, EPOCHS + 1):
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
        history["epoch"].append(ep); history["train_loss"].append(train_loss)
        history["test_acc"].append(acc); history["secs"].append(dt)
        print(f"  ep {ep:>2}/{EPOCHS}: train_loss={train_loss:.3f} "
              f"test_acc={acc*100:5.2f}% lr={opt.param_groups[0]['lr']:.2e} [{dt:4.1f}s]",
              flush=True)
        sched.step()
    total = time.time() - t_start

    # Save per-config checkpoint
    torch.save({"config":     {"output_dim": 10, "arch": "mid"},
                "state_dict": model.state_dict()}, ckpt_path)

    final_acc = history["test_acc"][-1]
    best_acc  = max(history["test_acc"])

    # Per-config plot
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.plot(history["epoch"], history["train_loss"], "o-", color="#c0392b")
    a1.set_xlabel("epoch"); a1.set_ylabel("train loss"); a1.set_title("Training loss")
    a1.grid(alpha=0.3)
    a2.plot(history["epoch"], [a*100 for a in history["test_acc"]], "o-", color="#27ae60")
    a2.set_xlabel("epoch"); a2.set_ylabel("test acc (%)"); a2.set_title("Test accuracy")
    a2.grid(alpha=0.3)
    fig.suptitle(f"CIFAR Mid — {tag} (lr={lr}, wd={wd}) — final {final_acc*100:.2f}% "
                 f"(best {best_acc*100:.2f}%) in {total:.0f}s", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "plot.png"), dpi=150,
                                    bbox_inches="tight")
    plt.close(fig)

    # Per-config summary
    lines = [
        f"CIFAR Mid sweep — {tag} (lr={lr}, wd={wd})",
        f"device={device}  params={n_params:,}  batch={BATCH_SIZE}  epochs={EPOCHS}",
        f"optimizer=AdamW(lr={lr}, wd={wd}) + cosine  aug=True  BN=yes",
        "-" * 56,
    ] + [f"  ep {e:>3} | loss {l:.3f} | acc {a*100:5.2f}% | {s:5.1f}s"
         for e, l, a, s in zip(history["epoch"], history["train_loss"],
                               history["test_acc"], history["secs"])] + [
        "-" * 56,
        f"final test acc : {final_acc*100:.2f}%",
        f"best  test acc : {best_acc*100:.2f}%",
        f"total time     : {total:.1f}s",
        f"checkpoint     : {ckpt_path}",
    ]
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    return {"tag": tag, "lr": lr, "wd": wd, "final_acc": final_acc,
            "best_acc": best_acc, "history": history, "ckpt_path": ckpt_path,
            "total_time": total}


def main():
    device = pick_device(DEVICE_PREF)
    print(f"Device: {device}  sweep configs: {len(SWEEP_CONFIGS)}", flush=True)
    os.makedirs(SWEEP_ROOT, exist_ok=True)
    os.makedirs(os.path.dirname(CANONICAL_CKPT), exist_ok=True)

    # Build loaders once (the train transform includes random aug — fine to re-iter)
    train_loader, test_loader = get_loaders(BATCH_SIZE, NUM_WORKERS)

    results = []
    for tag, lr, wd in SWEEP_CONFIGS:
        res = train_one_config(tag, lr, wd, device, train_loader, test_loader)
        results.append(res)

    # ── Pick best by FINAL test acc (not best-during-training, since we'd ship final) ─
    best = max(results, key=lambda r: r["final_acc"])
    print(f"\n──────────────────── SWEEP RESULTS ────────────────────")
    print(f"{'tag':<12} {'lr':>8} {'wd':>8} {'final':>9} {'best':>9} {'time':>7}")
    for r in results:
        marker = "  ← BEST" if r is best else ""
        print(f"{r['tag']:<12} {r['lr']:>8} {r['wd']:>8} "
              f"{r['final_acc']*100:>8.2f}% {r['best_acc']*100:>8.2f}% "
              f"{r['total_time']:>6.0f}s{marker}")

    # Copy best checkpoint to canonical location
    shutil.copyfile(best["ckpt_path"], CANONICAL_CKPT)
    print(f"\nCopied best checkpoint → {CANONICAL_CKPT}")

    # Write sweep-level summary
    lines = [
        "=" * 60,
        "CIFAR Mid HP SWEEP — final acc per config",
        "=" * 60,
        f"{'tag':<12} {'lr':>8} {'wd':>8} {'final':>9} {'best':>9} {'time_s':>8}",
    ]
    for r in results:
        marker = "  ← BEST → cifar_mid.pt" if r is best else ""
        lines.append(f"{r['tag']:<12} {r['lr']:>8} {r['wd']:>8} "
                     f"{r['final_acc']*100:>8.2f}% {r['best_acc']*100:>8.2f}% "
                     f"{r['total_time']:>7.0f}{marker}")
    with open(os.path.join(SWEEP_ROOT, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
