"""
Task-floor vs fixed-fraction test.

[1024,1024] prunes ~79% → ~427 neurons survive. Take a network that is ALREADY
~the size of that surviving subnetwork: [205,205] (=410 total, ~20% of the 2048
budget). Train fresh, run BiLSTM (sw=0.5), and look at the ABSOLUTE surviving
neuron count.

  - Task-floor      : task needs ~N neurons absolutely → [205,205] already near
                      the floor → prunes ~little, keeps ~all 410.
  - Fixed-fraction  : prunability ∝ overparameterization → [205,205] prunes ~79%
                      too (→ ~86 survive); subnetwork size scales with net size.

Output: experiments/latest/hypernetwork/shape_narrow205x2/{summary.txt, plot.png, run.log}
Run from project root:
    venv/bin/python scripts/hypernetwork/narrow_model_prune.py
"""

import os
import sys
import random
import datetime
import yaml
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(".")
from src.model import MLP
from src.dataset import get_mnist_loaders
from src.train import train_epoch, evaluate
from src.pruners.bilstm import Pruner as BiLSTMPruner
from src.prune_train import pruner_step, get_hidden_weights
from src.interpretability import analyze_pruner, evaluate_with_gates


# ── config ────────────────────────────────────────────────────────────────────
HIDDEN        = [205, 205]                # ~20% of 1024, two layers; 410 total
TRAIN_EPOCHS  = 10
TRAIN_LR      = 1e-3
DROPOUT       = 0.1
CKPT          = "experiments/checkpoints/mnist_narrow205x2.pt"

SEEDS         = [0, 1, 2]
SPARSITY_W    = 0.5
PRUNER_STEPS  = 1000
PRUNER_LR     = 1e-3
SAMPLES       = 64

CONFIG_PATH   = "configs/config.yaml"
OUT_DIR       = "experiments/latest/hypernetwork/shape_narrow205x2"

# reference: [1024,1024] settled on ~427 surviving neurons (79.17% of 2048 pruned)
BASELINE_KEPT = 427


def set_seed(s):
    torch.manual_seed(s); np.random.seed(s); random.seed(s)


def train_model(cfg, device):
    train_loader, test_loader = get_mnist_loaders(**cfg["data"])
    mcfg = {"input_dim": 784, "hidden_dims": HIDDEN, "output_dim": 10, "dropout": DROPOUT}
    model = MLP(**mcfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=TRAIN_LR)
    for ep in range(1, TRAIN_EPOCHS + 1):
        tl, ta = train_epoch(model, train_loader, opt, device)
        vl, va = evaluate(model, test_loader, device)
        print(f"  [train narrow] epoch {ep:>2} | train acc {ta:.4f} | test acc {va:.4f}", flush=True)
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": mcfg}, CKPT)


def load_frozen(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    m = MLP(**ckpt["config"]).to(device)
    m.load_state_dict(ckpt["state_dict"]); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def train_bilstm(model, train_loader, test_loader, device, baseline_acc):
    layer_shapes = [(w.shape[0], w.shape[1]) for w in get_hidden_weights(model)]
    pruner = BiLSTMPruner(layer_shapes).to(device)
    opt = torch.optim.Adam(pruner.parameters(), lr=PRUNER_LR)
    it = iter(train_loader)
    for _ in range(PRUNER_STEPS):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader); x, y = next(it)
        x, y = x[:SAMPLES].to(device), y[:SAMPLES].to(device)
        pruner_step(pruner, model, opt, x, y, SPARSITY_W)
    pruner.eval()
    with torch.no_grad():
        gates = pruner(get_hidden_weights(model))
    test_acc = evaluate_with_gates(model, gates, test_loader, device)
    res = analyze_pruner(model, gates=gates, calib_loader=train_loader,
                         device=device, n_calib_batches=5)
    n_total = sum(p["n_total"] for p in res["per_layer"])
    n_kept  = sum(p["n_kept"]  for p in res["per_layer"])
    alive, dead = res["mean_act_alive_overall"], res["mean_act_dead_overall"]
    return {
        "test_acc": test_acc, "drop": baseline_acc - test_acc,
        "frac_pruned": res["frac_pruned_total"],
        "n_total": n_total, "n_kept": n_kept,
        "ratio": dead / alive if alive else float("nan"),
    }


def stats(v):
    a = np.array(v, float)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else 0.0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print(f"=== Training narrow model {HIDDEN} (784 -> 205 -> 205 -> 10) ===")
    set_seed(0)
    train_model(cfg, device)
    print()

    train_loader, test_loader = get_mnist_loaders(**cfg["data"])
    base = load_frozen(CKPT, device)
    sizes = [L.out_features for L in
             [m for m in base.modules() if isinstance(m, torch.nn.Linear)][:-1]]
    full = [torch.ones(s, dtype=torch.bool, device=device) for s in sizes]
    baseline_acc = evaluate_with_gates(base, full, test_loader, device)
    print(f"Narrow model: {len(sizes)} hidden layers, {sum(sizes)} hidden neurons total")
    print(f"Baseline test acc (full set): {baseline_acc*100:.2f}%\n")

    print(f"=== BiLSTM pruner, {len(SEEDS)} seeds, sw={SPARSITY_W} ===")
    results = []
    for seed in SEEDS:
        model = load_frozen(CKPT, device); set_seed(seed)
        t0 = datetime.datetime.now()
        r = train_bilstm(model, train_loader, test_loader, device, baseline_acc)
        r["seed"] = seed; r["secs"] = (datetime.datetime.now() - t0).total_seconds()
        results.append(r)
        print(f"  seed {seed}: pruned={r['frac_pruned']*100:.2f}%  "
              f"kept={r['n_kept']}/{r['n_total']}  drop={r['drop']*100:.2f}pp  "
              f"ratio={r['ratio']:.3f}  [{r['secs']:.0f}s]", flush=True)
        _write_summary(results, baseline_acc)

    pm, ps = stats([r["frac_pruned"]*100 for r in results])
    km, ks = stats([r["n_kept"] for r in results])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(["narrow [205,205]\n(410 total)", "[1024,1024]\n(2048 total)"],
           [km, BASELINE_KEPT], color=["#e74c3c", "#2980b9"], alpha=0.85)
    ax.axhline(BASELINE_KEPT, color="#2980b9", ls="--", alpha=0.5,
               label=f"[1024,1024] survivors ≈ {BASELINE_KEPT}")
    ax.set_ylabel("Surviving neurons (absolute count)")
    ax.set_title("Task-floor vs fixed-fraction: absolute survivors", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); plt.savefig(f"{OUT_DIR}/plot.png", dpi=150); plt.close(fig)
    _write_summary(results, baseline_acc, final=True)


def _write_summary(results, baseline_acc, final=False):
    pm, ps = stats([r["frac_pruned"]*100 for r in results]) if results else (0, 0)
    km, ks = stats([r["n_kept"] for r in results]) if results else (0, 0)
    dm, ds = stats([r["drop"]*100 for r in results]) if results else (0, 0)
    rm, rs = stats([r["ratio"] for r in results]) if results else (0, 0)
    lines = [
        "=" * 74,
        f"NARROW-MODEL TEST — 784 -> {HIDDEN} -> 10  ({sum(HIDDEN)} hidden, ~20% of 2048)",
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 74,
        "",
        f"Baseline test accuracy: {baseline_acc*100:.2f}%",
        f"BiLSTM: sw={SPARSITY_W}, {PRUNER_STEPS} steps, {len(results)} seeds",
        "",
        f"{'seed':>4} | {'% Pruned':>9} | {'Kept':>11} | {'Test Acc':>9} | {'Drop pp':>8} | {'Ratio':>7}",
        "-" * 74,
    ]
    for r in results:
        lines.append(f"{r['seed']:>4} | {r['frac_pruned']*100:8.2f}% | {r['n_kept']:>4}/{r['n_total']:<5} | "
                     f"{r['test_acc']*100:8.2f}% | {r['drop']*100:7.2f}  | {r['ratio']:>7.3f}")
    lines += [
        "-" * 74,
        f"mean | {pm:7.2f}±{ps:.2f}% | {km:.0f}±{ks:.0f} kept | {'':>9} | {dm:7.2f}±{ds:.2f} | {rm:.3f}±{rs:.3f}",
        "",
        f"REFERENCE: [1024,1024] (2048 total) prunes 79.17% → ~{BASELINE_KEPT} survive.",
        "",
        "INTERPRETATION:",
        f"  - kept ≈ 410 (prunes ~0%)  → TASK-FLOOR: [205,205] already near minimal subnet.",
        f"  - kept ≈ 86  (prunes ~79%) → FIXED-FRACTION: subnet size scales with net size.",
        f"  - this run: kept ≈ {km:.0f} → see which it lands near.",
        "=" * 74,
    ]
    with open(f"{OUT_DIR}/summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    if final:
        print("\n".join(lines))


if __name__ == "__main__":
    main()
