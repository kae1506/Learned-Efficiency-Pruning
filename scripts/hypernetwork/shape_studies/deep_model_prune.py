"""
Depth axis of the shape→prunability study (companion to wide_model_prune.py).

Same 2048 hidden-neuron budget, but DEEP/NARROW: 784 -> 512 -> 512 -> 512 -> 512 -> 10
(4 layers × 512), vs baseline 2-layer [1024,1024] and wide 1-layer [2048].
Trains fresh on MNIST, runs BiLSTM pruner 3 seeds at sw=0.5, reports how much
it can prune.

Hypothesis (from "width concentrates redundancy"): narrower layers expose LESS
redundancy, so 4×512 should prune notably less than [1024,1024] (79%) and far
less than [2048] (93.5%) at the same neuron budget.

Output: experiments/latest/hypernetwork/shape_deep4x512/{summary.txt, plot.png, run.log}
Run from project root:
    venv/bin/python scripts/hypernetwork/deep_model_prune.py
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
HIDDEN        = [512, 512, 512, 512]      # 4 layers × 512 = 2048 total
TRAIN_EPOCHS  = 10
TRAIN_LR      = 1e-3
DROPOUT       = 0.1
CKPT          = "experiments/checkpoints/mnist_deep4x512.pt"

SEEDS         = [0, 1, 2]
SPARSITY_W    = 0.5
PRUNER_STEPS  = 1000
PRUNER_LR     = 1e-3
SAMPLES       = 64

CONFIG_PATH   = "configs/config.yaml"
OUT_DIR       = "experiments/latest/hypernetwork/shape_deep4x512"


def set_seed(s):
    torch.manual_seed(s); np.random.seed(s); random.seed(s)


def train_deep_model(cfg, device):
    train_loader, test_loader = get_mnist_loaders(**cfg["data"])
    mcfg = {"input_dim": 784, "hidden_dims": HIDDEN, "output_dim": 10, "dropout": DROPOUT}
    model = MLP(**mcfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=TRAIN_LR)
    for ep in range(1, TRAIN_EPOCHS + 1):
        tl, ta = train_epoch(model, train_loader, opt, device)
        vl, va = evaluate(model, test_loader, device)
        print(f"  [train deep] epoch {ep:>2} | train acc {ta:.4f} | test acc {va:.4f}", flush=True)
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": mcfg}, CKPT)
    print(f"  saved deep model to {CKPT}")


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
    alive, dead = res["mean_act_alive_overall"], res["mean_act_dead_overall"]
    return {
        "test_acc": test_acc, "drop": baseline_acc - test_acc,
        "frac_pruned": res["frac_pruned_total"],
        "ratio": dead / alive if alive else float("nan"),
        "per_layer": [p["frac_pruned"] for p in res["per_layer"]],
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

    print(f"=== Training deep model {HIDDEN} (784 -> {'->'.join(map(str,HIDDEN))} -> 10) ===")
    set_seed(0)
    train_deep_model(cfg, device)
    print()

    train_loader, test_loader = get_mnist_loaders(**cfg["data"])
    base = load_frozen(CKPT, device)
    sizes = [L.out_features for L in
             [m for m in base.modules() if isinstance(m, torch.nn.Linear)][:-1]]
    full = [torch.ones(s, dtype=torch.bool, device=device) for s in sizes]
    baseline_acc = evaluate_with_gates(base, full, test_loader, device)
    print(f"Deep model: {len(sizes)} hidden layers, {sum(sizes)} hidden neurons total")
    print(f"Baseline test acc (full set): {baseline_acc*100:.2f}%\n")

    print(f"=== BiLSTM pruner, {len(SEEDS)} seeds, sw={SPARSITY_W} ===")
    results = []
    for seed in SEEDS:
        model = load_frozen(CKPT, device); set_seed(seed)
        t0 = datetime.datetime.now()
        r = train_bilstm(model, train_loader, test_loader, device, baseline_acc)
        r["seed"] = seed; r["secs"] = (datetime.datetime.now() - t0).total_seconds()
        results.append(r)
        pl = "  ".join(f"L{i+1}:{p*100:.0f}%" for i, p in enumerate(r["per_layer"]))
        print(f"  seed {seed}: pruned={r['frac_pruned']*100:.2f}%  "
              f"drop={r['drop']*100:.2f}pp  ratio={r['ratio']:.3f}  [{pl}]  [{r['secs']:.0f}s]", flush=True)
        _write_summary(results, baseline_acc, sizes)

    dm, ds = stats([r["drop"]*100 for r in results])
    pm, ps = stats([r["frac_pruned"]*100 for r in results])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(["deep [512×4]\n(this)", "[1024,1024]", "wide [2048]"],
           [pm, 79.17, 93.47], yerr=[ps, 0.16, 0.31],
           color=["#e67e22", "#2980b9", "#9b59b6"], alpha=0.85, capsize=8)
    ax.set_ylabel("% pruned at sw=0.5")
    ax.set_title("Shape vs prunability — same 2048 neuron budget", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); plt.savefig(f"{OUT_DIR}/plot.png", dpi=150); plt.close(fig)
    _write_summary(results, baseline_acc, sizes, final=True)


def _write_summary(results, baseline_acc, sizes, final=False):
    dm, ds = stats([r["drop"]*100 for r in results]) if results else (0, 0)
    pm, ps = stats([r["frac_pruned"]*100 for r in results]) if results else (0, 0)
    rm, rs = stats([r["ratio"] for r in results]) if results else (0, 0)
    lines = [
        "=" * 72,
        f"DEEP-MODEL BiLSTM TEST — 784 -> {HIDDEN} -> 10  ({sum(sizes)} hidden)",
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 72,
        "",
        f"Same 2048 hidden-neuron budget, DEEP/NARROW shape (4 layers × 512).",
        f"Baseline test accuracy: {baseline_acc*100:.2f}%",
        f"BiLSTM: sw={SPARSITY_W}, {PRUNER_STEPS} steps, {len(results)} seeds",
        "",
        f"{'seed':>4} | {'% Pruned':>9} | {'Test Acc':>9} | {'Drop pp':>8} | {'Ratio':>7}",
        "-" * 72,
    ]
    for r in results:
        lines.append(f"{r['seed']:>4} | {r['frac_pruned']*100:8.2f}% | {r['test_acc']*100:8.2f}% | "
                     f"{r['drop']*100:7.2f}  | {r['ratio']:>7.3f}")
    lines += [
        "-" * 72,
        f"mean | {pm:7.2f}±{ps:.2f}% | {'':>9} | {dm:7.2f}±{ds:.2f} | {rm:.3f}±{rs:.3f}",
        "",
        "SHAPE COMPARISON (same 2048 neurons, sw=0.5):",
        f"  wide   [2048]        : 93.47% pruned",
        f"  medium [1024,1024]   : 79.17% pruned",
        f"  deep   [512×4]       : {pm:.2f}% pruned  (this)",
        "",
        "=" * 72,
    ]
    with open(f"{OUT_DIR}/summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    if final:
        print("\n".join(lines))


if __name__ == "__main__":
    main()
