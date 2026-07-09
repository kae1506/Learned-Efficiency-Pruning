"""
Architecture-generalization test for the BiLSTM pruner (#2-lite).

Same hidden-neuron budget (2048) but a DIFFERENT shape: one wide layer
784 -> 2048 -> 10, vs the baseline two-layer 784 -> 1024 -> 1024 -> 10.
Trains the wide model fresh on MNIST, then runs the BiLSTM pruner for 3 seeds
at sw=0.5 and compares prunability / accuracy-drop to the baseline.

Question: is the BiLSTM's ~79%-prune-at-3.7pp result a property of the neuron
BUDGET, or of the 2-layer SHAPE? Holds neuron count fixed, varies shape.

Output: experiments/latest/hypernetwork/transfer_wide2048/{summary.txt, plot.png, run.log}
Run from project root:
    venv/bin/python scripts/wide_model_prune.py
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
WIDE_HIDDEN   = [2048]            # one wide layer; same 2048 total as [1024,1024]
TRAIN_EPOCHS  = 10
TRAIN_LR      = 1e-3
DROPOUT       = 0.1
WIDE_CKPT     = "experiments/checkpoints/mnist_wide2048.pt"

SEEDS         = [0, 1, 2]
SPARSITY_W    = 0.5
PRUNER_STEPS  = 1000
PRUNER_LR     = 1e-3
SAMPLES       = 64

CONFIG_PATH   = "configs/config.yaml"
OUT_DIR       = "experiments/latest/hypernetwork/transfer_wide2048"


def set_seed(s):
    torch.manual_seed(s); np.random.seed(s); random.seed(s)


def train_wide_model(cfg, device):
    train_loader, test_loader = get_mnist_loaders(**cfg["data"])
    mcfg = {"input_dim": 784, "hidden_dims": WIDE_HIDDEN, "output_dim": 10, "dropout": DROPOUT}
    model = MLP(**mcfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=TRAIN_LR)
    for ep in range(1, TRAIN_EPOCHS + 1):
        tl, ta = train_epoch(model, train_loader, opt, device)
        vl, va = evaluate(model, test_loader, device)
        print(f"  [train wide] epoch {ep:>2} | train acc {ta:.4f} | test acc {va:.4f}", flush=True)
    os.makedirs(os.path.dirname(WIDE_CKPT), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": mcfg}, WIDE_CKPT)
    print(f"  saved wide model to {WIDE_CKPT}")
    return mcfg


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

    print(f"=== Training wide model {WIDE_HIDDEN} (784 -> {WIDE_HIDDEN[0]} -> 10) ===")
    set_seed(0)
    mcfg = train_wide_model(cfg, device)
    print()

    train_loader, test_loader = get_mnist_loaders(**cfg["data"])
    base = load_frozen(WIDE_CKPT, device)
    sizes = [L.out_features for L in
             [m for m in base.modules() if isinstance(m, torch.nn.Linear)][:-1]]
    full = [torch.ones(s, dtype=torch.bool, device=device) for s in sizes]
    baseline_acc = evaluate_with_gates(base, full, test_loader, device)
    n_hidden = sum(sizes)
    print(f"Wide model: {len(sizes)} hidden layer(s), {n_hidden} hidden neurons total")
    print(f"Baseline test acc (full set): {baseline_acc*100:.2f}%\n")

    print(f"=== BiLSTM pruner, {len(SEEDS)} seeds, sw={SPARSITY_W} ===")
    results = []
    for seed in SEEDS:
        model = load_frozen(WIDE_CKPT, device); set_seed(seed)
        t0 = datetime.datetime.now()
        r = train_bilstm(model, train_loader, test_loader, device, baseline_acc)
        r["seed"] = seed; r["secs"] = (datetime.datetime.now() - t0).total_seconds()
        results.append(r)
        print(f"  seed {seed}: pruned={r['frac_pruned']*100:.2f}%  "
              f"test_acc={r['test_acc']*100:.2f}%  drop={r['drop']*100:.2f}pp  "
              f"ratio={r['ratio']:.3f}  [{r['secs']:.0f}s]", flush=True)
        _write_summary(results, baseline_acc, n_hidden)

    dm, ds = stats([r["drop"]*100 for r in results])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["wide [2048]\n(this)", "[1024,1024]\n(baseline)"],
           [dm, 3.68], yerr=[ds, 0.79],
           color=["#9b59b6", "#2980b9"], alpha=0.85, capsize=8)
    for r in results:
        ax.scatter([0], [r["drop"]*100], color="k", s=30, zorder=3)
    ax.set_ylabel("Full-test drop (pp)")
    ax.set_title("Same neuron budget (2048), different shape — BiLSTM @ sw=0.5",
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); plt.savefig(f"{OUT_DIR}/plot.png", dpi=150); plt.close(fig)
    _write_summary(results, baseline_acc, n_hidden, final=True)


def _write_summary(results, baseline_acc, n_hidden, final=False):
    dm, ds = stats([r["drop"]*100 for r in results]) if results else (0, 0)
    pm, ps = stats([r["frac_pruned"]*100 for r in results]) if results else (0, 0)
    rm, rs = stats([r["ratio"] for r in results]) if results else (0, 0)
    lines = [
        "=" * 72,
        f"WIDE-MODEL BiLSTM TEST — 784 -> {WIDE_HIDDEN} -> 10  ({n_hidden} hidden)",
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 72,
        "",
        f"Same hidden-neuron budget as baseline (2048), different shape.",
        f"Baseline (this wide model) test accuracy: {baseline_acc*100:.2f}%",
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
        f"mean | {pm:7.2f}±{ps:.2f} | {'':>9} | {dm:7.2f}±{ds:.2f} | {rm:.3f}±{rs:.3f}",
        "",
        "BASELINE [1024,1024] (5-seed): 79.17±0.16% pruned, 3.68±0.79pp drop, ratio 0.308±0.007",
        "",
        "Interpretation: if wide-model prune% / drop match baseline → prunability is a",
        "property of the neuron BUDGET; if they differ → it depends on SHAPE/params.",
        "=" * 72,
    ]
    with open(f"{OUT_DIR}/summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    if final:
        print("\n".join(lines))


if __name__ == "__main__":
    main()
