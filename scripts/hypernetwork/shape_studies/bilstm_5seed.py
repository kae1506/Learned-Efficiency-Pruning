"""
BiLSTM hypernetwork pruning — 5-seed re-run at sw=0.5 (≈80% sparsity).

Fresh-seed confirmation of the headline BiLSTM number (prior: 3.68±0.79 from
multi_seed_compare.py). Trains a fresh pruner per seed, applies the final
binary mask, evaluates on the FULL MNIST test set, records drop + dead/alive
activation ratio.

Output: experiments/latest/hypernetwork/bilstm_5seed/{summary.txt, plot.png, run.log}
Run from project root:
    venv/bin/python scripts/bilstm_5seed.py
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
from src.pruners.bilstm import Pruner as BiLSTMPruner
from src.prune_train import pruner_step, get_hidden_weights
from src.interpretability import analyze_pruner, evaluate_with_gates


SEEDS         = [0, 1, 2, 3, 4]
SPARSITY_W    = 0.5
PRUNER_STEPS  = 1000
PRUNER_LR     = 1e-3
SAMPLES       = 64
CKPT_PATH     = "experiments/checkpoints/mnist_model.pt"
CONFIG_PATH   = "configs/config.yaml"
OUT_DIR       = "experiments/latest/hypernetwork/bilstm_5seed"


def set_seed(s):
    torch.manual_seed(s); np.random.seed(s); random.seed(s)


def load_model(device):
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
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
    }


def stats(v):
    a = np.array(v, float)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else 0.0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  BiLSTM sw={SPARSITY_W}  |  seeds={SEEDS}\n")

    train_loader, test_loader = get_mnist_loaders(**cfg["data"])
    base = load_model(device)
    sizes = [L.out_features for L in
             [m for m in base.modules() if isinstance(m, torch.nn.Linear)][:-1]]
    full = [torch.ones(s, dtype=torch.bool, device=device) for s in sizes]
    baseline_acc = evaluate_with_gates(base, full, test_loader, device)
    print(f"Baseline test acc (full set): {baseline_acc*100:.2f}%\n")

    results = []
    for seed in SEEDS:
        model = load_model(device); set_seed(seed)
        t0 = datetime.datetime.now()
        r = train_bilstm(model, train_loader, test_loader, device, baseline_acc)
        r["seed"] = seed; r["secs"] = (datetime.datetime.now() - t0).total_seconds()
        results.append(r)
        print(f"  seed {seed}: pruned={r['frac_pruned']*100:.2f}%  "
              f"test_acc={r['test_acc']*100:.2f}%  drop={r['drop']*100:.2f}pp  "
              f"ratio={r['ratio']:.3f}  [{r['secs']:.0f}s]", flush=True)
        _write_summary(results, baseline_acc)

    dm, ds = stats([r["drop"]*100 for r in results])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["BiLSTM sw=0.5"], [dm], yerr=[ds], color="#2980b9", alpha=0.85, capsize=8)
    for r in results:
        ax.scatter([0], [r["drop"]*100], color="k", s=30, zorder=3)
    ax.set_ylabel("Full-test drop (pp)")
    ax.set_title("BiLSTM 5-seed re-run (sw=0.5, ~80%)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); plt.savefig(f"{OUT_DIR}/plot.png", dpi=150); plt.close(fig)
    _write_summary(results, baseline_acc, final=True)


def _write_summary(results, baseline_acc, final=False):
    dm, ds = stats([r["drop"]*100 for r in results]) if results else (0, 0)
    pm, ps = stats([r["frac_pruned"]*100 for r in results]) if results else (0, 0)
    rm, rs = stats([r["ratio"] for r in results]) if results else (0, 0)
    lines = [
        "=" * 70,
        f"BiLSTM 5-SEED RE-RUN — sw={SPARSITY_W}, {PRUNER_STEPS} steps",
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        f"Baseline test accuracy: {baseline_acc*100:.2f}%",
        "",
        f"{'seed':>4} | {'% Pruned':>9} | {'Test Acc':>9} | {'Drop pp':>8} | {'Ratio':>7}",
        "-" * 70,
    ]
    for r in results:
        lines.append(f"{r['seed']:>4} | {r['frac_pruned']*100:8.2f}% | {r['test_acc']*100:8.2f}% | "
                     f"{r['drop']*100:7.2f}  | {r['ratio']:>7.3f}")
    lines += [
        "-" * 70,
        f"mean | {pm:7.2f}±{ps:.2f} | {'':>9} | {dm:7.2f}±{ds:.2f} | {rm:.3f}±{rs:.3f}",
        "",
        "PRIOR: BiLSTM 5-seed (multi_seed_compare) 3.68±0.79",
        "=" * 70,
    ]
    with open(f"{OUT_DIR}/summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    if final:
        print("\n".join(lines))


if __name__ == "__main__":
    main()
