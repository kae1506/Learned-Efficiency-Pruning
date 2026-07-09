"""
TRUE transfer test: does a trained BiLSTM pruner generalize to networks it
never saw, with NO retraining?

Setup (all models are 784 -> 1024 -> 1024 -> 10):
  - Model A = the existing checkpoint. Train pruner P_A on A (sw=0.5).
  - Models B1..B3 = fresh MLPs trained on MNIST with different seeds.
  - Conditions:
      in-distribution : P_A applied to A
      TRANSFER        : P_A applied to each Bi  (frozen pruner, no retrain)
      oracle          : a fresh pruner P_Bi trained on each Bi
  Same layer shapes mean P_A's per-layer row encoders apply directly to B.

Question: is the weight->mask readout a GENERAL theory of redundancy (transfer ≈
oracle) or overfit to A's specific weights (transfer >> oracle)?

Output: experiments/latest/hypernetwork/transfer_frozen_pruner/{summary.txt, plot.png, run.log}
Run from project root:
    venv/bin/python scripts/hypernetwork/transfer_test.py
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
MODEL_A_CKPT  = "experiments/checkpoints/mnist_model.pt"
B_SEEDS       = [1, 2, 3]
TRAIN_EPOCHS  = 10
TRAIN_LR      = 1e-3
DROPOUT       = 0.1

PRUNER_SEED   = 0
SPARSITY_W    = 0.5
PRUNER_STEPS  = 1000
PRUNER_LR     = 1e-3
SAMPLES       = 64

CONFIG_PATH   = "configs/config.yaml"
OUT_DIR       = "experiments/latest/hypernetwork/transfer_frozen_pruner"


def set_seed(s):
    torch.manual_seed(s); np.random.seed(s); random.seed(s)


def load_frozen(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    m = MLP(**ckpt["config"]).to(device)
    m.load_state_dict(ckpt["state_dict"]); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def train_base(seed, cfg, device):
    """Fresh 784->1024->1024->10 MLP trained on MNIST with given seed (frozen on return)."""
    set_seed(seed)
    train_loader, test_loader = get_mnist_loaders(**cfg["data"])
    mcfg = {"input_dim": 784, "hidden_dims": [1024, 1024], "output_dim": 10, "dropout": DROPOUT}
    model = MLP(**mcfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=TRAIN_LR)
    for _ in range(TRAIN_EPOCHS):
        train_epoch(model, train_loader, opt, device)
    _, va = evaluate(model, test_loader, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, va


def train_pruner(model, train_loader, device, seed=PRUNER_SEED):
    set_seed(seed)
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
    return pruner


def apply_and_eval(pruner, model, train_loader, test_loader, device):
    """Apply a (frozen) pruner to model's weights, eval on full test set."""
    baseline = evaluate_with_gates(
        model, [torch.ones(w.shape[0], dtype=torch.bool, device=device)
                for w in get_hidden_weights(model)], test_loader, device)
    with torch.no_grad():
        gates = pruner(get_hidden_weights(model))
    acc = evaluate_with_gates(model, gates, test_loader, device)
    res = analyze_pruner(model, gates=gates, calib_loader=train_loader,
                         device=device, n_calib_batches=5)
    alive, dead = res["mean_act_alive_overall"], res["mean_act_dead_overall"]
    return {
        "baseline": baseline, "acc": acc, "drop": baseline - acc,
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
    print(f"Device: {device}\n")
    train_loader, test_loader = get_mnist_loaders(**cfg["data"])

    # ── Model A + pruner P_A ──────────────────────────────────────────────────
    print("=== Model A (existing checkpoint) + pruner P_A ===")
    model_A = load_frozen(MODEL_A_CKPT, device)
    P_A = train_pruner(model_A, train_loader, device)
    indist = apply_and_eval(P_A, model_A, train_loader, test_loader, device)
    print(f"  P_A -> A (in-distribution): pruned={indist['frac_pruned']*100:.2f}%  "
          f"drop={indist['drop']*100:.2f}pp  ratio={indist['ratio']:.3f}\n")

    # ── Models B + transfer/oracle ────────────────────────────────────────────
    transfer_rows, oracle_rows = [], []
    for bs in B_SEEDS:
        print(f"=== Model B (seed {bs}) ===")
        model_B, vaB = train_base(bs, cfg, device)
        print(f"  trained B: test acc {vaB*100:.2f}%")

        tr = apply_and_eval(P_A, model_B, train_loader, test_loader, device)   # frozen P_A
        tr["seed"] = bs; transfer_rows.append(tr)
        print(f"  TRANSFER P_A -> B{bs}: pruned={tr['frac_pruned']*100:.2f}%  "
              f"drop={tr['drop']*100:.2f}pp  ratio={tr['ratio']:.3f}")

        P_B = train_pruner(model_B, train_loader, device)                      # oracle
        orc = apply_and_eval(P_B, model_B, train_loader, test_loader, device)
        orc["seed"] = bs; oracle_rows.append(orc)
        print(f"  ORACLE   P_B{bs} -> B{bs}: pruned={orc['frac_pruned']*100:.2f}%  "
              f"drop={orc['drop']*100:.2f}pp  ratio={orc['ratio']:.3f}\n", flush=True)
        _write_summary(indist, transfer_rows, oracle_rows)

    # ── plot ──────────────────────────────────────────────────────────────────
    tdm, tds = stats([r["drop"]*100 for r in transfer_rows])
    odm, ods = stats([r["drop"]*100 for r in oracle_rows])
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["P_A → A\n(in-dist)", "P_A → B\n(TRANSFER)", "P_B → B\n(oracle)"]
    means  = [indist["drop"]*100, tdm, odm]
    errs   = [0, tds, ods]
    ax.bar(labels, means, yerr=errs, color=["#2980b9", "#9b59b6", "#27ae60"], alpha=0.85, capsize=8)
    for r in transfer_rows:
        ax.scatter([1], [r["drop"]*100], color="k", s=25, zorder=3)
    for r in oracle_rows:
        ax.scatter([2], [r["drop"]*100], color="k", s=25, zorder=3)
    ax.axhline(3.68, color="#c0392b", ls="--", alpha=0.6, label="BiLSTM 5-seed (3.68)")
    ax.set_ylabel("Full-test drop (pp)")
    ax.set_title("Frozen-pruner transfer: P_A applied to unseen models B",
                 fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); plt.savefig(f"{OUT_DIR}/plot.png", dpi=150); plt.close(fig)
    _write_summary(indist, transfer_rows, oracle_rows, final=True)


def _write_summary(indist, transfer_rows, oracle_rows, final=False):
    tdm, tds = stats([r["drop"]*100 for r in transfer_rows]) if transfer_rows else (0, 0)
    odm, ods = stats([r["drop"]*100 for r in oracle_rows]) if oracle_rows else (0, 0)
    tpm, tps = stats([r["frac_pruned"]*100 for r in transfer_rows]) if transfer_rows else (0, 0)
    trm, trs = stats([r["ratio"] for r in transfer_rows]) if transfer_rows else (0, 0)
    lines = [
        "=" * 76,
        "FROZEN-PRUNER TRANSFER TEST — BiLSTM, all models 784->1024->1024->10",
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 76,
        "",
        f"Pruner P_A trained on model A (sw={SPARSITY_W}, {PRUNER_STEPS} steps), then",
        f"applied FROZEN to fresh models B (seeds {B_SEEDS}) with NO retraining.",
        "",
        f"IN-DISTRIBUTION  P_A -> A : pruned={indist['frac_pruned']*100:.2f}%  "
        f"drop={indist['drop']*100:.2f}pp  ratio={indist['ratio']:.3f}",
        "",
        f"{'seed':>4} | {'cond':>9} | {'% Pruned':>9} | {'Drop pp':>8} | {'Ratio':>7}",
        "-" * 76,
    ]
    for r in transfer_rows:
        lines.append(f"{r['seed']:>4} | {'TRANSFER':>9} | {r['frac_pruned']*100:8.2f}% | "
                     f"{r['drop']*100:7.2f}  | {r['ratio']:>7.3f}")
    for r in oracle_rows:
        lines.append(f"{r['seed']:>4} | {'oracle':>9} | {r['frac_pruned']*100:8.2f}% | "
                     f"{r['drop']*100:7.2f}  | {r['ratio']:>7.3f}")
    lines += [
        "-" * 76,
        f"TRANSFER mean: pruned={tpm:.2f}±{tps:.2f}%  drop={tdm:.2f}±{tds:.2f}pp  ratio={trm:.3f}±{trs:.3f}",
        f"ORACLE   mean: drop={odm:.2f}±{ods:.2f}pp",
        "",
        "READ: transfer≈oracle≈3.68 → readout is a GENERAL theory of redundancy (transfers).",
        "      transfer >> oracle → pruner overfit to model A's specific weights.",
        "=" * 76,
    ]
    with open(f"{OUT_DIR}/summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    if final:
        print("\n".join(lines))


if __name__ == "__main__":
    main()
