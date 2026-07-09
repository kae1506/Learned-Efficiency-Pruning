"""
Experiment A — collapse probe for the STE-top-K curriculum (B1 negative writeup).

We have established the top-K curriculum collapses to a deterministic 88.32pp
one-class mask at the 50%-kept stage, robust to T-schedule / normalization /
Adam-carry. Hypothesised proximate cause: the ROW ENCODER collapses to ~constant
output -> standardisation maps constant -> all-zeros -> ties -> torch.topk falls
to index order -> garbage mask -> no symmetry-breaking gradient -> absorbing.

This script INSTRUMENTS the (v4) curriculum and logs, every step:
  - raw row-encoder output std per layer (PRE-standardisation) — the smoking gun;
    if it crashes to ~0 the scores go constant.
  - pre-clip gradient norm — to see whether a KICK (spike) drives the crash, vs a
    gradual EROSION across warm-started stages.
  - loss (CE_pruned - CE_orig).
Single seed (collapse is deterministic). Same config as v4.

Output: experiments/latest/hypernetwork/topk_curriculum/collapse_probe.{png,txt}
Run:    venv/bin/python scripts/hypernetwork/topk_collapse_probe.py
"""

import os
import sys
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(".")
from src.model import MLP
from src.dataset import get_mnist_loaders
from src.pruners.bilstm_topk import TopKPruner, topk_ste
from src.prune_train import get_hidden_weights, masked_forward
from src.interpretability import evaluate_with_gates

CKPT, CONFIG_PATH = "experiments/checkpoints/mnist_model.pt", "configs/config.yaml"
OUT_DIR = "experiments/latest/hypernetwork/topk_curriculum"
SCHEDULE_PCT = [90, 75, 60, 50, 42, 36, 31, 27, 24, 21, 18]
SEED, SAMPLES = 0, 64
BASE_LR, CLIFF_LR, CLIFF_FRAC_PCT = 1e-3, 5e-4, 27
STEPS_EASY, STEPS_CLIFF, CLIFF_STEPS_PCT = 300, 600, 36
T_START, T_END = 4.0, 1.0


def set_seed(s):
    torch.manual_seed(s); np.random.seed(s); random.seed(s)


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_mnist_loaders(**cfg["data"])
    ck = torch.load(CKPT, map_location=dev, weights_only=True)
    model = MLP(**ck["config"]).to(dev); model.load_state_dict(ck["state_dict"]); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layer_shapes = [(w.shape[0], w.shape[1]) for w in get_hidden_weights(model)]
    full = [torch.ones(w.shape[0], device=dev) for w in get_hidden_weights(model)]
    baseline = evaluate_with_gates(model, full, test_loader, dev)
    n_total = int(sum(w.shape[0] for w in get_hidden_weights(model)))

    set_seed(SEED)
    # BOUND_CONTEXT=True re-adds the tanh on the per-layer context bias (the
    # allocation guardrail) — the only change vs the v4 collapse probe.
    BOUND_CONTEXT, NODE_NORM, WD = True, "std_detach", 1.0
    pruner = TopKPruner(layer_shapes, use_layernorm=True, node_norm=NODE_NORM,
                        bound_context=BOUND_CONTEXT).to(dev)
    print(f"bound_context={BOUND_CONTEXT} | node_norm={NODE_NORM} | AdamW wd={WD}")
    # AdamW: weight decay pins the scale-invariant encoder norm at an equilibrium
    # (η_eff ∝ √(η·wd), van Laarhoven) → σ stays bounded → 1/σ gradient does NOT
    # vanish → no freeze, while std_detach keeps scores unit-variance (band matched).
    opt = torch.optim.AdamW(pruner.parameters(), lr=BASE_LR, weight_decay=WD)   # carried
    total_steps = sum(STEPS_CLIFF if p <= CLIFF_STEPS_PCT else STEPS_EASY for p in SCHEDULE_PCT)

    rec_step, rec_rawL1, rec_rawL2, rec_grad, rec_loss = [], [], [], [], []
    rec_survL1, rec_survL2 = [], []           # per-layer survivor counts per step
    boundaries = []   # (global_step, pct)
    it = iter(train_loader); cum = 0
    print(f"baseline {baseline*100:.2f}% | n_total {n_total} | total_steps {total_steps}")

    for pct in SCHEDULE_PCT:
        k = round(pct / 100 * n_total)
        steps = STEPS_CLIFF if pct <= CLIFF_STEPS_PCT else STEPS_EASY
        for pg in opt.param_groups:
            pg["lr"] = CLIFF_LR if pct <= CLIFF_FRAC_PCT else BASE_LR
        boundaries.append((cum, pct))
        for s in range(steps):
            g = cum + s
            T = T_START + (T_END - T_START) * (g / (total_steps - 1))
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(train_loader); x, y = next(it)
            x, y = x[:SAMPLES].to(dev), y[:SAMPLES].to(dev)
            hw = get_hidden_weights(model)
            # raw row-encoder output std per layer (PRE-standardisation), no graph
            with torch.no_grad():
                raw_std = [float(enc(W).squeeze(-1).std()) for enc, W in zip(pruner.row_encoders, hw)]
            opt.zero_grad()
            gates = topk_ste(pruner.node_scores(hw), k, temp=T, center=True)
            # per-layer survivor counts (gate forward value = hard 0/1)
            survs = [int(gg.detach().sum().item()) for gg in gates]
            with torch.no_grad():
                ce_orig = F.cross_entropy(model(x), y)
            loss = F.cross_entropy(masked_forward(model, gates, x), y) - ce_orig
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(pruner.parameters(), 1.0))  # pre-clip
            opt.step()
            rec_step.append(g); rec_rawL1.append(raw_std[0]); rec_rawL2.append(raw_std[1])
            rec_grad.append(gnorm); rec_loss.append(loss.item())
            rec_survL1.append(survs[0]); rec_survL2.append(survs[1])
        cum += steps
        # end-of-stage eval
        with torch.no_grad():
            gates = pruner(get_hidden_weights(model), k)
        per_layer = [int(gg.sum().item()) for gg in gates]
        surv = int(sum(per_layer))
        acc = evaluate_with_gates(model, gates, test_loader, dev)
        print(f"  keep {pct:>2}% K={k:>4} surv={surv:>4} (L1={per_layer[0]:>4},L2={per_layer[1]:>4}) "
              f"drop={(baseline-acc)*100:6.2f}pp rawstd=({raw_std[0]:.3f},{raw_std[1]:.3f})", flush=True)

    # ── find the trigger step (first step raw std of EITHER layer < 1e-2) ──────
    arrL1, arrL2 = np.array(rec_rawL1), np.array(rec_rawL2)
    collapsed = np.where((arrL1 < 1e-2) | (arrL2 < 1e-2))[0]
    trig = int(rec_step[collapsed[0]]) if len(collapsed) else None
    trig_stage = None
    if trig is not None:
        for (b, pct) in boundaries:
            if b <= trig:
                trig_stage = pct
    # grad spike: max grad norm in the 20 steps before the trigger
    spike = None
    if trig is not None:
        lo = max(0, collapsed[0] - 20)
        spike = float(np.max(rec_grad[lo:collapsed[0] + 1]))

    # ── plot ─────────────────────────────────────────────────────────────────
    fig, (ax1, ax3, ax2) = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    # per-layer survivors panel (the layer-starvation test)
    ax3.plot(rec_step, rec_survL1, color="#2980b9", lw=1.2, label="survivors L1")
    ax3.plot(rec_step, rec_survL2, color="#c0392b", lw=1.2, label="survivors L2")
    ax3.set_ylabel("per-layer survivors")
    ax3.set_title("Per-layer survivors — layer-starvation test (each layer has 1024 neurons)",
                  fontweight="bold")
    for (b, pct) in boundaries:
        ax3.axvline(b, color="gray", ls=":", alpha=0.5)
    ax3.legend(fontsize=9, loc="upper right"); ax3.grid(alpha=0.3)
    ax1.plot(rec_step, rec_rawL1, color="#2980b9", lw=1.2, label="raw enc std L1")
    ax1.plot(rec_step, rec_rawL2, color="#c0392b", lw=1.2, label="raw enc std L2")
    ax1.set_yscale("log"); ax1.set_ylabel("raw row-encoder output std (log)")
    ax1.set_title("Row-encoder output variance — collapse probe (v4 config)", fontweight="bold")
    for (b, pct) in boundaries:
        ax1.axvline(b, color="gray", ls=":", alpha=0.5)
        ax1.text(b, ax1.get_ylim()[1], f"{pct}%", fontsize=7, va="top", color="gray")
    if trig is not None:
        ax1.axvline(trig, color="k", lw=1.5, label=f"collapse step {trig} (stage {trig_stage}%)")
    ax1.legend(fontsize=9, loc="lower left"); ax1.grid(alpha=0.3, which="both")

    ax2.plot(rec_step, rec_grad, color="#8e44ad", lw=1.0, label="pre-clip grad norm")
    axb = ax2.twinx()
    axb.plot(rec_step, rec_loss, color="#27ae60", lw=0.8, alpha=0.6, label="loss (CE diff)")
    axb.set_ylabel("loss (CE_pruned − CE_orig)", color="#27ae60")
    for (b, pct) in boundaries:
        ax2.axvline(b, color="gray", ls=":", alpha=0.5)
    if trig is not None:
        ax2.axvline(trig, color="k", lw=1.5)
    ax2.set_xlabel("global step"); ax2.set_ylabel("pre-clip grad norm", color="#8e44ad")
    ax2.set_title("Gradient norm (kick?) + loss", fontweight="bold")
    ax2.legend(fontsize=9, loc="upper left"); ax2.grid(alpha=0.3)

    fig.tight_layout()
    plot_path = os.path.join(OUT_DIR, "collapse_probe_adamw1.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight"); plt.close(fig)

    # ── summary ──────────────────────────────────────────────────────────────
    lines = [
        "STE-TOP-K COLLAPSE PROBE (v4 config: centered STE, global T, std_detach, carry-Adam)",
        f"baseline {baseline*100:.2f}%  n_total {n_total}  seed {SEED}",
        "",
        f"Stage boundaries (global_step -> %kept): "
        + ", ".join(f"{b}->{pct}%" for b, pct in boundaries),
        "",
    ]
    if trig is not None:
        lines += [
            f"COLLAPSE TRIGGER: step {trig} (within the {trig_stage}% stage; "
            f"stage starts at step {[b for b,p in boundaries if p==trig_stage][0]}).",
            f"  raw enc std at trigger: L1={arrL1[collapsed[0]]:.4f}  L2={arrL2[collapsed[0]]:.4f}",
            f"  max pre-clip grad norm in 20 steps before trigger: {spike:.3f}",
            f"  -> {'KICK (grad spike precedes crash)' if spike and spike > 5 else 'EROSION (no large spike; gradual decline)'}",
        ]
    else:
        lines += ["No collapse detected (raw std stayed > 1e-2 all run)."]
    summary = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "collapse_probe_adamw1.txt"), "w") as f:
        f.write(summary + "\n")
    print("\n" + summary)
    print(f"\nSaved {plot_path}")


if __name__ == "__main__":
    main()
