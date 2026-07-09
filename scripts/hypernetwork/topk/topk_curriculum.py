"""
STE-top-K curriculum — v2 (centered STE + per-stage temperature anneal +
per-layer node-score standardisation).

First run (plain STE, early-stop) underperformed badly (41pp @ K=492 vs the λ
floor's 2pp). The diagnostic (topk_diagnostic.py) showed:
  - the curriculum was NOT the culprit — one-shot plain top-K @492 was also 43pp;
  - the STE surrogate was: plain σ(s) puts the gradient peak at s=0, but the
    top-K keep/kill boundary is the K-th-largest score, far from 0 -> borderline
    neurons sit on σ's saturated tail -> frozen ranking.
  - CENTERED STE σ((s-thresh)/T) fixed the moderate regime (50% pruned: 13.9 ->
    2.1pp) but COLD-collapsed at aggressive K (76% pruned: 88pp) with T=1.

This v2 folds in the fixes:
  - centered STE (σ((s-thresh)/T));
  - temperature ANNEAL T: 4 -> 1 within each stage (start wide so the whole
    ranking can still move, narrow down to commit) — dodges the cold collapse;
  - per-layer standardisation of the node scores (zero-mean/unit-var) so the
    main path can't drift/saturate and layers are comparable for global top-K;
  - warm-start + re-rank across decreasing K (your 90->...->18% scheme),
    fresh optimiser per stage, FIXED step budget per stage (more for cliff
    stages) so the T-anneal has a well-defined horizon.

Run:  venv/bin/python scripts/hypernetwork/topk_curriculum.py
"""

import os
import sys
import time
import random

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(".")
from src.model import MLP
from src.dataset import get_mnist_loaders
from src.pruners.bilstm_topk import TopKPruner
from src.topk_train import topk_pruner_step
from src.prune_train import get_hidden_weights
from src.interpretability import evaluate_with_gates

# ── config ──────────────────────────────────────────────────────────────────
CKPT          = "experiments/checkpoints/mnist_model.pt"   # medium [1024,1024]
OUT_DIR       = "experiments/latest/hypernetwork/topk_curriculum"
CONFIG_PATH   = "configs/config.yaml"

SCHEDULE_PCT  = [90, 75, 60, 50, 42, 36, 31, 27, 24, 21, 18]   # % kept
SEEDS         = [0, 1]

SAMPLES       = 64
BASE_LR       = 1e-3
CLIFF_LR      = 5e-4
CLIFF_FRAC_PCT = 27          # stages <= this use CLIFF_LR

T_START, T_END = 4.0, 1.0    # per-stage temperature anneal

STEPS_EASY    = 300          # stages with %kept > CLIFF_STEPS_PCT
STEPS_CLIFF   = 600          # stages with %kept <= CLIFF_STEPS_PCT (need more)
CLIFF_STEPS_PCT = 36

# λ-pruner reference (medium, iso_accuracy_retrain): survivors @2pp.
LAMBDA_SURVIVORS_2PP  = 490
LAMBDA_PCT_PRUNED_2PP = 76.1

# previous plain-STE curriculum (for visual reference): (%pruned, drop pp)
PREV_PLAIN = [(10, -0.11), (25, 0.81), (40, 1.92), (50, 4.85), (58, 12.75),
              (64, 19.83), (69, 26.07), (73, 35.76), (76, 40.86), (79, 50.12),
              (82, 57.55)]


def set_seed(s):
    torch.manual_seed(s); np.random.seed(s); random.seed(s)


def load_frozen(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    m = MLP(**ckpt["config"]).to(device)
    m.load_state_dict(ckpt["state_dict"]); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def train_stage(pruner, model, opt, train_iter, train_loader, k, lr, stage_steps,
                cum_start, total_steps):
    """Train one keep-budget K (warm-started). T anneals GLOBALLY across the
    whole curriculum (cumulative step index) — it NEVER resets per stage, so
    late/cliff stages stay in the narrow band where the committed scores live.
    `opt` is CARRIED across stages (Adam m/v moments persist) — only its lr is
    reset per stage. This removes the fresh-Adam boundary kick (the large,
    undamped first step on a committed θ that knocked v3 into the tie-attractor)."""
    for pg in opt.param_groups:
        pg["lr"] = lr                                    # per-stage lr; moments carried
    pruner.train()
    ema, trace = None, []
    for step in range(stage_steps):
        g = cum_start + step                             # global step index
        T = T_START + (T_END - T_START) * (g / (total_steps - 1))   # global 4 -> 1
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader); x, y = next(train_iter)
        x, y = x[:SAMPLES].to(DEVICE), y[:SAMPLES].to(DEVICE)
        m = topk_pruner_step(pruner, model, opt, x, y, k, temp=T)
        ema = m["ce_drop"] if ema is None else 0.9 * ema + 0.1 * m["ce_drop"]
        trace.append(ema)
    return train_iter, trace


@torch.no_grad()
def eval_stage(pruner, model, test_loader, k, baseline_acc):
    """Hard top-K mask (T-independent) -> full-test drop + survivors."""
    pruner.eval()
    gates = pruner(get_hidden_weights(model), k)         # default temp; hard mask
    survivors = int(sum(int(g.sum().item()) for g in gates))
    test_acc = evaluate_with_gates(model, gates, test_loader, DEVICE)
    n_total = int(sum(int(g.numel()) for g in gates))
    return {"k": k, "survivors": survivors, "n_total": n_total,
            "frac_pruned": 1 - survivors / n_total, "drop": (baseline_acc - test_acc) * 100}


def run_seed(seed, model, train_loader, test_loader, n_total, baseline_acc):
    set_seed(seed)
    layer_shapes = [(w.shape[0], w.shape[1]) for w in get_hidden_weights(model)]
    # Best stable top-K config: center-only node norm (no scale-invariance freeze)
    # + tanh-bounded context bias (no layer-starvation). Adam carried across stages.
    pruner = TopKPruner(layer_shapes, use_layernorm=True,
                        node_norm="center", bound_context=True).to(DEVICE)
    opt = torch.optim.Adam(pruner.parameters(), lr=BASE_LR)   # ONE optimiser, carried
    train_iter = iter(train_loader)
    rows, traces = [], []
    # Total curriculum steps (for the GLOBAL temperature anneal horizon).
    total_steps = sum(STEPS_CLIFF if pct <= CLIFF_STEPS_PCT else STEPS_EASY
                      for pct in SCHEDULE_PCT)
    cum = 0
    for pct in SCHEDULE_PCT:
        k = round(pct / 100 * n_total)
        lr = CLIFF_LR if pct <= CLIFF_FRAC_PCT else BASE_LR
        steps = STEPS_CLIFF if pct <= CLIFF_STEPS_PCT else STEPS_EASY
        t_lo = T_START + (T_END - T_START) * (cum / (total_steps - 1))
        t_hi = T_START + (T_END - T_START) * ((cum + steps - 1) / (total_steps - 1))
        t0 = time.time()
        train_iter, trace = train_stage(pruner, model, opt, train_iter, train_loader,
                                        k, lr, steps, cum, total_steps)
        cum += steps
        r = eval_stage(pruner, model, test_loader, k, baseline_acc)
        r["pct_kept"], r["steps"], r["secs"] = pct, steps, time.time() - t0
        rows.append(r); traces.append((pct, trace))
        print(f"  seed{seed} keep {pct:>2}% (K={k:>4}): survivors={r['survivors']:>4} "
              f"pruned={r['frac_pruned']*100:5.1f}% drop={r['drop']:6.2f}pp "
              f"T={t_lo:.2f}->{t_hi:.2f} steps={steps} [{r['secs']:4.1f}s]", flush=True)
    return rows, traces


def surv_at_drop(pruned_pct, survivors, drops, target=2.0):
    for i in range(len(drops) - 1):
        d0, d1 = drops[i], drops[i + 1]
        if (d0 - target) * (d1 - target) <= 0 and d0 != d1:
            t = (target - d0) / (d1 - d0)
            return (survivors[i] + t * (survivors[i + 1] - survivors[i]),
                    pruned_pct[i] + t * (pruned_pct[i + 1] - pruned_pct[i]))
    return None, None


def main():
    global DEVICE
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}  |  centered STE, T {T_START}->{T_END}/stage  |  "
          f"schedule(%kept)={SCHEDULE_PCT}  |  seeds={SEEDS}\n")

    train_loader, test_loader = get_mnist_loaders(**cfg["data"])
    model = load_frozen(CKPT, DEVICE)
    n_total = int(sum(int(w.shape[0]) for w in get_hidden_weights(model)))
    full = [torch.ones(w.shape[0], device=DEVICE) for w in get_hidden_weights(model)]
    baseline_acc = evaluate_with_gates(model, full, test_loader, DEVICE)
    print(f"Baseline acc: {baseline_acc*100:.2f}%   n_hidden={n_total}\n")

    all_rows, seed0_traces = {}, None
    for s in SEEDS:
        rows, traces = run_seed(s, model, train_loader, test_loader, n_total, baseline_acc)
        all_rows[s] = rows
        if s == SEEDS[0]:
            seed0_traces = traces

    pct_kept   = SCHEDULE_PCT
    pruned_pct = [np.mean([all_rows[s][i]["frac_pruned"] * 100 for s in SEEDS]) for i in range(len(pct_kept))]
    drop_mean  = [np.mean([all_rows[s][i]["drop"] for s in SEEDS]) for i in range(len(pct_kept))]
    drop_lo    = [np.min ([all_rows[s][i]["drop"] for s in SEEDS]) for i in range(len(pct_kept))]
    drop_hi    = [np.max ([all_rows[s][i]["drop"] for s in SEEDS]) for i in range(len(pct_kept))]
    surv_mean  = [np.mean([all_rows[s][i]["survivors"] for s in SEEDS]) for i in range(len(pct_kept))]
    surv2pp, pp2pp = surv_at_drop(pruned_pct, surv_mean, drop_mean, 2.0)

    # ── plot ─────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot([p for p, _ in PREV_PLAIN], [d for _, d in PREV_PLAIN], "s--",
             color="#bbbbbb", lw=1.5, ms=5, label="v1 plain STE (reference)")
    ax1.fill_between(pruned_pct, drop_lo, drop_hi, color="#2980b9", alpha=0.18)
    ax1.plot(pruned_pct, drop_mean, "o-", color="#2980b9", lw=2, ms=7,
             label="center+tanh (best top-K, mean 2 seeds)")
    ax1.axhline(2.0, color="k", ls="--", alpha=0.6, label="2pp budget")
    ax1.scatter([LAMBDA_PCT_PRUNED_2PP], [2.0], marker="*", s=320, color="#e67e22",
                zorder=5, label=f"λ floor: {LAMBDA_SURVIVORS_2PP} surv @2pp")
    if surv2pp is not None:
        ax1.scatter([pp2pp], [2.0], marker="D", s=90, color="#27ae60", zorder=6,
                    label=f"v2 @2pp: {surv2pp:.0f} surv ({pp2pp:.1f}% pruned)")
    for x, y, pc in zip(pruned_pct, drop_mean, pct_kept):
        ax1.annotate(f"{pc}%", (x, y), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8, alpha=0.7)
    ax1.set_xlabel("Neurons pruned (%)"); ax1.set_ylabel("Full-test accuracy drop (pp)")
    ax1.set_title("STE-top-K curriculum v2 vs v1 vs λ floor — medium [1024,1024]",
                  fontweight="bold")
    ax1.legend(fontsize=9, loc="upper left"); ax1.grid(alpha=0.3)
    ax1.set_ylim(-2, max(60, max(drop_mean) + 5))

    cmap = plt.cm.viridis(np.linspace(0, 1, len(seed0_traces)))
    for (pct, trace), c in zip(seed0_traces, cmap):
        ax2.plot(trace, color=c, lw=1.4, label=f"{pct}%")
    ax2.set_xlabel("Step within stage"); ax2.set_ylabel("EMA(CE_pruned − CE_orig)")
    ax2.set_title("Per-stage convergence (seed 0) — T anneals 4→1", fontweight="bold")
    ax2.legend(fontsize=7, ncol=2, title="% kept"); ax2.grid(alpha=0.3)

    fig.tight_layout()
    plot_path = os.path.join(OUT_DIR, "plot_final.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight"); plt.close(fig)

    # ── summary ──────────────────────────────────────────────────────────────
    lines = [
        "=" * 72,
        "STE-TOP-K CURRICULUM v2 — centered STE + T-anneal + node standardisation",
        "medium [1024,1024] MNIST MLP",
        "=" * 72,
        f"Baseline acc      : {baseline_acc*100:.2f}%    hidden={n_total}",
        f"Schedule (% kept) : {SCHEDULE_PCT}    seeds={SEEDS}",
        f"STE               : centered σ((s-thresh)/T), T anneals {T_START}->{T_END} GLOBALLY (whole curriculum)",
        f"Node scores       : per-layer CENTER-only (no /σ; avoids scale-invariance freeze)",
        f"Context bias      : tanh-bounded (avoids layer-starvation)",
        f"Optimizer         : Adam CARRIED across stages (m/v persist; lr reset per stage)",
        f"Hacks dropped     : +2.0 bias, tanh   (LayerNorm on context kept)",
        f"Steps/stage       : {STEPS_EASY} (easy) / {STEPS_CLIFF} (cliff, %kept<={CLIFF_STEPS_PCT})",
        "-" * 72,
        f"{'%kept':>6} | {'K':>5} | {'survivors':>9} | {'%pruned':>8} | {'drop(pp)':>11}",
        "-" * 72,
    ]
    for i, pc in enumerate(pct_kept):
        band = (drop_hi[i] - drop_lo[i]) / 2
        lines.append(f"{pc:>6} | {round(pc/100*n_total):>5} | {surv_mean[i]:>9.0f} | "
                     f"{pruned_pct[i]:>7.1f}% | {drop_mean[i]:>6.2f}±{band:>4.2f}")
    lines += ["-" * 72]
    if surv2pp is not None:
        lines += [
            "ISO-ACCURACY @2pp (interpolated):",
            f"  center+tanh    : {surv2pp:.0f} survivors  ({pp2pp:.1f}% pruned)",
            f"  λ floor (ref)  : {LAMBDA_SURVIVORS_2PP} survivors  ({LAMBDA_PCT_PRUNED_2PP:.1f}% pruned)",
            f"  center vs λ    : {surv2pp - LAMBDA_SURVIVORS_2PP:+.0f} survivors "
            f"({'MATCHES' if abs(surv2pp - LAMBDA_SURVIVORS_2PP) <= 40 else 'differs from'} λ floor)",
        ]
    else:
        lines += ["ISO-ACCURACY @2pp: schedule did not bracket 2pp."]
    lines.append("=" * 72)
    summary = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "summary_final.txt"), "w") as f:
        f.write(summary + "\n")
    print("\n" + summary)
    print(f"\nSaved: {plot_path}")


if __name__ == "__main__":
    main()
