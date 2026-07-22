"""
GPT-2 small vs OPT-125M — % pruned vs CE, one plot, RAW (unnormalized) CE on
both curves. Deliberately does NOT rescale/shift either curve to a common
baseline -- the point is to see the two models' absolute CE values differ
(OPT-125M's WikiText-2 CE is much worse than GPT-2's at every sparsity,
including 0% -- see diary/crisp-findings.md F19 discussion of why).

Marks the 0%-pruned (original, unpruned) CE per model as a star, deliberately
NOT connected into the pruned-points line -- keeps "what pruning costs you"
(the line) visually separate from "where each model starts" (the star). CE
there = ln(mean orig_ppl) for that model, averaged across all that model's
runs (GPT-2's v1-reeval vs v2-native baselines differ by ~1%, a known
precision artifact -- see F18 appendix, negligible against what's being
compared here).

Reads the already-reconciled sweep CSVs (no re-eval, no training):
    experiments/latest/gpt2_results/reconciled_gpt2_sweep.csv
    experiments/latest/opt125m_results/reconciled_opt125m_sweep.csv

Usage (defaults assume running from experiments/latest/):
    python compare_gpt2_opt125m.py \
        [--gpt2_csv gpt2_results/reconciled_gpt2_sweep.csv] \
        [--opt125m_csv opt125m_results/reconciled_opt125m_sweep.csv] \
        [--out_dir .]
"""
import argparse
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_curve(csv_path):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "lambda": float(r["lambda"]),
                "pct_pruned": float(r["pct_pruned"]),
                "orig_ppl": float(r["orig_ppl"]),
                "pruned_ppl": float(r["pruned_ppl"]),
            })

    orig_ce_mean = float(np.mean([np.log(r["orig_ppl"]) for r in rows]))

    lambdas = sorted(set(r["lambda"] for r in rows))
    points = []   # pruned points only -- the 0%-pruned anchor is marked separately, not on this line
    for lam in lambdas:
        rs = [r for r in rows if r["lambda"] == lam]
        pct_pruned_mean = float(np.mean([r["pct_pruned"] for r in rs]))
        ce_mean = float(np.mean([np.log(r["pruned_ppl"]) for r in rs]))
        points.append((pct_pruned_mean, ce_mean))

    points.sort(key=lambda p: p[0])
    return points, orig_ce_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt2_csv",    default="gpt2_results/reconciled_gpt2_sweep.csv")
    ap.add_argument("--opt125m_csv", default="opt125m_results/reconciled_opt125m_sweep.csv")
    ap.add_argument("--out_dir",     default=".")
    args = ap.parse_args()

    gpt2_points, gpt2_orig_ce = load_curve(args.gpt2_csv)
    opt_points,  opt_orig_ce  = load_curve(args.opt125m_csv)

    print(f"GPT-2 small   orig CE = {gpt2_orig_ce:.4f} nats  (ppl = {np.exp(gpt2_orig_ce):.3f})")
    print(f"OPT-125M      orig CE = {opt_orig_ce:.4f} nats  (ppl = {np.exp(opt_orig_ce):.3f})")
    print(f"gap at 0% pruned: {opt_orig_ce - gpt2_orig_ce:+.4f} nats "
          f"(OPT-125M {np.exp(opt_orig_ce - gpt2_orig_ce):.2f}x GPT-2's perplexity)")

    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    gx, gy = zip(*gpt2_points)
    ox, oy = zip(*opt_points)
    ax.plot(gx, gy, "o-", color="steelblue", lw=1.8, markersize=7, label="GPT-2 small")
    ax.plot(ox, oy, "o-", color="tomato",    lw=1.8, markersize=7, label="OPT-125M")

    # 0%-pruned (original, unpruned) points -- marked only, deliberately NOT
    # connected into the pruned-points line above.
    ax.scatter([0], [gpt2_orig_ce], marker="*", s=500, color="steelblue", zorder=5,
              edgecolors="black", linewidths=1.2, label="GPT-2 unpruned (0%)")
    ax.scatter([0], [opt_orig_ce], marker="*", s=500, color="tomato", zorder=5,
              edgecolors="black", linewidths=1.2, label="OPT-125M unpruned (0%)")
    ax.annotate(f"GPT-2 unpruned\nCE={gpt2_orig_ce:.3f}", (0, gpt2_orig_ce),
               xytext=(12, -8), textcoords="offset points", fontsize=8)
    ax.annotate(f"OPT-125M unpruned\nCE={opt_orig_ce:.3f}", (0, opt_orig_ce),
               xytext=(12, 8), textcoords="offset points", fontsize=8)

    ax.set_xlabel("% pruned")
    ax.set_ylabel("CE (nats) — raw, NOT normalized between models")
    ax.set_title("GPT-2 small vs OPT-125M — % pruned vs CE (WikiText-2)", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_png = f"{args.out_dir}/gpt2_vs_opt125m_pruned_vs_ce.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_png}")


if __name__ == "__main__":
    main()
