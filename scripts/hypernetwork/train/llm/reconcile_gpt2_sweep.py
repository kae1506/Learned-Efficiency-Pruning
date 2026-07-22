"""
One-off reconciliation: merge v1 (re-evaluated under the new sliding-window
protocol, via reeval_gpt2_checkpoints.py -> reeval.csv) with v2 (already run
under the new protocol) into one λ=0.01->3.2 curve, compute efficiency =
pct_pruned / exp(ΔCE) for every point, and find the peak.

Run from experiments/latest/gpt2_results/ (or pass --v1_dir/--v2_dir):
    python reconcile_gpt2_sweep.py
"""
import argparse
import csv
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_v1_reeval(v1_dir):
    path = os.path.join(v1_dir, "reeval.csv")
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "lambda": float(r["lambda"]), "seed": int(r["seed"]),
                "pct_pruned": float(r["pct_pruned"]),
                "orig_ppl": float(r["orig_ppl_new_protocol"]),
                "pruned_ppl": float(r["pruned_ppl_new_protocol"]),
                "source": "v1 (re-evaluated)",
            })
    return rows


def load_v2_summary(v2_dir):
    path = os.path.join(v2_dir, "summary.txt")
    pat = re.compile(r"^\s*([\d.]+)\s+(\d+)\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|")
    rows = []
    with open(path) as f:
        for line in f:
            m = pat.match(line)
            if m:
                rows.append({
                    "lambda": float(m.group(1)), "seed": int(m.group(2)),
                    "pct_pruned": float(m.group(3)),
                    "orig_ppl": float(m.group(4)),
                    "pruned_ppl": float(m.group(5)),
                    "source": "v2 (native)",
                })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1_dir", default="v1")
    ap.add_argument("--v2_dir", default="v2")
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()

    rows = load_v1_reeval(args.v1_dir) + load_v2_summary(args.v2_dir)
    for r in rows:
        r["delta_ce"] = float(np.log(r["pruned_ppl"] / r["orig_ppl"]))
        r["efficiency"] = r["pct_pruned"] / float(np.exp(r["delta_ce"]))
    rows.sort(key=lambda r: (r["lambda"], r["seed"]))

    # per-lambda aggregate (mean over seeds)
    lambdas = sorted(set(r["lambda"] for r in rows))
    agg = []
    for lam in lambdas:
        rs = [r for r in rows if r["lambda"] == lam]
        agg.append({
            "lambda": lam,
            "pct_pruned_mean": float(np.mean([r["pct_pruned"] for r in rs])),
            "delta_ce_mean": float(np.mean([r["delta_ce"] for r in rs])),
            "efficiency_mean": float(np.mean([r["efficiency"] for r in rs])),
            "source": rs[0]["source"],
        })

    # ---- table ----
    print(f"{'lambda':>7} {'source':>18} | {'%pruned':>8} | {'ΔCE':>8} | {'efficiency':>10}")
    print("-" * 62)
    for a in agg:
        print(f"{a['lambda']:>7} {a['source']:>18} | {a['pct_pruned_mean']:>7.2f}% | "
              f"{a['delta_ce_mean']:>+8.4f} | {a['efficiency_mean']:>10.2f}")

    peak = max(agg, key=lambda a: a["efficiency_mean"])
    print("\n" + "=" * 62)
    print(f"PEAK EFFICIENCY: λ={peak['lambda']}  ->  {peak['pct_pruned_mean']:.2f}% pruned, "
          f"efficiency={peak['efficiency_mean']:.2f}  ({peak['source']})")
    print("=" * 62)

    # ---- csv ----
    out_csv = os.path.join(args.out_dir, "reconciled_gpt2_sweep.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {out_csv}")

    # ---- graph ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    v1_agg = [a for a in agg if a["source"].startswith("v1")]
    v2_agg = [a for a in agg if a["source"].startswith("v2")]

    ax = axes[0]
    ax.plot([a["lambda"] for a in v1_agg], [a["pct_pruned_mean"] for a in v1_agg],
            "o-", color="steelblue", label="v1 (re-evaluated, new protocol)")
    ax.plot([a["lambda"] for a in v2_agg], [a["pct_pruned_mean"] for a in v2_agg],
            "o-", color="tomato", label="v2 (native, new protocol)")
    ax.set_xscale("log"); ax.set_xlabel("λ (log scale)"); ax.set_ylabel("% pruned")
    ax.set_title("% pruned vs λ (one consistent eval protocol)")
    ax.grid(alpha=0.3, which="both"); ax.legend()

    ax = axes[1]
    ax.plot([a["lambda"] for a in v1_agg], [a["efficiency_mean"] for a in v1_agg],
            "o-", color="steelblue", label="v1 (re-evaluated, new protocol)")
    ax.plot([a["lambda"] for a in v2_agg], [a["efficiency_mean"] for a in v2_agg],
            "o-", color="tomato", label="v2 (native, new protocol)")
    ax.axvline(peak["lambda"], color="darkorange", ls="--", lw=1.2,
               label=f"peak: λ={peak['lambda']}")
    ax.set_xscale("log"); ax.set_xlabel("λ (log scale)")
    ax.set_ylabel("efficiency = %pruned / exp(ΔCE)")
    ax.set_title("Reconciled efficiency curve, λ=0.01→3.2, one protocol")
    ax.grid(alpha=0.3, which="both"); ax.legend()

    fig.tight_layout()
    out_png = os.path.join(args.out_dir, "reconciled_gpt2_sweep.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
