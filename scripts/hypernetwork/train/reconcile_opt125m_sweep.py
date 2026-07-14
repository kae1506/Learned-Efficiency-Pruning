"""
Reconcile OPT-125M v1 (λ=0.01-0.40) and v2 (λ=0.55-1.8) sweeps into one
λ=0.01->1.8 curve. Both were run under the same eval protocol
(train_pruner_opt125m.py's evaluate(), max_length=2048/stride=1024), so
unlike the GPT-2 reconciliation this needs no re-eval step first -- just
parse both sets of per-run summary.txt files directly and merge.

Produces:
  reconciled_opt125m_sweep.csv        per-run rows: lambda, seed, source,
                                       pct_pruned, orig_ppl, pruned_ppl, ce,
                                       delta_ce, efficiency
  opt125m_lambda_vs_efficiency.png    λ vs efficiency (log-x)
  opt125m_pruned_vs_loss.png          % pruned vs CE (nats), orig-CE line
  opt125m_pruned_vs_ppl.png           % pruned vs perplexity, orig-ppl line

efficiency = %pruned / exp(ΔCE), ΔCE = ln(pruned_ppl / orig_ppl) -- same
formula as train_pruner_opt125m.py's plot_efficiency() / the GPT-2
reconciliation, for consistency across the project.

Run from experiments/latest/opt125m_results/ (or pass --v1_dir/--v2_dir):
    python reconcile_opt125m_sweep.py
"""
import argparse
import csv
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_sweep(sweep_dir, source_label):
    rows = []
    for path in sorted(glob.glob(os.path.join(sweep_dir, "lambda_*", "seed_*", "summary.txt"))):
        text = open(path).read()
        header = re.search(r"λ=([\d.]+),\s*seed=(\d+)", text)
        lam, seed = float(header.group(1)), int(header.group(2))
        pct_pruned = float(re.search(r"final % FFN neurons pruned\s*:\s*([\d.]+)%", text).group(1))
        orig_ppl   = float(re.search(r"original\s+ppl\s*:\s*([\d.]+)", text).group(1))
        pruned_ppl = float(re.search(r"pruned\s+ppl\s*:\s*([\d.]+)", text).group(1))
        rows.append({
            "lambda": lam, "seed": seed, "source": source_label,
            "pct_pruned": pct_pruned, "orig_ppl": orig_ppl, "pruned_ppl": pruned_ppl,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1_dir",  default="opt125m_lambda_sweep")
    ap.add_argument("--v2_dir",  default="v2/opt125m_lambda_sweep_v2")
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()

    rows = load_sweep(args.v1_dir, "v1") + load_sweep(args.v2_dir, "v2")
    if not rows:
        raise SystemExit(f"No summary.txt files found under {args.v1_dir} or {args.v2_dir} "
                         f"-- check --v1_dir/--v2_dir or run this from experiments/latest/opt125m_results/.")

    orig_ppls = sorted(set(round(r["orig_ppl"], 3) for r in rows))
    if len(orig_ppls) > 1:
        print(f"WARNING: orig_ppl is not constant across runs: {orig_ppls} "
              f"-- eval protocol may differ between v1/v2, treat results with caution.\n")
    orig_ppl = rows[0]["orig_ppl"]
    orig_ce  = float(np.log(orig_ppl))

    for r in rows:
        r["ce"]         = float(np.log(r["pruned_ppl"]))
        r["delta_ce"]   = r["ce"] - orig_ce
        r["efficiency"] = r["pct_pruned"] / float(np.exp(r["delta_ce"]))
    rows.sort(key=lambda r: (r["lambda"], r["seed"]))

    # per-lambda aggregate (mean +/- std over seeds; std=0 where only 1 seed exists)
    lambdas = sorted(set(r["lambda"] for r in rows))
    agg = []
    for lam in lambdas:
        rs = [r for r in rows if r["lambda"] == lam]
        agg.append({
            "lambda":          lam,
            "source":          rs[0]["source"],
            "n_seeds":         len(rs),
            "pct_pruned_mean": float(np.mean([r["pct_pruned"] for r in rs])),
            "pct_pruned_std":  float(np.std([r["pct_pruned"]  for r in rs])),
            "ce_mean":         float(np.mean([r["ce"] for r in rs])),
            "ce_std":          float(np.std([r["ce"]  for r in rs])),
            "pruned_ppl_mean": float(np.mean([r["pruned_ppl"] for r in rs])),
            "pruned_ppl_std":  float(np.std([r["pruned_ppl"]  for r in rs])),
            "efficiency_mean": float(np.mean([r["efficiency"] for r in rs])),
            "efficiency_std":  float(np.std([r["efficiency"]  for r in rs])),
        })

    # ---- table ----
    print(f"orig ppl = {orig_ppl:.3f}  (orig CE = {orig_ce:.4f} nats)\n")
    print(f"{'lambda':>7} {'src':>4} {'n':>2} | {'%pruned':>8} | {'CE':>8} | {'ppl':>9} | {'efficiency':>10}")
    print("-" * 68)
    for a in agg:
        print(f"{a['lambda']:>7} {a['source']:>4} {a['n_seeds']:>2} | "
              f"{a['pct_pruned_mean']:>7.2f}% | {a['ce_mean']:>8.4f} | "
              f"{a['pruned_ppl_mean']:>9.3f} | {a['efficiency_mean']:>10.2f}")

    peak = max(agg, key=lambda a: a["efficiency_mean"])
    print("\n" + "=" * 68)
    print(f"PEAK EFFICIENCY: λ={peak['lambda']}  ->  {peak['pct_pruned_mean']:.2f}% pruned, "
          f"efficiency={peak['efficiency_mean']:.2f}  ({peak['source']})")
    print("=" * 68)

    # ---- csv (per-run, not aggregated) ----
    out_csv = os.path.join(args.out_dir, "reconciled_opt125m_sweep.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {out_csv}")

    colors = {"v1": "steelblue", "v2": "tomato"}
    v1_agg = [a for a in agg if a["source"] == "v1"]
    v2_agg = [a for a in agg if a["source"] == "v2"]

    def plot_series(ax, xkey, ykey, xerr_key, yerr_key, annotate=False):
        for grp, label in [(v1_agg, "v1"), (v2_agg, "v2")]:
            if not grp:
                continue
            xs   = [a[xkey] for a in grp]
            ys   = [a[ykey] for a in grp]
            xerr = [a[xerr_key] for a in grp] if xerr_key else None
            yerr = [a[yerr_key] for a in grp] if yerr_key else None
            ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, fmt="o-", color=colors[label],
                        capsize=4, lw=1.5, markersize=7, label=label)
            if annotate:
                for a, x, y in zip(grp, xs, ys):
                    ax.annotate(f"λ={a['lambda']}", (x, y), xytext=(6, 4),
                               textcoords="offset points", fontsize=7)

    # ---- plot 1: lambda vs efficiency ----
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    plot_series(ax, "lambda", "efficiency_mean", None, "efficiency_std")
    ax.axvline(peak["lambda"], color="darkorange", ls="--", lw=1.2, label=f"peak: λ={peak['lambda']}")
    ax.set_xscale("log"); ax.set_xlabel("λ (log scale)")
    ax.set_ylabel("efficiency = %pruned / exp(ΔCE)")
    ax.set_title("OPT-125M — efficiency vs λ (v1+v2 reconciled)", fontweight="bold")
    ax.grid(alpha=0.3, which="both"); ax.legend()
    fig.tight_layout()
    p1 = os.path.join(args.out_dir, "opt125m_lambda_vs_efficiency.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {p1}")

    # ---- plot 2: % pruned vs loss (CE), orig-CE reference line ----
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    plot_series(ax, "pct_pruned_mean", "ce_mean", "pct_pruned_std", "ce_std", annotate=True)
    ax.axhline(orig_ce, color="gray", ls="--", lw=1.2, label=f"orig CE = {orig_ce:.3f}")
    ax.set_xlabel("% FFN neurons pruned"); ax.set_ylabel("pruned CE (nats)")
    ax.set_title("OPT-125M — % pruned vs loss (v1+v2 reconciled)", fontweight="bold")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    p2 = os.path.join(args.out_dir, "opt125m_pruned_vs_loss.png")
    fig.savefig(p2, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {p2}")

    # ---- plot 3: % pruned vs perplexity, orig-ppl reference line ----
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    plot_series(ax, "pct_pruned_mean", "pruned_ppl_mean", "pct_pruned_std", "pruned_ppl_std", annotate=True)
    ax.axhline(orig_ppl, color="gray", ls="--", lw=1.2, label=f"orig ppl = {orig_ppl:.2f}")
    ax.set_xlabel("% FFN neurons pruned"); ax.set_ylabel("pruned perplexity")
    ax.set_title("OPT-125M — % pruned vs perplexity (v1+v2 reconciled)", fontweight="bold")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    p3 = os.path.join(args.out_dir, "opt125m_pruned_vs_ppl.png")
    fig.savefig(p3, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {p3}")


if __name__ == "__main__":
    main()
