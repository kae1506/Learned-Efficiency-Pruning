"""
Pareto-style plot for the CNN/DailyMail sweep (λ=0.05, 0.1, 0.3, 0.8; seed=0):
x = % FFN neurons pruned, y = ROUGE-L relative % change (pruned vs orig,
(pruned-orig)/orig*100). Reads directly from each lambda_*/summary.txt under
experiments/latest/llama2_7b_cnndailymail/ -- no hardcoded numbers, so this
stays correct if more points are added later.

Y-axis metric confirmed with the user (2026-08-19): relative % change, not
absolute percentage-point change -- standard reading of "percentage change"
given ROUGE-L is itself already a percentage-like metric.
"""
import os
import re
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "experiments", "latest", "llama2_7b_cnndailymail")

_PATTERNS = {
    "pct_pruned": r"final % FFN neurons pruned\s*:\s*([\d.]+)%",
    "rouge_orig_rougeL": r"original\s+rouge1/rouge2/rougeL\s*:\s*[\d.]+/[\d.]+/([\d.]+)",
    "rouge_pruned_rougeL": r"pruned\s+rouge1/rouge2/rougeL\s*:\s*[\d.]+/[\d.]+/([\d.]+)",
}


def load_points(out_dir):
    points = []
    for d in sorted(glob.glob(os.path.join(out_dir, "lambda_*"))):
        summary_path = os.path.join(d, "summary.txt")
        if not os.path.isdir(d) or not os.path.exists(summary_path):
            continue
        lam = float(os.path.basename(d).removeprefix("lambda_"))
        text = open(summary_path).read()
        vals = {}
        for key, pat in _PATTERNS.items():
            m = re.search(pat, text)
            if not m:
                print(f"  WARNING: could not parse {key!r} out of {summary_path}, skipping")
                vals = None
                break
            vals[key] = float(m.group(1))
        if vals is None:
            continue
        pct_pruned = vals["pct_pruned"]
        orig, pruned = vals["rouge_orig_rougeL"], vals["rouge_pruned_rougeL"]
        rel_change = (pruned - orig) / orig * 100
        points.append({"lambda": lam, "pct_pruned": pct_pruned, "orig": orig,
                       "pruned": pruned, "rel_change": rel_change})
    points.sort(key=lambda p: p["pct_pruned"])
    return points


def plot(points, save_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = [p["pct_pruned"] for p in points]
    ys = [p["rel_change"] for p in points]
    ax.plot(xs, ys, color="steelblue", lw=1.5, marker="o", markersize=8, zorder=3)
    ax.axhline(0, color="gray", ls="--", lw=0.8, zorder=1)
    for p in points:
        ax.annotate(f"λ={p['lambda']:g}", (p["pct_pruned"], p["rel_change"]),
                   textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.set_xlabel("% FFN neurons pruned")
    ax.set_ylabel("ROUGE-L relative change vs dense (%)\n(pruned − orig) / orig × 100")
    ax.set_title("Llama-2-7B — CNN/DailyMail — pruning vs ROUGE-L change\n"
                 "(all 4 points hit max_steps=12000 uncoverged -- see summary.txt)",
                 fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {save_path}")


def main():
    points = load_points(OUT_DIR)
    if not points:
        print("No lambda_*/summary.txt found under", OUT_DIR)
        return
    print(f"{'lambda':>7} | {'% pruned':>9} | {'orig R-L':>9} | {'pruned R-L':>10} | {'rel change':>10}")
    for p in points:
        print(f"{p['lambda']:>7} | {p['pct_pruned']:>8.2f}% | {p['orig']:>8.2f}% | "
              f"{p['pruned']:>9.2f}% | {p['rel_change']:>+9.2f}%")
    save_path = os.path.join(OUT_DIR, "pareto_pruned_vs_rougeL_change.png")
    plot(points, save_path)


if __name__ == "__main__":
    main()
