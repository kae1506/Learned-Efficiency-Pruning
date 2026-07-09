"""
Pruning efficiency vs λ — 5-net head-to-head (LeNet / MNIST-wide / MNIST-medium / MNIST-deep / CIFAR_big).

Reads existing summary.txt files (no new training). One figure, log-log scale.
Per-net annotations: λ_opt^overall (filled circle + dashed line) and "almost-optimal"
band (10% of peak, shaded marker).

Sources (all 15-epoch where multi-seed; 5-epoch where labelled):
  LeNet               cifar_lenet_lambda_fine_15ep/   (6 λ × 3 seeds × 15 ep)
                      cifar_lenet_lambda_extra_15ep/  (3 λ × 1 seed × 15 ep)
  MNIST Wide 1L       mnist_wide_sweep_15ep/          (6 λ × 3 seeds × 15 ep)
  MNIST Medium 2L     mnist_lambda_sweep_15ep/        (6 λ × 3 seeds × 15 ep)
  MNIST Deep 4L       mnist_deep_sweep_15ep/          (6 λ × 3 seeds × 15 ep)
  CIFAR_big           cifar_lambda_fine/              (4 λ × 3 seeds × 5 ep)
                      cifar_lambda_sweep/             (3 λ × 1 seed × 5 ep)

efficiency(λ) = (mean % pruned) / max(mean drop_pp, 0.5)

Run from project root:
  venv/bin/python scripts/hypernetwork/efficiency_compare.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


OUT_PATH = "experiments/latest/hypernetwork/efficiency_compare.png"


# (λ,    mean_pct_pruned, mean_drop_pp, n_seed, epochs)
# Multi-seed where seeds≥3; single-seed extras to widen the curve range.

LENET = [
    (0.04,  46.74,   8.66, 3, 15),
    (0.06,  47.79,   8.87, 3, 15),
    (0.08,  47.14,   8.75, 3, 15),
    (0.10,  52.34,  13.28, 3, 15),
    (0.15,  53.26,  13.16, 3, 15),
    (0.20,  52.86,  10.67, 3, 15),
    (0.18,  54.30,  11.60, 1, 15),
    (0.25,  53.52,  10.61, 1, 15),
    (0.30,  55.86,  11.76, 1, 15),
]

MNIST_WIDE = [   # [2048], 1L
    (0.01,  68.64,  -0.12, 3, 15),
    (0.02,  77.73,   0.16, 3, 15),
    (0.03,  79.46,   0.35, 3, 15),
    (0.05,  82.91,   0.61, 3, 15),
    (0.08,  85.16,   0.96, 3, 15),
    (0.12,  87.37,   1.63, 3, 15),
]

MNIST_MEDIUM = [ # [1024,1024], 2L
    (0.04,  64.10,   0.12, 3, 15),
    (0.06,  67.92,   0.30, 3, 15),
    (0.08,  69.35,   0.54, 3, 15),
    (0.10,  70.83,   0.67, 3, 15),
    (0.15,  72.74,   1.01, 3, 15),
    (0.20,  74.54,   1.24, 3, 15),
]

MNIST_DEEP = [   # [512×4], 4L
    (0.05,  40.66,   0.26, 3, 15),
    (0.10,  48.86,   0.07, 3, 15),
    (0.15,  51.84,   0.26, 3, 15),
    (0.25,  54.69,   0.46, 3, 15),
    (0.40,  57.49,   0.77, 3, 15),
    (0.55,  60.03,   1.80, 3, 15),
]

CIFAR_BIG = [
    (0.02,  64.32,   1.67, 3, 5),
    (0.03,  70.90,   1.48, 3, 5),
    (0.05,  74.33,   2.12, 3, 5),
    (0.07,  76.22,   2.52, 3, 5),
    (0.01,  33.14,   1.65, 1, 5),
    (0.10,  77.21,   3.29, 1, 5),
    (0.30,  79.52,   3.91, 1, 5),
]

# Pulled from iso_accuracy_retrain/run.log (2 seeds, 30 ep) — only existing MNIST
# datapoint with hidden_dims well below the over-parameterised regime; useful
# 5th datapoint at N_L=2 for the H2 analysis. Drops are 2-seed means.
MNIST_NARROW = [   # [205,205], 2L, 410 hidden neurons
    (0.01,   7.65,   0.19, 2, 30),
    (0.03,   6.70,   0.19, 2, 30),
    (0.06,  24.00,   1.64, 2, 30),
    (0.10,  33.05,   1.89, 2, 30),
    (0.20,  40.15,   4.68, 2, 30),
]


NETS = [
    # (label,            data,         color,         N_layers, S_layer, n_params)
    ("MNIST Wide [2048]",     MNIST_WIDE,   "#8e44ad",   1, 2048,  1_625_610),
    ("LeNet (CIFAR, 63K)",    LENET,        "darkorange",2,   96,     63_106),
    ("MNIST Narrow [205×2]",  MNIST_NARROW, "#16a085",   2,  205,    202_565),
    ("MNIST Medium [1024×2]", MNIST_MEDIUM, "seagreen",  2, 1024,  1_863_690),
    ("CIFAR_big",             CIFAR_BIG,    "steelblue", 3,  597, 10_379_658),
    ("MNIST Deep [512×4]",    MNIST_DEEP,   "#c0392b",   4,  512,  1_189_898),
]


def compute_eff(rows):
    rows = sorted(rows, key=lambda r: r[0])
    lams   = [r[0] for r in rows]
    eff    = [r[1] / max(r[2], 0.5) for r in rows]
    n_seed = [r[3] for r in rows]
    return lams, eff, n_seed


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(12, 7.5))

    summary_rows = []   # collected for the print table

    for label, rows, color, N_L, S_L, n_params in NETS:
        lams, eff, n_seed = compute_eff(rows)

        # Mark "almost-optimal" band: any point within 10% of peak.
        peak = max(eff)
        almost = [e >= 0.9 * peak for e in eff]
        i_peak = eff.index(peak)

        # Line through all points; filled circles for multi-seed, open for single-seed.
        ax.plot(lams, eff, "-", color=color, lw=1.5, alpha=0.55, zorder=2)
        for lam, e, n, is_alm in zip(lams, eff, n_seed, almost):
            if is_alm and n >= 3:
                ax.plot([lam], [e], "o", color=color, markersize=14,
                        mec="black", mew=1.5, zorder=5)
            elif n >= 3:
                ax.plot([lam], [e], "o", color=color, markersize=9, zorder=4)
            else:
                ax.plot([lam], [e], "o", mfc="white", mec=color, mew=2,
                        markersize=9, zorder=4)

        # Dashed vertical line at the overall peak
        ax.axvline(lams[i_peak], color=color, ls=":", lw=1, alpha=0.5)

        # Label the peak
        ax.annotate(f"{label}\nλ_opt={lams[i_peak]}, eff={peak:.1f}",
                    (lams[i_peak], peak),
                    xytext=(8, 18 if "Wide" in label else (-25 if "LeNet" in label else 14)),
                    textcoords="offset points",
                    fontsize=9, color=color, fontweight="bold",
                    ha="left")

        # Identify almost-optimal band's λ range for the table
        alm_lams = [lams[i] for i, ok in enumerate(almost) if ok]
        alm_min, alm_max = (min(alm_lams), max(alm_lams)) if alm_lams else (lams[i_peak], lams[i_peak])
        summary_rows.append({
            "label": label, "N_L": N_L, "S_L": S_L, "params": n_params,
            "lambda_opt": lams[i_peak], "peak_eff": peak,
            "alm_min": alm_min, "alm_max": alm_max,
            "alm_count": sum(almost),
            "is_bimodal": is_bimodal(lams, eff),
        })

    # Legend
    legend_items = [Line2D([0], [0], color=c, lw=2, marker="o", markersize=8, label=lbl)
                    for lbl, _, c, *_ in NETS]
    legend_items += [
        Line2D([0], [0], color="gray", lw=0, marker="o", markersize=12, mec="black", mew=1.5,
               label="within 10% of peak (almost-optimal)"),
        Line2D([0], [0], color="gray", lw=0, marker="o", markersize=8, mfc="white", mec="gray", mew=2,
               label="single-seed point"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=9, ncol=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("λ  (log scale)", fontsize=11)
    ax.set_ylabel("efficiency  =  (% pruned) / max(drop pp, 0.5)   (log scale)",
                  fontsize=11)
    ax.set_title("Pruning efficiency vs λ — 5 nets, almost-optimal bands shown",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT_PATH}")

    # ── Table for chat ───────────────────────────────────────────────────────
    print("\nPer-net summary (15-epoch, 3-seed where available):")
    print(f"{'net':<25} {'N_L':>3} {'S_L':>5} {'params':>11} {'λ_opt':>7} {'peak_eff':>9} "
          f"{'10%-band':>14} {'bimodal':>8} {'λ·N_L':>7} {'λ·S_L':>7}")
    for r in summary_rows:
        band = f"[{r['alm_min']:.2f}, {r['alm_max']:.2f}]"
        print(f"{r['label']:<25} {r['N_L']:>3} {r['S_L']:>5} {r['params']:>11,} "
              f"{r['lambda_opt']:>7} {r['peak_eff']:>9.2f} {band:>14} "
              f"{'Y' if r['is_bimodal'] else 'N':>8} "
              f"{r['lambda_opt'] * r['N_L']:>7.3f} {r['lambda_opt'] * r['S_L']:>7.1f}")


def is_bimodal(lams, eff) -> bool:
    """Detect a TRUE bimodal: interior dip with a HIGHER value strictly to its
    right (not just the descending tail of a single peak).

    Specifically: there exists an interior index i s.t.
      - eff[i] ≤ 0.9 · peak  (dip below the 10% band)
      - max(eff[:i]) ≥ 0.9 · peak  (a peak exists to the left)
      - max(eff[i+1:]) ≥ eff[i] + 0.05·peak  (recovery: value strictly RISES
        again after the dip — distinguishes a real local max from a monotone tail)
    """
    if len(eff) < 4:
        return False
    peak = max(eff)
    for i in range(1, len(eff) - 1):
        if eff[i] > 0.9 * peak:
            continue
        left_max  = max(eff[:i]) if i > 0 else 0.0
        right_max = max(eff[i + 1:])
        if left_max >= 0.9 * peak and right_max >= eff[i] + 0.05 * peak:
            return True
    return False


if __name__ == "__main__":
    main()
