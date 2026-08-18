"""
Run the next 3 CNN/DailyMail sweep points -- λ=0.05, 0.3, 0.8; seed=0 --
directly on this machine. No RunPod orchestration: just invokes
train_pruner_llama2_7b_cnndailymail.py with the confirmed args (2026-08-18)
as a subprocess, then merges the results with the existing λ=0.1 point.
Run this on whatever machine has the CUDA GPU (already-provisioned pod,
local box, wherever):

    python3 scripts/hypernetwork/train/llm/run_next3_cnndailymail.py

λ=0.1 is already done (experiments/latest/llama2_7b_cnndailymail/lambda_0.1/,
9275.9s training loop, capped at max_steps=12000 without converging). These
3 values were confirmed with the user to bracket it low/mid/high; max_steps
kept at 12000 to match, for cross-point consistency, even though 0.1 hit
that cap non-converged -- some of these 3 may cap out too.

MERGE STEP -- train_pruner_llama2_7b_cnndailymail.py's own out_dir/summary.txt
and out_dir/gap_diagnostic_all.csv are written covering only the --lambdas
passed to THAT invocation, in "w" mode -- so a bare run here would silently
overwrite the combined view and drop the existing 0.1 row. merge_local_summary()
below regenerates both files afterward by scanning every lambda_*/ directory
actually present on disk, so 0.1 stays represented alongside these 3.

HF TOKEN -- pass --hf_token explicitly on the command line:
    python3 run_next3_cnndailymail.py --hf_token hf_xxx
(falls back to the HF_TOKEN env var if --hf_token is omitted, but the whole
point of the CLI arg is that you don't have to export anything.)
"""
import os
import re
import csv
import sys
import glob
import argparse
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "train_pruner_llama2_7b_cnndailymail.py")
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "experiments", "latest", "llama2_7b_cnndailymail")

LAMBDAS = ["0.05", "0.3", "0.8"]
SEEDS = ["0"]
MAX_STEPS = "12000"

_SUMMARY_PATTERNS = {
    "steps_taken": r"steps taken\s*:\s*(\d+)",
    "converged": r"converged\s*:\s*(True|False)",
    "pct_pruned": r"final % FFN neurons pruned\s*:\s*([\d.]+)%",
    "test_ppl_orig": r"original\s+ppl\s*:\s*([\d.]+)",
    "test_ppl_pruned": r"pruned\s+ppl\s*:\s*([\d.]+)",
    "rouge_orig_rougeL": r"original\s+rouge1/rouge2/rougeL\s*:\s*[\d.]+/[\d.]+/([\d.]+)",
    "rouge_pruned_rougeL": r"pruned\s+rouge1/rouge2/rougeL\s*:\s*[\d.]+/[\d.]+/([\d.]+)",
}


def parse_run_summary(summary_path):
    text = open(summary_path).read()
    out = {}
    for key, pat in _SUMMARY_PATTERNS.items():
        m = re.search(pat, text)
        if not m:
            print(f"  WARNING: could not parse {key!r} out of {summary_path}", flush=True)
            return None
        out[key] = m.group(1)
    return out


def merge_local_summary(out_dir):
    """Regenerate out_dir/summary.txt and out_dir/gap_diagnostic_all.csv from
    EVERY lambda_*/ (and lambda_*/seed_*/) directory found on disk, not just
    this run's 3 -- so the pre-existing lambda_0.1 point stays represented."""
    run_dirs = sorted(set(glob.glob(os.path.join(out_dir, "lambda_*"))) |
                      set(glob.glob(os.path.join(out_dir, "lambda_*", "seed_*"))))
    run_dirs = [d for d in run_dirs if os.path.isdir(d) and
               os.path.exists(os.path.join(d, "summary.txt"))]

    def lam_seed_from_path(d):
        parts = d.replace(out_dir, "").strip(os.sep).split(os.sep)
        lam = float(parts[0].removeprefix("lambda_"))
        seed = int(parts[1].removeprefix("seed_")) if len(parts) > 1 else 0
        return lam, seed

    rows = []
    for d in run_dirs:
        parsed = parse_run_summary(os.path.join(d, "summary.txt"))
        if parsed is None:
            continue
        lam, seed = lam_seed_from_path(d)
        rows.append((lam, seed, parsed))
    rows.sort(key=lambda r: (r[0], r[1]))

    sep = "-" * 100
    lines = [f"Llama-2-7B convergence+LR-decay sweep | CNN/DailyMail | max_steps=12000 "
            f"(regenerated locally from {len(rows)} lambda_*/ dirs on disk, merging this "
            f"run with pre-existing points)", sep,
            f"{'lambda':>7} {'seed':>5} | {'steps':>7} {'conv?':>6} | {'% pruned':>9} | "
            f"{'orig ppl':>9} | {'pruned ppl':>10} | {'orig R-L':>9} | {'pruned R-L':>10}", sep]
    for lam, seed, p in rows:
        lines.append(
            f"{lam:>7} {seed:>5} | {p['steps_taken']:>7} "
            f"{'YES' if p['converged'] == 'True' else 'NO':>6} | {float(p['pct_pruned']):>8.2f}% | "
            f"{float(p['test_ppl_orig']):>9.3f} | {float(p['test_ppl_pruned']):>10.3f} | "
            f"{float(p['rouge_orig_rougeL'])*100:>8.2f}% | {float(p['rouge_pruned_rougeL'])*100:>9.2f}%")
    summary_str = "\n".join(lines)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(summary_str + "\n")
    print("\n" + summary_str, flush=True)

    gap_rows = []
    fieldnames = None
    for d in run_dirs:
        gap_csv = os.path.join(d, "gap_diagnostic.csv")
        if not os.path.exists(gap_csv):
            continue
        with open(gap_csv, newline="") as in_f:
            reader = csv.DictReader(in_f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            gap_rows.extend(list(reader))
    if fieldnames:
        combined_path = os.path.join(out_dir, "gap_diagnostic_all.csv")
        with open(combined_path, "w", newline="") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(gap_rows)
        print(f"Combined gap diagnostic ({len(gap_rows)} rows across {len(run_dirs)} runs) -> "
              f"{combined_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_token", type=str, default=None,
                    help="HF token with ACCEPTED access to meta-llama/Llama-2-7b-hf. "
                         "Falls back to the HF_TOKEN env var if omitted.")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token is None:
        print("WARNING: no --hf_token passed and HF_TOKEN not set in this shell -- "
              "meta-llama/Llama-2-7b-hf is gate-licensed, the training script will fail "
              "at model load without one.", flush=True)

    cmd = [sys.executable, "-u", TRAIN_SCRIPT,
          "--lambdas", *LAMBDAS, "--seeds", *SEEDS, "--max_steps", MAX_STEPS,
          "--out_dir", OUT_DIR]
    if hf_token:
        cmd += ["--hf_token", hf_token]
    print(f"$ {' '.join(cmd[:-1] + ['<redacted>'] if hf_token else cmd)}", flush=True)
    subprocess.run(cmd, check=True)

    print("\nRegenerating combined summary.txt / gap_diagnostic_all.csv from ALL lambda_*/ "
          "dirs on disk (existing 0.1 + this run's 3) ...", flush=True)
    merge_local_summary(OUT_DIR)
    print(f"\nDone -> {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
