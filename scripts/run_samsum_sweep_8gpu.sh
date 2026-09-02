#!/usr/bin/env bash
# SAMSum pruner sweep -- 8 lambdas, one per A6000, run in parallel.
#
# Pre-flight numbers MEASURED on this box (2026-08-26) before launching:
#   step time              : 878-908 ms/step (1xA6000, batch=4)
#   peak GPU mem, typical  : 36.38 GB / 48 GB
#   peak GPU mem, WORST    : 36.62 GB / 48 GB  (4 longest dialogues, 518 tok)
#   pruner params          : 50,604,833
#   sanity check           : PASS (identity-gate diff = +0.000000)
#
# Peak memory is dominated by constants (13.5 GB bf16 model + 11.5 GB fp32
# weight copies), NOT activations -- SAMSum sequences are short enough that
# the worst-case batch costs only 0.24 GB over typical. Hence ~11 GB headroom
# per GPU and no length-distribution OOM tail.
#
# lambda grid: 0.25 dropped from the 9-value default to fit 8 GPUs (most
# redundant point -- sits between 0.2 and 0.3). max_steps=12000 kept as a
# safety CAP, not a target: the convergence check stops runs early on its own.
# Both confirmed with the user before launch.
#
# OFFLINE MODE is deliberate: engineering_decisions.md:192-201 records the B5
# run dying ~20s in when 7 ranks re-resolved the repo against the hub at once.
# Everything needed is verified cached. Trade-off accepted: a genuinely
# missing cache entry now fails loudly instead of silently downloading.
set -euo pipefail

REPO=/data/home/keshava/work/Learned-Efficiency-Pruning
cd "$REPO"

export HF_HOME="$REPO/huggingface"     # absolute -- the script's own default is
                                        # relative ("./huggingface"), which would
                                        # split the cache if cwd ever differed
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false     # 8 procs x tokenizer threads on one box
export $(grep -v '^#' env_variables.txt | xargs)

OUT_DIR="$REPO/experiments/latest/llama2_7b_samsum"
LOG_DIR="$REPO/logs/samsum_sweep"
SCRIPT="$REPO/scripts/hypernetwork/train/llm/train_pruner_llama2_7b_samsum.py"
mkdir -p "$OUT_DIR" "$LOG_DIR"

LAMBDAS=(0.01 0.05 0.1 0.2 0.3 0.4 0.8 1.6)

echo "Launching ${#LAMBDAS[@]} runs, one per GPU. out_dir=$OUT_DIR"
for i in "${!LAMBDAS[@]}"; do
  LAM="${LAMBDAS[$i]}"
  LOG="$LOG_DIR/lambda_${LAM}.log"
  echo "  GPU $i -> lambda=$LAM -> $LOG"
  CUDA_VISIBLE_DEVICES=$i nohup python3 -u "$SCRIPT" \
      --lambdas "$LAM" \
      --seeds 0 \
      --max_steps 12000 \
      --out_dir "$OUT_DIR" \
      --hf_token "$HF_TOKEN" \
      > "$LOG" 2>&1 &
  echo "$!" >> "$LOG_DIR/pids.txt"
  # stagger: 8 concurrent cold reads of a 13 GB checkpoint would hammer nvme;
  # after the first, page cache (947 GB free) serves the rest.
  sleep 25
done

echo "All ${#LAMBDAS[@]} launched. PIDs in $LOG_DIR/pids.txt"
echo "NOTE: every process writes the same top-level summary.txt /"
echo "gap_diagnostic_all.csv (each holding only its own lambda, last writer"
echo "wins). Per-lambda subdirs are complete and correct -- aggregate the"
echo "sweep from those, not from the top-level roll-up."
