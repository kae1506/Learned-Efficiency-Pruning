#!/usr/bin/env bash
# SAMSum B5 control -- pruned(lambda=1.6) vs dense, both LoRA fine-tuned.
# 4 runs, sequential, each using all 8 A6000s via torchrun.
#
#   noeos/  --append_eos OFF -- faithful to the CNN/DailyMail convention,
#           directly comparable to F25's numbers.
#   eos/    --append_eos ON  -- fixes the no-stopping flaw the SAMSum
#           generation-length diagnostic surfaced (2026-08-27): without EOS on
#           the supervised target nothing teaches the model to stop, so every
#           generation saturates max_new_tokens and pads with degenerate
#           repetition, and ROUGE-L mixes "did it summarise" with "how much
#           junk followed".
#
# Both arms within a pair are matched on every knob except whether physical
# surgery ran first; --append_eos is set identically across the arms of a pair.
# Confirmed with the user before launch.
set -euo pipefail

REPO=/data/home/keshava/work/Learned-Efficiency-Pruning
cd "$REPO"

export HF_HOME="$REPO/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export $(grep -v '^#' env_variables.txt | xargs)

SCRIPT="$REPO/scripts/hypernetwork/train/llm/lora_samsum_ddp.py"
CKPT="$REPO/experiments/latest/llama2_7b_samsum/lambda_1.6/pruner.pt"
BASE="$REPO/experiments/latest/llama2_7b_samsum_lora_b5"
LOG_DIR="$REPO/logs/samsum_lora_b5"
mkdir -p "$LOG_DIR"

run () {   # $1=arm  $2=eos_flag_or_empty  $3=subdir
  local arm="$1" eos="$2" sub="$3"
  local log="$LOG_DIR/${sub}_${arm}.log"
  echo "=== $(date +%H:%M:%S)  arm=$arm  variant=$sub  -> $log"
  local extra=()
  [ -n "$eos" ] && extra+=(--append_eos)
  [ "$arm" = "pruned" ] && extra+=(--pruner_ckpt "$CKPT")
  torchrun --standalone --nproc_per_node=8 "$SCRIPT" \
      --arm "$arm" \
      --out_dir "$BASE/$sub" \
      --hf_token "$HF_TOKEN" \
      "${extra[@]}" > "$log" 2>&1
  echo "    done $(date +%H:%M:%S)"
}

run pruned "" noeos
run dense  "" noeos
run pruned eos eos
run dense  eos eos

echo "ALL 4 RUNS COMPLETE $(date +%H:%M:%S)"
