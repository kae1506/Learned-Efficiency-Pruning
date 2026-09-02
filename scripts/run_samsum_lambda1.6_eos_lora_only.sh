#!/usr/bin/env bash
# STEP 2 ONLY, retried standalone: LoRA on the already-trained eos-pruner
# checkpoint (experiments/latest/llama2_7b_samsum_eos/lambda_1.6/pruner.pt,
# finished cleanly -- 80.59% pruned, non-converged at the 12000-step cap,
# same shape as the original). The first combined-script attempt died here
# with "torchrun: command not found" (PYTHONPATH was set but not PATH) --
# fixed below, prune step not rerun.
set -euo pipefail

REPO=/data/home/keshava/work/Learned-Efficiency-Pruning
cd "$REPO"

export HF_HOME="$REPO/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/data/pypackages${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/data/pypackages/bin:$PATH"
export $(grep -v '^#' env_variables.txt | xargs)

LORA_SCRIPT="$REPO/scripts/hypernetwork/train/llm/lora_samsum_ddp.py"
PRUNE_OUT="$REPO/experiments/latest/llama2_7b_samsum_eos"
LORA_OUT="$REPO/experiments/latest/llama2_7b_samsum_lora_b5_eos_pruner/eos"
LOG_DIR="$REPO/logs/samsum_eos_pruner"
mkdir -p "$LORA_OUT" "$LOG_DIR"

echo "=== $(date +%H:%M:%S) STEP 2/2 (retry): LoRA on the eos-trained pruned model, --append_eos ==="
torchrun --standalone --nproc_per_node=8 "$LORA_SCRIPT" \
    --arm pruned \
    --append_eos \
    --pruner_ckpt "$PRUNE_OUT/lambda_1.6/pruner.pt" \
    --out_dir "$LORA_OUT" \
    --hf_token "$HF_TOKEN" \
    > "$LOG_DIR/2_lora_pruned_eos.log" 2>&1
echo "    done $(date +%H:%M:%S)"

echo "ALL DONE $(date +%H:%M:%S)"
echo "Pruned+LoRA(eos-pruner) results: $LORA_OUT/pruned_lora_lambda_1.6/summary.txt"
echo "Compare against: $REPO/experiments/latest/llama2_7b_samsum_lora_b5/eos/dense_lora/summary.txt"
