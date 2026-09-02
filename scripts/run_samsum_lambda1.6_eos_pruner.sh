#!/usr/bin/env bash
# One-off: retrain the SAMSum pruner at lambda=1.6 WITH --append_eos (the
# pruner-training script never had this flag before -- added alongside this
# run so the CE-delta objective itself sees a stopping signal, not just
# downstream LoRA), then LoRA-finetune the resulting pruned model with
# --append_eos, sequentially (two separate scripts/processes, not one).
#
# Writes to a NEW path (llama2_7b_samsum_eos/) so the original 8-lambda
# sweep's lambda_1.6/pruner.pt (noeos-trained) is untouched.
#
# Compare target (already computed, not rerun): dense+LoRA+eos at
# experiments/latest/llama2_7b_samsum_lora_b5/eos/dense_lora/summary.txt
# (R-L 42.22, ppl 3.802, mean gen tok 26.7/96) -- the dense arm doesn't
# depend on the pruner, so it's unaffected by this checkpoint being new.
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

PRUNE_SCRIPT="$REPO/scripts/hypernetwork/train/llm/train_pruner_llama2_7b_samsum.py"
LORA_SCRIPT="$REPO/scripts/hypernetwork/train/llm/lora_samsum_ddp.py"
PRUNE_OUT="$REPO/experiments/latest/llama2_7b_samsum_eos"
LORA_OUT="$REPO/experiments/latest/llama2_7b_samsum_lora_b5_eos_pruner/eos"
LOG_DIR="$REPO/logs/samsum_eos_pruner"
mkdir -p "$PRUNE_OUT" "$LORA_OUT" "$LOG_DIR"

echo "=== $(date +%H:%M:%S) STEP 1/2: pruner training, lambda=1.6, --append_eos ==="
CUDA_VISIBLE_DEVICES=0 python3 -u "$PRUNE_SCRIPT" \
    --lambdas 1.6 \
    --seeds 0 \
    --max_steps 12000 \
    --append_eos \
    --out_dir "$PRUNE_OUT" \
    --hf_token "$HF_TOKEN" \
    > "$LOG_DIR/1_prune_lambda1.6_eos.log" 2>&1
echo "    done $(date +%H:%M:%S) -- pruner.pt at $PRUNE_OUT/lambda_1.6/pruner.pt"

echo "=== $(date +%H:%M:%S) STEP 2/2: LoRA on the eos-trained pruned model, --append_eos ==="
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
