#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# B5 dense-fine-tune control, CNN/DailyMail, Llama-2-7B: run BOTH LoRA arms
# back-to-back with matched settings, then compare final ROUGE-L.
#
#   arm A  physically-pruned (from pruner.pt) + LoRA   <- prune_and_lora_cnndailymail_ddp.py
#   arm B  dense (unpruned)                  + LoRA   <- lora_finetune_dense_cnndailymail_ddp.py
#
# Both arms are launched with the SAME --max_steps / --batch_size /
# --grad_accum_steps / --rouge_eval_examples / --seed / LoRA config and the
# same world_size, so they see the same number of CNN/DailyMail examples and
# are scored on the byte-identical test sample. That is the whole point --
# the only difference between the arms is whether pruning happened first.
# Do not change one arm's flags here without changing the other's.
#
# TRAIN-SPLIT COVERAGE (CNN/DailyMail train = 287,113 examples):
#   examples/step = BATCH_SIZE * GRAD_ACCUM * NPROC = 4 * 4 * 8 = 128
#     MAX_STEPS=500   ->  64,000 ex  =  22.3%  (0.22 epochs)
#     MAX_STEPS=1122  -> 143,616 ex  =  50.0%  (0.50 epochs)
#     MAX_STEPS=2244  -> 287,232 ex  = 100.0%  (1.00 epoch)
#
# USAGE:
#   ./run_lora_pruned_vs_dense.sh                  # uses the defaults below
#   MAX_STEPS=1122 ROUGE_N=1500 ./run_lora_pruned_vs_dense.sh
#   ARMS=dense ./run_lora_pruned_vs_dense.sh       # run one arm only
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

# ── knobs ───────────────────────────────────────────────────────────────────
# Confirmed with the user 2026-08-20: 1122 steps = 50% train coverage per arm (~9.7h for both
# arms on 8x A6000), ROUGE on 3000 test examples (26% of the test split) to keep the sampling
# error on the arm-vs-arm ROUGE-L gap well under the gap sizes seen so far -- generation costs
# ~0.25s/example at world_size=8, so the bigger eval sample is the cheap half of this budget.
MAX_STEPS="${MAX_STEPS:-1122}"          # 1122 * 128 ex/step = 143,616 ex = 50.0% of train
ROUGE_N="${ROUGE_N:-3000}"              # test examples scored by generation, per arm
NPROC="${NPROC:-8}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
EVAL_EVERY="${EVAL_EVERY:-250}"
SEED="${SEED:-0}"
ARMS="${ARMS:-both}"                    # both | pruned | dense
PRUNER_CKPT="${PRUNER_CKPT:-experiments/latest/llama2_7b_cnndailymail/lambda_0.3/pruner.pt}"
OUT_ROOT="${OUT_ROOT:-experiments/latest/llama2_7b_cnndailymail_lora_b5}"

# ── env ─────────────────────────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-$REPO_ROOT/huggingface}"
export TOKENIZERS_PARALLELISM=false

# OFFLINE, deliberately (2026-08-20). rank0_first_call() serializes rank 0 against the other
# ranks, but NOT those ranks against each other -- after the barrier all 7 call
# from_pretrained() simultaneously and each independently re-resolves the repo against the
# hub. That killed the first launch of this run: rank 2 (only rank 2; the other 7 loaded
# fine) died with "meta-llama/Llama-2-7b-hf does not appear to have a file named
# model-00001-of-00002.safetensors" while that shard was present and intact in the local
# cache -- a transient hub-resolution failure under 7-way concurrency, not a missing file.
# The 2-rank sanity checks never exercised this (one concurrent caller, not seven).
# Everything this run needs is already cached locally and verified to load with these set:
# the Llama-2-7B snapshot (both shards, no .incomplete blobs), CNN/DailyMail, and the rouge
# metric module. Offline removes the network from the critical path entirely, so there is no
# resolution race and no rate-limit surface. NOTE: this means a genuinely missing cache entry
# now fails immediately and loudly instead of silently downloading -- which is what you want
# for a 10-hour run.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
if [[ -z "${HF_TOKEN:-}" ]]; then
    HF_TOKEN="$(grep -oP 'HF_TOKEN=\K.*' "$REPO_ROOT/env_variables.txt")"
fi
export HF_TOKEN

SCRIPTS="scripts/hypernetwork/train/llm"
PRUNED_DIR="$OUT_ROOT/pruned_lora"
DENSE_DIR="$OUT_ROOT/dense_lora"
mkdir -p "$OUT_ROOT" logs

TRAIN_SIZE=287113
EX_PER_STEP=$(( BATCH_SIZE * GRAD_ACCUM * NPROC ))
EX_SEEN=$(( MAX_STEPS * EX_PER_STEP ))
COVERAGE=$(python3 -c "print(f'{100*$EX_SEEN/$TRAIN_SIZE:.1f}')")

echo "════════════════════════════════════════════════════════════════════════"
echo " B5 control — pruned+LoRA vs dense+LoRA — CNN/DailyMail — Llama-2-7B"
echo "════════════════════════════════════════════════════════════════════════"
echo "  arms            : $ARMS"
echo "  world_size      : $NPROC"
echo "  optimizer steps : $MAX_STEPS  ($EX_PER_STEP ex/step, global effective batch $EX_PER_STEP)"
echo "  train coverage  : $EX_SEEN examples = ${COVERAGE}% of the ${TRAIN_SIZE}-example train split"
echo "  ROUGE sample    : $ROUGE_N test examples (identical prefix slice in both arms)"
echo "  pruner ckpt     : $PRUNER_CKPT"
echo "  out root        : $OUT_ROOT"
echo "════════════════════════════════════════════════════════════════════════"

COMMON_ARGS=(
    --hf_token "$HF_TOKEN"
    --max_steps "$MAX_STEPS"
    --batch_size "$BATCH_SIZE"
    --grad_accum_steps "$GRAD_ACCUM"
    --eval_every "$EVAL_EVERY"
    --rouge_eval_examples "$ROUGE_N"
    --seed "$SEED"
)

# ── arm A: physically-pruned + LoRA ─────────────────────────────────────────
if [[ "$ARMS" == "both" || "$ARMS" == "pruned" ]]; then
    echo -e "\n[$(date +%H:%M:%S)] ARM A — physically-pruned + LoRA ..."
    torchrun --standalone --nproc_per_node="$NPROC" \
        "$SCRIPTS/prune_and_lora_cnndailymail_ddp.py" \
        --pruner_ckpt "$PRUNER_CKPT" \
        --out_dir "$PRUNED_DIR" \
        "${COMMON_ARGS[@]}" 2>&1 | tee "logs/b5_pruned_arm.log"
    echo "[$(date +%H:%M:%S)] ARM A done."
fi

# ── arm B: dense + LoRA ─────────────────────────────────────────────────────
if [[ "$ARMS" == "both" || "$ARMS" == "dense" ]]; then
    echo -e "\n[$(date +%H:%M:%S)] ARM B — dense (unpruned) + LoRA ..."
    torchrun --standalone --nproc_per_node="$NPROC" \
        "$SCRIPTS/lora_finetune_dense_cnndailymail_ddp.py" \
        --out_dir "$DENSE_DIR" \
        "${COMMON_ARGS[@]}" 2>&1 | tee "logs/b5_dense_arm.log"
    echo "[$(date +%H:%M:%S)] ARM B done."
fi

# ── comparison ──────────────────────────────────────────────────────────────
# The pruned arm nests its output under a lambda_<x>/ subdir; the dense arm does not.
PRUNED_META="$(find "$PRUNED_DIR" -name meta.json -print -quit 2>/dev/null || true)"
DENSE_META="$DENSE_DIR/meta.json"

if [[ -f "$PRUNED_META" && -f "$DENSE_META" ]]; then
    echo -e "\n[$(date +%H:%M:%S)] Comparing arms ..."
    python3 "$SCRIPTS/compare_pruned_vs_dense_lora.py" \
        --pruned_meta "$PRUNED_META" \
        --dense_meta "$DENSE_META" \
        --out "$OUT_ROOT/comparison.txt"
else
    echo -e "\nSkipping comparison — need both arms' meta.json."
    echo "  pruned: ${PRUNED_META:-<not found>}"
    echo "  dense : $DENSE_META $([[ -f $DENSE_META ]] || echo '<not found>')"
fi
