#!/bin/bash
# ─── Fill in from RunPod console after pod is running ────────────────────────
POD_IP="YOUR_POD_IP"      # e.g. "213.181.9.12"
POD_PORT=22222            # SSH port shown in RunPod console (usually 10000-range)
KEY_PATH="$HOME/.ssh/id_rsa"
# ─────────────────────────────────────────────────────────────────────────────

REMOTE_DIR="/workspace/results/gpt2_lambda_sweep"
LOCAL_DIR="./experiments/runpod_results/gpt2_lambda_sweep"

mkdir -p "$LOCAL_DIR"

rsync -avz --progress \
    -e "ssh -p $POD_PORT -i $KEY_PATH -o StrictHostKeyChecking=no" \
    "root@$POD_IP:$REMOTE_DIR/" \
    "$LOCAL_DIR/"
