#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/kotoriee/llm/sentiment-distillation-pipeline/3_lora_training
source /home/kotoriee/miniconda3/etc/profile.d/conda.sh
conda activate base

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export LD_LIBRARY_PATH="/home/kotoriee/miniconda3/lib/python3.13/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

OUTPUT_DIR="outputs_grpo_9b_rewardfix_v4_soft_from_ckpt60_60"
LOG_FILE="${OUTPUT_DIR}/grpo_rewardfix_v4_soft_from_ckpt60_60.log"

mkdir -p "$OUTPUT_DIR"

echo "Started: $(date --iso-8601=seconds)"
python -u train_qwen35_9b_grpo_from_sft.py \
  --sft-lora outputs_grpo_9b_rewardfix_v3_nothink_continue_300/checkpoint-60 \
  --output-dir "$OUTPUT_DIR" \
  --max-steps 60 \
  --save-steps 15 \
  --correct-reward 4 \
  --wrong-reward -2 \
  --missing-reward -4 \
  --format-reward 1 \
  --learning-rate 5e-7 \
  --beta 0.05 \
  --temperature 0.7 \
  --num-generations 8 \
  --warmup-ratio 0.03 \
  --steps-per-generation 16 \
  --max-completion-length-cap 128 \
  2>&1 | tee "$LOG_FILE"
echo "Finished: $(date --iso-8601=seconds)"
