#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/kotoriee/llm/sentiment-distillation-pipeline/3_lora_training

source /home/kotoriee/miniconda3/etc/profile.d/conda.sh
conda activate base

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/home/kotoriee/miniconda3/lib/python3.13/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

OUT_DIR="outputs_grpo_9b_rewardfix_v2_100"
LOG_FILE="${OUT_DIR}/grpo_rewardfix_v2_100.log"

mkdir -p "${OUT_DIR}"

{
  echo "==== GRPO reward-fix v2 100-step run started: $(date -Is) ===="
  echo "Output dir: ${OUT_DIR}"
  echo "Prompt: original SFT/eval prompt + answer-first JSON reminder"
  echo "Reward: correctness +10/-10, answer-first JSON format +1, length penalty"
  echo "Generation: num_generations=8, max_completion_length=128"
} | tee "${LOG_FILE}"

python -u train_qwen35_9b_grpo_from_sft.py \
  --output-dir "${OUT_DIR}" \
  --max-steps 100 \
  2>&1 | tee -a "${LOG_FILE}"

echo "==== GRPO reward-fix v2 100-step run finished: $(date -Is) ====" | tee -a "${LOG_FILE}"
