#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/kotoriee/llm/sentiment-distillation-pipeline/3_lora_training

source /home/kotoriee/miniconda3/etc/profile.d/conda.sh
conda activate base

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

OUT_DIR=outputs_grpo_9b_from_sft_v3
LOG_FILE="${OUT_DIR}/grpo_full_v3.log"
mkdir -p "${OUT_DIR}"

{
  echo "=============================================="
  echo "GRPO full v3 start: $(date)"
  echo "PWD: $(pwd)"
  echo "HF_ENDPOINT: ${HF_ENDPOINT}"
  echo "Output dir: ${OUT_DIR}"
  echo "Log file: ${LOG_FILE}"
  echo "=============================================="
} >> "${LOG_FILE}"

exec python -u train_qwen35_9b_grpo_from_sft.py \
  --output-dir "${OUT_DIR}" \
  --max-steps 1000 \
  >> "${LOG_FILE}" 2>&1
