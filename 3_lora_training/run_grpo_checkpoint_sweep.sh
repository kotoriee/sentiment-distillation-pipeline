#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/kotoriee/llm/sentiment-distillation-pipeline/3_lora_training

source /home/kotoriee/miniconda3/etc/profile.d/conda.sh
conda activate base

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/home/kotoriee/miniconda3/lib/python3.13/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

BASE_DIR="outputs_grpo_9b_from_sft_v3"
RESULT_DIR="../6_experiments_results/grpo_v3_checkpoint_sweep"
LOG_FILE="${RESULT_DIR}/sweep.log"

mkdir -p "${RESULT_DIR}"

echo "==== GRPO v3 checkpoint sweep started: $(date -Is) ====" | tee "${LOG_FILE}"

for ckpt in 100 200 300 400 500 600 700 800 900 1000; do
  model_path="${BASE_DIR}/checkpoint-${ckpt}"
  output_path="${RESULT_DIR}/checkpoint-${ckpt}_eval.json"

  echo "" | tee -a "${LOG_FILE}"
  echo "==== Evaluating checkpoint-${ckpt}: $(date -Is) ====" | tee -a "${LOG_FILE}"

  python -u evaluate_lora.py \
    --model "${model_path}" \
    --data ../data/test_answer_first.json \
    --qwen \
    --batch-size 8 \
    --max-new-tokens 32 \
    --max-seq-length 512 \
    --max-input-length 512 \
    --output "${output_path}" 2>&1 | tee -a "${LOG_FILE}"
done

echo "" | tee -a "${LOG_FILE}"
echo "==== GRPO v3 checkpoint sweep finished: $(date -Is) ====" | tee -a "${LOG_FILE}"
