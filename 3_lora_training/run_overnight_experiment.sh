#!/bin/bash
# 过夜实验：答案优先训练集 — Gemma4 E4B + Qwen3.5-9B 微调
#
# 运行方式: bash run_overnight_experiment.sh
# 注意：先跑 Gemma4 E4B（较快），再跑 Qwen3.5-9B（较慢）

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo " 过夜实验开始"
echo " $(date)"
echo "=============================================="

# ============ 实验 1: Gemma 4 E4B ============

echo ""
echo "=============================================="
echo " 实验 1/2: Gemma 4 E4B 答案优先训练"
echo " $(date)"
echo "=============================================="

python train_gemma4_e4b_answer_first.py \
  --data ../data/train_answer_first.json \
  --output ../models/gemma4-e4b-answer-first \
  --epochs 3

echo ""
echo "=============================================="
echo " 实验 1 完成: Gemma 4 E4B"
echo " 模型: ../models/gemma4-e4b-answer-first/"
echo " $(date)"
echo "=============================================="

# ============ 实验 2: Qwen3.5-9B ============

echo ""
echo "=============================================="
echo " 实验 2/2: Qwen3.5-9B 答案优先训练"
echo " $(date)"
echo "=============================================="

python train_qwen35_9b_answer_first.py \
  --data ../data/train_answer_first.json \
  --output ../models/qwen35-9b-answer-first \
  --epochs 3

echo ""
echo "=============================================="
echo " 实验 2 完成: Qwen3.5-9B"
echo " 模型: ../models/qwen35-9b-answer-first/"
echo " $(date)"
echo "=============================================="

# ============ 全部完成 ============

echo ""
echo "=============================================="
echo " 过夜实验全部完成!"
echo " 模型输出:"
echo "   1. models/gemma4-e4b-answer-first/"
echo "   2. models/qwen35-9b-answer-first/"
echo " $(date)"
echo "=============================================="
