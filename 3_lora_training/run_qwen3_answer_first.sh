#!/bin/bash
# ==============================================================================
# Qwen3-4B 答案优先格式训练 - WSL 完整流程
# ==============================================================================

set -e

WSL_PROJECT="/mnt/d/0321/sentiment-distillation-pipeline"
DATA_DIR="${WSL_PROJECT}/data"

echo "=============================================================="
echo "Qwen3-4B 答案优先格式训练流程"
echo "=============================================================="

# ==============================================================================
# Step 1: 数据格式转换
# ==============================================================================

echo ""
echo "Step 1: 数据格式转换（答案优先）"
echo "=============================================================="

cd "${WSL_PROJECT}/3_lora_training"

# 转换训练数据
python3 convert_answer_first.py \
    --input "${DATA_DIR}/conversations/train_conversations.json" \
    --output "${DATA_DIR}/train_answer_first.json"

# 转换验证数据
python3 convert_answer_first.py \
    --input "${DATA_DIR}/conversations/val_conversations.json" \
    --output "${DATA_DIR}/val_answer_first.json"

# 转换测试数据
python3 convert_answer_first.py \
    --input "${DATA_DIR}/conversations/test_conversations.json" \
    --output "${DATA_DIR}/test_answer_first.json"

# 创建小批量测试集 (700条)
python3 << 'EOF'
import json
import random

with open("/mnt/d/0321/sentiment-distillation-pipeline/data/train_answer_first.json") as f:
    data = json.load(f)

# 分层采样
neg = [d for d in data if d['label'] == 0]
neu = [d for d in data if d['label'] == 1]
pos = [d for d in data if d['label'] == 2]

small = random.sample(neg, 233) + random.sample(neu, 233) + random.sample(pos, 234)

with open("/mnt/d/0321/sentiment-distillation-pipeline/data/train_700_answer_first.json", "w") as f:
    json.dump(small, f, ensure_ascii=False, indent=2)

print(f"Created train_700_answer_first.json: {len(small)} samples")
EOF

echo "数据转换完成 ✓"

# ==============================================================================
# Step 2: 小批量测试训练 (700条, 30步)
# ==============================================================================

echo ""
echo "Step 2: 小批量测试训练"
echo "=============================================================="

python3 train_qwen3_answer_first.py \
    --data "${DATA_DIR}/train_700_answer_first.json" \
    --output models/qwen3_test \
    --test

echo "测试训练完成 ✓"

# ==============================================================================
# Step 3: 全量训练 (7172条, 3epochs)
# ==============================================================================

echo ""
echo "Step 3: 全量训练"
echo "=============================================================="

python3 train_qwen3_answer_first.py \
    --data "${DATA_DIR}/train_answer_first.json" \
    --output models/qwen3-4b-answer-first \
    --epochs 3 \
    --batch 1 \
    --grad-acc 16 \
    --lr 2e-5 \
    --temperature 2.0 \
    --alpha 0.5

echo "全量训练完成 ✓"

# ==============================================================================
# Step 4: 评估（答案优先格式 - 极速）
# ==============================================================================

echo ""
echo "Step 4: 评估（答案优先格式）"
echo "=============================================================="

cd "${WSL_PROJECT}/4_evaluation"

python3 eval_ultra_fast.py \
    --model "${WSL_PROJECT}/3_lora_training/models/qwen3-4b-answer-first" \
    --data "${DATA_DIR}/test_answer_first.json" \
    --samples 897 \
    --output "${WSL_PROJECT}/6_experiments_results/qwen3_answer_first_eval.json"

echo "评估完成 ✓"

echo ""
echo "=============================================================="
echo "完整流程执行完毕"
echo "=============================================================="