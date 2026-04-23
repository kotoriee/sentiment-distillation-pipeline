#!/bin/bash
# ==============================================================================
# Qwen3-4B 软标签蒸馏训练 - WSL 完整执行脚本
# ==============================================================================

set -e  # 遇到错误立即退出

# ==============================================================================
# 配置区
# ==============================================================================

# Windows 项目路径
WIN_PROJECT="d:/0321/sentiment-distillation-pipeline"
# WSL 路径转换
WSL_PROJECT="/mnt/d/0321/sentiment-distillation-pipeline"

# 训练配置
MODEL="unsloth/Qwen3-4B-unsloth-bnb-4bit"
DATA_SMALL="${WSL_PROJECT}/data/train_700.json"
DATA_FULL="${WSL_PROJECT}/data/conversations/train_conversations.json"
TEST_DATA="${WSL_PROJECT}/data/conversations/test_conversations.json"

OUTPUT_DIR="${WSL_PROJECT}/3_lora_training/models/qwen3-4b-soft-full"
EPOCHS=3
BATCH=1
GRAD_ACC=16
LR=2e-5
TEMPERATURE=2.0
ALPHA=0.5

# ==============================================================================
# Step 0: 环境检查
# ==============================================================================

echo "=============================================================="
echo "Step 0: 环境检查"
echo "=============================================================="

# 检查 GPU
echo "检查 GPU..."
python3 -c "import torch; assert torch.cuda.is_available(), '需要 GPU！'; print(f'GPU: {torch.cuda.get_device_name(0)}')" || {
    echo "错误: 未检测到 GPU，请检查 CUDA 安装"
    exit 1
}

# 检查 unsloth
echo "检查 unsloth..."
python3 -c "from unsloth import FastLanguageModel; print('unsloth 可用')" || {
    echo "错误: unsloth 未安装，请运行: pip install unsloth"
    exit 1
}

# 检查数据文件
echo "检查数据文件..."
test -f "${DATA_SMALL}" || { echo "错误: ${DATA_SMALL} 不存在"; exit 1; }
test -f "${DATA_FULL}" || { echo "错误: ${DATA_FULL} 不存在"; exit 1; }
test -f "${TEST_DATA}" || { echo "错误: ${TEST_DATA} 不存在"; exit 1; }

echo "环境检查完成 ✓"

# ==============================================================================
# Step 1: 小批量测试训练 (700条, 30步)
# ==============================================================================

echo ""
echo "=============================================================="
echo "Step 1: 小批量测试训练 (30步验证)"
echo "=============================================================="

cd "${WSL_PROJECT}/3_lora_training"

python3 train_soft_label.py \
    --model "${MODEL}" \
    --data "${DATA_SMALL}" \
    --output models/qwen3_soft_test \
    --test \
    --temperature "${TEMPERATURE}" \
    --alpha "${ALPHA}"

echo "小批量测试完成 ✓"

# ==============================================================================
# Step 2: 全量软标签训练 (7172条, 3epochs)
# ==============================================================================

echo ""
echo "=============================================================="
echo "Step 2: 全量软标签训练"
echo "=============================================================="
echo "数据: 7172 条"
echo "Epochs: ${EPOCHS}"
echo "Temperature: ${TEMPERATURE}"
echo "Alpha: ${ALPHA}"
echo "预估时间: ~4小时"
echo "=============================================================="

python3 train_soft_label.py \
    --model "${MODEL}" \
    --data "${DATA_FULL}" \
    --output "${OUTPUT_DIR}" \
    --epochs "${EPOCHS}" \
    --batch "${BATCH}" \
    --grad-acc "${GRAD_ACC}" \
    --lr "${LR}" \
    --temperature "${TEMPERATURE}" \
    --alpha "${ALPHA}"

echo "全量训练完成 ✓"

# ==============================================================================
# Step 3: 模型评估
# ==============================================================================

echo ""
echo "=============================================================="
echo "Step 3: 模型评估"
echo "=============================================================="

cd "${WSL_PROJECT}/4_evaluation"

python3 eval_model.py \
    --model "${OUTPUT_DIR}" \
    --base-model "${MODEL}" \
    --data "${TEST_DATA}" \
    --samples 897 \
    --batch-size 8 \
    --output "${WSL_PROJECT}/6_experiments_results/soft_full_eval.json"

echo "评估完成 ✓"

# ==============================================================================
# Step 4: 结果对比
# ==============================================================================

echo ""
echo "=============================================================="
echo "Step 4: 结果对比"
echo "=============================================================="

EVAL_FILE="${WSL_PROJECT}/6_experiments_results/soft_full_eval.json"

if [ -f "${EVAL_FILE}" ]; then
    python3 << EOF
import json
with open("${EVAL_FILE}") as f:
    result = json.load(f)

print("软标签蒸馏结果:")
print(f"  准确率: {result['accuracy']:.2f}%")
print(f"  总样本: {result['total']}")
print(f"  正确: {result['correct']}")
print(f"  解析错误: {result['parse_errors']}")

# 计算各类召回率
from collections import Counter
cm = Counter()
for r in result['results']:
    if r['pred'] != -1:
        cm[(r['true'], r['pred'])] += 1

print("\n各类召回率:")
for label in [0, 1, 2]:
    total = sum(cm[(label, p)] for p in [0, 1, 2])
    correct = cm[(label, label)]
    recall = correct / total * 100 if total > 0 else 0
    name = ['负面', '中性', '正面'][label]
    print(f"  {name}: {recall:.1f}% ({correct}/{total})")

print("\n对比硬标签结果:")
print("  硬标签准确率: 79.69%")
print("  硬标签中性召回: 72.1%")
print("")
if result['accuracy'] > 79.69:
    print("✓ 软标签提升有效!")
else:
    print("! 软标签未超越硬标签")
EOF
fi

echo ""
echo "=============================================================="
echo "完整流程执行完毕"
echo "=============================================================="
echo "输出文件:"
echo "  模型: ${OUTPUT_DIR}"
echo "  评估: ${EVAL_FILE}"
echo "=============================================================="