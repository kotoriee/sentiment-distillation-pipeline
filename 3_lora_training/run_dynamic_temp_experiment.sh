#!/bin/bash
# 动态温度对比实验运行脚本

set -e

cd "$(dirname "$0")"

echo "============================================================"
echo "答案优先格式 + 动态温度对比实验"
echo "============================================================"

# 快速测试模式
if [ "$1" == "--test" ]; then
    echo "运行快速测试模式..."
    python3 train_dynamic_temp_comparison.py --test
    echo "快速测试完成!"
    exit 0
fi

# 完整训练
echo "运行完整训练..."
echo ""
echo "阶段1: 训练固定温度模型 (T=2.0)"
echo "阶段2: 训练动态温度模型 (自适应)"
echo ""

python3 train_dynamic_temp_comparison.py \
    --data ../data/train_answer_first.json \
    --output-dir models/dynamic_temp_comparison \
    --epochs 3 \
    --batch 1 \
    --grad-acc 16 \
    --lr 2e-5 \
    --alpha 0.5

echo ""
echo "============================================================"
echo "训练完成，开始评估..."
echo "============================================================"

cd ../4_evaluation

python3 eval_dynamic_temp_comparison.py \
    --fixed ../3_lora_training/models/dynamic_temp_comparison/fixed_temp \
    --adaptive ../3_lora_training/models/dynamic_temp_comparison/adaptive_temp \
    --output ../6_experiments_results/dynamic_temp_comparison.json

echo ""
echo "============================================================"
echo "实验完成!"
echo "============================================================"
echo "结果文件: ../6_experiments_results/dynamic_temp_comparison.json"