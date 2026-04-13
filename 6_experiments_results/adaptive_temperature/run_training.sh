#!/bin/bash
# 动态温度训练脚本（内存优化版）

echo "=========================================="
echo "动态温度蒸馏训练 - 内存优化版"
echo "=========================================="

# 清理GPU内存
echo "清理GPU缓存..."
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# 设置内存环境变量
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

# 训练参数
TRAIN_DATA="${1:-../../data/curriculum/train_600.json}"
VAL_DATA="${2:-../../data/curriculum/val_fixed.json}"

echo "训练数据: $TRAIN_DATA"
echo "验证数据: $VAL_DATA"
echo ""

# 启动训练（内存优化参数）
python3 adaptive_temperature.py \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --base_model unsloth/Qwen3-4B-unsloth-bnb-4bit \
    --output_dir ./results/adaptive_temp_model \
    --epochs 3 \
    --batch_size 4 \
    --lr 2e-4 \
    --alpha 0.5 \
    --lora_r 16 \
    2>&1 | tee training_$(date +%m%d_%H%M).log

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
