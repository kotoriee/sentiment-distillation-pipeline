#!/bin/bash
# ==============================================================================
# 快速测试脚本 - 仅30步验证 (约10分钟)
# ==============================================================================

set -e

# 路径配置
WSL_PROJECT="/mnt/d/0321/sentiment-distillation-pipeline"
MODEL="unsloth/Qwen3-4B-unsloth-bnb-4bit"
DATA="${WSL_PROJECT}/data/train_700.json"

echo "=============================================================="
echo "快速测试 - 验证软标签训练脚本兼容性"
echo "=============================================================="

# GPU 检查
echo "GPU 检查..."
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')" || exit 1

# 进入训练目录
cd "${WSL_PROJECT}/3_lora_training"

# 运行30步测试
echo ""
echo "开始测试训练 (30步)..."
echo ""

python3 train_soft_label.py \
    --model "${MODEL}" \
    --data "${DATA}" \
    --output models/quick_test \
    --test \
    --temperature 2.0 \
    --alpha 0.5

echo ""
echo "=============================================================="
echo "测试完成!"
echo "=============================================================="
echo ""
echo "如果输出显示:"
echo "  - 'Sentiment position found: ~700/700 samples'"
echo "  - 'sft=X.XX kl=X.XX total=X.XX'"
echo ""
echo "则说明训练脚本兼容，可以继续执行全量训练。"
echo ""
echo "全量训练命令:"
echo "  bash run_soft_label_distillation.sh"
echo "=============================================================="