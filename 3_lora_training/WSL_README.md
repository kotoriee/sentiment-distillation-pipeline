# WSL 执行说明

## 环境要求

- WSL2 + Ubuntu 22.04
- NVIDIA GPU + CUDA 12.x
- Python 3.10+

## 快速开始

### 1. 进入 WSL
```bash
wsl
```

### 2. 安装依赖 (首次运行)
```bash
pip install unsloth trl datasets transformers torch
```

### 3. 快速测试 (~10分钟)
验证脚本兼容性，仅运行30步：
```bash
cd /mnt/d/0321/sentiment-distillation-pipeline/3_lora_training
bash quick_test.sh
```

预期输出：
```
Sentiment position found: ~700/700 samples
[debug step 0] sft=... kl=... total=...
```

### 4. 全量训练 (~4小时)
```bash
bash run_soft_label_distillation.sh
```

## 执行流程

| Step | 内容 | 时间 |
|------|------|------|
| Step 0 | 环境检查 | 10秒 |
| Step 1 | 小批量测试 (30步) | ~10分钟 |
| Step 2 | 全量训练 (7172条, 3epochs) | ~4小时 |
| Step 3 | 模型评估 (897条) | ~15分钟 |
| Step 4 | 结果对比分析 | 5秒 |

## 预期结果对比

| 模型 | 准确率 | 中性召回率 |
|------|--------|------------|
| 硬标签基线 | 79.69% | 72.1% |
| **软标签目标** | >82% | >77% |

## 路径映射

| Windows | WSL |
|---------|-----|
| `d:/0321/sentiment-distillation-pipeline` | `/mnt/d/0321/sentiment-distillation-pipeline` |

## 输出文件

- 模型: `3_lora_training/models/qwen3-4b-soft-full/`
- 评估: `6_experiments_results/soft_full_eval.json`

## 常见问题

### Q: GPU 检测失败
```bash
nvidia-smi  # 检查 GPU 状态
pip install torch --upgrade  # 更新 PyTorch
```

### Q: unsloth 安装失败
```bash
pip install "unsloth[colab-new]"  # 使用新版安装方式
pip install --no-deps bitsandbytes
```

### Q: 数据文件找不到
确认 Windows 路径已挂载：
```bash
ls /mnt/d/0321/sentiment-distillation-pipeline/data/
```