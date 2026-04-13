# Sentiment Distillation Pipeline

电商评论情感分析 - 软标签蒸馏训练流程

## 项目概述

本项目实现了一个完整的情感分析蒸馏流程，包含4个核心模块：

1. **数据预处理** - HuggingFace流式加载 + 文本清洗
2. **软标注系统** - DeepSeek API + Cherry Studio标注
3. **LoRA微调** - 动态温度蒸馏 + 多模型支持
4. **评估系统** - 多格式输出解析 + 指标计算

## 目录结构

```
sentiment-distillation-pipeline/
├── 1_data_preprocessing/      # 模块1：数据预处理
│   ├── download_from_hf.py    # HuggingFace流式下载
│   ├── clean_text.py          # 文本清洗
│   ├── quality_check.py       # 数据质量检查
│   └── data_schema.py         # 数据契约定义
│
├── 2_soft_annotation/         # 模块2：软标注
│   ├── annotate_with_deepseek.py  # DeepSeek标注
│   ├── batch_annotator.py     # 批量标注
│   ├── merge_annotations.py   # 合并标注结果
│   └── prompts/               # 提示词模板
│
├── 3_lora_training/           # 模块3：LoRA微调
│   ├── train_soft_label.py    # 软标签训练（动态温度）
│   ├── train_qwen3.py         # Qwen3-4B训练
│   ├── train_gemma4.py        # Gemma 4 E2B训练
│   ├── preprocess_data.py     # 数据格式转换
│   └── config/                # 训练配置
│
├── 4_evaluation/              # 模块4：评估
│   ├── eval_model.py          # 统一评估入口
│   ├── eval_batch.py          # 批量推理
│   ├── metrics.py             # 指标计算
│   └── visualize.py           # 结果可视化
│
└── data/                      # 数据目录
    ├── train.json             # 训练集 (7,172条)
    ├── val.json               # 验证集 (896条)
    ├── test.json              # 测试集 (897条)
    ├── train_700.json         # 小批量测试集
    └── conversations/         # conversations格式数据
```

## 快速开始

### 1. 数据质量检查

```bash
cd 1_data_preprocessing
python quality_check.py --data ../data/train.json --check all
```

### 2. 软标注（可选）

如果需要重新标注：

```bash
cd 2_soft_annotation
python annotate_with_deepseek.py --batch-test 20  # 小批量测试
python annotate_with_deepseek.py                   # 全量标注
```

### 3. LoRA训练

**Qwen3-4B训练**：

```bash
cd 3_lora_training
python train_qwen3.py --data ../data/train_700.json --test  # 测试模式
python train_qwen3.py --data ../data/train.json --epochs 3 # 全量训练
```

**Gemma 4 E2B训练**：

```bash
python train_gemma4.py --data ../data/conversations/train_conversations.json --test
```

### 4. 评估

```bash
cd 4_evaluation
python eval_model.py --model ../3_lora_training/models/qwen3-4b-rationale --data ../data/test.json --samples 50
```

## 模型支持

| 模型 | 基座 | API | VRAM需求 |
|-----|------|-----|---------|
| Qwen3-4B | `Qwen/Qwen3-4B` | FastLanguageModel | ~5GB |
| Gemma 4 E2B | `unsloth/gemma-4-E2B-it` | FastModel | ~4GB |

## 训练参数

- **软标签训练**: 动态温度 (temperature=2.0)
- **LoRA rank**: 16
- **最大序列长度**: 512
- **Batch size**: 1 × gradient_accumulation=16

## 数据格式

### 原始格式 (train.json)

```json
{
  "text": "评论内容",
  "label": 0/1/2,
  "soft_labels": [0.2, 0.3, 0.5],
  "cot": "推理过程",
  "rating": 1-5
}
```

### Conversations格式 (Gemma 4)

```json
{
  "conversations": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Review: ..."},
    {"role": "assistant", "content": "<|channel>thought\n...<channel|>\n{JSON}"}
  ],
  "text": "...",
  "label": 0/1/2,
  "soft_labels": [...]
}
```

## 环境配置

**新电脑安装指南**: 请参考 [SETUP.md](SETUP.md)

快速安装（干净环境）：

```bash
# 1. 创建 conda 环境
conda create -n sentiment python=3.10 -y
conda activate sentiment

# 2. 安装 Unsloth（官方脚本，避免依赖冲突）
curl -fsSL https://unsloth.ai/install.sh | sh

# 3. 安装其他依赖
pip install -r requirements.txt

# 4. 克隆项目
git clone https://github.com/kotoriee/sentiment-distillation-pipeline.git
```

## 依赖安装

完整依赖清单见 [requirements.txt](requirements.txt)

推荐使用官方Unsloth安装脚本：

```bash
curl -fsSL https://unsloth.ai/install.sh | sh
```

⚠️ **不要使用 `pip install unsloth`**，会导致 transformers 版本冲突。

## Colab训练

如果本地VRAM不足，可使用Colab：

1. 上传 `data/train_700.json` 到Colab
2. 运行 `notebooks/colab_training.ipynb`
3. 下载训练好的LoRA adapter

## 项目来源

整理自以下原始目录：
- `ecommerce-review-analysis/` - 主项目
- `workspaces/rationale_distillation_9k/` - 9k数据标注
- `gamma4/` - Gemma 4实验