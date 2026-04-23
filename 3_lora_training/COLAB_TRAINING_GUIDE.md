# Qwen3.5-4B Colab 训练指南

## 训练方法对比

| 方法 | Notebook | 特点 |
|------|----------|------|
| **软标签蒸馏** | `Qwen3_5_Answer_First_Colab.ipynb` | KL Divergence + SFT 混合损失 |
| **GRPO强化学习** | `Qwen3_5_GRPO_Sentiment_Colab.ipynb` | 奖励函数驱动的强化学习 |

---

## 软标签蒸馏训练

### 需要上传的文件

| 文件 | 大小 | 用途 |
|------|------|------|
| `Qwen3_5_Answer_First_Colab.ipynb` | 20KB | Colab notebook |
| `train_answer_first.json` | 15.8MB | 训练数据 |

### 使用步骤

### 1. 上传 Notebook 到 Colab

1. 打开 [Google Colab](https://colab.research.google.com/)
2. 选择 **文件 → 上传笔记本**
3. 上传 `Qwen3_5_Answer_First_Colab.ipynb`

### 2. 启用 GPU

1. 选择 **运行时 → 更改运行时类型**
2. 选择 **T4 GPU**
3. 点击 **保存**

### 3. 运行训练

按顺序运行所有 cell：
1. **安装依赖** - 安装 unsloth 和相关库
2. **上传数据** - 上传 `train_answer_first.json`
3. **训练配置** - 确认参数
4. **加载模型** - 加载 Qwen3.5-4B
5. **数据预处理** - 预处理训练数据
6. **开始训练** - 运行软标签蒸馏训练 (~2小时)
7. **保存模型** - 保存 LoRA 适配器
8. **下载模型** - 下载到本地

### 4. 本地评估

训练完成后：
1. 解压下载的 `qwen35-4b-answer-first.zip`
2. 放到 `models/qwen35-4b-answer-first/`
3. 运行评估：

```bash
cd 4_evaluation
python3 eval_answer_first.py --model ../3_lora_training/models/qwen35-4b-answer-first
```

## 训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| MAX_SEQ_LENGTH | 512 | 最大序列长度 |
| LORA_RANK | 16 | LoRA秩 |
| EPOCHS | 3 | 训练轮数 |
| TEMPERATURE | 2.0 | KL温度 |
| ALPHA | 0.5 | KL权重 |

## 对比目标

| 模型 | 准确率 |
|------|--------|
| Qwen3-4B 固定温度 | 80.38% |
| Qwen3-4B 动态温度 | 80.71% |
| **Qwen3.5-4B (本实验)** | 待测试 |

## 可选：动态温度训练

在 cell 3 中设置 `USE_DYNAMIC_TEMP = True`，可以启用动态温度策略。

动态温度策略：
- 高置信度 (>0.9): T=1.5
- 中置信度 (0.6-0.9): T=2.0
- 低置信度 (<0.6): T=2.5-3.0

---

## GRPO 强化学习训练

### 需要上传的文件

| 文件 | 大小 | 用途 |
|------|------|------|
| `Qwen3_5_GRPO_Sentiment_Colab.ipynb` | 15KB | GRPO notebook |
| `train_answer_first.json` | 15.8MB | 训练数据 |

### 使用步骤

1. **上传 Notebook** - 上传 `Qwen3_5_GRPO_Sentiment_Colab.ipynb`
2. **启用 T4 GPU** - 运行时 → 更改运行时类型 → T4 GPU
3. **运行训练** - 按顺序运行所有 cell：
   - Cell 1: 安装依赖 (unsloth, vllm, trl)
   - Cell 2: 上传数据 (`train_answer_first.json`)
   - Cell 3: 配置参数
   - Cell 4: 加载模型
   - Cell 5: 数据预处理
   - Cell 6: 定义奖励函数
   - Cell 7: GRPO 训练 (~1小时，100步测试)
   - Cell 8: 保存 LoRA
   - Cell 9: 下载模型

### GRPO 训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| MAX_SEQ_LENGTH | 512 | 最大序列长度 |
| LORA_RANK | 32 | LoRA秩（GRPO使用更大） |
| MAX_STEPS | 100 | 测试步数 |
| NUM_GENERATIONS | 4 | 每prompt生成响应数 |
| TEMPERATURE | 1.0 | GRPO探索温度 |
| LEARNING_RATE | 5e-6 | 学习率 |

### GRPO 奖励函数

| 奖励函数 | 值 | 说明 |
|----------|-----|------|
| `match_format_exactly` | +3.0 | 格式完全正确 |
| `match_format_approximately` | +0.5/-1.0 | 格式部分正确 |
| `check_sentiment` | +5.0/-1.5 | 情感分类正确/错误 |
| `check_sentiment_printed` | +3.5/-1.5 | 打印输出的奖励 |

### 本地评估

```bash
cd 4_evaluation
python3 eval_answer_first.py --model ../3_lora_training/models/sentiment_grpo_lora
```

---

## 方法对比建议

| 场景 | 推荐方法 |
|------|----------|
| 有软标签数据 | **软标签蒸馏** |
| 无软标签数据 | **GRPO强化学习** |
| 需要更精确控制 | **GRPO强化学习** |

两种方法可以结合使用：先用软标签蒸馏预训练，再用GRPO强化学习优化。