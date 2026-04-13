# 动态温度缩放实验

## 最新成果

### 核心突破: 95.80% 准确率

| 指标 | 动态温度 (本实验) | 固定温度 (基线) | 提升 |
|------|-------------------|-----------------|------|
| **准确率** | **95.80%** | 86.50% | **+9.30%** |
| 中性类召回 | 94.53% | 80.5% | +14.03% |
| ECE (校准误差) | 0.0162 | - | 优秀 |
| 置信度-准确率差距 | 0.0122 | - | 优秀 |

**关键发现**: 动态温度策略显著优于固定温度，特别是在解决中性类识别难题上效果突出。

详细实验记录: [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md)

## 背景

当前训练使用固定温度 T=2.0 生成软标签。研究表明，根据样本置信度自适应调整温度可能获得更好的蒸馏效果。

## 核心思想

**来源**: EMNLP 2024 "Calibrating Long-form Generations From Large Language Models"

不同置信度的样本应该使用不同的温度：
- 高置信度样本：低温度保留锐利分布
- 低置信度样本：高温度平滑分布

## 实验方案

### 方案1：基于置信度的动态温度

```python
def adaptive_temperature(confidence: float) -> float:
    """
    根据样本置信度动态选择温度

    Args:
        confidence: 教师模型置信度 (max probability)

    Returns:
        temperature: 蒸馏温度
    """
    if confidence > 0.9:
        return 1.5  # 高置信度，锐利分布
    elif confidence > 0.6:
        return 2.0  # 中等置信度，当前设置
    else:
        return 2.5 + (0.6 - confidence) * 2  # 最高到3.0
```

### 方案2：可学习温度

将温度作为可训练参数：

```python
class AdaptiveDistillationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(2.0))

    def forward(self, student_logits, teacher_probs):
        temp = F.softplus(self.temperature) + 1.0  # 确保 > 1
        student_probs = F.softmax(student_logits / temp, dim=-1)
        return F.kl_div(student_probs.log(), teacher_probs, reduction='batchmean')
```

## 实验设计

| 实验 | 温度策略 | 预期效果 |
|------|----------|----------|
| baseline | T=2.0 固定 | 86.50% (当前基线) |
| exp1 | 基于置信度分段 | +1-2% |
| exp2 | 可学习温度 | +2-3% |

## 文件结构

```
adaptive_temperature/
├── README.md                 # 本文件
├── adaptive_temperature.py   # 动态温度训练脚本
├── learnable_temp.py        # 可学习温度实现
├── run_experiments.sh       # 批量运行脚本
└── results/                 # 实验结果
    ├── exp1_vs_baseline.json
    └── summary.md
```

## 快速开始

```bash
# 基础动态温度实验
python adaptive_temperature.py \
    --train_data ../../data/curriculum/train_full.json \
    --val_data ../../data/curriculum/val_fixed.json \
    --strategy confidence_based \
    --output_dir ./results/exp1

# 可学习温度实验
python learnable_temp.py \
    --train_data ../../data/curriculum/train_full.json \
    --val_data ../../data/curriculum/val_fixed.json \
    --epochs 3 \
    --output_dir ./results/exp2
```

## 评估指标

- 准确率（Accuracy）
- 各类别 F1 分数
- 期望校准误差（ECE）
- 训练时间对比

## 注意事项

1. 动态温度需要在数据加载阶段计算每个样本的置信度
2. 可学习温度可能需要调整学习率（建议降低至 1e-4）
3. 所有实验保持其他超参数一致（LoRA r=16, batch=16）
