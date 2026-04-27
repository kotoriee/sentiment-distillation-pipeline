# 实验日志：答案优先训练集 — 模型评估汇总

## 日期
2026-04-26

## 实验概述
在 `data/test_answer_first.json`（897 条，3 类均衡各 299）上评估多个模型。

---

## 1. Qwen3.5-9B QLoRA（答案优先训练）

| 指标 | 值 |
|------|-----|
| Accuracy | **82.27%** |
| Macro-F1 | **82.29%** |
| F1 Neg | 82.18% |
| F1 Neu | 82.12% |
| F1 Pos | 82.56% |
| 解析失败 | 0 |
| 正确/总数 | 738/897 |
| 硬标签基线 | 79.69% |
| **vs 基线** | **+2.58%** |

**结论**: 超过硬标签基线，答案优先+SFT+KL 蒸馏方案有效。

---

## 2. Gemma4 E4B QLoRA（答案优先训练）

| 指标 | 值 |
|------|-----|
| Accuracy | **33.33%** |
| Macro-F1 | **16.67%** |
| 解析失败 | 0 |
| 正确/总数 | 299/897 |
| 混淆矩阵 | 全部预测为 1 (Neutral) |

**结论**: 模型决策坍缩到中性类。不是评估流程问题，而是训练后输出退化。
**建议**: 检查训练数据质量、调整 alpha（KL 权重）、降低 temperature。

---

## 3. DeepSeek-V4-Flash (NVIDIA API)

| 指标 | 值 |
|------|-----|
| Accuracy | **77.04%** |
| Macro-F1 | **76.17%** |
| F1 Neg | 78.97% |
| F1 Neu | 60.85% |
| F1 Pos | 88.70% |
| API 错误 | 174/897 (19.4%) |
| 有效预测 | 723/897 |
| 混淆矩阵 | Neg: 231/6/2, Neu: 108/122/14, Pos: 7/29/204 |

**结论**: 零样本表现低于微调后的 Qwen3.5-9B（77.04% vs 82.27%），说明蒸馏微调有价值。Neutral 类最难判断（F1 仅 60.85%）。

---

## 评估脚本改进

### `evaluate_lora.py` 修复项
1. **Qwen3.5-9B**: processor 需要显式传 `text=batch`，否则会把 prompt 当图片输入
2. **评估长度**: 从 256 调到 512，使用左截断（`truncation_side = "left"`），避免长 review 把 assistant prompt 截没
3. **结果文件**: 保存 metrics + raw predictions，方便排查

### 结果文件
- `models/qwen35-9b-answer-first/qwen35_9b_answer_first_eval.json`
- `models/gemma4-e4b-answer-first/gemma4_e4b_answer_first_eval.json`

---

## 对比基线

| 方法 | Accuracy | Macro-F1 | 备注 |
|------|----------|----------|------|
| SVM Baseline | 59.0% | 0.595 | 硬标签 TF-IDF |
| DeepSeek-V4-Flash (零样本) | **77.04%** | **76.17%** | NVIDIA API, 723 有效 |
| 硬标签 Qwen3-4B | 79.69% | — | 原始训练方案 |
| **Qwen3.5-9B（答案优先）** | **82.27%** | **82.29%** | SFT+KL 蒸馏, 3 epochs |
| Gemma4 E4B（答案优先） | 33.33% | 16.67% | 坍缩到 Neutral |
