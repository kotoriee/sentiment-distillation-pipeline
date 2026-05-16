# 传统分类器延迟对比测试设计

**Date**: 2026-05-14
**Topic**: latency-comparison-traditional-classifiers

## 目标

对比三个传统分类算法在10k数据上的推理延迟：
- Logistic Regression
- MultinomialNB (朴素贝叶斯)
- LinearSVC (线性支持向量机)

## 背景

已有结果显示：
- Logistic Regression: Accuracy 0.7670, Neutral F1 0.6923
- MultinomialNB: Accuracy 0.7536, Neutral F1 0.6854
- LinearSVC: Accuracy 0.7402, Neutral F1 0.6493

需要补充延迟数据，完成传统基线的完整对比。

## 实现方案

### 脚本: `evaluate_traditional_latency.py`

位置: `05_experiments/10k_robustness/`

### 数据流程

```
train_answer_first.json (7172条) → 训练模型
eval_three_category_10k_fixed.json (9999条) → 测试延迟
```

### 测量方式

**排除特征提取时间，仅测量推理延迟**:

```python
# 1. 特征提取（训练+测试）
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), ...)
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)  # 记录此时间

# 2. 训练模型（记录训练时间）
model.fit(X_train, train_labels)

# 3. 推理延迟（精确测量）
t0 = time.perf_counter()
predictions = model.predict(X_test)
infer_time = time.perf_counter() - t0
```

### 模型配置

| 模型 | 配置 |
|------|------|
| Logistic Regression | max_iter=1000, class_weight="balanced", C=1.0 |
| MultinomialNB | alpha=1.0 (默认) |
| LinearSVC | max_iter=1000, class_weight="balanced", C=1.0 |

### 输出指标

| 指标 | 说明 |
|------|------|
| Accuracy | 准确率 |
| Macro F1 | 宏平均F1 |
| Neutral F1 | 中性类F1 |
| Feature extraction time | TF-IDF转换时间 |
| Training time | 训练耗时 |
| Inference time | 推理延迟（ms/条） |
| Throughput | 吞吐量（samples/s） |

### 输出格式

```
============================================================
传统分类器延迟对比 (10k)
============================================================

模型                 Accuracy   Macro-F1   Neu F1     Train(s)   Infer(ms)  Throughput
----------------------------------------------------------------------------------
Logistic Regression  0.7670     0.7683     0.6923     5.12       0.01       1000000
MultinomialNB        0.7536     0.7556     0.6854     0.05       0.02       500000
LinearSVC            0.7402     0.7404     0.6493     8.34       0.03       333333

结果已保存: results/traditional_latency_comparison.json
```

## 文件结构

```
05_experiments/10k_robustness/
├── evaluate_traditional_latency.py  # 新脚本
├── results/
│   └── traditional_latency_comparison.json
```

## 验证方式

1. 与已有Logistic Regression结果交叉验证（确保一致）
2. 测量多次取平均（减少测量误差）
3. 使用 `perf_counter()` 精确计时

## 时间估计

- 实现脚本: 10分钟
- 运行测试: 2分钟（三个模型训练+推理）
- 总计: 约15分钟