# 模块5：基线模型

传统机器学习和深度学习基线模型对比实验。

## 目录结构

```
5_baseline_models/
├── svm/                    # SVM分类器
│   ├── svm_classifier.py   # 核心SVM实现
│   ├── train_svm_denoising.py    # 去噪训练
│   ├── train_svm_optimized.py    # 优化版本
│   └── step2_train_svm_clean.py  # 清洗数据训练
│
├── bert/                   # BERT模型
│   ├── step3_train_bert_clean.py     # 清洗数据训练
│   ├── train_bert_denoising.py       # 去噪训练
│   ├── train_bert_optimized.py       # 优化版本
│   └ train_bert_soft_label.py        # 软标签训练
│
├── topic_modeling/         # 主题模型（GSDMM/LDA）
│   ├── gsdmm_model.py      # GSDMM实现
│   ├── lda_model.py        # LDA实现
│   └ train_gsdmm_baseline.py         # GSDMM训练
│
├── step1_prepare_clean_data.py  # 数据准备
├── step4_generate_report.py     # 报告生成
└ evaluate_with_latency.py       # 延迟评估
```

## 使用方法

### 1. 数据准备

```bash
python step1_prepare_clean_data.py
```

### 2. SVM训练

```bash
cd svm
python train_svm_optimized.py --data ../data/train.json
```

### 3. BERT训练

```bash
cd bert
python train_bert_optimized.py --data ../data/train.json --epochs 3
```

### 4. 主题模型

```bash
cd topic_modeling
python train_gsdmm_baseline.py --data ../data/train.json
```

## 历史实验结果

| 模型 | 准确率 | 训练时间 | 备注 |
|------|--------|---------|------|
| SVM + TF-IDF | 55.17% | ~5分钟 | 传统ML基线 |
| BERT-base | 67% | ~2小时 | 深度学习基线 |
| GSDMM | - | ~10分钟 | 主题聚类（非分类） |

详细对比结果见 `../6_experiments_results/THREE_WAY_COMPARISON_REPORT.md`