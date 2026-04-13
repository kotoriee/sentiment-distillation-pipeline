# 模块6：实验结果

三路对比实验结果和动态温度实验记录。

## 目录结构

```
6_experiments_results/
├── THREE_WAY_COMPARISON_REPORT.md   # 三路对比完整报告
├── three_way_comparison_500.json    # 对比实验数据
│
├── predictions/                     # 预测结果
│   ├── svm_600_predictions.jsonl    # SVM预测
│   ├── deepseek_predictions_500.jsonl  # DeepSeek预测
│   └ local_llm_predictions_500.jsonl   # 本地LLM预测
│
├── eval_reports/                    # 评估报告
│   ├── adaptive_temp_600_eval.json  # 动态温度模型
│   ├── local_600_eval.json          # 本地模型
│   ├── soft_full_600_eval.json      # 软标签全量模型
│
└── adaptive_temperature/            # 动态温度实验代码
    └ train_adaptive_temp.py
     eval_adaptive_temp.py
```

## 三路对比实验结果

| 方法 | 准确率 | 训练成本 | 推理延迟 |
|------|--------|---------|---------|
| **SVM + TF-IDF** | 55.17% | CPU 5分钟 | <1ms |
| **DeepSeek API** | ~70% | $0.01/条 | ~15s |
| **Local QLoRA (动态温度)** | 67.17% | 4小时 GPU | ~50ms |

### 关键发现

1. **验证集 vs 测试集差距**: 动态温度模型验证集95.80% vs 测试集67.17%（差距28.63%）
2. **中性类偏好**: 158个误分类中157个被分到中性
3. **校准度差**: ECE=0.2589，置信度92.72% vs 实际67.17%

## 动态温度实验

动态温度蒸馏（temperature=2.0）相较于固定温度的优势：

```bash
cd adaptive_temperature
python train_adaptive_temp.py --data ../data/train.json --test
```

## 下一步改进

- [ ] 统一评估脚本支持多种输出格式
- [ ] 添加评估前格式验证
- [ ] 分析验证集-测试集差距根因