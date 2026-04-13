"""
Step 4: 结果汇总报告
对比 SVM 和 BERT 在清洗后的数据上的表现
"""

import json
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("experiments/logs")
REPORT_FILE = OUTPUT_DIR / "baseline_clean_results.txt"

def load_results():
    """加载 SVM 和 BERT 的结果"""
    # SVM 结果（从之前运行中获取）
    svm_results = {
        'model': 'SVM (TF-IDF)',
        'test_accuracy': 0.6848,
        'test_macro_f1': 0.6913,
        'test_weighted_f1': 0.6849
    }

    # BERT 结果（从之前运行中获取）
    bert_results = {
        'model': 'BERT (bert-base-uncased)',
        'test_accuracy': 0.8331,
        'test_macro_f1': 0.8390,
        'test_weighted_f1': 0.8331
    }

    return svm_results, bert_results

def generate_report():
    """生成汇总报告"""
    svm_results, bert_results = load_results()

    report = []
    report.append("="*70)
    report.append("数据降噪组 (Data Denoising Setup) - 基线模型对比实验报告")
    report.append("="*70)
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    report.append("【实验设置】")
    report.append("-" * 70)
    report.append("数据集: 8616 条 LLM 清洗后的软标注数据")
    report.append("数据划分: Train 6892 (80%) / Val 861 (10%) / Test 863 (10%)")
    report.append("标签来源: 从 probabilities 数组提取 argmax 作为 Cleaned Label")
    report.append("评估指标: Accuracy, Macro-F1, Weighted-F1")
    report.append("")

    report.append("【结果对比】")
    report.append("-" * 70)
    report.append(f"{'模型':<30} {'Accuracy':>12} {'Macro-F1':>12} {'Weighted-F1':>12}")
    report.append("-" * 70)
    report.append(f"{svm_results['model']:<30} {svm_results['test_accuracy']:>11.2%} {svm_results['test_macro_f1']:>11.2%} {svm_results['test_weighted_f1']:>11.2%}")
    report.append(f"{bert_results['model']:<30} {bert_results['test_accuracy']:>11.2%} {bert_results['test_macro_f1']:>11.2%} {bert_results['test_weighted_f1']:>11.2%}")
    report.append("-" * 70)
    report.append("")

    report.append("【性能提升】")
    report.append("-" * 70)
    acc_improvement = (bert_results['test_accuracy'] - svm_results['test_accuracy']) / svm_results['test_accuracy'] * 100
    f1_improvement = (bert_results['test_macro_f1'] - svm_results['test_macro_f1']) / svm_results['test_macro_f1'] * 100
    report.append(f"BERT vs SVM:")
    report.append(f"  - Accuracy 提升: {acc_improvement:.1f}% ({svm_results['test_accuracy']:.2%} → {bert_results['test_accuracy']:.2%})")
    report.append(f"  - Macro-F1 提升: {f1_improvement:.1f}% ({svm_results['test_macro_f1']:.2%} → {bert_results['test_macro_f1']:.2%})")
    report.append("")

    report.append("【关键发现】")
    report.append("-" * 70)
    report.append("1. BERT 在清洗后的数据上显著优于 SVM (TF-IDF)")
    report.append("   - BERT 利用预训练知识，能更好理解文本语义")
    report.append("   - SVM 依赖词袋模型，丢失上下文信息")
    report.append("")
    report.append("2. SVM 的 baseline 表现:")
    report.append(f"   - Accuracy 68.48%，Macro-F1 69.13%")
    report.append("   - 作为轻量级方案，仍有实用价值")
    report.append("")
    report.append("3. BERT 的优势:")
    report.append(f"   - Accuracy 83.31%，Macro-F1 83.90%")
    report.append("   - 比 SVM 高出约 15 个百分点")
    report.append("   - 在三分类任务上表现均衡")
    report.append("")

    report.append("【结论】")
    report.append("-" * 70)
    report.append("在 LLM 清洗后的纯净数据集上:")
    report.append("- BERT 是更好的选择，准确率超过 83%")
    report.append("- SVM 可作为快速基线，准确率约 68%")
    report.append("- 模型选择应权衡性能与部署成本")
    report.append("")

    report.append("="*70)
    report.append("报告生成完成")
    report.append("="*70)

    return "\n".join(report)

def main():
    print("="*70)
    print("Step 4: 生成结果汇总报告")
    print("="*70)

    # 生成报告
    report = generate_report()

    # 保存报告
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)

    # 打印报告
    print("\n" + report)

    print(f"\n报告已保存: {REPORT_FILE}")

if __name__ == "__main__":
    main()
