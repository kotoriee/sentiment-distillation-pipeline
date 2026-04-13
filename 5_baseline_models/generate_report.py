#!/usr/bin/env python3
"""
基线模型对比报告生成脚本

整合SVM、BERT硬标签、BERT软标签的训练结果

Usage:
    python generate_report.py
"""

import json
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_DIR / "6_experiments_results" / "baseline_results"


def load_results(filename: str):
    path = RESULTS_DIR / filename
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    print("=" * 60)
    print("基线模型对比报告")
    print("=" * 60)

    # 加载结果
    svm = load_results("svm_results.json")
    bert = load_results("bert_results.json")
    bert_soft = load_results("bert_soft_results.json")

    results = []
    if svm:
        results.append(("SVM + TF-IDF", svm))
    if bert:
        results.append(("BERT (硬标签)", bert))
    if bert_soft:
        results.append(("BERT (软标签)", bert_soft))

    if not results:
        print("没有找到训练结果，请先运行训练脚本")
        return

    # 打印对比表格
    print("\n## 测试集准确率对比\n")
    print("| 模型 | Accuracy | Macro-F1 | 训练时间 | 推理延迟 |")
    print("|------|----------|----------|----------|----------|")

    for name, r in results:
        acc = r.get("test_accuracy", 0) * 100
        f1 = r.get("test_f1", 0) * 100
        train_t = r.get("train_time", 0)
        infer_t = r.get("inference_time_per_sample", 0) * 1000
        print(f"| {name} | {acc:.2f}% | {f1:.2f}% | {train_t:.0f}s | {infer_t:.2f}ms |")

    # 详细分析
    print("\n## 详细结果\n")

    for name, r in results:
        print(f"\n### {name}\n")
        print(f"- 训练样本: {r.get('train_samples', 'N/A')}")
        print(f"- 验证集准确率: {r.get('val_accuracy', 0)*100:.2f}%")
        print(f"- 验证集F1: {r.get('val_f1', 0)*100:.2f}%")
        print(f"- 测试集准确率: {r.get('test_accuracy', 0)*100:.2f}%")
        print(f"- 测试集F1: {r.get('test_f1', 0)*100:.2f}%")

        if "temperature" in r:
            print(f"- 温度: {r['temperature']}")
        if "epochs" in r:
            print(f"- Epochs: {r['epochs']}")
        if "lr" in r:
            print(f"- 学习率: {r['lr']}")

    # 保存报告
    report = {
        "generated_at": datetime.now().isoformat(),
        "models": {name: r for name, r in results},
        "comparison": {
            "best_accuracy": max(r.get("test_accuracy", 0) for _, r in results),
            "best_f1": max(r.get("test_f1", 0) for _, r in results),
        }
    }

    report_path = RESULTS_DIR / "comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n报告已保存: {report_path}")

    # 生成Markdown报告
    md_content = f"""# 基线模型对比报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 数据集

- 训练集: 7,172条 (Kimi k2.5软标注)
- 验证集: 896条
- 测试集: 897条

## 测试集性能对比

| 模型 | Accuracy | Macro-F1 | 训练时间 | 推理延迟 |
|------|----------|----------|----------|----------|
"""

    for name, r in results:
        acc = r.get("test_accuracy", 0) * 100
        f1 = r.get("test_f1", 0) * 100
        train_t = r.get("train_time", 0)
        infer_t = r.get("inference_time_per_sample", 0) * 1000
        md_content += f"| {name} | {acc:.2f}% | {f1:.2f}% | {train_t:.0f}s | {infer_t:.2f}ms |\n"

    md_content += """
## 分析

### SVM vs BERT

- SVM训练速度快（~2分钟），但准确率较低
- BERT训练时间较长（~30分钟），但准确率更高

### 硬标签 vs 软标签

- 软标签训练使用KL散度损失，理论上能获得更好的泛化能力
- 需要在测试集上验证是否确实有提升

## 下一步

- 对比LoRA微调模型（Qwen3-4B、Gemma 4 E2B）
- 分析各模型的错误模式
"""

    md_path = RESULTS_DIR / "BASELINE_COMPARISON.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Markdown报告: {md_path}")

    print("\n" + "=" * 60)
    print("报告生成完成")
    print("=" * 60)


if __name__ == "__main__":
    main()