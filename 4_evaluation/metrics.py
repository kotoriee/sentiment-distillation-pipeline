#!/usr/bin/env python3
"""
评估框架 - Phase 5 起点

计算情感分析三路对比指标：
  - F1-macro, Per-class F1
  - Confusion Matrix
  - 与 SVM baseline 对比

Usage:
    python evaluation/metrics.py \
        --predictions data/processed/test_predictions.jsonl \
        --ground-truth data/processed/test.json \
        --baseline results/svm_results.json \
        --output results/phase3_comparison.json
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional


LABEL_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}


# ============== 核心指标计算 ==============

def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    """计算分类指标（不依赖 sklearn，纯 Python 实现）"""
    labels = [0, 1, 2]
    n = len(labels)

    # 混淆矩阵
    cm = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n and 0 <= p < n:
            cm[t][p] += 1

    # Per-class precision / recall / F1
    per_class = {}
    for i in labels:
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in labels) - tp
        fn = sum(cm[i][j] for j in labels) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        per_class[LABEL_NAMES[i]] = {
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "support":   sum(cm[i]),
        }

    # Macro F1
    f1_macro = sum(v["f1"] for v in per_class.values()) / n

    # Accuracy
    accuracy = sum(cm[i][i] for i in labels) / len(y_true) if y_true else 0.0

    return {
        "accuracy":   round(accuracy, 4),
        "f1_macro":   round(f1_macro, 4),
        "per_class":  per_class,
        "confusion_matrix": cm,
        "total_samples": len(y_true),
    }


def print_report(metrics: Dict, title: str = "Results"):
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  F1-macro  : {metrics['f1_macro']:.4f}")
    print(f"\n  Per-class F1:")
    for name, vals in metrics["per_class"].items():
        print(f"    {name:8s}: P={vals['precision']:.3f}  R={vals['recall']:.3f}  "
              f"F1={vals['f1']:.3f}  (n={vals['support']})")
    print(f"\n  Confusion Matrix (rows=true, cols=pred):")
    print(f"             Neg  Neu  Pos")
    for i, row in enumerate(metrics["confusion_matrix"]):
        print(f"  {LABEL_NAMES[i]:8s}: {row[0]:4d} {row[1]:4d} {row[2]:4d}")


# ============== 数据加载 ==============

def load_predictions(pred_path: str) -> List[Dict]:
    """加载推理结果 JSONL"""
    with open(pred_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_ground_truth(gt_path: str) -> List[int]:
    """从 test.json 加载真实标签"""
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels = []
    for item in data:
        # 支持多种格式（优先级：ground_truth_label > sentiment_label > label > output）
        if "ground_truth_label" in item:
            labels.append(int(item["ground_truth_label"]))
        elif "sentiment_label" in item:
            labels.append(int(item["sentiment_label"]))
        elif "label" in item:
            labels.append(int(item["label"]))
        elif "output" in item:
            # *_3cls.json 格式：output 字段值为字符串 "0"/"1"/"2"
            labels.append(int(item["output"]))
    return labels


# ============== 主函数 ==============

def main():
    parser = argparse.ArgumentParser(description="情感分析评估 - 三路对比")
    parser.add_argument("--predictions", type=str,
                        default="data/processed/test_predictions.jsonl",
                        help="模型预测结果 JSONL（含 predicted_label 字段）")
    parser.add_argument("--ground-truth", type=str,
                        default="data/processed/test.json",
                        help="测试集真实标签")
    parser.add_argument("--baseline", type=str,
                        default=None,
                        help="SVM baseline 结果 JSON（可选，用于对比）")
    parser.add_argument("--output", type=str,
                        default="results/phase3_comparison.json",
                        help="输出对比结果 JSON")
    args = parser.parse_args()

    # 加载预测
    if not Path(args.predictions).exists():
        print(f"错误: 预测文件不存在: {args.predictions}")
        print("请先运行推理脚本生成预测结果")
        return

    predictions = load_predictions(args.predictions)
    y_pred = [p.get("predicted_label", p.get("sentiment", 1)) for p in predictions]

    # 加载真实标签
    if Path(args.ground_truth).exists():
        y_true = load_ground_truth(args.ground_truth)
    else:
        # 从预测文件中取 ground_truth
        y_true = [p.get("ground_truth_label", p.get("label", 1)) for p in predictions]

    if len(y_true) != len(y_pred):
        print(f"警告: 标签数量不匹配 true={len(y_true)}, pred={len(y_pred)}")
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

    # 计算 LLM 指标
    llm_metrics = compute_metrics(y_true, y_pred)
    print_report(llm_metrics, "Qwen3-4B (Fine-tuned)")

    # 对比 SVM baseline
    comparison = {"qwen3_4b_finetuned": llm_metrics}

    if args.baseline and Path(args.baseline).exists():
        with open(args.baseline, "r") as f:
            svm_metrics = json.load(f)
        comparison["svm_baseline"] = svm_metrics
        print_report(svm_metrics, "SVM Baseline")

        delta = llm_metrics["f1_macro"] - svm_metrics.get("f1_macro", 0)
        print(f"\n  F1-macro 提升: {delta:+.4f} "
              f"({'LLM 更好' if delta > 0 else 'SVM 更好'})")
        comparison["f1_macro_delta"] = round(delta, 4)

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {output_path}")


if __name__ == "__main__":
    main()
