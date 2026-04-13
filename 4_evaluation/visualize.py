#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化模块 — Phase 5

生成三路模型对比图表，保存到 reports/ 目录。

Usage:
    python evaluation/visualize.py --results reports/comparison_results.json
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script use
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


ROUTE_LABELS = {
    "svm":  "SVM+TF-IDF",
    "api":  "DeepSeek API",
    "qwen": "Qwen3-4B",
}

CLASS_LABELS = ["Neg", "Neu", "Pos"]
COLORS = ["#4e79a7", "#f28e2b", "#59a14f"]  # Blue, Orange, Green


# ─── F1 对比柱状图 ────────────────────────────────────────────────────────────

def plot_f1_comparison(results: Dict, output_path: str):
    """
    分组柱状图：3 类 F1 × 可用路线数。
    每组最后加一条 F1-macro 总体指标。
    """
    routes = list(results.keys())
    n_routes = len(routes)
    class_names = ["Negative", "Neutral", "Positive"]
    metric_names = class_names + ["Macro"]

    # Build data matrix: (n_metrics, n_routes)
    data = np.zeros((len(metric_names), n_routes))
    for j, route in enumerate(routes):
        m = results[route]
        for i, cls in enumerate(class_names):
            data[i][j] = m["per_class"][cls]["f1"]
        data[3][j] = m["f1_macro"]

    fig, ax = plt.subplots(figsize=(max(8, n_routes * 2.5), 5))
    x = np.arange(len(metric_names))
    width = 0.8 / n_routes

    for j, route in enumerate(routes):
        offset = (j - n_routes / 2 + 0.5) * width
        bars = ax.bar(x + offset, data[:, j], width,
                      label=ROUTE_LABELS.get(route, route),
                      color=COLORS[j % len(COLORS)],
                      alpha=0.85, edgecolor="white")
        # Value labels on bars
        for bar in bars:
            h = bar.get_height()
            if h > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Metric")
    ax.set_ylabel("F1 Score")
    ax.set_title("Three-Way Model Comparison — F1 Scores")
    ax.set_xticks(x)
    ax.set_xticklabels(["F1-Neg", "F1-Neu", "F1-Pos", "F1-Macro"], fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper right")
    ax.axvline(x=2.5, color="gray", linestyle="--", alpha=0.4)  # Separator before Macro
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ─── 混淆矩阵热力图 ───────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: List[List[int]], title: str, output_path: str):
    """
    3×3 混淆矩阵热力图（行=真实，列=预测）。
    使用 seaborn（若可用）或 matplotlib 纯实现。
    """
    cm_array = np.array(cm, dtype=float)
    # Normalize by row (recall view)
    row_sums = cm_array.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid div by zero
    cm_norm = cm_array / row_sums

    fig, ax = plt.subplots(figsize=(5, 4))

    if HAS_SEABORN:
        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
            ax=ax, vmin=0, vmax=1,
            annot_kws={"size": 11},
        )
        # Add raw counts as secondary annotation
        for i in range(3):
            for j in range(3):
                ax.text(j + 0.5, i + 0.75, f"({int(cm[i][j])})",
                        ha="center", va="center", fontsize=8, color="gray")
    else:
        im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues,
                       vmin=0, vmax=1)
        plt.colorbar(im, ax=ax)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm_norm[i][j]:.2f}\n({int(cm[i][j])})",
                        ha="center", va="center", fontsize=9,
                        color="white" if cm_norm[i][j] > 0.6 else "black")
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(CLASS_LABELS)
        ax.set_yticklabels(CLASS_LABELS)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {title}")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ─── 标签分布图 ───────────────────────────────────────────────────────────────

def plot_label_distribution(results: Dict, output_path: str):
    """
    各路线预测标签分布柱状图（比较各模型的预测分布偏差）。
    """
    routes = list(results.keys())
    n_routes = len(routes)
    class_names = ["Negative", "Neutral", "Positive"]

    fig, axes = plt.subplots(1, n_routes, figsize=(4 * n_routes, 4), sharey=True)
    if n_routes == 1:
        axes = [axes]

    for ax, route in zip(axes, routes):
        m = results[route]
        supports = [m["per_class"][cls]["support"] for cls in class_names]
        total = sum(supports)
        pcts = [s / total * 100 if total > 0 else 0 for s in supports]

        bars = ax.bar(CLASS_LABELS, pcts, color=COLORS[:3], alpha=0.85, edgecolor="white")
        for bar, pct in zip(bars, pcts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)

        ax.set_title(ROUTE_LABELS.get(route, route))
        ax.set_ylabel("Predicted (%)" if route == routes[0] else "")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Predicted Label Distribution by Model", fontsize=12, y=1.02)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ─── 批量生成所有图表 ─────────────────────────────────────────────────────────

def generate_all_charts(results: Dict, output_dir: str):
    """生成所有图表到指定目录。"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # F1 对比柱状图
    plot_f1_comparison(results, f"{output_dir}/f1_comparison.png")

    # 各路线混淆矩阵
    for route, m in results.items():
        name = ROUTE_LABELS.get(route, route)
        plot_confusion_matrix(
            m["confusion_matrix"], name,
            f"{output_dir}/confusion_matrix_{route}.png"
        )

    # 标签分布图
    plot_label_distribution(results, f"{output_dir}/label_distribution.png")

    print(f"\nAll charts saved to: {output_dir}/")


# ─── CLI 入口 ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison charts from evaluation results"
    )
    parser.add_argument(
        "--results", default="reports/comparison_results.json",
        help="Path to comparison_results.json"
    )
    parser.add_argument(
        "--output", default="reports/",
        help="Output directory for charts"
    )
    args = parser.parse_args()

    if not Path(args.results).exists():
        print(f"Error: results file not found: {args.results}")
        print("Run run_comparison.py first to generate comparison_results.json")
        sys.exit(1)

    with open(args.results, encoding="utf-8") as f:
        results = json.load(f)

    generate_all_charts(results, args.output)


if __name__ == "__main__":
    main()
