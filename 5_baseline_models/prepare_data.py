#!/usr/bin/env python3
"""
数据格式转换脚本 - 将JSON转换为CSV格式供基线模型使用

将 sentiment-distillation-pipeline/data/*.json 转换为：
1. CSV格式（id, text, label）供SVM/BERT硬标签训练
2. JSONL格式（id, text, probabilities, hard_label）供BERT软标签训练

Usage:
    python prepare_data.py
"""

import json
import csv
from pathlib import Path

# 配置
SCRIPT_DIR = Path(__file__).parent  # 5_baseline_models/
PROJECT_DIR = SCRIPT_DIR.parent  # sentiment-distillation-pipeline/
DATA_DIR = PROJECT_DIR / "data"  # sentiment-distillation-pipeline/data/
OUTPUT_DIR = DATA_DIR / "processed" / "baseline"

SPLITS = ["train", "val", "test"]

def convert_json_to_csv(split: str):
    """将JSON转换为CSV格式"""
    input_path = DATA_DIR / f"{split}.json"
    output_csv = OUTPUT_DIR / f"{split}.csv"

    if not input_path.exists():
        print(f"跳过: {input_path} 不存在")
        return 0

    # 加载JSON
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 写入CSV
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "label"])

        for item in data:
            writer.writerow([
                item.get("id", ""),
                item.get("text", ""),
                item.get("label", 1)
            ])

    print(f"转换完成: {input_path} → {output_csv} ({len(data)}条)")
    return len(data)


def convert_to_soft_labels_jsonl():
    """创建软标签JSONL供BERT蒸馏训练"""
    input_path = DATA_DIR / "train.json"
    output_jsonl = OUTPUT_DIR / "soft_labels.jsonl"

    if not input_path.exists():
        print(f"跳过: {input_path} 不存在")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in data:
            record = {
                "id": item.get("id", ""),
                "text": item.get("text", ""),
                "probabilities": item.get("soft_labels", [0.33, 0.33, 0.34]),
                "hard_label": item.get("label", 1),
                "confidence": item.get("confidence", 0.5),
                "split": "train"
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"软标签JSONL: {output_jsonl} ({len(data)}条)")


def main():
    print("=" * 60)
    print("数据格式转换 - 基线模型准备")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 转换CSV
    total = 0
    for split in SPLITS:
        total += convert_json_to_csv(split)

    # 创建软标签JSONL
    convert_to_soft_labels_jsonl()

    print(f"\n总计转换: {total}条")
    print("\n输出文件:")
    for f in OUTPUT_DIR.iterdir():
        print(f"  {f}")

    # 验证CSV格式
    print("\n验证CSV格式:")
    test_csv = OUTPUT_DIR / "train.csv"
    with open(test_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        print(f"  列名: {header}")
        first_row = next(reader)
        print(f"  首行: id={first_row[0][:20]}, label={first_row[2]}")

    print("\n下一步:")
    print("  cd 5_baseline_models/svm && python train_svm.py")
    print("  cd 5_baseline_models/bert && python train_bert.py")


if __name__ == "__main__":
    main()