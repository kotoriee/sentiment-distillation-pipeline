#!/usr/bin/env python3
"""
合并三品类 CoT 数据集

将 Beauty / Electronics / Pet_Supplies 三个 JSONL 文件合并为一个训练集。
保留 category 字段用于论文分析（各品类准确率对比）。

Usage:
    python code/cloud_agent/merge_datasets.py

    # 指定文件
    python code/cloud_agent/merge_datasets.py \
        --files data/processed/cot_allbeauty_3333.jsonl \
                data/processed/cot_electronics_3333.jsonl \
                data/processed/cot_petsupplies_3333.jsonl \
        --output data/processed/cot_merged_10000.jsonl
"""

import json
import random
import argparse
from pathlib import Path
from collections import Counter


DEFAULT_FILES = [
    "data/processed/cot_allbeauty_3333.jsonl",
    "data/processed/cot_electronics_3333.jsonl",
    "data/processed/cot_petsupplies_3333.jsonl",
]
DEFAULT_OUTPUT = "data/processed/cot_merged_10000.jsonl"


def main():
    parser = argparse.ArgumentParser(description="合并三品类 CoT 数据集")
    parser.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-confidence", type=float, default=0.65,
                        help="过滤低置信度样本 (default: 0.65)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    all_records = []
    category_stats = {}

    for filepath in args.files:
        p = Path(filepath)
        if not p.exists():
            print(f"⚠️  文件不存在，跳过: {filepath}")
            continue

        # 从文件名推断品类
        stem = p.stem  # e.g. "cot_allbeauty_3333"
        parts = stem.split("_")
        category = parts[1] if len(parts) >= 2 else "unknown"

        records = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        # 过滤低置信度
        before = len(records)
        records = [r for r in records if r.get("confidence", 0) >= args.min_confidence]

        # 添加品类字段
        for r in records:
            r["category"] = category

        all_records.extend(records)
        dist = Counter(r["predicted_label"] for r in records)
        category_stats[category] = {
            "total": len(records),
            "filtered": before - len(records),
            "neg": dist.get(0, 0),
            "pos": dist.get(1, 0),
        }
        print(f"  {category}: {len(records)} 条 (过滤 {before - len(records)} 低置信度)")

    if not all_records:
        print("错误：没有加载到任何数据")
        return

    # 去重（按 id）
    seen_ids = set()
    deduped = []
    for r in all_records:
        rid = r.get("id", "")
        if rid not in seen_ids:
            seen_ids.add(rid)
            deduped.append(r)
    print(f"\n去重: {len(all_records)} → {len(deduped)} 条")

    # 打乱
    random.shuffle(deduped)

    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 统计
    total_dist = Counter(r["predicted_label"] for r in deduped)
    print(f"\n{'='*50}")
    print(f"合并完成: {len(deduped)} 条 → {output_path}")
    print(f"标签分布: neg={total_dist[0]}, pos={total_dist[1]}")
    print(f"\n品类分布:")
    for cat, stats in category_stats.items():
        print(f"  {cat}: {stats['total']} 条 (neg={stats['neg']}, pos={stats['pos']})")
    print(f"\n下一步: python code/local_llm/data_formatter.py --input {output_path}")


if __name__ == "__main__":
    main()
