#!/usr/bin/env python3
"""
数据质量检查模块 - 整合所有check脚本功能

Usage:
    python quality_check.py --data path/to/data.json --check all
    python quality_check.py --data path/to/data.json --check distribution
    python quality_check.py --data path/to/data.json --check soft_labels
    python quality_check.py --data path/to/data.json --check cot
"""

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def load_data(data_path: str) -> list:
    """加载JSON或JSONL数据文件"""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    if path.suffix == '.jsonl':
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    else:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


# ============================================================
# 1. 分布检查 (合并 check_dist.py, check_train_full.py, check_balance.py)
# ============================================================
def check_distribution(data: list, verbose: bool = True) -> dict:
    """检查标签分布和rating分布"""
    if verbose:
        print("=" * 60)
        print("Distribution Check")
        print("=" * 60)

    stats = {}

    # 总样本数
    stats['total'] = len(data)
    if verbose:
        print(f"Total samples: {stats['total']}")

    # 标签分布
    labels = [item.get('label', item.get('hard_label', 1)) for item in data]
    label_dist = Counter(labels)
    stats['labels'] = dict(label_dist)

    if verbose:
        print("\nLabel distribution:")
        for k in sorted(label_dist.keys()):
            name = ['Negative', 'Neutral', 'Positive'][int(k)]
            pct = label_dist[k] / len(data) * 100
            print(f"  {name}({k}): {label_dist[k]} ({pct:.1f}%)")

    # Rating分布
    ratings = [item.get('rating', 3) for item in data]
    rating_dist = Counter(ratings)
    stats['ratings'] = dict(rating_dist)

    if verbose:
        print("\nRating distribution:")
        for r in sorted(rating_dist.keys()):
            pct = rating_dist[r] / len(data) * 100
            print(f"  {r} stars: {rating_dist[r]} ({pct:.1f}%)")

    # Rating-Label一致性检查
    consistent = 0
    for item in data:
        rating = item.get('rating', 3)
        label = item.get('label', 1)
        expected_label = 0 if rating <= 2 else (1 if rating == 3 else 2)
        if label == expected_label:
            consistent += 1

    stats['rating_label_consistency'] = consistent / len(data) * 100
    if verbose:
        print(f"\nRating-Label consistency: {consistent}/{len(data)} ({stats['rating_label_consistency']:.1f}%)")

    # 平衡性检查 - 如果需要扩展到目标数量
    if verbose and len(data) < 9000:
        print(f"\nTo reach 3000 per class:")
        for k in [0, 1, 2]:
            name = ['Negative', 'Neutral', 'Positive'][k]
            needed = 3000 - label_dist.get(k, 0)
            print(f"  {name}: +{needed} needed")

    return stats


# ============================================================
# 2. 软标签检查 (合并 check_quality.py)
# ============================================================
def check_soft_labels(data: list, verbose: bool = True) -> dict:
    """检查软标签与硬标签的对齐情况"""
    if verbose:
        print("=" * 60)
        print("Soft Labels Check")
        print("=" * 60)

    stats = {}

    # 检查是否有soft_labels
    has_soft = data and 'soft_labels' in data[0]
    stats['has_soft_labels'] = has_soft
    if verbose:
        print(f"Has soft_labels: {'Yes' if has_soft else 'No'}")

    if not has_soft:
        return stats

    # 检查argmax与label对齐
    mismatch = 0
    mismatch_samples = []
    for item in data:
        soft = item.get('soft_labels', [0.33, 0.33, 0.34])
        label = item.get('label', 1)
        predicted = soft.index(max(soft))
        if predicted != label:
            mismatch += 1
            if len(mismatch_samples) < 5:
                mismatch_samples.append({
                    'text': item.get('text', '')[:50],
                    'soft_labels': soft,
                    'label': label,
                    'predicted': predicted
                })

    stats['mismatch_count'] = mismatch
    stats['mismatch_pct'] = mismatch / len(data) * 100
    stats['mismatch_samples'] = mismatch_samples

    if verbose:
        print(f"Mismatches (argmax(soft) != label): {mismatch}/{len(data)} ({stats['mismatch_pct']:.2f}%)")
        if mismatch_samples:
            print("\nSample mismatches:")
            for s in mismatch_samples:
                print(f"  Text: {s['text']}...")
                print(f"  Soft: {s['soft_labels']}, Label: {s['label']}, Pred: {s['predicted']}")

    # 置信度分布
    confidences = [max(item.get('soft_labels', [0.33, 0.33, 0.34])) for item in data]
    stats['confidence_mean'] = statistics.mean(confidences) if confidences else 0
    stats['confidence_median'] = statistics.median(confidences) if confidences else 0
    stats['confidence_min'] = min(confidences) if confidences else 0
    stats['confidence_max'] = max(confidences) if confidences else 0

    if verbose:
        print(f"\nConfidence stats:")
        print(f"  Mean: {stats['confidence_mean']:.3f}")
        print(f"  Median: {stats['confidence_median']:.3f}")
        print(f"  Min: {stats['confidence_min']:.3f}")
        print(f"  Max: {stats['confidence_max']:.3f}")

        # 分段统计
        ranges = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
        print("\nConfidence distribution:")
        for lo, hi in ranges:
            count = sum(1 for c in confidences if lo <= c < hi)
            pct = count / len(confidences) * 100
            print(f"  [{lo:.1f}, {hi:.1f}): {count} ({pct:.1f}%)")

    # 低置信度样本
    low_conf = sum(1 for c in confidences if c < 0.6)
    stats['low_confidence_count'] = low_conf
    stats['low_confidence_pct'] = low_conf / len(data) * 100

    if verbose:
        print(f"\nLow confidence (<0.6): {low_conf} ({stats['low_confidence_pct']:.2f}%)")

    return stats


# ============================================================
# 3. CoT检查 (合并 check_cot_rating.py, check_cot_v2.py)
# ============================================================
def check_cot_section(data: list, verbose: bool = True) -> dict:
    """检查CoT是否引用rating/star信息"""
    if verbose:
        print("=" * 60)
        print("CoT Section Check")
        print("=" * 60)

    stats = {}

    # 统计CoT中提到rating的比例
    rating_mentioned = 0
    samples_with_cot = 0
    cot_samples = []

    for item in data:
        # 检查conversations格式
        conversations = item.get('conversations', [])
        for conv in conversations:
            if conv.get('role') == 'assistant':
                content = conv.get('content', '')
                samples_with_cot += 1

                # 检查是否提到星级/rating
                if re.search(r'(\\d星|星级|rating|给了|给.*分|star)', content, re.IGNORECASE):
                    rating_mentioned += 1

                if len(cot_samples) < 3:
                    # 尝试解析JSON格式
                    try:
                        resp = json.loads(content)
                        reasoning = resp.get('reasoning', content[:100])
                    except:
                        reasoning = content[:100]
                    cot_samples.append({
                        'rating': item.get('rating'),
                        'label': item.get('label'),
                        'reasoning': reasoning[:80]
                    })
                break

        # 或者检查cot字段
        if 'cot' in item:
            samples_with_cot += 1
            cot = item.get('cot', '')
            if re.search(r'(\\d星|星级|rating|给了|给.*分|star)', cot, re.IGNORECASE):
                rating_mentioned += 1

    stats['samples_with_cot'] = samples_with_cot
    stats['rating_mentioned'] = rating_mentioned
    stats['rating_mentioned_pct'] = rating_mentioned / samples_with_cot * 100 if samples_with_cot > 0 else 0
    stats['cot_samples'] = cot_samples

    if verbose:
        print(f"Samples with CoT: {samples_with_cot}")
        print(f"Rating mentioned: {rating_mentioned} ({stats['rating_mentioned_pct']:.1f}%)")

        if cot_samples:
            print("\nSample CoTs:")
            for i, s in enumerate(cot_samples):
                print(f"  [{i+1}] Rating={s['rating']}, Label={s['label']}")
                print(f"      Reasoning: {s['reasoning']}...")

    return stats


# ============================================================
# 4. 深度质量检查 (合并 verify_quality.py)
# ============================================================
def check_deep_quality(data: list, verbose: bool = True) -> dict:
    """深度验证标注质量 - rating与label对应关系"""
    if verbose:
        print("=" * 60)
        print("Deep Quality Check")
        print("=" * 60)

    stats = {}

    # 1. 检查soft_labels与label对齐
    mismatch = 0
    for item in data:
        soft = item.get('soft_labels', [0.33, 0.33, 0.34])
        label = item.get('label', 1)
        predicted = soft.index(max(soft))
        if predicted != label:
            mismatch += 1

    stats['alignment_mismatch'] = mismatch
    stats['alignment_pct'] = (len(data) - mismatch) / len(data) * 100

    if verbose:
        print(f"1. Alignment Check:")
        print(f"   Correct: {len(data) - mismatch}/{len(data)} ({stats['alignment_pct']:.2f}%)")

    # 2. 检查rating与label的对应
    rating_label_map = defaultdict(lambda: Counter())
    for item in data:
        rating = item.get('rating', 3)
        label = item.get('label', 1)
        rating_label_map[rating][label] += 1

    stats['rating_label_map'] = {r: dict(c) for r, c in rating_label_map.items()}

    if verbose:
        print("\n2. Rating -> Label Distribution:")
        for rating in sorted(rating_label_map.keys()):
            dist = rating_label_map[rating]
            total = sum(dist.values())
            print(f"   Rating {rating} (n={total}):")
            for label in sorted(dist.keys()):
                name = ['Neg', 'Neu', 'Pos'][label]
                pct = dist[label] / total * 100
                print(f"      {name}: {dist[label]} ({pct:.1f}%)")

    # 3. 置信度分布
    confidences = [max(item.get('soft_labels', [0.33, 0.33, 0.34])) for item in data]
    stats['confidence_mean'] = sum(confidences) / len(confidences) if confidences else 0

    if verbose:
        print(f"\n3. Confidence Distribution:")
        print(f"   Mean: {stats['confidence_mean']:.3f}")

        ranges = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
        for lo, hi in ranges:
            count = sum(1 for c in confidences if lo <= c < hi)
            pct = count / len(confidences) * 100
            print(f"   [{lo:.1f}, {hi:.1f}): {count} ({pct:.1f}%)")

    # 4. 抽样检查低置信度样本
    low_conf_samples = [
        {
            'text': item.get('text', '')[:80],
            'soft_labels': item.get('soft_labels'),
            'label': item.get('label')
        }
        for item in data
        if max(item.get('soft_labels', [0.33, 0.33, 0.34])) < 0.7
    ][:5]

    stats['low_conf_samples'] = low_conf_samples

    if verbose and low_conf_samples:
        print("\n4. Low Confidence Samples (<0.7):")
        for s in low_conf_samples:
            print(f"   Text: {s['text']}...")
            print(f"   Soft: {s['soft_labels']}, Label: {s['label']}")

    return stats


# ============================================================
# 5. 查找未使用样本 (合并 check_samples.py)
# ============================================================
def find_unused_samples(data: list, reference_path: str, verbose: bool = True) -> list:
    """查找不在参考数据中的未使用样本"""
    if verbose:
        print("=" * 60)
        print("Find Unused Samples")
        print("=" * 60)

    # 加载参考数据
    ref_data = load_data(reference_path)
    ref_ids = set()
    for item in ref_data:
        text = item.get('text', '')
        ref_ids.add(hash(text[:100]))

    # 筛选未使用的样本
    unused = []
    for item in data:
        text_hash = hash(item.get('text', '')[:100])
        if text_hash not in ref_ids:
            unused.append(item)

    if verbose:
        print(f"Reference data: {len(ref_data)} samples")
        print(f"Input data: {len(data)} samples")
        print(f"Unused samples: {len(unused)}")

        unused_dist = Counter(item.get('label', 1) for item in unused)
        print("\nUnused distribution:")
        for label, name in [(0, 'Negative'), (1, 'Neutral'), (2, 'Positive')]:
            count = unused_dist.get(label, 0)
            pct = count / len(unused) * 100 if unused else 0
            print(f"  {name}: {count} ({pct:.1f}%)")

    return unused


# ============================================================
# 主函数：运行所有检查
# ============================================================
def run_all_checks(data: list, verbose: bool = True) -> dict:
    """运行所有检查"""
    print("\n" + "=" * 60)
    print("Running All Quality Checks")
    print("=" * 60 + "\n")

    all_stats = {}

    all_stats['distribution'] = check_distribution(data, verbose)
    all_stats['soft_labels'] = check_soft_labels(data, verbose)
    all_stats['cot'] = check_cot_section(data, verbose)
    all_stats['deep_quality'] = check_deep_quality(data, verbose)

    print("\n" + "=" * 60)
    print("All Checks Completed")
    print("=" * 60)

    return all_stats


# ============================================================
# CLI入口
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="数据质量检查")
    parser.add_argument("--data", type=str, required=True, help="数据文件路径")
    parser.add_argument("--check", type=str, default="all",
                        choices=["all", "distribution", "soft_labels", "cot", "deep", "unused"],
                        help="检查类型")
    parser.add_argument("--reference", type=str, default=None,
                        help="参考数据路径（用于unused检查）")
    parser.add_argument("--quiet", action="store_true", help="静默模式")
    return parser.parse_args()


def main():
    args = parse_args()
    verbose = not args.quiet

    data = load_data(args.data)

    if args.check == "all":
        run_all_checks(data, verbose)
    elif args.check == "distribution":
        check_distribution(data, verbose)
    elif args.check == "soft_labels":
        check_soft_labels(data, verbose)
    elif args.check == "cot":
        check_cot_section(data, verbose)
    elif args.check == "deep":
        check_deep_quality(data, verbose)
    elif args.check == "unused":
        if not args.reference:
            print("Error: --reference required for unused check")
            return
        find_unused_samples(data, args.reference, verbose)


if __name__ == "__main__":
    main()