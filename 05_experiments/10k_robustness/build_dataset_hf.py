#!/usr/bin/env python3
"""
构建 10k 泛化评估数据集 - 使用 HuggingFace 流式加载

从 Amazon Reviews'23 流式加载采样构建：
1. 三品类 10k（Electronics, All_Beauty, Clothing_Shoes_and_Jewelry）
2. 随机品类 10k（从多个品类随机采样）

Rating → Sentiment 映射：
- 1-2 星 → Negative(0)
- 3 星 → Neutral(1)
- 4-5 星 → Positive(2)

Usage:
    cd 05_experiments/10k_robustness
    python build_dataset_hf.py --mode streaming
"""

import json
import random
import argparse
from pathlib import Path
from collections import Counter
from tqdm import tqdm

OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = 10000
PER_CLASS = TARGET_SIZE // 3

# 三品类目标
THREE_CATEGORIES = [
    "raw_review_Electronics",
    "raw_review_All_Beauty",
    "raw_review_Clothing_Shoes_and_Jewelry",
]

# 随机品类候选（更多品类）
RANDOM_CATEGORIES = [
    "raw_review_Electronics",
    "raw_review_All_Beauty",
    "raw_review_Clothing_Shoes_and_Jewelry",
    "raw_review_Home_and_Kitchen",
    "raw_review_Books",
    "raw_review_Movies_and_TV",
    "raw_review_Sports_and_Outdoors",
    "raw_review_Toys_and_Games",
    "raw_review_Video_Games",
    "raw_review_Automotive",
]

SYSTEM_PROMPT = """You are a professional e-commerce review sentiment analysis expert.

## Task
Analyze the review and output:
1. Sentiment classification (negative/neutral/positive)
2. Reasoning chain explaining your analysis

## Output Format (JSON)
{
    "sentiment": 0/1/2
}"""


def rating_to_sentiment(rating):
    """Rating → Sentiment 映射"""
    if rating <= 2:
        return 0  # Negative
    elif rating == 3:
        return 1  # Neutral
    else:
        return 2  # Positive


def convert_to_answer_first_format(text, label, category):
    """转换为 answer_first 格式"""
    user_content = f"Review: {text}"

    return {
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": json.dumps({"sentiment": label})},
        ],
        "text": text,
        "label": label,
        "soft_labels": [0.33, 0.33, 0.34] if label == 1 else ([0.8, 0.15, 0.05] if label == 0 else [0.05, 0.15, 0.8]),
        "category": category,
    }


def stream_category(category_name, max_samples=None, target_labels=None):
    """流式加载单个品类数据"""
    from datasets import load_dataset

    print(f"  加载 {category_name} (streaming=True, split='full')...")

    try:
        ds = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            category_name,
            split="full",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"    加载失败: {e}")
        return []

    samples = []
    label_counts = {0: 0, 1: 0, 2: 0}

    # 流式采样
    for item in tqdm(ds, desc=f"采样 {category_name}", total=max_samples * 3 if max_samples else None):
        rating = item.get('rating', 0)
        text = item.get('text', '')

        if not text or rating < 1 or rating > 5:
            continue

        label = rating_to_sentiment(rating)

        # 均衡采样控制
        if target_labels:
            if label_counts[label] >= target_labels:
                continue

        samples.append(convert_to_answer_first_format(
            text, label, category_name.replace("raw_review_", "")
        ))
        label_counts[label] += 1

        # 达到目标数量时停止
        if max_samples and len(samples) >= max_samples * 3:
            break
        if target_labels and all(label_counts[l] >= target_labels for l in [0, 1, 2]):
            break

    print(f"    采样完成: {len(samples)} 条 (Neg={label_counts[0]}, Neu={label_counts[1]}, Pos={label_counts[2]})")
    return samples


def build_three_category_10k_streaming():
    """流式构建三品类 10k"""
    print("\n" + "=" * 60)
    print("构建三品类 10k（流式加载）")
    print("=" * 60)
    print(f"目标品类: {THREE_CATEGORIES}")
    print(f"目标数量: {TARGET_SIZE} (每类 {PER_CLASS})")

    all_samples = []

    # 从每个品类采样
    samples_per_category = PER_CLASS // len(THREE_CATEGORIES)

    for cat in THREE_CATEGORIES:
        samples = stream_category(cat, max_samples=samples_per_category * 2)
        all_samples.extend(samples)

    # 均衡采样到目标数量
    neg = [s for s in all_samples if s['label'] == 0]
    neu = [s for s in all_samples if s['label'] == 1]
    pos = [s for s in all_samples if s['label'] == 2]

    print(f"\n总采样: {len(all_samples)} 条")
    print(f"  Negative: {len(neg)}")
    print(f"  Neutral: {len(neu)}")
    print(f"  Positive: {len(pos)}")

    # 均衡截取
    sampled_neg = random.sample(neg, min(PER_CLASS, len(neg)))
    sampled_neu = random.sample(neu, min(PER_CLASS, len(neu)))
    sampled_pos = random.sample(pos, min(PER_CLASS, len(pos)))

    result = sampled_neg + sampled_neu + sampled_pos
    random.shuffle(result)

    # 统计
    final_cat = Counter(s['category'] for s in result)
    final_label = Counter(s['label'] for s in result)

    print(f"\n最终数据集:")
    print(f"  总计: {len(result)}")
    print(f"  品类分布: {dict(final_cat)}")
    print(f"  标签分布: Negative={final_label[0]}, Neutral={final_label[1]}, Positive={final_label[2]}")

    # 保存
    output_path = OUTPUT_DIR / "eval_three_category_10k.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {output_path}")

    return result


def build_random_category_10k_streaming():
    """流式构建随机品类 10k"""
    print("\n" + "=" * 60)
    print("构建随机品类 10k（流式加载）")
    print("=" * 60)
    print(f"候选品类: {RANDOM_CATEGORIES}")
    print(f"目标数量: {TARGET_SIZE} (每类 {PER_CLASS})")

    all_samples = []

    # 从多个品类随机采样
    samples_per_category = PER_CLASS // len(RANDOM_CATEGORIES)

    for cat in RANDOM_CATEGORIES:
        samples = stream_category(cat, max_samples=samples_per_category * 2)
        all_samples.extend(samples)

    # 均衡采样
    neg = [s for s in all_samples if s['label'] == 0]
    neu = [s for s in all_samples if s['label'] == 1]
    pos = [s for s in all_samples if s['label'] == 2]

    print(f"\n总采样: {len(all_samples)} 条")
    print(f"  Negative: {len(neg)}")
    print(f"  Neutral: {len(neu)}")
    print(f"  Positive: {len(pos)}")

    # 均衡截取
    sampled_neg = random.sample(neg, min(PER_CLASS, len(neg)))
    sampled_neu = random.sample(neu, min(PER_CLASS, len(neu)))
    sampled_pos = random.sample(pos, min(PER_CLASS, len(pos)))

    result = sampled_neg + sampled_neu + sampled_pos
    random.shuffle(result)

    # 统计
    final_cat = Counter(s['category'] for s in result)
    final_label = Counter(s['label'] for s in result)

    print(f"\n最终数据集:")
    print(f"  总计: {len(result)}")
    print(f"  品类分布: {dict(final_cat)}")
    print(f"  标签分布: Negative={final_label[0]}, Neutral={final_label[1]}, Positive={final_label[2]}")

    # 保存
    output_path = OUTPUT_DIR / "eval_random_category_10k.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {output_path}")

    return result


def build_smoke_subset(data, n=100):
    """构建 Smoke 测试子集"""
    print("\n" + "=" * 60)
    print(f"构建 Smoke 测试子集 ({n} 条)")
    print("=" * 60)

    # 均衡采样
    neg = [d for d in data if d['label'] == 0]
    neu = [d for d in data if d['label'] == 1]
    pos = [d for d in data if d['label'] == 2]

    per_class = n // 3
    sampled = (
        random.sample(neg, min(per_class, len(neg))) +
        random.sample(neu, min(per_class, len(neu))) +
        random.sample(pos, min(per_class, len(pos)))
    )
    random.shuffle(sampled)

    label_dist = Counter(d['label'] for d in sampled)
    print(f"  总计: {len(sampled)}")
    print(f"  标签分布: Negative={label_dist[0]}, Neutral={label_dist[1]}, Positive={label_dist[2]}")

    output_path = OUTPUT_DIR / "smoke_subset_100.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)
    print(f"  保存: {output_path}")

    return sampled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["streaming"], default="streaming",
                        help="使用流式加载")
    args = parser.parse_args()

    print("=" * 60)
    print("构建 10k 泛化评估数据集（HuggingFace 流式加载）")
    print("=" * 60)

    random.seed(42)

    # 构建三品类 10k
    three_cat = build_three_category_10k_streaming()

    # 构建随机品类 10k
    random_cat = build_random_category_10k_streaming()

    # 构建 Smoke 子集
    smoke = build_smoke_subset(three_cat, 100)

    print("\n" + "=" * 60)
    print("构建完成")
    print("=" * 60)
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"  eval_three_category_10k.json: {len(three_cat)} 条")
    print(f"  eval_random_category_10k.json: {len(random_cat)} 条")
    print(f"  smoke_subset_100.json: {len(smoke)} 条")


if __name__ == "__main__":
    main()