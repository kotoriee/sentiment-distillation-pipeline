#!/usr/bin/env python3
"""
构建 10k 泛化评估数据集 - 流式下载解压 JSONL.gz

方案B：流式解压 .jsonl.gz 文件，边下载边读取采样（无需完整下载）

数据来源：
- 三品类10k: Electronics, Clothing_Shoes_and_Jewelry, Beauty_and_Personal_Care（与训练数据一致）
- 随机品类10k: 从10个品类随机采样

Rating → Sentiment 映射：
- 1-2 星 → Negative(0)
- 3 星 → Neutral(1)
- 4-5 星 → Positive(2)

Usage:
    cd 05_experiments/10k_robustness
    python build_dataset_streaming.py
"""

import json
import random
import gzip
import argparse
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import urllib.request
import io

OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = 10000
PER_CLASS = TARGET_SIZE // 3

# 三品类目标（与训练数据一致）
THREE_CATEGORIES = [
    "Electronics",
    "Clothing_Shoes_and_Jewelry",
    "Beauty_and_Personal_Care",
]

# 随机品类候选
RANDOM_CATEGORIES = [
    "Electronics",
    "Clothing_Shoes_and_Jewelry",
    "Beauty_and_Personal_Care",
    "Home_and_Kitchen",
    "Books",
    "Movies_and_TV",
    "Sports_and_Outdoors",
    "Toys_and_Games",
    "Video_Games",
    "Automotive",
]

# UCSD 直接下载链接
BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories"

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


def stream_category_jsonl_gz(category_name, max_per_label=None, total_max=None):
    """
    流式下载解压 JSONL.gz 文件并采样

    Args:
        category_name: 品类名（不含 raw_review_ 前缀）
        max_per_label: 每个标签最多采样数量
        total_max: 总采样上限

    Returns:
        samples: 采样后的数据列表
    """
    url = f"{BASE_URL}/{category_name}.jsonl.gz"
    print(f"  流式下载: {url}")

    samples = []
    label_counts = {0: 0, 1: 0, 2: 0}

    try:
        # 流式下载
        response = urllib.request.urlopen(url, timeout=60)

        # 使用 gzip 解压流
        decompressed = gzip.GzipFile(fileobj=response)

        # 包装为文本流
        text_stream = io.TextIOWrapper(decompressed, encoding='utf-8')

        # 逐行读取
        for line in tqdm(text_stream, desc=f"采样 {category_name}"):
            try:
                item = json.loads(line.strip())

                rating = item.get('rating', 0)
                text = item.get('text', '')

                if not text or rating < 1 or rating > 5:
                    continue

                label = rating_to_sentiment(rating)

                # 均衡采样控制
                if max_per_label and label_counts[label] >= max_per_label:
                    continue

                samples.append(convert_to_answer_first_format(text, label, category_name))
                label_counts[label] += 1

                # 达到目标数量时停止
                if total_max and len(samples) >= total_max:
                    break
                if max_per_label and all(label_counts[l] >= max_per_label for l in [0, 1, 2]):
                    break

            except json.JSONDecodeError:
                continue

        print(f"    采样完成: {len(samples)} 条 (Neg={label_counts[0]}, Neu={label_counts[1]}, Pos={label_counts[2]})")

    except Exception as e:
        print(f"    下载失败: {e}")

    return samples


def build_three_category_10k_streaming():
    """流式构建三品类 10k（与训练数据品类一致）"""
    print("\n" + "=" * 60)
    print("构建三品类 10k（流式下载）")
    print("=" * 60)
    print(f"目标品类: {THREE_CATEGORIES}")
    print(f"目标数量: {TARGET_SIZE} (每类 {PER_CLASS})")

    all_samples = []

    # 从每个品类均衡采样
    samples_per_category = PER_CLASS // len(THREE_CATEGORIES)

    for cat in THREE_CATEGORIES:
        samples = stream_category_jsonl_gz(cat, max_per_label=samples_per_category)
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
    print("构建随机品类 10k（流式下载）")
    print("=" * 60)
    print(f"候选品类: {RANDOM_CATEGORIES}")
    print(f"目标数量: {TARGET_SIZE} (每类 {PER_CLASS})")

    all_samples = []

    # 从多个品类采样
    samples_per_category = PER_CLASS // len(RANDOM_CATEGORIES)

    for cat in RANDOM_CATEGORIES:
        samples = stream_category_jsonl_gz(cat, max_per_label=samples_per_category)
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
    parser.add_argument("--proxy", default=None, help="HTTP proxy URL (e.g. http://192.168.0.195:7897)")
    args = parser.parse_args()

    # 设置代理
    if args.proxy:
        print(f"使用代理: {args.proxy}")
        import os
        os.environ['http_proxy'] = args.proxy
        os.environ['https_proxy'] = args.proxy

    print("=" * 60)
    print("构建 10k 泛化评估数据集（流式下载解压）")
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