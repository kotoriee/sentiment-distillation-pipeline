#!/usr/bin/env python3
"""
构建 10k 泛化评估数据集

功能：
1. 三品类 10k（Electronics, Clothing, Beauty 相关品类）
2. 随机品类 10k（从 Amazon Reviews'23 全品类随机采样）

数据来源：
- 优先使用本地已处理数据（data/train_answer_first.json）
- 若需更多品类，从 HuggingFace McAuley-Lab/Amazon-Reviews-2023 加载

Usage:
    cd 05_experiments/10k_robustness
    python build_dataset.py [--mode local|huggingface]
"""

import json
import random
import argparse
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 目标配置
TARGET_SIZE = 10000
PER_CLASS = TARGET_SIZE // 3  # 3333 per class

# 三品类目标（包含子品类）
THREE_CATEGORY_PATTERNS = [
    "Electronics",
    "Clothing", "Clothing_Shoes_and_Jewelry",
    "Beauty", "All_Beauty",
]


def load_local_data():
    """加载本地已处理数据"""
    train_path = DATA_DIR / "train_answer_first.json"
    test_path = DATA_DIR / "test_answer_first.json"

    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    all_data = train_data + test_data

    # 统计分布
    cat_dist = Counter(d['category'] for d in all_data)
    label_dist = Counter(d['label'] for d in all_data)

    print(f"本地数据总计: {len(all_data)}")
    print(f"品类分布: {dict(cat_dist)}")
    print(f"标签分布: Negative={label_dist[0]}, Neutral={label_dist[1]}, Positive={label_dist[2]}")

    return all_data


def build_three_category_10k_local(data: list):
    """从本地数据构建三品类 10k"""
    print("\n" + "=" * 60)
    print("构建三品类 10k（本地数据）")
    print("=" * 60)

    # 筛选三品类数据
    three_cat_data = []
    for d in data:
        cat = d['category']
        if any(pattern in cat for pattern in THREE_CATEGORY_PATTERNS):
            three_cat_data.append(d)

    print(f"三品类数据: {len(three_cat_data)}")

    # 按标签分组均衡采样
    neg = [d for d in three_cat_data if d['label'] == 0]
    neu = [d for d in three_cat_data if d['label'] == 1]
    pos = [d for d in three_cat_data if d['label'] == 2]

    print(f"  Negative: {len(neg)}")
    print(f"  Neutral: {len(neu)}")
    print(f"  Positive: {len(pos)}")

    # 均衡采样（若某类不足，则使用全部）
    sampled_neg = random.sample(neg, min(PER_CLASS, len(neg)))
    sampled_neu = random.sample(neu, min(PER_CLASS, len(neu)))
    sampled_pos = random.sample(pos, min(PER_CLASS, len(pos)))

    result = sampled_neg + sampled_neu + sampled_pos
    random.shuffle(result)

    # 统计最终分布
    final_cat = Counter(d['category'] for d in result)
    final_label = Counter(d['label'] for d in result)

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


def build_random_category_10k_local(data: list):
    """从本地数据构建随机品类 10k（使用所有可用品类）"""
    print("\n" + "=" * 60)
    print("构建随机品类 10k（本地数据）")
    print("=" * 60)

    # 使用全部本地数据（代表所有可用品类）
    print(f"可用数据: {len(data)}")

    # 按标签分组均衡采样
    neg = [d for d in data if d['label'] == 0]
    neu = [d for d in data if d['label'] == 1]
    pos = [d for d in data if d['label'] == 2]

    print(f"  Negative: {len(neg)}")
    print(f"  Neutral: {len(neu)}")
    print(f"  Positive: {len(pos)}")

    # 均衡采样
    sampled_neg = random.sample(neg, min(PER_CLASS, len(neg)))
    sampled_neu = random.sample(neu, min(PER_CLASS, len(neu)))
    sampled_pos = random.sample(pos, min(PER_CLASS, len(pos)))

    result = sampled_neg + sampled_neu + sampled_pos
    random.shuffle(result)

    # 统计最终分布
    final_cat = Counter(d['category'] for d in result)
    final_label = Counter(d['label'] for d in result)

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


def build_smoke_subset(data: list, n: int = 100):
    """构建 Smoke 测试 100 条样本"""
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


def load_huggingface_data():
    """从 HuggingFace 加载 Amazon Reviews'23（更多品类）"""
    print("\n" + "=" * 60)
    print("从 HuggingFace 加载 Amazon Reviews'23")
    print("=" * 60)

    try:
        from datasets import load_dataset
    except ImportError:
        print("错误: 需要安装 datasets 库")
        print("  pip install datasets")
        return None

    # McAuley-Lab/Amazon-Reviews-2023
    # 这是一个大数据集，需要指定子集
    print("加载 raw_review 子集（可能需要几分钟）...")

    # 尝试加载多个品类的子数据集
    categories_to_load = [
        "raw_review_Electronics",
        "raw_review_Clothing_Shoes_and_Jewelry",
        "raw_review_Beauty",
        "raw_review_Home_and_Kitchen",
        "raw_review_Books",
    ]

    all_reviews = []
    for cat in categories_to_load:
        try:
            print(f"  加载 {cat}...")
            ds = load_dataset("McAuley-Lab/Amazon-Reviews-2023", cat, split="train")
            for item in ds:
                # 提取必要字段
                rating = item.get('rating', 0)
                text = item.get('text', '')

                # Rating → Label 映射
                if rating <= 2:
                    label = 0  # Negative
                elif rating == 3:
                    label = 1  # Neutral
                else:
                    label = 2  # Positive

                all_reviews.append({
                    'text': text,
                    'rating': rating,
                    'label': label,
                    'category': cat.replace('raw_review_', ''),
                })

            print(f"    {cat}: {len(ds)} 条")

        except Exception as e:
            print(f"    加载失败: {e}")
            continue

    print(f"\n总计加载: {len(all_reviews)} 条")

    # 转换为 answer_first 格式
    converted = convert_to_answer_first_format(all_reviews)

    return converted


def convert_to_answer_first_format(reviews: list):
    """转换为 answer_first conversations 格式"""
    SYSTEM_PROMPT = """You are a professional e-commerce review sentiment analysis expert.

## Task
Analyze the review and output:
1. Sentiment classification (negative/neutral/positive)
2. Reasoning chain explaining your analysis

## Output Format (JSON)
{
    "sentiment": 0/1/2
}"""

    converted = []
    for r in reviews:
        user_content = f"Review: {r['text']}"

        converted.append({
            "conversations": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": json.dumps({"sentiment": r['label']})},
            ],
            "text": r['text'],
            "label": r['label'],
            "soft_labels": [0.33, 0.33, 0.34] if r['label'] == 1 else ([0.8, 0.15, 0.05] if r['label'] == 0 else [0.05, 0.15, 0.8]),
            "category": r['category'],
        })

    return converted


def build_three_category_10k_hf(data: list):
    """从 HuggingFace 数据构建三品类 10k"""
    print("\n" + "=" * 60)
    print("构建三品类 10k（HuggingFace 数据）")
    print("=" * 60)

    # 筛选三个核心品类
    target_cats = ["Electronics", "Clothing_Shoes_and_Jewelry", "Beauty"]
    filtered = [d for d in data if d['category'] in target_cats]

    print(f"三品类数据: {len(filtered)}")

    # 均衡采样
    neg = [d for d in filtered if d['label'] == 0]
    neu = [d for d in filtered if d['label'] == 1]
    pos = [d for d in filtered if d['label'] == 2]

    sampled = (
        random.sample(neg, min(PER_CLASS, len(neg))) +
        random.sample(neu, min(PER_CLASS, len(neu))) +
        random.sample(pos, min(PER_CLASS, len(pos)))
    )
    random.shuffle(sampled)

    output_path = OUTPUT_DIR / "eval_three_category_10k.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)
    print(f"保存: {output_path}")

    return sampled


def build_random_category_10k_hf(data: list):
    """从 HuggingFace 数据构建随机品类 10k"""
    print("\n" + "=" * 60)
    print("构建随机品类 10k（HuggingFace 数据）")
    print("=" * 60)

    # 使用所有品类
    print(f"全部品类数据: {len(data)}")

    # 统计品类分布
    cat_dist = Counter(d['category'] for d in data)
    print(f"品类分布: {dict(cat_dist)}")

    # 均衡采样
    neg = [d for d in data if d['label'] == 0]
    neu = [d for d in data if d['label'] == 1]
    pos = [d for d in data if d['label'] == 2]

    sampled = (
        random.sample(neg, min(PER_CLASS, len(neg))) +
        random.sample(neu, min(PER_CLASS, len(neu))) +
        random.sample(pos, min(PER_CLASS, len(pos)))
    )
    random.shuffle(sampled)

    output_path = OUTPUT_DIR / "eval_random_category_10k.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)
    print(f"保存: {output_path}")

    return sampled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["local", "huggingface"], default="local",
                        help="数据来源模式")
    args = parser.parse_args()

    print("=" * 60)
    print("构建 10k 泛化评估数据集")
    print("=" * 60)

    random.seed(42)  # 固定随机种子保证可复现

    if args.mode == "local":
        data = load_local_data()
        three_cat = build_three_category_10k_local(data)
        random_cat = build_random_category_10k_local(data)
        smoke = build_smoke_subset(data, 100)
    else:
        data = load_huggingface_data()
        if data:
            three_cat = build_three_category_10k_hf(data)
            random_cat = build_random_category_10k_hf(data)
            smoke = build_smoke_subset(data, 100)
        else:
            print("HuggingFace 加载失败，回退到本地数据")
            data = load_local_data()
            three_cat = build_three_category_10k_local(data)
            random_cat = build_random_category_10k_local(data)
            smoke = build_smoke_subset(data, 100)

    print("\n" + "=" * 60)
    print("构建完成")
    print("=" * 60)
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"  eval_three_category_10k.json: {len(three_cat)} 条")
    print(f"  eval_random_category_10k.json: {len(random_cat)} 条")
    print(f"  smoke_subset_100.json: {len(smoke)} 条")


if __name__ == "__main__":
    main()