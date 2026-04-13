"""
Step 1: 构建清洗后的 Baseline 数据集
从 soft_labels_reviewed.jsonl 中提取 Cleaned Label，划分 train/val/test (8:1:1)
"""

import json
import random
from pathlib import Path
import pandas as pd

# 设置随机种子
random.seed(42)

# 路径配置
INPUT_FILE = Path("data/processed/soft_labels_reviewed.jsonl")
OUTPUT_DIR = Path("data/processed/baseline_clean")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_soft_labels(filepath):
    """加载软标注数据"""
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            # 从 probabilities 提取 argmax 作为 Cleaned Label
            probs = data['probabilities']
            cleaned_label = probs.index(max(probs))  # 0=Negative, 1=Neutral, 2=Positive

            samples.append({
                'id': data['id'],
                'text': data['text'],
                'cleaned_label': cleaned_label,
                'probabilities': probs,
                'confidence': data['confidence'],
                'original_split': data.get('split', 'unknown')
            })
    return samples

def split_data(samples, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """按 8:1:1 划分数据集"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # 打乱数据
    random.shuffle(samples)
    total = len(samples)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_data = samples[:train_end]
    val_data = samples[train_end:val_end]
    test_data = samples[val_end:]

    return train_data, val_data, test_data

def save_to_csv(data, filepath):
    """保存为 CSV 格式"""
    df = pd.DataFrame(data)
    df = df[['id', 'text', 'cleaned_label']]  # 只保留需要的列
    df.to_csv(filepath, index=False, encoding='utf-8')
    print(f"  保存: {filepath} ({len(df)} 条)")

def main():
    print("="*60)
    print("Step 1: 构建清洗后的 Baseline 数据集")
    print("="*60)

    # 1. 加载数据
    print(f"\n加载数据: {INPUT_FILE}")
    samples = load_soft_labels(INPUT_FILE)
    print(f"  总样本数: {len(samples)}")

    # 2. 划分数据集
    print("\n划分数据集 (8:1:1)...")
    train_data, val_data, test_data = split_data(samples)
    print(f"  Train: {len(train_data)} ({len(train_data)/len(samples)*100:.1f}%)")
    print(f"  Val:   {len(val_data)} ({len(val_data)/len(samples)*100:.1f}%)")
    print(f"  Test:  {len(test_data)} ({len(test_data)/len(samples)*100:.1f}%)")

    # 3. 保存数据
    print("\n保存数据...")
    save_to_csv(train_data, OUTPUT_DIR / "baseline_clean_train.csv")
    save_to_csv(val_data, OUTPUT_DIR / "baseline_clean_val.csv")
    save_to_csv(test_data, OUTPUT_DIR / "baseline_clean_test.csv")

    # 4. 统计标签分布
    print("\n标签分布统计:")
    for split_name, data_split in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
        labels = [s['cleaned_label'] for s in data_split]
        label_counts = {0: labels.count(0), 1: labels.count(1), 2: labels.count(2)}
        print(f"  {split_name}: Neg={label_counts[0]}, Neu={label_counts[1]}, Pos={label_counts[2]}")

    # 5. 保存完整信息（包含 probabilities）的 JSONL（用于 BERT 训练）
    print("\n保存完整 JSONL 格式（用于 BERT）...")
    for split_name, data_list in [('train', train_data), ('val', val_data), ('test', test_data)]:
        output_file = OUTPUT_DIR / f"baseline_clean_{split_name}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data_list:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"  保存: {output_file}")

    print("\n" + "="*60)
    print("Step 1 完成!")
    print(f"输出目录: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
