"""
在降噪组设置下重新训练 SVM：
- 训练：清洗后的标签
- 测试：原始硬标签
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent))
from svm_classifier import SVMSentimentClassifier

DATA_DIR = Path("experiments/denoising_setup")
OUTPUT_DIR = Path("experiments/denoising_setup/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data(split, use_original_label=False):
    """加载数据"""
    if split in ['train', 'val']:
        filepath = DATA_DIR / f"{split}_cleaned.csv"
        df = pd.read_csv(filepath)
        return df['text'].tolist(), df['label'].tolist()  # 清洗标签
    else:  # test
        filepath = DATA_DIR / "test_original.csv"
        df = pd.read_csv(filepath)
        if use_original_label:
            return df['text'].tolist(), df['label'].tolist()  # 原始硬标签
        else:
            return df['text'].tolist(), df['cleaned_label'].tolist()  # 清洗标签（用于对比）

def main():
    print("="*60)
    print("SVM (TF-IDF) - 数据降噪组设置")
    print("训练: 清洗标签 | 测试: 原始硬标签")
    print("="*60)

    # 加载数据
    print("\n[1/3] 加载数据...")
    train_texts, train_labels = load_data('train')
    val_texts, val_labels = load_data('val')
    test_texts, test_labels = load_data('test', use_original_label=True)  # 关键！用原始硬标签

    print(f"  Train: {len(train_texts)} 条")
    print(f"  Val:   {len(val_texts)} 条")
    print(f"  Test:  {len(test_texts)} 条 (原始硬标签)")

    # 训练
    print("\n[2/3] 训练 SVM 模型...")
    clf = SVMSentimentClassifier()
    clf.fit(train_texts, train_labels)
    print("  训练完成")

    # 测试
    print("\n[3/3] 测试集评估...")
    test_pred = clf.predict(test_texts)

    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

    test_acc = accuracy_score(test_labels, test_pred)
    test_f1_macro = f1_score(test_labels, test_pred, average='macro')
    test_f1_weighted = f1_score(test_labels, test_pred, average='weighted')

    print(f"\n  Test Accuracy: {test_acc:.4f}")
    print(f"  Test Macro-F1: {test_f1_macro:.4f}")
    print(f"  Test Weighted-F1: {test_f1_weighted:.4f}")

    print("\n  Classification Report:")
    print(classification_report(test_labels, test_pred,
                               target_names=['Negative', 'Neutral', 'Positive']))

    print("\n  Confusion Matrix:")
    cm = confusion_matrix(test_labels, test_pred)
    print(f"    True\\Pred  Neg  Neu  Pos")
    for i, row in enumerate(cm):
        label = ['Neg', 'Neu', 'Pos'][i]
        print(f"    {label}        {row[0]:3d}  {row[1]:3d}  {row[2]:3d}")

    # 保存结果
    results = {
        'model': 'SVM (TF-IDF)',
        'test_accuracy': float(test_acc),
        'test_macro_f1': float(test_f1_macro),
        'test_weighted_f1': float(test_f1_weighted)
    }

    with open(OUTPUT_DIR / "svm_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # 保存模型
    clf.save(str(OUTPUT_DIR / "svm_model.pkl"))

    print("\n" + "="*60)
    print("SVM 评估完成!")
    print(f"结果保存: {OUTPUT_DIR}")
    print("="*60)

    return results

if __name__ == "__main__":
    results = main()
    print(f"\n结果: {results}")
