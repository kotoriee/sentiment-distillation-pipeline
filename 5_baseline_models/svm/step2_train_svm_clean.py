"""
Step 2: 使用清洗后的数据训练 SVM (TF-IDF)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

# 复用现有的 SVM 分类器
sys.path.insert(0, str(Path(__file__).parent))
from svm_classifier import SVMSentimentClassifier

# 路径配置
DATA_DIR = Path("data/processed/baseline_clean")
OUTPUT_DIR = Path("experiments/logs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data(split):
    """加载 CSV 数据"""
    filepath = DATA_DIR / f"baseline_clean_{split}.csv"
    df = pd.read_csv(filepath)
    return df['text'].tolist(), df['cleaned_label'].tolist()

def train_and_evaluate():
    """训练并评估 SVM"""
    print("="*60)
    print("Step 2: SVM (TF-IDF) 基线训练与评估")
    print("="*60)

    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    train_texts, train_labels = load_data('train')
    val_texts, val_labels = load_data('val')
    test_texts, test_labels = load_data('test')

    print(f"  Train: {len(train_texts)} 条")
    print(f"  Val:   {len(val_texts)} 条")
    print(f"  Test:  {len(test_texts)} 条")

    # 2. 训练 SVM
    print("\n[2/4] 训练 SVM 模型...")
    clf = SVMSentimentClassifier()
    clf.fit(train_texts, train_labels)
    print("  训练完成")

    # 3. 验证集评估
    print("\n[3/4] 验证集评估...")
    val_pred = clf.predict(val_texts)
    val_acc = np.mean(np.array(val_pred) == np.array(val_labels))
    print(f"  Val Accuracy: {val_acc:.4f}")

    # 4. 测试集评估
    print("\n[4/4] 测试集评估...")
    test_pred = clf.predict(test_texts)

    # 计算指标
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

    test_acc = accuracy_score(test_labels, test_pred)
    test_f1_macro = f1_score(test_labels, test_pred, average='macro')
    test_f1_weighted = f1_score(test_labels, test_pred, average='weighted')

    print(f"\n  Test Accuracy: {test_acc:.4f}")
    print(f"  Test Macro-F1: {test_f1_macro:.4f}")
    print(f"  Test Weighted-F1: {test_f1_weighted:.4f}")

    # 详细分类报告
    print("\n  Classification Report:")
    print(classification_report(test_labels, test_pred,
                               target_names=['Negative', 'Neutral', 'Positive']))

    # 混淆矩阵
    print("\n  Confusion Matrix:")
    cm = confusion_matrix(test_labels, test_pred)
    print(f"    True\Pred  Neg  Neu  Pos")
    for i, row in enumerate(cm):
        label = ['Neg', 'Neu', 'Pos'][i]
        print(f"    {label}        {row[0]:3d}  {row[1]:3d}  {row[2]:3d}")

    # 5. 保存预测结果
    predictions = []
    for text, pred, true in zip(test_texts, test_pred, test_labels):
        predictions.append({
            'text': text,
            'predicted': int(pred),
            'true': int(true)
        })

    pred_file = OUTPUT_DIR / "svm_clean_predictions.json"
    with open(pred_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"\n  预测结果已保存: {pred_file}")

    # 6. 保存模型
    model_file = OUTPUT_DIR / "svm_clean_model.pkl"
    clf.save(str(model_file))
    print(f"  模型已保存: {model_file}")

    print("\n" + "="*60)
    print("Step 2 完成!")
    print("="*60)

    return {
        'model': 'SVM',
        'test_accuracy': test_acc,
        'test_macro_f1': test_f1_macro,
        'test_weighted_f1': test_f1_weighted
    }

if __name__ == "__main__":
    results = train_and_evaluate()
    print(f"\n最终结果: {results}")
