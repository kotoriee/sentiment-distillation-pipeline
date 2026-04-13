#!/usr/bin/env python3
"""
SVM基线模型训练脚本

使用TF-IDF + SVM对情感分类任务进行训练

Usage:
    python train_svm.py
"""

import csv
import json
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
import time

# 配置
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed" / "baseline"
RESULTS_DIR = PROJECT_DIR / "6_experiments_results" / "baseline_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_csv(path: Path):
    """加载CSV数据"""
    texts, labels, ids = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            labels.append(int(row["label"]))
            ids.append(row["id"])
    return texts, labels, ids


def main():
    print("=" * 60)
    print("SVM基线模型训练 (TF-IDF)")
    print("=" * 60)

    # 加载数据
    print("\n加载数据...")
    train_texts, train_labels, train_ids = load_csv(DATA_DIR / "train.csv")
    val_texts, val_labels, val_ids = load_csv(DATA_DIR / "val.csv")
    test_texts, test_labels, test_ids = load_csv(DATA_DIR / "test.csv")

    print(f"  train: {len(train_texts)}条")
    print(f"  val: {len(val_texts)}条")
    print(f"  test: {len(test_texts)}条")

    # TF-IDF特征提取
    print("\nTF-IDF特征提取...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    start_time = time.time()
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)
    feat_time = time.time() - start_time

    print(f"  特征维度: {X_train.shape[1]}")
    print(f"  特征提取耗时: {feat_time:.2f}s")

    # SVM训练
    print("\n训练SVM...")
    svm = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        probability=True
    )

    start_time = time.time()
    svm.fit(X_train, train_labels)
    train_time = time.time() - start_time

    print(f"  训练耗时: {train_time:.2f}s")

    # 验证集评估
    print("\n验证集评估...")
    val_pred = svm.predict(X_val)
    val_acc = accuracy_score(val_labels, val_pred)
    val_f1 = f1_score(val_labels, val_pred, average="macro")

    print(f"  Accuracy: {val_acc:.4f}")
    print(f"  Macro-F1: {val_f1:.4f}")

    # 测试集评估
    print("\n测试集评估...")
    start_time = time.time()
    test_pred = svm.predict(X_test)
    infer_time = time.time() - start_time

    test_acc = accuracy_score(test_labels, test_pred)
    test_f1 = f1_score(test_labels, test_pred, average="macro")

    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Macro-F1: {test_f1:.4f}")
    print(f"  推理耗时: {infer_time:.2f}s ({infer_time/len(test_texts)*1000:.2f}ms/条)")

    # 分类报告
    print("\n分类报告:")
    print(classification_report(test_labels, test_pred, target_names=["Negative", "Neutral", "Positive"]))

    # 保存结果
    results = {
        "model": "SVM + TF-IDF",
        "train_samples": len(train_texts),
        "val_accuracy": val_acc,
        "val_f1": val_f1,
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "train_time": train_time,
        "inference_time_per_sample": infer_time / len(test_texts),
        "feature_dim": X_train.shape[1]
    }

    results_path = RESULTS_DIR / "svm_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {results_path}")

    # 保存预测结果
    predictions = []
    for i, (id_, true, pred) in enumerate(zip(test_ids, test_labels, test_pred)):
        predictions.append({
            "id": id_,
            "true_label": int(true),
            "pred_label": int(pred),
            "correct": bool(true == pred)
        })

    pred_path = RESULTS_DIR / "svm_predictions.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"预测结果: {pred_path}")

    print("\n" + "=" * 60)
    print("SVM训练完成")
    print("=" * 60)


if __name__ == "__main__":
    main()