#!/usr/bin/env python3
"""
传统分类器延迟对比测试

对比三个传统分类算法在10k数据上的推理延迟：
- Logistic Regression
- MultinomialNB (朴素贝叶斯)
- LinearSVC (线性支持向量机)

Usage:
    python evaluate_traditional_latency.py
"""

import json
import time
import pickle
from pathlib import Path
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

# 配置
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"
TRAIN_PATH = DATA_DIR / "train_answer_first.json"
TEST_897_PATH = DATA_DIR / "test_answer_first.json"
TEST_10K_PATH = SCRIPT_DIR / "data" / "eval_three_category_10k_fixed.json"


def load_answer_first_data(path: Path):
    """加载 answer_first 格式数据，提取 text 和 label"""
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        convs = item["conversations"]
        user_content = convs[1]["content"]
        if user_content.startswith("Review: "):
            text = user_content[8:]
        else:
            text = user_content
        texts.append(text)
        labels.append(item["label"])

    return texts, labels


def measure_latency(model, X_test, name: str, n_runs: int = 5):
    """多次测量推理延迟，返回平均值"""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        predictions = model.predict(X_test)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_time = sum(times) / len(times)
    return predictions, avg_time


def train_and_evaluate_model(model_class, model_name, model_config,
                              X_train, train_labels, X_test, test_labels, n_runs=5):
    """训练模型并评估延迟"""
    print(f"\n训练 {model_name}...")

    # 创建模型
    model = model_class(**model_config)

    # 训练
    t0 = time.perf_counter()
    model.fit(X_train, train_labels)
    train_time = time.perf_counter() - t0
    print(f"  训练耗时: {train_time:.2f}s")

    # 推理（多次测量）
    predictions, infer_time = measure_latency(model, X_test, model_name, n_runs)

    # 计算指标
    accuracy = accuracy_score(test_labels, predictions)
    f1_macro = f1_score(test_labels, predictions, average="macro")
    f1_per_class = f1_score(test_labels, predictions, average=None)

    throughput = len(test_labels) / infer_time

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Macro F1: {f1_macro:.4f}")
    print(f"  Neutral F1: {f1_per_class[1]:.4f}")
    print(f"  推理延迟: {infer_time*1000:.2f}ms ({infer_time*1000/len(test_labels):.4f}ms/条)")
    print(f"  吞吐量: {throughput:.0f} samples/s")

    return {
        "model": model_name,
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_class_0": round(f1_per_class[0], 4),
        "f1_class_1": round(f1_per_class[1], 4),
        "f1_class_2": round(f1_per_class[2], 4),
        "train_time_sec": round(train_time, 2),
        "infer_time_ms": round(infer_time * 1000, 2),
        "infer_time_per_sample_ms": round(infer_time * 1000 / len(test_labels), 4),
        "throughput": round(throughput, 0),
        "n_runs": n_runs
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set", choices=["897", "10k"], default="897",
                        help="选择测试集: 897 (原始测试集) 或 10k (泛化测试集)")
    args = parser.parse_args()

    print("=" * 60)
    print("传统分类器延迟对比测试")
    print("=" * 60)

    # 加载训练数据
    print(f"\n加载训练数据: {TRAIN_PATH}")
    train_texts, train_labels = load_answer_first_data(TRAIN_PATH)
    print(f"  训练样本数: {len(train_texts)}")

    # 选择测试数据
    if args.test_set == "897":
        test_path = TEST_897_PATH
        print(f"\n使用原始测试集 (897条)")
    else:
        test_path = TEST_10K_PATH
        print(f"\n使用10k泛化测试集 (9999条)")

    print(f"加载测试数据: {test_path}")
    test_texts, test_labels = load_answer_first_data(test_path)
    print(f"  测试样本数: {len(test_texts)}")
    label_counts = Counter(test_labels)
    print(f"  标签分布: 0={label_counts[0]}, 1={label_counts[1]}, 2={label_counts[2]}")

    # TF-IDF 特征提取
    print("\nTF-IDF 特征提取...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    t0 = time.perf_counter()
    X_train = vectorizer.fit_transform(train_texts)
    feature_train_time = time.perf_counter() - t0
    print(f"  训练集特征提取: {feature_train_time:.2f}s")

    t0 = time.perf_counter()
    X_test = vectorizer.transform(test_texts)
    feature_test_time = time.perf_counter() - t0
    print(f"  测试集特征提取: {feature_test_time:.2f}s")
    print(f"  特征维度: {X_train.shape[1]}")

    # 评估三个模型
    results = {}

    # Logistic Regression
    results["lr"] = train_and_evaluate_model(
        LogisticRegression,
        "Logistic Regression",
        {"max_iter": 1000, "class_weight": "balanced", "C": 1.0, "random_state": 42},
        X_train, train_labels, X_test, test_labels,
        n_runs=5
    )

    # MultinomialNB
    results["nb"] = train_and_evaluate_model(
        MultinomialNB,
        "MultinomialNB",
        {"alpha": 1.0},
        X_train, train_labels, X_test, test_labels,
        n_runs=5
    )

    # LinearSVC
    results["svc"] = train_and_evaluate_model(
        LinearSVC,
        "LinearSVC",
        {"max_iter": 1000, "class_weight": "balanced", "C": 1.0, "random_state": 42},
        X_train, train_labels, X_test, test_labels,
        n_runs=5
    )

    # 特征提取信息
    results["feature_extraction"] = {
        "train_time_sec": round(feature_train_time, 2),
        "test_time_sec": round(feature_test_time, 2),
        "dimension": X_train.shape[1]
    }

    # 打印对比表
    print("\n" + "=" * 60)
    print("传统分类器延迟对比汇总")
    print("=" * 60)
    print(f"\n{'模型':<20} {'Accuracy':<10} {'Macro-F1':<10} {'Neu F1':<10} {'Train(s)':<10} {'Infer(ms)':<10} {'Throughput':<15}")
    print("-" * 85)

    for key, data in [("lr", results["lr"]), ("nb", results["nb"]), ("svc", results["svc"])]:
        print(f"{data['model']:<20} {data['accuracy']:<10.4f} {data['f1_macro']:<10.4f} "
              f"{data['f1_class_1']:<10.4f} {data['train_time_sec']:<10.2f} "
              f"{data['infer_time_ms']:<10.2f} {data['throughput']:<15.0f}")

    # 保存结果
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "traditional_latency_comparison.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_path}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()