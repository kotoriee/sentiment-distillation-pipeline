#!/usr/bin/env python3
"""
10k 泛化能力评估 - 基线模型 (BERT + SVM)

使用已训练的模型在 10k 数据上评估泛化能力，不重新训练。
BERT: 使用已保存的 bert_answer_first_best.pt
SVM: 用原始训练数据训练后保存，然后在 10k 上评估

Usage:
    python evaluate_baseline_10k.py
"""

import os
import json
import time
import torch
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm import tqdm

# sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

# transformers
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader

# 配置
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "05_experiments" / "10k_robustness" / "results" / "three_category_10k"
BERT_MODEL_PATH = PROJECT_DIR / "6_experiments_results" / "baseline_results" / "bert_answer_first_best.pt"
SVM_MODEL_PATH = RESULTS_DIR.parent.parent / "svm_model.pkl"
LR_MODEL_PATH = RESULTS_DIR.parent.parent / "lr_model.pkl"

MAX_LEN = 256
BATCH_SIZE = 16
BERT_MODEL_NAME = "bert-base-uncased"


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


class BertSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def evaluate_bert_10k(data_path: Path, device):
    """评估 BERT 模型在 10k 数据上"""
    print("\n" + "=" * 60)
    print("BERT 10k 评估")
    print("=" * 60)

    # 加载模型
    print("加载模型...")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=3)

    if BERT_MODEL_PATH.exists():
        print(f"加载已训练模型: {BERT_MODEL_PATH}")
        model.load_state_dict(torch.load(BERT_MODEL_PATH, map_location=device))
    else:
        print("警告: 未找到已训练模型，使用预训练模型")

    model.to(device)

    # 加载 10k 数据
    print(f"\n加载 10k 数据: {data_path}")
    texts, labels = load_answer_first_data(data_path)
    print(f"  样本数: {len(texts)}")
    label_counts = Counter(labels)
    print(f"  标签分布: 0={label_counts[0]}, 1={label_counts[1]}, 2={label_counts[2]}")

    dataset = BertSentimentDataset(texts, labels, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    # 评估
    print("\n开始评估...")
    model.eval()
    predictions, true_labels = [], []

    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm(loader, desc="BERT 评估"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            lbls = batch["label"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(lbls.numpy())
    infer_time = time.time() - start_time

    # 计算指标
    accuracy = accuracy_score(true_labels, predictions)
    f1_macro = f1_score(true_labels, predictions, average="macro")
    f1_per_class = f1_score(true_labels, predictions, average=None)

    print(f"\n结果:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Macro F1:  {f1_macro:.4f}")
    print(f"  F1 Class 0 (Negative): {f1_per_class[0]:.4f}")
    print(f"  F1 Class 1 (Neutral):  {f1_per_class[1]:.4f}")
    print(f"  F1 Class 2 (Positive): {f1_per_class[2]:.4f}")
    print(f"  推理耗时: {infer_time:.2f}s ({infer_time/len(texts)*1000:.2f}ms/条)")

    return {
        "model": "bert-base-uncased",
        "test_dataset": data_path.name,
        "total_samples": len(texts),
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_class_0": round(f1_per_class[0], 4),
        "f1_class_1": round(f1_per_class[1], 4),
        "f1_class_2": round(f1_per_class[2], 4),
        "infer_time_sec": round(infer_time, 2),
        "throughput": round(len(texts) / infer_time, 2)
    }


def train_and_evaluate_svm_10k(train_path: Path, test_10k_path: Path):
    """训练 SVM（用原始训练数据）并在 10k 上评估"""
    print("\n" + "=" * 60)
    print("SVM 10k 评估")
    print("=" * 60)

    # 加载原始训练数据 (answer_first 格式)
    print(f"加载原始训练数据: {train_path}")
    train_texts, train_labels = load_answer_first_data(train_path)
    print(f"  训练样本数: {len(train_texts)}")

    # 加载 10k 测试数据
    print(f"\n加载 10k 测试数据: {test_10k_path}")
    test_texts, test_labels = load_answer_first_data(test_10k_path)
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

    start_time = time.time()
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    feat_time = time.time() - start_time
    print(f"  特征维度: {X_train.shape[1]}")
    print(f"  特征提取耗时: {feat_time:.2f}s")

    # 检查是否有已保存的模型
    if SVM_MODEL_PATH.exists():
        print(f"\n加载已保存的 SVM 模型: {SVM_MODEL_PATH}")
        import pickle
        with open(SVM_MODEL_PATH, 'rb') as f:
            svm = pickle.load(f)
    else:
        # 训练 SVM
        print("\n训练 SVM...")
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

        # 保存模型
        import pickle
        with open(SVM_MODEL_PATH, 'wb') as f:
            pickle.dump(svm, f)
        print(f"  模型已保存: {SVM_MODEL_PATH}")

    # 在 10k 上评估
    print("\n10k 评估...")
    start_time = time.time()
    predictions = svm.predict(X_test)
    infer_time = time.time() - start_time

    accuracy = accuracy_score(test_labels, predictions)
    f1_macro = f1_score(test_labels, predictions, average="macro")
    f1_per_class = f1_score(test_labels, predictions, average=None)

    print(f"\n结果:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Macro F1:  {f1_macro:.4f}")
    print(f"  F1 Class 0 (Negative): {f1_per_class[0]:.4f}")
    print(f"  F1 Class 1 (Neutral):  {f1_per_class[1]:.4f}")
    print(f"  F1 Class 2 (Positive): {f1_per_class[2]:.4f}")
    print(f"  推理耗时: {infer_time:.2f}s ({infer_time/len(test_texts)*1000:.4f}ms/条)")

    return {
        "model": "SVM + TF-IDF",
        "test_dataset": test_10k_path.name,
        "total_samples": len(test_texts),
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_class_0": round(f1_per_class[0], 4),
        "f1_class_1": round(f1_per_class[1], 4),
        "f1_class_2": round(f1_per_class[2], 4),
        "infer_time_sec": round(infer_time, 2),
        "throughput": round(len(test_texts) / infer_time, 2)
    }


def train_and_evaluate_lr_10k(train_path: Path, test_10k_path: Path):
    """训练 Logistic Regression 并在 10k 上评估"""
    print("\n" + "=" * 60)
    print("Logistic Regression 10k 评估")
    print("=" * 60)

    # 加载原始训练数据
    print(f"加载原始训练数据: {train_path}")
    train_texts, train_labels = load_answer_first_data(train_path)
    print(f"  训练样本数: {len(train_texts)}")

    # 加载 10k 测试数据
    print(f"\n加载 10k 测试数据: {test_10k_path}")
    test_texts, test_labels = load_answer_first_data(test_10k_path)
    print(f"  测试样本数: {len(test_texts)}")

    # TF-IDF 特征提取
    print("\nTF-IDF 特征提取...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    print(f"  特征维度: {X_train.shape[1]}")

    # 检查是否有已保存的模型
    if LR_MODEL_PATH.exists():
        print(f"\n加载已保存的 LR 模型: {LR_MODEL_PATH}")
        import pickle
        with open(LR_MODEL_PATH, 'rb') as f:
            lr = pickle.load(f)
    else:
        # 训练 LR
        print("\n训练 Logistic Regression...")
        lr = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=1.0,
            random_state=42
        )

        start_time = time.time()
        lr.fit(X_train, train_labels)
        train_time = time.time() - start_time
        print(f"  训练耗时: {train_time:.2f}s")

        # 保存模型
        import pickle
        with open(LR_MODEL_PATH, 'wb') as f:
            pickle.dump(lr, f)
        print(f"  模型已保存: {LR_MODEL_PATH}")

    # 在 10k 上评估
    print("\n10k 评估...")
    start_time = time.time()
    predictions = lr.predict(X_test)
    infer_time = time.time() - start_time

    accuracy = accuracy_score(test_labels, predictions)
    f1_macro = f1_score(test_labels, predictions, average="macro")
    f1_per_class = f1_score(test_labels, predictions, average=None)

    print(f"\n结果:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Macro F1:  {f1_macro:.4f}")
    print(f"  F1 Class 0 (Negative): {f1_per_class[0]:.4f}")
    print(f"  F1 Class 1 (Neutral):  {f1_per_class[1]:.4f}")
    print(f"  F1 Class 2 (Positive): {f1_per_class[2]:.4f}")
    print(f"  推理耗时: {infer_time:.2f}s ({infer_time/len(test_texts)*1000:.4f}ms/条)")

    return {
        "model": "Logistic Regression + TF-IDF",
        "test_dataset": test_10k_path.name,
        "total_samples": len(test_texts),
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_class_0": round(f1_per_class[0], 4),
        "f1_class_1": round(f1_per_class[1], 4),
        "f1_class_2": round(f1_per_class[2], 4),
        "infer_time_sec": round(infer_time, 2),
        "throughput": round(len(test_texts) / infer_time, 2)
    }


def main():
    print("=" * 60)
    print("10k 泛化能力评估 - 全模型对比")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 数据路径
    train_path = DATA_DIR / "train_answer_first.json"
    test_10k_path = PROJECT_DIR / "05_experiments" / "10k_robustness" / "data" / "eval_three_category_10k_fixed.json"

    if not test_10k_path.exists():
        print(f"错误: 10k 测试数据不存在: {test_10k_path}")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 评估各模型
    results = {}

    # BERT
    results["bert"] = evaluate_bert_10k(test_10k_path, device)

    # SVM
    results["svm"] = train_and_evaluate_svm_10k(train_path, test_10k_path)

    # Logistic Regression
    results["lr"] = train_and_evaluate_lr_10k(train_path, test_10k_path)

    # 加载已有的 SFT 和 GRPO 结果
    sft_path = RESULTS_DIR / "sft_results.json"
    grpo_path = PROJECT_DIR / "3_lora_training" / "outputs_grpo_9b_rewardfix_v3_nothink_continue_300" / "checkpoint-90" / "grpo_eval_results.json"

    if sft_path.exists():
        with open(sft_path, 'r') as f:
            sft_data = json.load(f)
        results["sft"] = {
            "model": "SFT (Qwen3.5-9B)",
            "accuracy": sft_data["metrics"]["accuracy"],
            "f1_macro": sft_data["metrics"]["f1_macro"],
            "f1_class_0": sft_data["metrics"]["f1_class_0"],
            "f1_class_1": sft_data["metrics"]["f1_class_1"],
            "f1_class_2": sft_data["metrics"]["f1_class_2"],
            "throughput": sft_data["metrics"]["speed"]
        }

    if grpo_path.exists():
        with open(grpo_path, 'r') as f:
            grpo_data = json.load(f)
        results["grpo"] = {
            "model": "GRPO (Qwen3.5-9B)",
            "accuracy": grpo_data["metrics"]["accuracy"],
            "f1_macro": grpo_data["metrics"]["f1_macro"],
            "f1_class_0": grpo_data["metrics"]["f1_class_0"],
            "f1_class_1": grpo_data["metrics"]["f1_class_1"],
            "f1_class_2": grpo_data["metrics"]["f1_class_2"],
            "throughput": grpo_data["metrics"]["speed"]
        }

    # 生成对比表
    print("\n" + "=" * 60)
    print("10k 泛化能力对比汇总")
    print("=" * 60)
    print(f"\n{'模型':<30} {'Accuracy':<10} {'Macro-F1':<10} {'Neg F1':<10} {'Neu F1':<10} {'Pos F1':<10} {'Throughput':<10}")
    print("-" * 90)

    for name, data in results.items():
        print(f"{data['model']:<30} {data['accuracy']:<10.4f} {data['f1_macro']:<10.4f} {data['f1_class_0']:<10.4f} {data['f1_class_1']:<10.4f} {data['f1_class_2']:<10.4f} {data.get('throughput', 'N/A'):<10}")

    # 保存汇总结果
    summary_path = RESULTS_DIR / "all_models_10k_comparison.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n汇总结果已保存: {summary_path}")

    print("\n" + "=" * 60)
    print("评估完成")
    print("=" * 60)


if __name__ == "__main__":
    main()