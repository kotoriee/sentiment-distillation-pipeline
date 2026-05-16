#!/usr/bin/env python3
"""
BERT baseline 评估脚本 - 使用 answer_first 测试集

确保与 LoRA 实验使用相同的测试集 (test_answer_first.json)
输出 class-wise F1 以便对比

Usage:
    python eval_bert_answer_first.py
"""

import json
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report, f1_score, accuracy_score
from tqdm import tqdm


# 配置
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "6_experiments_results" / "baseline_results"
MODEL_PATH = RESULTS_DIR / "bert_best_model.pt"

MAX_LEN = 256
BATCH_SIZE = 16
MODEL_NAME = "bert-base-uncased"


class SentimentDataset(Dataset):
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


def load_answer_first_data(path: Path):
    """加载 answer_first 格式数据，提取 text 和 label"""
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        # 从 conversations 中提取原始 review text
        convs = item["conversations"]
        # user content 包含原始 review
        user_content = convs[1]["content"]
        # 提取 review text (去掉 "Review: " prefix if present)
        if user_content.startswith("Review: "):
            text = user_content[8:]
        else:
            text = user_content

        texts.append(text)
        labels.append(item["label"])

    return texts, labels


def evaluate(model, dataloader, device):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.numpy())

    return predictions, true_labels


def main():
    print("=" * 60)
    print("BERT baseline 评估 (answer_first 测试集)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载模型
    print("\n加载模型...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    if MODEL_PATH.exists():
        print(f"加载训练好的模型: {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("警告: 未找到训练好的模型，使用预训练模型 (未fine-tune)")

    model.to(device)

    # 加载 answer_first 测试集
    print("\n加载测试数据...")
    test_path = DATA_DIR / "test_answer_first.json"
    test_texts, test_labels = load_answer_first_data(test_path)
    print(f"  测试集: {len(test_texts)} 条")

    # 统计标签分布
    from collections import Counter
    label_counts = Counter(test_labels)
    print(f"  标签分布: 0={label_counts[0]}, 1={label_counts[1]}, 2={label_counts[2]}")

    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 评估
    print("\n开始评估...")
    predictions, true_labels = evaluate(model, test_loader, device)

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

    print("\n分类报告:")
    print(classification_report(true_labels, predictions, target_names=["Negative", "Neutral", "Positive"]))

    # 保存结果
    results = {
        "model": "bert-base-uncased",
        "test_dataset": "test_answer_first.json",
        "total_samples": len(test_texts),
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_class_0": f1_per_class[0],
        "f1_class_1": f1_per_class[1],
        "f1_class_2": f1_per_class[2],
        "classification_report": classification_report(true_labels, predictions, target_names=["Negative", "Neutral", "Positive"], output_dict=True)
    }

    results_path = RESULTS_DIR / "bert_answer_first_eval.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {results_path}")

    print("\n" + "=" * 60)
    print("评估完成")
    print("=" * 60)


if __name__ == "__main__":
    main()