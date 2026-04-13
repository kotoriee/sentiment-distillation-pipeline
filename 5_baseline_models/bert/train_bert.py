#!/usr/bin/env python3
"""
BERT基线模型训练脚本 - 硬标签

使用bert-base-uncased进行三分类情感分析

Usage:
    python train_bert.py --epochs 3
"""

import csv
import json
import argparse
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW  # 从torch导入
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import time

# 配置
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed" / "baseline"
RESULTS_DIR = PROJECT_DIR / "6_experiments_results" / "baseline_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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


def load_csv(path: Path):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            labels.append(int(row["label"]))
    return texts, labels


def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    return total_loss / len(dataloader)


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    print("=" * 60)
    print("BERT基线模型训练 (硬标签)")
    print("=" * 60)
    print(f"Epochs: {args.epochs}, LR: {args.lr}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载tokenizer和模型
    print("\n加载模型...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    model.to(device)

    # 加载数据
    print("\n加载数据...")
    train_texts, train_labels = load_csv(DATA_DIR / "train.csv")
    val_texts, val_labels = load_csv(DATA_DIR / "val.csv")
    test_texts, test_labels = load_csv(DATA_DIR / "test.csv")

    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    print(f"  train: {len(train_dataset)}条")
    print(f"  val: {len(val_dataset)}条")
    print(f"  test: {len(test_dataset)}条")

    # 训练配置
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # 训练
    print("\n开始训练...")
    start_time = time.time()
    best_val_f1 = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"  Train Loss: {train_loss:.4f}")

        # 验证
        val_preds, val_true = evaluate(model, val_loader, device)
        val_acc = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds, average="macro")
        print(f"  Val Accuracy: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), RESULTS_DIR / "bert_best_model.pt")

    train_time = time.time() - start_time
    print(f"\n训练总耗时: {train_time:.2f}s")

    # 测试评估
    print("\n测试集评估...")
    model.load_state_dict(torch.load(RESULTS_DIR / "bert_best_model.pt"))

    start_time = time.time()
    test_preds, test_true = evaluate(model, test_loader, device)
    infer_time = time.time() - start_time

    test_acc = accuracy_score(test_true, test_preds)
    test_f1 = f1_score(test_true, test_preds, average="macro")

    print(f"\n测试集结果:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Macro-F1: {test_f1:.4f}")
    print(f"  推理耗时: {infer_time:.2f}s ({infer_time/len(test_texts)*1000:.2f}ms/条)")

    print("\n分类报告:")
    print(classification_report(test_true, test_preds, target_names=["Negative", "Neutral", "Positive"]))

    # 保存结果
    results = {
        "model": "bert-base-uncased",
        "train_samples": len(train_dataset),
        "epochs": args.epochs,
        "lr": args.lr,
        "val_accuracy": val_acc,
        "val_f1": best_val_f1,
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "train_time": train_time,
        "inference_time_per_sample": infer_time / len(test_texts),
        "max_len": MAX_LEN,
        "batch_size": BATCH_SIZE
    }

    results_path = RESULTS_DIR / "bert_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {results_path}")

    print("\n" + "=" * 60)
    print("BERT训练完成")
    print("=" * 60)


if __name__ == "__main__":
    main()