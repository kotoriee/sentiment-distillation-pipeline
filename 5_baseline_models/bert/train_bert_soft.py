#!/usr/bin/env python3
"""
BERT软标签训练脚本 - 知识蒸馏

使用KL散度损失进行软标签蒸馏训练

Usage:
    python train_bert_soft.py --epochs 3 --temperature 2.0
"""

import json
import argparse
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
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


class SoftLabelDataset(Dataset):
    """软标签数据集"""
    def __init__(self, path: Path, tokenizer, max_len=MAX_LEN):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.data.append({
                    "text": item["text"],
                    "soft_labels": item["probabilities"],
                    "hard_label": item["hard_label"]
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["text"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "soft_labels": torch.tensor(item["soft_labels"], dtype=torch.float),
            "hard_label": torch.tensor(item["hard_label"], dtype=torch.long)
        }


def soft_cross_entropy_loss(logits, soft_labels, temperature=2.0):
    """软标签KL散度损失"""
    # logits: [batch, num_classes]
    # soft_labels: [batch, num_classes]

    # 温度缩放
    logits_soft = logits / temperature
    soft_labels_soft = soft_labels

    # 计算KL散度
    log_probs = F.log_softmax(logits_soft, dim=-1)
    loss = F.kl_div(log_probs, soft_labels_soft, reduction="batchmean") * (temperature ** 2)

    return loss


def train_epoch(model, dataloader, optimizer, scheduler, device, temperature):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        soft_labels = batch["soft_labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 软标签损失
        loss = soft_cross_entropy_loss(outputs.logits, soft_labels, temperature)
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """使用硬标签评估"""
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            hard_labels = batch["hard_label"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(hard_labels.numpy())

    return predictions, true_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=2.0)
    args = parser.parse_args()

    print("=" * 60)
    print("BERT软标签训练 (知识蒸馏)")
    print("=" * 60)
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Temperature: {args.temperature}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载tokenizer和模型
    print("\n加载模型...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    model.to(device)

    # 加载软标签数据
    print("\n加载软标签数据...")
    train_dataset = SoftLabelDataset(DATA_DIR / "soft_labels.jsonl", tokenizer)

    # 加载测试集（硬标签）
    test_texts, test_labels = [], []
    import csv
    with open(DATA_DIR / "test.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_texts.append(row["text"])
            test_labels.append(int(row["label"]))

    class TestDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoding = self.tokenizer(
                self.texts[idx],
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "hard_label": torch.tensor(self.labels[idx], dtype=torch.long)
            }

    test_dataset = TestDataset(test_texts, test_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    print(f"  train (软标签): {len(train_dataset)}条")
    print(f"  test (硬标签): {len(test_dataset)}条")

    # 训练配置
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # 训练
    print("\n开始训练...")
    start_time = time.time()

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, args.temperature)
        print(f"  Train Loss (KL): {train_loss:.4f}")

    train_time = time.time() - start_time
    print(f"\n训练总耗时: {train_time:.2f}s")

    # 测试评估
    print("\n测试集评估...")
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
        "model": "bert-base-uncased (soft label)",
        "train_samples": len(train_dataset),
        "epochs": args.epochs,
        "lr": args.lr,
        "temperature": args.temperature,
        "loss": "KL-divergence",
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "train_time": train_time,
        "inference_time_per_sample": infer_time / len(test_texts),
        "max_len": MAX_LEN,
        "batch_size": BATCH_SIZE
    }

    results_path = RESULTS_DIR / "bert_soft_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {results_path}")

    # 保存模型
    torch.save(model.state_dict(), RESULTS_DIR / "bert_soft_model.pt")
    print(f"模型已保存: {RESULTS_DIR / 'bert_soft_model.pt'}")

    print("\n" + "=" * 60)
    print("BERT软标签训练完成")
    print("=" * 60)


if __name__ == "__main__":
    main()