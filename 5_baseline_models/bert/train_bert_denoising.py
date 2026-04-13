"""
在降噪组设置下重新训练 BERT：
- 训练：清洗后的标签
- 测试：原始硬标签
"""

import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import json
from tqdm import tqdm

DATA_DIR = Path("experiments/denoising_setup")
OUTPUT_DIR = Path("experiments/denoising_setup/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_data(split, use_original_label=False):
    """加载数据"""
    if split in ['train', 'val']:
        filepath = DATA_DIR / f"{split}_cleaned.csv"
        df = pd.read_csv(filepath)
        return df['text'].tolist(), df['label'].tolist()
    else:  # test
        filepath = DATA_DIR / "test_original.csv"
        df = pd.read_csv(filepath)
        if use_original_label:
            return df['text'].tolist(), df['label'].tolist()
        else:
            return df['text'].tolist(), df['cleaned_label'].tolist()

def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    return total_loss / len(data_loader)

def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return np.array(predictions), np.array(true_labels)

def main():
    print("="*60)
    print("BERT - 数据降噪组设置")
    print("训练: 清洗标签 | 测试: 原始硬标签")
    print("="*60)

    # 超参数
    MAX_LEN = 256
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5

    # 加载数据
    print("\n[1/4] 加载数据...")
    train_texts, train_labels = load_data('train')
    val_texts, val_labels = load_data('val')
    test_texts, test_labels = load_data('test', use_original_label=True)  # 关键！用原始硬标签

    print(f"  Train: {len(train_texts)} 条")
    print(f"  Val:   {len(val_texts)} 条")
    print(f"  Test:  {len(test_texts)} 条 (原始硬标签)")

    # 加载模型
    print("\n[2/4] 加载 BERT 模型...")
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model = model.to(device)

    # 数据集
    print("\n[3/4] 准备数据集...")
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, MAX_LEN)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 训练
    print("\n[4/4] 训练模型...")
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_val_f1 = 0
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"  Train Loss: {train_loss:.4f}")

        val_pred, val_true = evaluate(model, val_loader, device)
        val_acc = accuracy_score(val_true, val_pred)
        val_f1 = f1_score(val_true, val_pred, average='macro')
        print(f"  Val Accuracy: {val_acc:.4f}, Val Macro-F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), OUTPUT_DIR / "bert_best_model.pt")
            print(f"  保存最佳模型 (Val Macro-F1: {val_f1:.4f})")

    # 测试
    print("\n[5/5] 测试集评估...")
    model.load_state_dict(torch.load(OUTPUT_DIR / "bert_best_model.pt"))
    test_pred, test_true = evaluate(model, test_loader, device)

    test_acc = accuracy_score(test_true, test_pred)
    test_f1_macro = f1_score(test_true, test_pred, average='macro')
    test_f1_weighted = f1_score(test_true, test_pred, average='weighted')

    print(f"\n  Test Accuracy: {test_acc:.4f}")
    print(f"  Test Macro-F1: {test_f1_macro:.4f}")
    print(f"  Test Weighted-F1: {test_f1_weighted:.4f}")

    print("\n  Classification Report:")
    print(classification_report(test_true, test_pred,
                               target_names=['Negative', 'Neutral', 'Positive']))

    print("\n  Confusion Matrix:")
    cm = confusion_matrix(test_true, test_pred)
    print(f"    True\\Pred  Neg  Neu  Pos")
    for i, row in enumerate(cm):
        label = ['Neg', 'Neu', 'Pos'][i]
        print(f"    {label}        {row[0]:3d}  {row[1]:3d}  {row[2]:3d}")

    # 保存结果
    results = {
        'model': 'BERT (bert-base-uncased)',
        'test_accuracy': float(test_acc),
        'test_macro_f1': float(test_f1_macro),
        'test_weighted_f1': float(test_f1_weighted)
    }

    with open(OUTPUT_DIR / "bert_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    torch.save(model.state_dict(), OUTPUT_DIR / "bert_model.pt")

    print("\n" + "="*60)
    print("BERT 评估完成!")
    print(f"结果保存: {OUTPUT_DIR}")
    print("="*60)

    return results

if __name__ == "__main__":
    results = main()
    print(f"\n结果: {results}")
