"""
Step 3: 使用清洗后的数据训练 BERT
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import sys
from tqdm import tqdm

# 路径配置
DATA_DIR = Path("data/processed/baseline_clean")
OUTPUT_DIR = Path("experiments/logs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

class SentimentDataset(Dataset):
    """情感分析数据集"""
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

def load_data(split):
    """加载 CSV 数据"""
    filepath = DATA_DIR / f"baseline_clean_{split}.csv"
    df = pd.read_csv(filepath)
    return df['text'].tolist(), df['cleaned_label'].tolist()

def train_epoch(model, data_loader, optimizer, scheduler, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    return total_loss / len(data_loader)

def evaluate(model, data_loader, device):
    """评估模型"""
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return np.array(predictions), np.array(true_labels)

def train_and_evaluate():
    """训练并评估 BERT"""
    print("="*60)
    print("Step 3: BERT 基线训练与评估")
    print("="*60)

    # 超参数
    MAX_LEN = 256
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    train_texts, train_labels = load_data('train')
    val_texts, val_labels = load_data('val')
    test_texts, test_labels = load_data('test')

    print(f"  Train: {len(train_texts)} 条")
    print(f"  Val:   {len(val_texts)} 条")
    print(f"  Test:  {len(test_texts)} 条")

    # 2. 加载 Tokenizer 和模型
    print("\n[2/5] 加载 BERT 模型...")
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        output_attentions=False,
        output_hidden_states=False
    )
    model = model.to(device)
    print(f"  模型: {model_name}")

    # 3. 创建数据集和数据加载器
    print("\n[3/5] 准备数据集...")
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, MAX_LEN)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 4. 训练
    print("\n[4/5] 训练模型...")
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    best_val_f1 = 0
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"  Train Loss: {train_loss:.4f}")

        # 验证
        val_pred, val_true = evaluate(model, val_loader, device)
        val_acc = accuracy_score(val_true, val_pred)
        val_f1 = f1_score(val_true, val_pred, average='macro')
        print(f"  Val Accuracy: {val_acc:.4f}, Val Macro-F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # 保存最佳模型
            model_save_path = OUTPUT_DIR / "bert_clean_best_model.pt"
            torch.save(model.state_dict(), model_save_path)
            print(f"  保存最佳模型 (Val Macro-F1: {val_f1:.4f})")

    # 5. 测试集评估
    print("\n[5/5] 测试集评估...")
    # 加载最佳模型
    model.load_state_dict(torch.load(OUTPUT_DIR / "bert_clean_best_model.pt"))
    test_pred, test_true = evaluate(model, test_loader, device)

    test_acc = accuracy_score(test_true, test_pred)
    test_f1_macro = f1_score(test_true, test_pred, average='macro')
    test_f1_weighted = f1_score(test_true, test_pred, average='weighted')

    print(f"\n  Test Accuracy: {test_acc:.4f}")
    print(f"  Test Macro-F1: {test_f1_macro:.4f}")
    print(f"  Test Weighted-F1: {test_f1_weighted:.4f}")

    # 详细分类报告
    print("\n  Classification Report:")
    print(classification_report(test_true, test_pred,
                               target_names=['Negative', 'Neutral', 'Positive']))

    # 混淆矩阵
    print("\n  Confusion Matrix:")
    cm = confusion_matrix(test_true, test_pred)
    print(f"    True\\Pred  Neg  Neu  Pos")
    for i, row in enumerate(cm):
        label = ['Neg', 'Neu', 'Pos'][i]
        print(f"    {label}        {row[0]:3d}  {row[1]:3d}  {row[2]:3d}")

    # 6. 保存预测结果
    predictions = []
    for text, pred, true in zip(test_texts, test_pred, test_true):
        predictions.append({
            'text': text,
            'predicted': int(pred),
            'true': int(true)
        })

    pred_file = OUTPUT_DIR / "bert_clean_predictions.json"
    with open(pred_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"\n  预测结果已保存: {pred_file}")

    # 保存完整模型
    model_save_path = OUTPUT_DIR / "bert_clean_model.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"  模型已保存: {model_save_path}")

    print("\n" + "="*60)
    print("Step 3 完成!")
    print("="*60)

    return {
        'model': 'BERT',
        'test_accuracy': test_acc,
        'test_macro_f1': test_f1_macro,
        'test_weighted_f1': test_f1_weighted
    }

if __name__ == "__main__":
    results = train_and_evaluate()
    print(f"\n最终结果: {results}")
