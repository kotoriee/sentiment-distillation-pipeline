"""
优化版 BERT 训练 - 学习率调优 + warmup + 5 epochs + early stopping
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
OUTPUT_DIR = Path("experiments/denoising_setup/results_optimized")
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

def get_optimizer_with_layer_lr(model, lr, layer_decay=0.95):
    """
    分层学习率：顶层分类器用较高LR，底层encoder用较低LR
    """
    no_decay = ['bias', 'LayerNorm.weight']

    # 分类器层 - 最高学习率
    classifier_params = [
        {'params': [p for n, p in model.classifier.named_parameters()
                    if not any(nd in n for nd in no_decay)], 'lr': lr, 'weight_decay': 0.01},
        {'params': [p for n, p in model.classifier.named_parameters()
                    if any(nd in n for nd in no_decay)], 'lr': lr, 'weight_decay': 0.0}
    ]

    # BERT encoder层 - 逐层衰减学习率
    encoder_params = []
    num_layers = len(model.bert.encoder.layer)

    for i, layer in enumerate(model.bert.encoder.layer):
        layer_lr = lr * (layer_decay ** (num_layers - i))  # 越底层学习率越小
        encoder_params.append({
            'params': [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
            'lr': layer_lr, 'weight_decay': 0.01
        })
        encoder_params.append({
            'params': [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': layer_lr, 'weight_decay': 0.0
        })

    # embeddings层 - 最低学习率
    embed_lr = lr * (layer_decay ** num_layers)
    encoder_params.append({
        'params': [p for n, p in model.bert.embeddings.named_parameters()
                   if not any(nd in n for nd in no_decay)],
        'lr': embed_lr, 'weight_decay': 0.01
    })
    encoder_params.append({
        'params': [p for n, p in model.bert.embeddings.named_parameters()
                   if any(nd in n for nd in no_decay)],
        'lr': embed_lr, 'weight_decay': 0.0
    })

    all_params = classifier_params + encoder_params
    return AdamW(all_params)

def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        num_batches += 1

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    return total_loss / num_batches

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
    print("BERT 优化版 - 数据降噪组设置")
    print("优化: 分层学习率 + warmup + 5 epochs + early stopping")
    print("训练: 清洗标签 | 测试: 原始硬标签")
    print("="*60)

    # 优化后的超参数
    MAX_LEN = 256
    BATCH_SIZE = 16
    EPOCHS = 5  # 增加到5轮
    LEARNING_RATE = 2e-5
    WARMUP_RATIO = 0.1  # 10% warmup
    LAYER_DECAY = 0.95  # 分层学习率衰减
    EARLY_STOPPING_PATIENCE = 2  # 早停耐心值

    # 加载数据
    print("\n[1/4] 加载数据...")
    train_texts, train_labels = load_data('train')
    val_texts, val_labels = load_data('val')
    test_texts, test_labels = load_data('test', use_original_label=True)

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

    # 训练 - 使用分层学习率
    print("\n[4/4] 训练模型...")
    print(f"  学习率: {LEARNING_RATE}, warmup比例: {WARMUP_RATIO}, 分层衰减: {LAYER_DECAY}")

    optimizer = get_optimizer_with_layer_lr(model, LEARNING_RATE, LAYER_DECAY)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    print(f"  总步数: {total_steps}, warmup步数: {warmup_steps}")

    best_val_f1 = 0
    epochs_no_improve = 0

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
            epochs_no_improve = 0
            torch.save(model.state_dict(), OUTPUT_DIR / "bert_optimized_best_model.pt")
            print(f"  ✓ 保存最佳模型 (Val Macro-F1: {val_f1:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  验证F1未提升 ({epochs_no_improve}/{EARLY_STOPPING_PATIENCE})")

            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"  触发早停，提前结束训练")
                break

    # 测试
    print("\n[5/5] 测试集评估...")
    model.load_state_dict(torch.load(OUTPUT_DIR / "bert_optimized_best_model.pt"))
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
        'model': 'BERT Optimized (分层LR + warmup + 5epochs + early stopping)',
        'config': {
            'max_len': MAX_LEN,
            'batch_size': BATCH_SIZE,
            'epochs': epoch + 1,  # 实际运行的epoch数
            'learning_rate': LEARNING_RATE,
            'warmup_ratio': WARMUP_RATIO,
            'layer_decay': LAYER_DECAY,
            'early_stopping_patience': EARLY_STOPPING_PATIENCE
        },
        'test_accuracy': float(test_acc),
        'test_macro_f1': float(test_f1_macro),
        'test_weighted_f1': float(test_f1_weighted),
        'best_val_f1': float(best_val_f1)
    }

    with open(OUTPUT_DIR / "bert_optimized_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    torch.save(model.state_dict(), OUTPUT_DIR / "bert_optimized_model.pt")

    print("\n" + "="*60)
    print("BERT 优化版评估完成!")
    print(f"结果保存: {OUTPUT_DIR}")
    print("="*60)

    return results

if __name__ == "__main__":
    results = main()
    print(f"\n结果: {results}")
