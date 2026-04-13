"""
BERT 软标签训练（Soft Label Training / Knowledge Distillation）

使用LLM提供的概率分布 [p_neg, p_neu, p_pos] 进行训练，
而非argmax后的硬标签。

损失函数：KL散度（Soft Target Distillation）
"""

import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("experiments/denoising_setup")
OUTPUT_DIR = Path("experiments/denoising_setup/results_soft_label")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


class SoftLabelDataset(Dataset):
    """支持软标签的数据集"""
    def __init__(self, texts, soft_labels, tokenizer, max_len=256):
        """
        Args:
            texts: 文本列表
            soft_labels: 软标签列表，每个元素是 [p_neg, p_neu, p_pos]
            tokenizer: BERT tokenizer
            max_len: 最大序列长度
        """
        self.texts = texts
        self.soft_labels = soft_labels  # 保持概率分布
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        soft_label = self.soft_labels[idx]  # [p_neg, p_neu, p_pos]

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
            'soft_label': torch.tensor(soft_label, dtype=torch.float32),  # 软标签
            'hard_label': torch.argmax(torch.tensor(soft_label)).item()  # 用于监控
        }


def load_data_with_soft_labels(split):
    """
    加载带软标签的数据

    从 soft_labels_reviewed.jsonl 加载原始概率分布
    """
    SOFT_LABELS_FILE = Path("data/processed/soft_labels_reviewed.jsonl")

    samples = []
    with open(SOFT_LABELS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if data.get('split') == split:
                samples.append({
                    'id': data['id'],
                    'text': data['text'],
                    'soft_label': data['probabilities'],  # [p_neg, p_neu, p_pos]
                    'hard_label': data['hard_label']  # 原始硬标签（仅用于测试）
                })

    return samples


def soft_cross_entropy_loss(pred_logits, target_soft, temperature=1.0):
    """
    软标签交叉熵损失

    Args:
        pred_logits: 模型输出的logits [batch_size, num_classes]
        target_soft: 软标签概率分布 [batch_size, num_classes]
        temperature: 温度参数（蒸馏温度）

    Returns:
        loss: 标量损失值
    """
    # 应用温度缩放
    pred_soft = F.softmax(pred_logits / temperature, dim=-1)

    # KL散度损失（等价于软标签交叉熵）
    # KL(P_target || P_pred) = Σ P_target * log(P_target / P_pred)
    # 忽略 target_soft * log(target_soft) 常数项
    loss = F.kl_div(
        pred_soft.log(),
        target_soft,
        reduction='batchmean'
    ) * (temperature ** 2)  # 温度缩放补偿

    return loss


def train_epoch_soft(model, data_loader, optimizer, scheduler, device, temperature=2.0):
    """
    软标签训练的一个epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(data_loader, desc="Training (Soft Label)")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        soft_labels = batch['soft_label'].to(device)  # [batch_size, 3]

        optimizer.zero_grad()

        # 前向传播
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [batch_size, 3]

        # 软标签损失
        loss = soft_cross_entropy_loss(logits, soft_labels, temperature)

        total_loss += loss.item()
        num_batches += 1

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # 更新进度条
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / num_batches


def evaluate_soft(model, data_loader, device):
    """
    评估模型（使用硬标签计算准确率，但可以看到软标签预测的分布）
    """
    model.eval()
    predictions = []
    true_labels = []
    soft_distributions = []  # 记录预测的软分布

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # 兼容软标签数据集和硬标签数据集
            if 'hard_label' in batch:
                hard_labels = batch['hard_label']  # 软标签数据集
            elif 'label' in batch:
                hard_labels = batch['label']  # 硬标签数据集
            else:
                raise KeyError("Batch must contain 'hard_label' or 'label' key")

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # 预测概率分布
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(hard_labels.cpu().numpy() if torch.is_tensor(hard_labels) else hard_labels.numpy())
            soft_distributions.extend(probs.cpu().numpy())

    return np.array(predictions), np.array(true_labels), np.array(soft_distributions)


def analyze_uncertainty(soft_distributions, true_labels, predictions):
    """
    分析模型预测的不确定性
    """
    print("\n" + "="*60)
    print("不确定性分析")
    print("="*60)

    # 计算每个样本的最大概率（置信度）
    max_probs = np.max(soft_distributions, axis=1)

    # 按置信度分桶统计准确率
    buckets = {
        'high_conf (>0.9)': (max_probs > 0.9),
        'med_conf (0.7-0.9)': ((max_probs >= 0.7) & (max_probs <= 0.9)),
        'low_conf (<0.7)': (max_probs < 0.7)
    }

    print("\n按置信度分层的准确率:")
    print(f"{'置信度区间':<20} {'样本数':>10} {'准确率':>10}")
    print("-" * 45)

    for name, mask in buckets.items():
        if mask.sum() > 0:
            acc = accuracy_score(true_labels[mask], predictions[mask])
            print(f"{name:<20} {mask.sum():>10} {acc:>10.4f}")

    # 校准度分析
    from sklearn.calibration import calibration_curve

    # 将多分类转为二分类（预测类别的概率 vs 实际准确率）
    pred_probs = np.max(soft_distributions, axis=1)
    correct = (predictions == true_labels).astype(float)

    print(f"\n平均置信度: {pred_probs.mean():.4f}")
    print(f"实际准确率: {correct.mean():.4f}")
    print(f"校准差距: {pred_probs.mean() - correct.mean():.4f}")


def main():
    print("="*70)
    print("BERT 软标签训练 (Soft Label Training / Knowledge Distillation)")
    print("="*70)
    print("训练: LLM软标签分布 | 测试: 原始硬标签")
    print("="*70)

    # 超参数
    MAX_LEN = 256
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    TEMPERATURE = 2.0  # 蒸馏温度（软标签训练关键参数）

    print(f"\n配置参数:")
    print(f"  温度 T: {TEMPERATURE} (控制软标签平滑度)")
    print(f"  学习率: {LEARNING_RATE}")
    print(f"  Epochs: {EPOCHS}")

    # 加载数据（带软标签）
    print("\n[1/4] 加载软标签数据...")
    train_samples = load_data_with_soft_labels('train')
    val_samples = load_data_with_soft_labels('val')

    # 测试集加载原始硬标签
    test_df = pd.read_csv(DATA_DIR / "test_original.csv")
    test_texts = test_df['text'].tolist()
    test_labels = test_df['label'].tolist()

    print(f"  Train: {len(train_samples)} 条 (软标签)")
    print(f"  Val:   {len(val_samples)} 条 (软标签)")
    print(f"  Test:  {len(test_texts)} 条 (原始硬标签)")

    # 查看软标签示例
    print("\n软标签示例:")
    for i in range(min(3, len(train_samples))):
        s = train_samples[i]
        print(f"  {i+1}. Text: {s['text'][:50]}...")
        print(f"     Soft: [{s['soft_label'][0]:.2f}, {s['soft_label'][1]:.2f}, {s['soft_label'][2]:.2f}]")
        print(f"     Hard: {s['hard_label']} ({['Neg','Neu','Pos'][s['hard_label']]})")

    # 加载模型
    print("\n[2/4] 加载 BERT 模型...")
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model = model.to(device)

    # 准备数据集
    print("\n[3/4] 准备数据集...")
    train_dataset = SoftLabelDataset(
        [s['text'] for s in train_samples],
        [s['soft_label'] for s in train_samples],
        tokenizer, MAX_LEN
    )
    val_dataset = SoftLabelDataset(
        [s['text'] for s in val_samples],
        [s['soft_label'] for s in val_samples],
        tokenizer, MAX_LEN
    )

    # 测试集使用普通Dataset（只有硬标签）
    from train_bert_denoising import SentimentDataset
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 训练
    print("\n[4/4] 训练模型 (软标签)...")
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    best_val_f1 = 0
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        train_loss = train_epoch_soft(
            model, train_loader, optimizer, scheduler, device, TEMPERATURE
        )
        print(f"  Train Loss: {train_loss:.4f}")

        # 验证（使用硬标签评估）
        val_pred, val_true, val_soft = evaluate_soft(model, val_loader, device)
        val_acc = accuracy_score(val_true, val_pred)
        val_f1 = f1_score(val_true, val_pred, average='macro')
        print(f"  Val Accuracy: {val_acc:.4f}, Val Macro-F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), OUTPUT_DIR / "bert_soft_best_model.pt")
            print(f"  ✓ 保存最佳模型 (Val Macro-F1: {val_f1:.4f})")

    # 测试
    print("\n[5/5] 测试集评估...")
    model.load_state_dict(torch.load(OUTPUT_DIR / "bert_soft_best_model.pt"))
    test_pred, test_true, test_soft = evaluate_soft(model, test_loader, device)

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

    # 不确定性分析
    analyze_uncertainty(test_soft, test_true, test_pred)

    # 保存结果
    results = {
        'model': 'BERT (Soft Label Training / Knowledge Distillation)',
        'config': {
            'max_len': MAX_LEN,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'temperature': TEMPERATURE
        },
        'test_accuracy': float(test_acc),
        'test_macro_f1': float(test_f1_macro),
        'test_weighted_f1': float(test_f1_weighted),
        'best_val_f1': float(best_val_f1)
    }

    with open(OUTPUT_DIR / "bert_soft_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    torch.save(model.state_dict(), OUTPUT_DIR / "bert_soft_model.pt")

    print("\n" + "="*70)
    print("BERT 软标签训练完成!")
    print(f"结果保存: {OUTPUT_DIR}")
    print("="*70)

    return results


if __name__ == "__main__":
    results = main()
    print(f"\n结果: {results}")
