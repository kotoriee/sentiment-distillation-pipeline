#!/usr/bin/env python3
"""
快速调参实验脚本 - 600条样本
测试不同超参数组合，找到最优配置后再全量训练
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
import os
from functools import partial

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'


class QuickDataset(Dataset):
    """简化数据集"""
    def __init__(self, data_path: str, tokenizer, max_seq_len: int = 64):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        with open(data_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
        self.instruction = "Sentiment (0=neg,1=neu,2=pos):"
        for sample in self.samples:
            soft_label = sample.get('soft_label', sample.get('soft_labels', [0.33, 0.33, 0.34]))
            sample['confidence'] = max(soft_label)
            sample['soft_labels'] = soft_label

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample.get('text') or sample.get('input', '')
        prompt = f"{self.instruction}\n{text}\nAnswer:"
        encoding = self.tokenizer(
            prompt, max_length=self.max_seq_len,
            padding=False, truncation=True, return_tensors='pt'
        )
        label = sample.get('label')
        if label is None:
            label = int(sample.get('output', 0))  # Alpaca格式
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(int(label), dtype=torch.long),
            'soft_labels': torch.tensor(sample['soft_labels'], dtype=torch.float32),
            'confidence': torch.tensor(sample['confidence'], dtype=torch.float32)
        }


def collate_fn(batch, tokenizer):
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = torch.stack([item['labels'] for item in batch])
    soft_labels = torch.stack([item['soft_labels'] for item in batch])
    confidences = torch.stack([item['confidence'] for item in batch])

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_masks, batch_first=True, padding_value=0
    )

    return {
        'input_ids': input_ids, 'attention_mask': attention_mask,
        'labels': labels, 'soft_labels': soft_labels, 'confidence': confidences
    }


def adaptive_temperature_v1(confidence: float) -> float:
    """版本1: 原始策略"""
    if confidence > 0.9:
        return 1.5
    elif confidence > 0.6:
        return 2.0
    else:
        return min(2.5 + (0.6 - confidence) * 2, 3.0)


def adaptive_temperature_v2(confidence: float) -> float:
    """版本2: 更激进的温度调整"""
    if confidence > 0.85:
        return 1.2
    elif confidence > 0.6:
        return 1.8
    else:
        return min(2.0 + (0.6 - confidence) * 3, 4.0)


def adaptive_temperature_v3(confidence: float) -> float:
    """版本3: 线性插值"""
    # 置信度0->温度3.0, 置信度1->温度1.0
    return 3.0 - confidence * 2.0


class AdaptiveDistillationLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, temp_fn=None):
        super().__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.temp_fn = temp_fn or adaptive_temperature_v1

    def forward(self, logits, hard_labels, soft_labels, confidences):
        batch_size = logits.size(0)
        ce = self.ce_loss(logits, hard_labels)

        kl_total = 0.0
        temps_used = []

        for i in range(batch_size):
            temp = self.temp_fn(confidences[i].item())
            temps_used.append(temp)
            student_probs = F.log_softmax(logits[i] / temp, dim=-1)
            teacher_probs = soft_labels[i]
            kl = F.kl_div(student_probs.unsqueeze(0), teacher_probs.unsqueeze(0), reduction='batchmean')
            kl_total += kl

        kl_avg = kl_total / batch_size
        total = self.alpha * kl_avg + (1 - self.alpha) * ce

        return total, {
            'loss': total.item(), 'ce': ce.item(), 'kl': kl_avg.item(),
            'avg_temp': np.mean(temps_used)
        }


class QuickTrainer:
    def __init__(self, model, config: Dict):
        self.model = model
        self.config = config
        self.criterion = AdaptiveDistillationLoss(
            alpha=config.get('alpha', 0.5),
            temp_fn=config.get('temp_fn', adaptive_temperature_v1)
        )
        self.grad_accum_steps = config.get('grad_accum_steps', 4)
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.get('lr', 2e-4), weight_decay=0.01
        )

    def train_epoch(self, dataloader, epoch: int, max_steps: int = 50):
        self.model.train()
        total_loss = 0.0
        all_metrics = []
        self.optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            if max_steps and step >= max_steps:
                break

            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['labels'].cuda()
            soft_labels = batch['soft_labels'].cuda()
            confidences = batch['confidence'].cuda()

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, -1, :3]
                loss, metrics = self.criterion(logits, labels, soft_labels, confidences)
                loss = loss / self.grad_accum_steps

            loss.backward()
            if (step + 1) % self.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum_steps
            all_metrics.append(metrics)

            if step % 10 == 0:
                torch.cuda.empty_cache()

            pbar.set_postfix({'loss': f"{metrics['loss']:.4f}", 'avg_temp': f"{metrics['avg_temp']:.2f}"})

        return {
            'loss': total_loss / len(all_metrics) if all_metrics else 0,
            'avg_temp': np.mean([m['avg_temp'] for m in all_metrics])
        }


def quick_eval(model, tokenizer, data_path: str, max_samples: int = 100):
    """快速评估"""
    model.eval()
    with open(data_path, 'r') as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    correct = 0
    total = 0

    with torch.no_grad():
        for item in tqdm(data, desc="Eval"):
            text = item.get('text') or item.get('input', '')
            prompt = f"Sentiment (0=neg,1=neu,2=pos):\n{text}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt", max_length=64, truncation=True).to(model.device)
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :3]
            pred = logits.argmax(dim=-1).item()

            label = item.get('label')
            if label is None:
                label = int(item.get('output', 0))

            if pred == int(label):
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy


def run_experiment(config: Dict, train_data: str, val_data: str):
    """运行单次实验"""
    print(f"\n{'='*60}")
    print(f"实验: {config['name']}")
    print(f"LR={config['lr']}, Alpha={config['alpha']}, Temp={config.get('temp_name', 'v1')}")
    print(f"{'='*60}")

    # 加载模型
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        'unsloth/Qwen3-4B-unsloth-bnb-4bit',
        quantization_config=bnb_config, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained('unsloth/Qwen3-4B-unsloth-bnb-4bit')
    tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()

    # 数据
    train_dataset = QuickDataset(train_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                              collate_fn=partial(collate_fn, tokenizer=tokenizer))

    # 训练
    trainer = QuickTrainer(model, config)
    best_acc = 0

    for epoch in range(3):
        metrics = trainer.train_epoch(train_loader, epoch, max_steps=50)
        print(f"Epoch {epoch}: Loss={metrics['loss']:.4f}, AvgTemp={metrics['avg_temp']:.2f}")

        # 快速评估
        acc = quick_eval(model, tokenizer, val_data, max_samples=100)
        print(f"  验证准确率: {acc:.4f} ({acc*100:.2f}%)")
        best_acc = max(best_acc, acc)

    del model
    torch.cuda.empty_cache()

    return {
        'name': config['name'],
        'lr': config['lr'],
        'alpha': config['alpha'],
        'temp': config.get('temp_name', 'v1'),
        'best_acc': best_acc
    }


def main():
    """主函数 - 运行多个实验"""
    train_data = "../../data/curriculum/train_600.json"
    val_data = "../../data/curriculum/val_fixed.json"

    # 实验配置
    experiments = [
        # 学习率实验
        {'name': 'LR_1e-4', 'lr': 1e-4, 'alpha': 0.5, 'temp_fn': adaptive_temperature_v1, 'temp_name': 'v1'},
        {'name': 'LR_3e-4', 'lr': 3e-4, 'alpha': 0.5, 'temp_fn': adaptive_temperature_v1, 'temp_name': 'v1'},
        {'name': 'LR_5e-4', 'lr': 5e-4, 'alpha': 0.5, 'temp_fn': adaptive_temperature_v1, 'temp_name': 'v1'},

        # Alpha实验
        {'name': 'Alpha_0.3', 'lr': 2e-4, 'alpha': 0.3, 'temp_fn': adaptive_temperature_v1, 'temp_name': 'v1'},
        {'name': 'Alpha_0.7', 'lr': 2e-4, 'alpha': 0.7, 'temp_fn': adaptive_temperature_v1, 'temp_name': 'v1'},

        # 温度策略实验
        {'name': 'Temp_v2', 'lr': 2e-4, 'alpha': 0.5, 'temp_fn': adaptive_temperature_v2, 'temp_name': 'v2'},
        {'name': 'Temp_v3', 'lr': 2e-4, 'alpha': 0.5, 'temp_fn': adaptive_temperature_v3, 'temp_name': 'v3'},

        # 最佳组合猜测
        {'name': 'Best_Guess', 'lr': 3e-4, 'alpha': 0.7, 'temp_fn': adaptive_temperature_v2, 'temp_name': 'v2'},
    ]

    results = []
    for config in experiments:
        try:
            result = run_experiment(config, train_data, val_data)
            results.append(result)
        except Exception as e:
            print(f"实验 {config['name']} 失败: {e}")
            import traceback
            traceback.print_exc()

    # 结果汇总
    print(f"\n{'='*60}")
    print("实验结果汇总")
    print(f"{'='*60}")
    results.sort(key=lambda x: x['best_acc'], reverse=True)
    for r in results:
        print(f"{r['name']:15s} | LR={r['lr']:.0e} | Alpha={r['alpha']:.1f} | Temp={r['temp']} | Acc={r['best_acc']:.4f}")

    best = results[0] if results else None
    if best:
        print(f"\n最佳配置: {best['name']}")
        print(f"  学习率: {best['lr']}")
        print(f"  Alpha: {best['alpha']}")
        print(f"  温度策略: {best['temp']}")
        print(f"  最佳准确率: {best['best_acc']:.4f}")


if __name__ == '__main__':
    main()
