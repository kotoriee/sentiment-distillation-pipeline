#!/usr/bin/env python3
"""
答案优先格式 + 动态温度蒸馏训练

直接复用 adaptive_temperature.py 的成熟训练逻辑，
仅修改数据格式为答案优先格式。

对比目标：
- 已有固定温度结果: 80.38% (models/qwen3-4b-answer-first)
- 本脚本: 动态温度训练，评估后对比

Usage:
    python train_answer_first_dynamic_temp.py --epochs 3

    # 快速测试
    python train_answer_first_dynamic_temp.py --test
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# 设置内存分配配置避免碎片化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

# ============== 配置 ==============

DEFAULT_MODEL = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 512
LORA_RANK = 16
RANDOM_STATE = 3407


# ============== 动态温度策略（直接复用原始代码）==============

def adaptive_temperature(confidence: float) -> float:
    """
    根据置信度选择温度

    Args:
        confidence: 样本置信度 (0-1)

    Returns:
        temperature: 蒸馏温度
    """
    if confidence > 0.9:
        return 1.5
    elif confidence > 0.6:
        return 2.0
    else:
        # 低置信度使用更高温度，最高3.0
        return min(2.5 + (0.6 - confidence) * 2, 3.0)


class AdaptiveDistillationLoss(nn.Module):
    """
    自适应温度蒸馏损失（直接复用原始代码）

    混合损失: alpha * KL + (1-alpha) * CE
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self,
                logits: torch.Tensor,
                hard_labels: torch.Tensor,
                soft_labels: torch.Tensor,
                confidences: torch.Tensor) -> tuple:
        """
        计算混合损失

        Args:
            logits: 学生模型输出 (B, 3) - 仅 sentiment 三分类
            hard_labels: 硬标签 (B,)
            soft_labels: 软标签 (B, 3)
            confidences: 置信度 (B,)

        Returns:
            total_loss: 总损失
            metrics: 损失分量
        """
        batch_size = logits.size(0)

        # 硬标签交叉熵
        ce = self.ce_loss(logits, hard_labels)

        # 为每个样本计算温度并应用
        kl_total = 0.0
        temps_used = []

        for i in range(batch_size):
            temp = adaptive_temperature(confidences[i].item())
            temps_used.append(temp)

            # 该样本的KL散度
            student_probs = F.log_softmax(logits[i] / temp, dim=-1)
            teacher_probs = soft_labels[i]

            kl = F.kl_div(student_probs.unsqueeze(0),
                         teacher_probs.unsqueeze(0),
                         reduction='batchmean')
            kl_total += kl

        kl_avg = kl_total / batch_size
        avg_temp = np.mean(temps_used)

        # 混合损失
        total = self.alpha * kl_avg + (1 - self.alpha) * ce

        metrics = {
            'loss': total.item(),
            'ce': ce.item(),
            'kl': kl_avg.item(),
            'avg_temp': avg_temp,
            'min_temp': min(temps_used),
            'max_temp': max(temps_used)
        }

        return total, metrics


# ============== 数据集（答案优先格式）==============

class AnswerFirstDataset(torch.utils.data.Dataset):
    """
    答案优先格式数据集

    输出格式: {"sentiment": X}
    从 soft_labels 提取置信度用于动态温度
    """

    def __init__(self, data_path: str, tokenizer, max_seq_len: int = MAX_SEQ_LENGTH):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        with open(data_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)

        # 预计算置信度
        for sample in self.samples:
            soft_labels = sample.get('soft_labels', [0.33, 0.33, 0.34])
            sample['confidence'] = max(soft_labels)
            sample['soft_labels'] = soft_labels

        print(f"加载 {len(self.samples)} 条样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        conv = sample['conversations']

        # ChatML 格式构建 prompt（仅输入部分）
        prompt = self.tokenizer.apply_chat_template(
            conv[:-1],  # system + user
            tokenize=False,
            add_generation_prompt=True
        )

        # 编码
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_seq_len,
            padding=False,
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(sample['label'], dtype=torch.long),
            'soft_labels': torch.tensor(sample['soft_labels'], dtype=torch.float32),
            'confidence': torch.tensor(sample['confidence'], dtype=torch.float32)
        }


def collate_fn(batch, tokenizer):
    """动态padding到batch最大长度"""
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = torch.stack([item['labels'] for item in batch])
    soft_labels = torch.stack([item['soft_labels'] for item in batch])
    confidences = torch.stack([item['confidence'] for item in batch])

    # 动态padding
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id or 0
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_masks, batch_first=True, padding_value=0
    )

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'soft_labels': soft_labels,
        'confidence': confidences
    }


# ============== 训练器（复用原始逻辑）==============

class AdaptiveTrainer:
    """
    动态温度训练器（复用原始 adaptive_temperature.py 逻辑）

    核心改动：答案优先格式数据加载
    训练逻辑完全复用成熟代码
    """

    def __init__(self, model, tokenizer, config: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.criterion = AdaptiveDistillationLoss(alpha=config.get('alpha', 0.5))
        self.grad_accum_steps = config.get('grad_accum_steps', 16)

        # 只训练LoRA参数
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.get('lr', 2e-5),
            weight_decay=0.01
        )

        # 学习率调度器
        total_steps = config.get('total_steps', 1347)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.get('epochs', 3)
        )

    def train_epoch(self, dataloader, epoch: int, max_steps: int = None) -> dict:
        """训练一个epoch（带梯度累积和混合精度）"""
        self.model.train()
        total_loss = 0.0
        all_metrics = []

        self.optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            if max_steps and step >= max_steps:
                break

            # 移动到GPU
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['labels'].cuda()
            soft_labels = batch['soft_labels'].cuda()
            confidences = batch['confidence'].cuda()

            # 混合精度前向传播
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                # 取最后一个位置的 logits，仅取前3个（sentiment分类）
                logits = outputs.logits[:, -1, :3]
                loss, metrics = self.criterion(logits, labels, soft_labels, confidences)
                loss = loss / self.grad_accum_steps  # 缩放损失

            # 反向传播
            loss.backward()

            # 梯度累积步数到达后更新
            if (step + 1) % self.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum_steps
            all_metrics.append(metrics)

            # 每20步清理一次GPU缓存
            if step % 20 == 0:
                torch.cuda.empty_cache()

            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'avg_temp': f"{metrics['avg_temp']:.2f}"
            })

        # 更新学习率
        self.scheduler.step()

        # 聚合指标
        actual_steps = len(all_metrics) if all_metrics else 1
        return {
            'loss': total_loss / actual_steps,
            'avg_temp': np.mean([m['avg_temp'] for m in all_metrics]),
            'ce': np.mean([m['ce'] for m in all_metrics]),
            'kl': np.mean([m['kl'] for m in all_metrics])
        }


def main():
    args = parse_args()

    print("=" * 60)
    print("答案优先格式 + 动态温度蒸馏训练")
    print("=" * 60)
    print(f"训练数据: {args.train_data}")
    print(f"输出目录: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Alpha: {args.alpha}")
    print(f"学习率: {args.lr}")
    print("=" * 60)

    # 加载模型和tokenizer
    print("\n加载模型...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model

    # 配置4-bit量化
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 配置LoRA（与原始成功训练保持一致）
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],  # 与固定温度训练一致
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # 关闭梯度检查点（避免与use_cache冲突）
    # model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    # 数据集
    print("\n加载数据集...")
    train_dataset = AnswerFirstDataset(args.train_data, tokenizer)

    # DataLoader
    from functools import partial
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
        num_workers=0,
        pin_memory=False
    )

    # 计算总步数
    total_steps = len(train_loader) // args.grad_accum_steps * args.epochs
    print(f"总步数: {total_steps}")

    # 训练配置
    config = {
        'lr': args.lr,
        'alpha': args.alpha,
        'epochs': args.epochs,
        'grad_accum_steps': args.grad_accum_steps,
        'total_steps': total_steps
    }
    trainer = AdaptiveTrainer(model, tokenizer, config)

    # 训练循环
    print("\n开始训练...")

    # 快速验证模式或完整训练
    if args.test:
        max_steps = 50
        print(f"快速测试模式：每epoch最多{max_steps}步")
    else:
        max_steps = None
        print(f"完整训练模式：每epoch全部数据 ({len(train_loader)} 步)")

    start_time = datetime.now()

    for epoch in range(args.epochs):
        train_metrics = trainer.train_epoch(train_loader, epoch, max_steps=max_steps)
        print(f"\nEpoch {epoch} 训练指标:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  CE: {train_metrics['ce']:.4f}")
        print(f"  KL: {train_metrics['kl']:.4f}")
        print(f"  Avg Temp: {train_metrics['avg_temp']:.2f}")

    train_time = (datetime.now() - start_time).total_seconds()

    # 保存模型
    print(f"\n保存模型到 {args.output_dir}...")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 配置记录
    config_out = {
        "model": args.model,
        "data": args.train_data,
        "epochs": args.epochs,
        "temperature_strategy": "adaptive",
        "alpha": args.alpha,
        "lr": args.lr,
        "format": "answer_first",
        "train_time_seconds": train_time,
    }
    with open(Path(args.output_dir) / "train_config.json", "w") as f:
        json.dump(config_out, f, indent=2)

    print(f"\n训练完成!")
    print(f"训练时间: {train_time/60:.1f} 分钟")
    print(f"模型保存: {args.output_dir}")

    # 提示评估对比
    print("\n" + "=" * 60)
    print("下一步: 运行评估对比")
    print("=" * 60)
    print(f"已有固定温度结果: 80.38%")
    print(f"评估命令:")
    print(f"  cd 4_evaluation")
    print(f"  python3 eval_answer_first.py --model ../3_lora_training/models/qwen3-4b-adaptive-temp")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train-data", default="../data/train_answer_first.json")
    parser.add_argument("--output-dir", default="models/qwen3-4b-adaptive-temp")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)  # 降低batch避免OOM
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()