"""
动态温度缩放训练脚本

基于置信度自适应调整蒸馏温度的训练实现
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
import os

# 设置内存分配配置避免碎片化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'


class ConfidenceAwareSoftLabelDataset(Dataset):
    """
    支持动态温度的软标签数据集（优化版）
    """

    def __init__(self, data_path: str, tokenizer, max_seq_len: int = 128):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # 加载数据 (JSON array format)
        with open(data_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)

        # 预计算prompt和置信度（避免重复构建）
        self.instruction = "Sentiment (0=neg,1=neu,2=pos):"
        for sample in self.samples:
            soft_label = sample.get('soft_label', sample.get('soft_labels', [0.33, 0.33, 0.34]))
            sample['confidence'] = max(soft_label)
            sample['soft_labels'] = soft_label  # 统一字段名

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict:
        sample = self.samples[idx]
        text = sample['text']
        prompt = f"{self.instruction}\n{text}\nAnswer:"

        # 编码（不填充，由collate_fn处理）
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_seq_len,
            padding=False,  # 动态padding
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(sample['label'], dtype=torch.long),
            'soft_labels': torch.tensor(sample['soft_labels'], dtype=torch.float32),
            'confidence': torch.tensor(sample.get('confidence', max(sample['soft_labels'])), dtype=torch.float32)
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
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
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
    自适应温度蒸馏损失
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self,
                logits: torch.Tensor,
                hard_labels: torch.Tensor,
                soft_labels: torch.Tensor,
                confidences: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        计算混合损失

        Args:
            logits: 学生模型输出 (B, 3)
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


class AdaptiveTrainer:
    """
    动态温度训练器（优化版）
    """

    def __init__(self, model, tokenizer, config: Dict):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.criterion = AdaptiveDistillationLoss(alpha=config.get('alpha', 0.5))
        self.grad_accum_steps = config.get('grad_accum_steps', 4)

        # 只训练LoRA参数
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.get('lr', 2e-4),
            weight_decay=0.01
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.get('epochs', 3)
        )

    def train_epoch(self, dataloader: DataLoader, epoch: int, max_steps: int = None) -> Dict:
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
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
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
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', required=True, help='训练数据路径')
    parser.add_argument('--val_data', required=True, help='验证数据路径')
    parser.add_argument('--base_model', default='unsloth/Qwen3-4B-unsloth-bnb-4bit')
    parser.add_argument('--output_dir', default='./results/adaptive_temp_model')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--alpha', type=float, default=0.5, help='KL权重')
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--full_train', action='store_true', help='Full training without step limit')
    args = parser.parse_args()

    print("="*60)
    print("动态温度蒸馏训练")
    print("="*60)
    print(f"训练数据: {args.train_data}")
    print(f"验证数据: {args.val_data}")
    print(f"Alpha: {args.alpha}")
    print(f"Epochs: {args.epochs}")
    print("="*60)

    # 加载模型和tokenizer
    print("\n加载模型...")
    # 配置4-bit量化
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # 配置LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # 启用梯度检查点节省内存
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    # 数据集
    print("\n加载数据集...")
    train_dataset = ConfidenceAwareSoftLabelDataset(args.train_data, tokenizer, max_seq_len=64)
    val_dataset = ConfidenceAwareSoftLabelDataset(args.val_data, tokenizer, max_seq_len=64)

    # 使用动态padding的collate_fn
    from functools import partial
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,  # 减少batch size避免OOM
        shuffle=True,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
        num_workers=0,
        pin_memory=False  # 关闭pin_memory节省内存
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )

    # 训练器
    config = {
        'lr': args.lr,
        'alpha': args.alpha,
        'epochs': args.epochs,
        'grad_accum_steps': 4  # 梯度累积步数
    }
    trainer = AdaptiveTrainer(model, tokenizer, config)

    # 训练循环
    print("\n开始训练...")
    best_val_loss = float('inf')

    # 快速验证模式或完整训练
    if args.full_train:
        max_steps = None
        print(f"完整训练模式：每epoch全部数据 ({len(train_loader)} 步)")
    else:
        max_steps = 100
        print(f"快速验证模式：每epoch最多{max_steps}步")

    for epoch in range(args.epochs):
        train_metrics = trainer.train_epoch(train_loader, epoch, max_steps=max_steps)
        print(f"\nEpoch {epoch} 训练指标:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  CE: {train_metrics['ce']:.4f}")
        print(f"  KL: {train_metrics['kl']:.4f}")
        print(f"  Avg Temp: {train_metrics['avg_temp']:.2f}")

    # 保存模型
    print(f"\n保存模型到 {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n训练完成!")


if __name__ == '__main__':
    main()
