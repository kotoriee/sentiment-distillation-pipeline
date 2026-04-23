#!/usr/bin/env python3
"""
答案优先格式 + 动态温度蒸馏训练

使用 unsloth 框架（与成功的固定温度训练一致）
复用原始动态温度脚本的温度策略和损失计算逻辑

对比目标：
- 已有固定温度结果: 80.38% (models/qwen3-4b-answer-first)
- 本脚本: 动态温度训练

Usage:
    python train_answer_first_dynamic_temp_unsloth.py --epochs 3
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ============== 配置 ==============

DEFAULT_MODEL = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 512
LORA_RANK = 16


# ============== 动态温度策略（复用原始代码）==============

def adaptive_temperature(confidence: float) -> float:
    """根据置信度选择温度"""
    if confidence > 0.9:
        return 1.5
    elif confidence > 0.6:
        return 2.0
    else:
        return min(2.5 + (0.6 - confidence) * 2, 3.0)


def train():
    args = parse_args()

    print("=" * 60)
    print("答案优先格式 + 动态温度蒸馏训练 (Unsloth)")
    print("=" * 60)

    # 加载数据
    with open(args.train_data, encoding="utf-8") as f:
        data = json.load(f)

    print(f"数据: {len(data)} 条")

    # 加载模型
    print(f"\n加载模型: {args.model}")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    # LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_RANK,
        lora_dropout=0,
        bias="none",
        random_state=3407,
    )

    # 预处理数据
    print("\n预处理数据...")

    sentiment_ids = [
        tokenizer.encode('0', add_special_tokens=False)[0],
        tokenizer.encode('1', add_special_tokens=False)[0],
        tokenizer.encode('2', add_special_tokens=False)[0],
    ]
    print(f"Sentiment token IDs: {sentiment_ids}")

    records = []
    for item in data:
        conv = item['conversations']

        # ChatML 格式
        full_text = tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=False
        )
        prefix_text = tokenizer.apply_chat_template(
            conv[:-1], tokenize=False, add_generation_prompt=True
        )

        full_ids = tokenizer.encode(full_text, add_special_tokens=False)[:MAX_SEQ_LENGTH]
        prefix_len = min(len(tokenizer.encode(prefix_text, add_special_tokens=False)), len(full_ids))

        labels = [-100] * prefix_len + full_ids[prefix_len:]

        # 定位 sentiment token
        sentiment_pos = -1
        target_str = '"sentiment": '
        idx = full_text.find(target_str)
        if idx != -1:
            pos = len(tokenizer.encode(full_text[:idx + len(target_str)], add_special_tokens=False))
            if pos < len(full_ids) and full_ids[pos] in sentiment_ids:
                sentiment_pos = pos

        # 置信度
        soft_labels = item.get('soft_labels', [0.33, 0.33, 0.34])
        confidence = max(soft_labels)

        records.append({
            'input_ids': full_ids,
            'attention_mask': [1] * len(full_ids),
            'labels': labels,
            'soft_labels': soft_labels,
            'sentiment_pos': sentiment_pos,
            'confidence': confidence,
        })

    print(f"Sentiment 定位成功: {sum(1 for r in records if r['sentiment_pos'] != -1)}/{len(records)}")

    # 创建 Dataset
    from datasets import Dataset as HFDataset
    dataset = HFDataset.from_list(records)

    # DataCollator
    class DataCollator:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def __call__(self, features):
            soft_labels = [f.pop('soft_labels') for f in features]
            sentiment_pos = [f.pop('sentiment_pos') for f in features]
            confidence = [f.pop('confidence') for f in features]

            max_len = max(len(f['input_ids']) for f in features)
            pad_id = self.tokenizer.pad_token_id or 0

            input_ids = torch.tensor([
                f['input_ids'] + [pad_id] * (max_len - len(f['input_ids']))
                for f in features
            ], dtype=torch.long)

            attention_mask = torch.tensor([
                f['attention_mask'] + [0] * (max_len - len(f['attention_mask']))
                for f in features
            ], dtype=torch.long)

            labels = torch.tensor([
                f['labels'] + [-100] * (max_len - len(f['labels']))
                for f in features
            ], dtype=torch.long)

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'soft_labels': torch.tensor(soft_labels, dtype=torch.float32),
                'sentiment_pos': torch.tensor(sentiment_pos, dtype=torch.long),
                'confidence': torch.tensor(confidence, dtype=torch.float32),
            }

    # 自定义 compute_loss（动态温度 KL）
    def compute_loss(model, inputs, return_outputs=False, **kwargs):
        soft_labels = inputs.pop("soft_labels", None)
        sentiment_pos = inputs.pop("sentiment_pos", None)
        labels = inputs.pop("labels", None)
        confidences = inputs.pop("confidence", None)

        outputs = model(**inputs)
        logits = outputs.logits

        # SFT Loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        if (shift_labels != -100).sum() == 0:
            sft_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        else:
            sft_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        # KL Loss（动态温度）
        kl_losses = []
        temps_used = []

        if soft_labels is not None and sentiment_pos is not None:
            soft_labels = soft_labels.to(logits.device)

            for b in range(logits.size(0)):
                pos = sentiment_pos[b].item()
                if pos == -1:
                    continue

                target_pos = pos - 1
                if target_pos < 0 or target_pos >= logits.size(1):
                    continue

                # 动态温度
                confidence = confidences[b].item() if confidences is not None else 0.5
                temp = adaptive_temperature(confidence)
                temps_used.append(temp)

                # KL 计算
                sent_logits = logits[b, target_pos, sentiment_ids].float()
                sent_logits = sent_logits / temp
                student_log_probs = F.log_softmax(sent_logits, dim=-1)
                teacher_probs = soft_labels[b]

                kl = F.kl_div(student_log_probs, teacher_probs, reduction='sum')
                kl = kl * (temp ** 2)
                kl_losses.append(kl)

        if kl_losses:
            kl_loss = torch.stack(kl_losses).mean()
            total_loss = args.alpha * kl_loss + (1 - args.alpha) * sft_loss
        else:
            kl_loss = torch.tensor(0.0, device=logits.device)
            total_loss = sft_loss

        # Debug
        if not hasattr(compute_loss, '_step'):
            compute_loss._step = 0
        if compute_loss._step < 5:
            avg_temp = np.mean(temps_used) if temps_used else 2.0
            print(f"[{compute_loss._step}] sft={sft_loss.item():.4f} kl={kl_loss.item():.4f} "
                  f"total={total_loss.item():.4f} avg_temp={avg_temp:.2f}")
            compute_loss._step += 1

        return (total_loss, outputs) if return_outputs else total_loss

    # 训练参数
    n_examples = len(data)
    total_steps = (n_examples // (args.batch_size * args.grad_accum)) * args.epochs
    warmup_steps = max(1, int(total_steps * 0.05))

    from transformers import TrainingArguments, Trainer

    training_args = TrainingArguments(
        output_dir=args.output_dir + "_checkpoints",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=warmup_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        learning_rate=args.lr,
        logging_steps=20,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
        bf16=True,
        fp16=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollator(tokenizer),
    )

    # 替换 compute_loss
    trainer.compute_loss = compute_loss

    print(f"\n开始训练:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch: {args.batch_size} x {args.grad_accum}")
    print(f"  Alpha: {args.alpha}")
    print(f"  温度策略: adaptive (1.5-3.0)")

    start_time = datetime.now()
    trainer.train()
    train_time = (datetime.now() - start_time).total_seconds()

    # 保存模型
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 配置记录
    config = {
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
        json.dump(config, f, indent=2)

    print(f"\n训练完成!")
    print(f"训练时间: {train_time/60:.1f} 分钟")
    print(f"模型保存: {args.output_dir}")

    print("\n" + "=" * 60)
    print("下一步评估:")
    print(f"  cd 4_evaluation")
    print(f"  python3 eval_answer_first.py --model ../3_lora_training/models/qwen3-4b-adaptive-temp")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train-data", default="../data/train_answer_first.json")
    parser.add_argument("--output-dir", default="models/qwen3-4b-adaptive-temp")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--test", action="store_true", help="Quick test mode")
    return parser.parse_args()


if __name__ == "__main__":
    train()