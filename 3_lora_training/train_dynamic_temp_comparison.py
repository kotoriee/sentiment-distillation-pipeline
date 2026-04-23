#!/usr/bin/env python3
"""
答案优先格式 + 动态温度蒸馏对比实验

实验设计：
1. 固定温度 T=2.0（当前方案，作为基线）
2. 动态温度（基于置信度自适应调整）

数据格式：答案优先格式 {"sentiment": X}
输出：JSON sentiment token + 可选 CoT

Usage:
    # 运行对比实验（快速测试模式）
    python train_dynamic_temp_comparison.py --test

    # 完整训练
    python train_dynamic_temp_comparison.py --epochs 3
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np

# ============== 配置 ==============

DEFAULT_MODEL = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 512  # 答案优先格式需要更长序列
LORA_RANK = 16
RANDOM_STATE = 3407


# ============== 动态温度策略 ==============

def adaptive_temperature(confidence: float) -> float:
    """
    根据置信度自适应选择温度

    高置信度样本：低温度保留锐利分布
    低置信度样本：高温度平滑分布

    Args:
        confidence: 教师模型置信度 (max probability, 0-1)

    Returns:
        temperature: 蒸馏温度 (1.5-3.0)
    """
    if confidence > 0.9:
        return 1.5  # 高置信度，锐利分布
    elif confidence > 0.6:
        return 2.0  # 中等置信度
    else:
        # 低置信度使用更高温度，最高3.0
        return min(2.5 + (0.6 - confidence) * 2, 3.0)


def fixed_temperature(confidence: float) -> float:
    """固定温度策略（基线）"""
    return 2.0


# ============== 训练器 ==============

class AnswerFirstDynamicTempTrainer:
    """答案优先格式 + 动态温度训练器"""

    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        args,
        temperature_strategy: str = "fixed",  # "fixed" or "adaptive"
        alpha: float = 0.5,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.args = args
        self.temperature_strategy = temperature_strategy
        self.alpha = alpha

        # sentiment token IDs
        self.id_0 = tokenizer.encode('0', add_special_tokens=False)[0]
        self.id_1 = tokenizer.encode('1', add_special_tokens=False)[0]
        self.id_2 = tokenizer.encode('2', add_special_tokens=False)[0]
        self.sentiment_token_ids = [self.id_0, self.id_1, self.id_2]

        # 温度函数
        self.temp_func = adaptive_temperature if temperature_strategy == "adaptive" else fixed_temperature

        # 统计
        self.temp_stats = {"temps_used": [], "avg_temp": 0.0}

        print(f"Temperature strategy: {temperature_strategy}")
        print(f"Sentiment token IDs: 0={self.id_0}, 1={self.id_1}, 2={self.id_2}")
        print(f"Alpha: {alpha}")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """混合 Loss: SFT + KL（仅 sentiment token）- 支持动态温度"""

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
            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
            sft_loss = loss_fct(
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

                # 获取该样本的置信度
                confidence = confidences[b].item() if confidences is not None else 0.5

                # 动态温度
                temp = self.temp_func(confidence)
                temps_used.append(temp)

                # KL 计算（带温度缩放）
                sentiment_logits = logits[b, target_pos, self.sentiment_token_ids].float()
                sentiment_logits = sentiment_logits / temp
                student_log_probs = F.log_softmax(sentiment_logits, dim=-1)
                teacher_probs = soft_labels[b]

                kl = F.kl_div(student_log_probs, teacher_probs, reduction='sum')
                kl = kl * (temp ** 2)  # T² 校正
                kl_losses.append(kl)

        if kl_losses:
            kl_loss = torch.stack(kl_losses).mean()
            total_loss = self.alpha * kl_loss + (1 - self.alpha) * sft_loss

            # 记录温度统计
            if temps_used:
                self.temp_stats["temps_used"].extend(temps_used)
                self.temp_stats["avg_temp"] = np.mean(temps_used)
        else:
            kl_loss = torch.tensor(0.0, device=logits.device)
            total_loss = sft_loss

        if not hasattr(self, '_debug_step'):
            self._debug_step = 0
        if self._debug_step < 5:
            avg_temp = np.mean(temps_used) if temps_used else 2.0
            print(f"\n  [debug {self._debug_step}] sft={sft_loss.item():.4f} kl={kl_loss.item():.4f} "
                  f"total={total_loss.item():.4f} avg_temp={avg_temp:.2f}")
            self._debug_step += 1

        return (total_loss, outputs) if return_outputs else total_loss

    def _pre_tokenize_dataset(self, raw_data):
        """Pre-tokenize，答案优先格式 + ChatML"""

        records = []
        raw_list = raw_data if isinstance(raw_data, list) else list(raw_data)
        n_valid = 0

        for item in raw_list:
            conv = item['conversations']

            # ChatML 格式
            full_text = self.tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False
            )
            prefix_text = self.tokenizer.apply_chat_template(
                conv[:-1], tokenize=False, add_generation_prompt=True
            )

            full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
            prefix_len = len(self.tokenizer.encode(prefix_text, add_special_tokens=False))

            full_ids = full_ids[:MAX_SEQ_LENGTH]
            prefix_len = min(prefix_len, len(full_ids))
            labels = [-100] * prefix_len + full_ids[prefix_len:]

            # 定位 sentiment token（答案在开头）
            sentiment_pos = -1
            target_str = '"sentiment": '
            first_idx = full_text.find(target_str)
            if first_idx != -1:
                exact_prefix = full_text[:first_idx + len(target_str)]
                pos = len(self.tokenizer.encode(exact_prefix, add_special_tokens=False))
                if pos < len(full_ids) and full_ids[pos] in self.sentiment_token_ids:
                    sentiment_pos = pos
                    n_valid += 1

            # 置信度（用于动态温度）
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

        print(f"  Sentiment found: {n_valid}/{len(records)} samples")
        return records

    def train(self):
        """训练"""
        from datasets import Dataset as HFDataset
        from transformers import TrainingArguments, Trainer

        print("Pre-tokenizing...")
        records = self._pre_tokenize_dataset(self.train_dataset)
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
                labels = torch.tensor([
                    f['labels'] + [-100] * (max_len - len(f['labels']))
                    for f in features
                ], dtype=torch.long)
                attention_mask = torch.tensor([
                    f['attention_mask'] + [0] * (max_len - len(f['attention_mask']))
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

        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            warmup_steps=self.args.warmup_steps,
            num_train_epochs=self.args.num_train_epochs,
            max_steps=self.args.max_steps,
            learning_rate=self.args.learning_rate,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=RANDOM_STATE,
            report_to="none",
            bf16=True,
            fp16=False,
            dataloader_num_workers=0,
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollator(self.tokenizer),
        )

        trainer.compute_loss = lambda model, inputs, return_outputs=False, **kwargs: \
            self.compute_loss(model, inputs, return_outputs, **kwargs)

        return trainer.train()


def run_experiment(
    data_path: str,
    output_dir: str,
    model_name: str,
    temperature_strategy: str,
    epochs: int,
    batch: int,
    grad_acc: int,
    lr: float,
    alpha: float,
    max_steps: int = -1,
):
    """运行单个实验"""

    print("\n" + "=" * 60)
    print(f"实验: {temperature_strategy.upper()} 温度策略")
    print("=" * 60)

    # 加载数据
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    print(f"数据: {len(data)} 条")

    # 加载模型
    print(f"\n加载模型: {model_name}")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
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
        random_state=RANDOM_STATE,
    )

    # 训练配置
    n_examples = len(data)
    total_steps = (n_examples // (batch * grad_acc)) * epochs
    warmup_steps = max(1, int(total_steps * 0.05))

    from trl import SFTConfig
    sft_config = SFTConfig(
        output_dir=output_dir + "_checkpoints",
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_acc,
        warmup_steps=warmup_steps,
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=lr,
        bf16=True,
        seed=RANDOM_STATE,
    )

    trainer = AnswerFirstDynamicTempTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=data,
        args=sft_config,
        temperature_strategy=temperature_strategy,
        alpha=alpha,
    )

    print(f"\n开始训练:")
    print(f"  温度策略: {temperature_strategy}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch: {batch} x {grad_acc}")
    print(f"  Alpha: {alpha}")

    start_time = datetime.now()
    trainer.train()
    train_time = (datetime.now() - start_time).total_seconds()

    # 保存模型
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 配置记录
    config = {
        "model": model_name,
        "data": data_path,
        "epochs": epochs,
        "temperature_strategy": temperature_strategy,
        "alpha": alpha,
        "format": "answer_first",
        "train_time_seconds": train_time,
        "avg_temp_used": trainer.temp_stats.get("avg_temp", 2.0),
    }
    with open(Path(output_dir) / "train_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n完成! 模型保存: {output_dir}")
    print(f"训练时间: {train_time/60:.1f} 分钟")

    # 清理内存
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return config


def main():
    args = parse_args()

    print("=" * 60)
    print("答案优先格式 + 动态温度对比实验")
    print("=" * 60)

    # 实验配置
    base_output = args.output_dir

    # 1. 固定温度实验（基线）
    fixed_config = run_experiment(
        data_path=args.data,
        output_dir=f"{base_output}/fixed_temp",
        model_name=args.model,
        temperature_strategy="fixed",
        epochs=args.epochs,
        batch=args.batch,
        grad_acc=args.grad_acc,
        lr=args.lr,
        alpha=args.alpha,
        max_steps=30 if args.test else -1,
    )

    # 2. 动态温度实验
    adaptive_config = run_experiment(
        data_path=args.data,
        output_dir=f"{base_output}/adaptive_temp",
        model_name=args.model,
        temperature_strategy="adaptive",
        epochs=args.epochs,
        batch=args.batch,
        grad_acc=args.grad_acc,
        lr=args.lr,
        alpha=args.alpha,
        max_steps=30 if args.test else -1,
    )

    # 保存对比结果
    comparison = {
        "fixed_temp": fixed_config,
        "adaptive_temp": adaptive_config,
        "timestamp": datetime.now().isoformat(),
    }

    with open(f"{base_output}/comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "=" * 60)
    print("对比实验完成")
    print("=" * 60)
    print(f"\n固定温度模型: {base_output}/fixed_temp")
    print(f"动态温度模型: {base_output}/adaptive_temp")
    print(f"对比结果: {base_output}/comparison.json")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", default="../data/train_answer_first.json")
    parser.add_argument("--output-dir", default="models/dynamic_temp_comparison")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grad-acc", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()