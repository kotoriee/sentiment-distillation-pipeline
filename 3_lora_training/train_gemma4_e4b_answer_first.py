#!/usr/bin/env python3
"""
Gemma 4 E4B 训练脚本 - 答案优先格式

使用 FastModel (多模态加载) + 纯文本训练。
参考: unslothai/unsloth Gemma4 fine-tuning guide

Usage:
    python train_gemma4_e4b_answer_first.py --data ../data/train_answer_first.json --test
    python train_gemma4_e4b_answer_first.py --data ../data/train_answer_first.json --epochs 3
"""

import os
import json
import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

DEFAULT_MODEL = "unsloth/gemma-4-E4B-it"
DEFAULT_OUTPUT = "models/gemma4-e4b-answer-first"
MAX_SEQ_LENGTH = 256
LORA_RANK = 16
RANDOM_STATE = 3407
TEMPERATURE = 2.0


class AnswerFirstTrainer:
    def __init__(self, model, tokenizer, train_dataset, args,
                 temperature=TEMPERATURE, use_soft_labels=True, alpha=0.5):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.args = args
        self.temperature = temperature
        self.use_soft_labels = use_soft_labels
        self.alpha = alpha

        self.id_0 = tokenizer.encode('0', add_special_tokens=False)[0]
        self.id_1 = tokenizer.encode('1', add_special_tokens=False)[0]
        self.id_2 = tokenizer.encode('2', add_special_tokens=False)[0]
        self.sentiment_token_ids = [self.id_0, self.id_1, self.id_2]
        print(f"Sentiment token IDs: 0={self.id_0}, 1={self.id_1}, 2={self.id_2}")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        soft_labels = inputs.pop("soft_labels", None) if self.use_soft_labels else None
        sentiment_pos = inputs.pop("sentiment_pos", None)
        labels = inputs.pop("labels", None)

        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        if (shift_labels != -100).sum() == 0:
            sft_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        else:
            sft_loss = torch.nn.CrossEntropyLoss(reduction='mean')(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        kl_losses = []
        if self.use_soft_labels and soft_labels is not None and sentiment_pos is not None:
            soft_labels = soft_labels.to(logits.device)
            for b in range(logits.size(0)):
                pos = sentiment_pos[b].item()
                if pos == -1:
                    continue
                target_pos = pos - 1
                if target_pos < 0 or target_pos >= logits.size(1):
                    continue
                sentiment_logits = logits[b, target_pos, self.sentiment_token_ids].float()
                sentiment_logits = sentiment_logits / self.temperature
                student_log_probs = F.log_softmax(sentiment_logits, dim=-1)
                kl = F.kl_div(student_log_probs, soft_labels[b], reduction='sum')
                kl = kl * (self.temperature ** 2)
                kl_losses.append(kl)

        if kl_losses:
            kl_loss = torch.stack(kl_losses).mean()
            total_loss = self.alpha * kl_loss + (1 - self.alpha) * sft_loss
        else:
            kl_loss = torch.tensor(0.0, device=logits.device)
            total_loss = sft_loss

        if not hasattr(self, '_debug_step'):
            self._debug_step = 0
        if self._debug_step < 5:
            print(f"  [debug {self._debug_step}] sft={sft_loss.item():.4f} "
                  f"kl={kl_loss.item():.4f} total={total_loss.item():.4f}")
            self._debug_step += 1

        return (total_loss, outputs) if return_outputs else total_loss

    def _pre_tokenize_dataset(self, raw_data):
        records = []
        raw_list = raw_data if isinstance(raw_data, list) else list(raw_data)
        n_valid = 0

        for item in raw_list:
            conv = item['conversations']

            full_text = self.tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False, enable_thinking=True
            )
            prefix_text = self.tokenizer.apply_chat_template(
                conv[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=True
            )

            full_text = full_text.removeprefix('<bos>')
            prefix_text = prefix_text.removeprefix('<bos>')

            full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
            prefix_len = len(self.tokenizer.encode(prefix_text, add_special_tokens=False))

            full_ids = full_ids[:MAX_SEQ_LENGTH]
            prefix_len = min(prefix_len, len(full_ids))
            labels = [-100] * prefix_len + full_ids[prefix_len:]

            sentiment_pos = -1
            target_str = '"sentiment": '
            first_idx = full_text.find(target_str)
            if first_idx != -1:
                exact_prefix = full_text[:first_idx + len(target_str)]
                pos = len(self.tokenizer.encode(exact_prefix, add_special_tokens=False))
                if pos < len(full_ids) and full_ids[pos] in self.sentiment_token_ids:
                    sentiment_pos = pos
                    n_valid += 1

            records.append({
                'input_ids': full_ids,
                'attention_mask': [1] * len(full_ids),
                'labels': labels,
                'soft_labels': item.get('soft_labels', [0.33, 0.33, 0.33]),
                'sentiment_pos': sentiment_pos,
            })

        print(f"  Sentiment found: {n_valid}/{len(records)} samples")
        return records

    def train(self):
        from datasets import Dataset as HFDataset
        from transformers import TrainingArguments, Trainer

        print("Pre-tokenizing...")
        records = self._pre_tokenize_dataset(self.train_dataset)
        dataset = HFDataset.from_list(records)

        class DataCollator:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer

            def __call__(self, features):
                soft_labels = [f.pop('soft_labels') for f in features]
                sentiment_pos = [f.pop('sentiment_pos') for f in features]
                max_len = max(len(f['input_ids']) for f in features)
                pad_id = self.tokenizer.pad_token_id or 0

                return {
                    'input_ids': torch.tensor([
                        f['input_ids'] + [pad_id] * (max_len - len(f['input_ids']))
                        for f in features
                    ], dtype=torch.long),
                    'attention_mask': torch.tensor([
                        f['attention_mask'] + [0] * (max_len - len(f['attention_mask']))
                        for f in features
                    ], dtype=torch.long),
                    'labels': torch.tensor([
                        f['labels'] + [-100] * (max_len - len(f['labels']))
                        for f in features
                    ], dtype=torch.long),
                    'soft_labels': torch.tensor(soft_labels, dtype=torch.float32),
                    'sentiment_pos': torch.tensor(sentiment_pos, dtype=torch.long),
                }

        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            warmup_steps=self.args.warmup_steps,
            num_train_epochs=self.args.num_train_epochs,
            max_steps=self.args.max_steps,
            learning_rate=self.args.learning_rate,
            max_grad_norm=self.args.max_grad_norm,
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


def main():
    args = parse_args()

    print("=" * 60)
    print("Gemma 4 E4B 答案优先格式训练")
    print("=" * 60)

    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    print(f"\n数据: {len(data)} 条")
    if 'soft_labels' in data[0]:
        print(f"软标签: {data[0]['soft_labels']}")

    # 加载模型
    print(f"\n加载模型: {args.model}")
    model, processor = FastModel.from_pretrained(
        model_name=args.model,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    # 设置 chat template 并提取 tokenizer
    processor = get_chat_template(processor, chat_template="gemma-4-thinking")
    tokenizer = processor.tokenizer

    # 训练模式
    FastModel.for_training(model)

    # LoRA
    model = FastModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules="all-linear",
        lora_alpha=LORA_RANK,
        lora_dropout=0,
        bias="none",
        random_state=RANDOM_STATE,
        finetune_language_layers=True,
        finetune_vision_layers=False,
    )

    # 训练配置
    n_examples = len(data)
    steps_per_epoch = math.ceil(n_examples / (args.batch * args.grad_acc))
    total_steps = 30 if args.test else steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    from trl import SFTConfig
    sft_config = SFTConfig(
        output_dir=args.output + "_checkpoints",
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_acc,
        warmup_steps=warmup_steps,
        num_train_epochs=args.epochs if not args.test else 1,
        max_steps=30 if args.test else -1,
        learning_rate=args.lr,
        max_grad_norm=args.max_grad_norm,
        bf16=True,
        seed=RANDOM_STATE,
    )

    trainer = AnswerFirstTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=data,
        args=sft_config,
        temperature=args.temperature,
        alpha=args.alpha,
    )

    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_mem = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2)
        max_mem = round(gpu_stats.total_memory / 1024 ** 3, 2)
        print(f"\nGPU: {gpu_stats.name} | 显存: {max_mem} GB | 已用: {start_mem} GB")

    print(f"\n开始训练:")
    print(f"  Epochs: {args.epochs if not args.test else 'test'}")
    print(f"  Batch: {args.batch} x {args.grad_acc}")
    print(f"  LR: {args.lr:g} | Warmup: {warmup_steps}/{total_steps} steps ({args.warmup_ratio:.1%})")
    print(f"  Max grad norm: {args.max_grad_norm}")

    trainer_stats = trainer.train()

    if torch.cuda.is_available():
        used_mem = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2)
        print(f"\n训练时间: {trainer_stats.metrics['train_runtime']:.0f}s "
              f"({trainer_stats.metrics['train_runtime']/60:.1f} min)")
        print(f"峰值显存: {used_mem} GB")

    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    config = {
        "model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "temperature": args.temperature,
        "alpha": args.alpha,
        "learning_rate": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "warmup_steps": warmup_steps,
        "max_grad_norm": args.max_grad_norm,
        "batch": args.batch,
        "grad_acc": args.grad_acc,
        "format": "answer_first",
    }
    with open(Path(args.output) / "train_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n完成! 模型保存: {args.output}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", default="../data/train_answer_first.json")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grad-acc", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
