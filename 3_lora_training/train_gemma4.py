#!/usr/bin/env python3
"""
Gemma 4 训练脚本 - 适配 rationale distillation 流程 (E2B 版本)

Usage:
    # 软标签训练（测试模式）
    python train_soft_v2.py --data data/train_700.json --test

    # 硬标签对比
    python train_soft_v2.py --data data/train_700.json --test --hard-label

    # 全量训练
    python train_soft_v2.py --data data/train_conversations.json --epochs 3
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# ============== 配置 ==============

# Gemma 4 E2B - 适配 RTX 3070 8GB VRAM
DEFAULT_MODEL = "unsloth/gemma-4-E2B-it"
DEFAULT_OUTPUT = "models/gemma4-e2b-rationale"
MAX_SEQ_LENGTH = 512
LORA_RANK = 16
RANDOM_STATE = 3407
TEMPERATURE = 2.0  # 温度缩放，软化分布


class SoftLabelTrainer:
    """自定义训练器，支持软标签蒸馏

    通过剥离 labels 再传入模型（无 labels → Unsloth 跳过 Triton fast_cross_entropy 内核），
    获取原生完整 logits，手动计算 SFT loss + KL loss，彻底避免显存冲突。
    无需 lm_head hook，无需 UNSLOTH_RETURN_LOGITS 环境变量。
    """

    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        args,
        temperature: float = TEMPERATURE,
        use_soft_labels: bool = True,
        alpha: float = 0.5,  # soft_loss 权重，(1-alpha) 给 sft_loss
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.args = args
        self.temperature = temperature
        self.use_soft_labels = use_soft_labels
        self.alpha = alpha

        # 获取 sentiment token IDs
        self.id_0 = tokenizer.encode('0', add_special_tokens=False)[0]
        self.id_1 = tokenizer.encode('1', add_special_tokens=False)[0]
        self.id_2 = tokenizer.encode('2', add_special_tokens=False)[0]
        self.sentiment_token_ids = [self.id_0, self.id_1, self.id_2]

        print(f"Sentiment token IDs: 0={self.id_0}, 1={self.id_1}, 2={self.id_2}")
        print(f"Temperature: {temperature}, Alpha: {alpha}, Use soft labels: {use_soft_labels}")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        混合 Loss：SFT（全序列 CE）+ KL（仅 sentiment token）。

        - labels 在 forward 前剥离：Unsloth 无 labels → 跳过 fast_cross_entropy → 返回原生 BF16 logits
        - BF16 logits 直接用于 SFT CE（PyTorch log_softmax 内核内部稳定，无需整体转 FP32）
        - KL loss 仅提取 3 个 sentiment token logits 再转 FP32（节省 ~165 MB/step）
        - sentiment_pos：由 _pre_tokenize_dataset 在字符串级别精确计算，消除 BPE 上下文歧义
        """
        soft_labels = inputs.pop("soft_labels", None) if self.use_soft_labels else None
        sentiment_pos = inputs.pop("sentiment_pos", None)
        labels = inputs.pop("labels", None)

        outputs = model(**inputs)
        logits = outputs.logits  # 保持 BF16，不整体转 FP32（节省 ~165 MB/step）

        # ---------------------------------------------------------
        # 1. SFT Loss（BF16 直接输入，PyTorch log_softmax 内核内部数值稳定）
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # 2. KL Loss（精确坐标，仅 sentiment token）
        # ---------------------------------------------------------
        kl_losses = []
        if self.use_soft_labels and soft_labels is not None and sentiment_pos is not None:
            soft_labels = soft_labels.to(logits.device)

            for b in range(logits.size(0)):
                pos = sentiment_pos[b].item()
                if pos == -1:
                    continue  # 该样本被截断，未找到 sentiment token

                # Causal LM off-by-one：预测 labels[pos] 需要用 logits[pos-1]
                target_logit_pos = pos - 1
                if target_logit_pos < 0 or target_logit_pos >= logits.size(1):
                    continue

                # 仅提取 3 个 sentiment token logits 后转 FP32
                # 内存：[3]×4 bytes = 12 bytes，而非整词表 [152000]×4 = 608 KB
                sentiment_logits = logits[b, target_logit_pos, self.sentiment_token_ids].float()
                sentiment_logits = sentiment_logits / self.temperature
                student_log_probs = F.log_softmax(sentiment_logits, dim=-1)
                teacher_probs = soft_labels[b]

                kl = F.kl_div(student_log_probs, teacher_probs, reduction='sum')
                kl = kl * (self.temperature ** 2)
                kl_losses.append(kl)

        # ---------------------------------------------------------
        # 3. 合并 + 调试
        # ---------------------------------------------------------
        if kl_losses:
            kl_loss = torch.stack(kl_losses).mean()
            total_loss = self.alpha * kl_loss + (1 - self.alpha) * sft_loss
        else:
            kl_loss = torch.tensor(0.0, device=logits.device)
            total_loss = sft_loss

        if not hasattr(self, '_debug_step'):
            self._debug_step = 0
        if self._debug_step < 5:
            # 注意：E2B 的 loss 在 13-15 范围是正常的（多模态模型特性）
            print(f"\n  [debug step {self._debug_step}] sft={sft_loss.item():.4f}  "
                  f"kl={kl_loss.item():.4f}  total={total_loss.item():.4f}")
            if total_loss.item() > 10 and self._debug_step == 0:
                print("  注意: Gemma 4 E2B 的 loss 在 13-15 范围是正常的（多模态模型特性）")
            self._debug_step += 1

        return (total_loss, outputs) if return_outputs else total_loss

    def _pre_tokenize_dataset(self, raw_data):
        """手动 tokenize，保留 soft_labels 和预计算的 sentiment_pos。

        Gemma 4 thinking 格式：
        - 使用 tokenizer.apply_chat_template 启用 enable_thinking=True
        - 格式：<bos><|turn>system\n<|think|>...<turn|>\n<|turn>user\n...<turn|>\n<|turn>model\n<|channel>thought\n...<channel|>\n{JSON}

        sentiment_pos 的计算方式：
        - 用 Python rfind() 在字符串层面找到 '"sentiment": ' 最后一次出现的位置
        - 对 full_text[:该位置+len(target)] 做完整编码，得到精确的 token 数量
        """
        records = []
        raw_list = raw_data if isinstance(raw_data, list) else list(raw_data)
        n_valid = 0

        for item in raw_list:
            conv = item['conversations']

            # 使用 Gemma 4 thinking chat template
            full_text = self.tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=True,  # 启用思考模式
            )
            prefix_text = self.tokenizer.apply_chat_template(
                conv[:-1],  # 不包含 assistant 输出
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )

            # 移除 <bos> token（训练时 tokenizer 会自动添加）
            full_text = full_text.removeprefix('<bos>')
            prefix_text = prefix_text.removeprefix('<bos>')

            full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
            prefix_len = len(self.tokenizer.encode(prefix_text, add_special_tokens=False))

            full_ids = full_ids[:MAX_SEQ_LENGTH]
            prefix_len = min(prefix_len, len(full_ids))
            labels = [-100] * prefix_len + full_ids[prefix_len:]

            # 字符串级别精确定位 sentiment token 绝对索引
            sentiment_pos = -1
            target_str = '"sentiment": '
            last_idx = full_text.rfind(target_str)
            if last_idx != -1:
                exact_prefix = full_text[:last_idx + len(target_str)]
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

        print(f"  Sentiment position found: {n_valid}/{len(records)} samples")
        return records

    def train(self):
        """手动 pre-tokenize + 自定义 DataCollator，确保 soft_labels 正确传入 compute_loss"""
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset as HFDataset
        from transformers import TrainingArguments

        # Pre-tokenize（保留 soft_labels）
        print("Pre-tokenizing dataset...")
        records = self._pre_tokenize_dataset(self.train_dataset)
        dataset = HFDataset.from_list(records)
        print(f"  Pre-tokenized: {len(dataset)} examples, columns: {dataset.column_names}")

        # 验证第一条样本
        sample = dataset[0]
        assistant_tokens = [t for t in sample['labels'] if t != -100]
        print(f"  Sample: input_len={len(sample['input_ids'])}, "
              f"supervised_len={len(assistant_tokens)}, "
              f"soft_labels={sample['soft_labels']}")

        # 自定义 DataCollator：保留 soft_labels，padding 其余字段
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0

        class SoftLabelDataCollator:
            def __call__(self, features):
                soft_labels = [f.pop('soft_labels') for f in features]
                sentiment_pos = [f.pop('sentiment_pos') for f in features]
                max_len = max(len(f['input_ids']) for f in features)

                input_ids = torch.tensor([
                    f['input_ids'] + [pad_id] * (max_len - len(f['input_ids']))
                    for f in features
                ], dtype=torch.long)
                labels_t = torch.tensor([
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
                    'labels': labels_t,
                    'soft_labels': torch.tensor(soft_labels, dtype=torch.float32),
                    'sentiment_pos': torch.tensor(sentiment_pos, dtype=torch.long),
                }

        # 从 SFTConfig 提取 TrainingArguments 兼容的参数
        # （SFTConfig 继承自 TrainingArguments，可以直接用）
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            warmup_steps=self.args.warmup_steps,
            num_train_epochs=self.args.num_train_epochs,
            max_steps=self.args.max_steps,
            learning_rate=self.args.learning_rate,
            logging_steps=self.args.logging_steps,
            optim=self.args.optim,
            weight_decay=self.args.weight_decay,
            lr_scheduler_type=self.args.lr_scheduler_type,
            seed=self.args.seed,
            report_to=self.args.report_to,
            bf16=self.args.bf16,
            fp16=self.args.fp16,
            dataloader_num_workers=self.args.dataloader_num_workers,
            dataloader_pin_memory=self.args.dataloader_pin_memory,
            remove_unused_columns=False,  # 必须保留 soft_labels
        )

        from transformers import Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=SoftLabelDataCollator(),
        )

        # 替换 compute_loss
        trainer.compute_loss = lambda model, inputs, return_outputs=False, **kwargs: \
            self.compute_loss(model, inputs, return_outputs, **kwargs)

        return trainer.train()


def parse_args():
    parser = argparse.ArgumentParser(description="软标签蒸馏训练")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=str, default="data/train_700.json")
    parser.add_argument("--output", type=str, default="models/rationale-soft-v2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grad-acc", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--test", action="store_true", help="测试模式：30步")
    parser.add_argument("--hard-label", action="store_true", help="使用硬标签（标准SFT）")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--alpha", type=float, default=0.5, help="软标签 loss 权重")
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载依赖
    try:
        from unsloth import FastModel  # Gemma 4 使用 FastModel
        from unsloth.chat_templates import get_chat_template
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset as HFDataset
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("请使用官方安装脚本: curl -fsSL https://unsloth.ai/install.sh | sh")
        return

    # 加载数据
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"错误: 训练数据不存在: {data_path}")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"加载训练数据: {len(raw_data)} 条")
    print(f"使用 {'硬标签' if args.hard_label else '软标签'} 训练")

    # 验证 soft_labels 存在
    if not args.hard_label:
        sample = raw_data[0]
        if 'soft_labels' not in sample:
            print("警告: 数据中没有 soft_labels，将使用硬标签")
            args.hard_label = True
        else:
            print(f"Soft labels 示例: {sample['soft_labels']}")

    # 加载模型（使用 FastModel API）
    print(f"\n加载模型: {args.model}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        full_finetuning=False,
    )

    # 应用 Gemma 4 thinking chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma-4-thinking",
    )

    # 添加 LoRA adapters（Gemma 4 配置）
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,    # 仅文本训练
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=LORA_RANK,
        lora_alpha=LORA_RANK,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=RANDOM_STATE,
        use_rslora=False,
        loftq_config=None,
    )

    # 调试：打印模型结构以确认 lm_head 名称
    print("\n模型顶层模块:")
    for name, _ in model.named_children():
        print(f"  {name}")

    # 显示内存状态
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_mem = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2)
        max_mem = round(gpu_stats.total_memory / 1024 ** 3, 2)
        print(f"\nGPU: {gpu_stats.name} | 总显存: {max_mem} GB | 已用: {start_mem} GB")

    # 训练步数（用于 warmup 计算）
    n_examples = sum(1 for _ in open(args.data)) if args.data.endswith('.jsonl') \
        else len(json.load(open(args.data, encoding='utf-8')))
    total_steps = (n_examples // (args.batch * args.grad_acc)) * (args.epochs if not args.test else 1)
    warmup_steps = max(1, int(total_steps * 0.05))

    # 训练配置（使用 SFTConfig 作为参数容器，train() 内部会转为 TrainingArguments）
    sft_config = SFTConfig(
        dataset_text_field="text",  # train() 内部不会用到这个字段
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_acc,
        warmup_steps=warmup_steps,
        num_train_epochs=args.epochs if not args.test else 1,
        max_steps=30 if args.test else -1,
        learning_rate=args.lr,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=RANDOM_STATE,
        output_dir=args.output + "_checkpoints",
        report_to="none",
        bf16=True,
        fp16=False,
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    # 创建自定义训练器
    trainer = SoftLabelTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=raw_data,
        args=sft_config,
        temperature=args.temperature,
        use_soft_labels=not args.hard_label,
        alpha=args.alpha,
    )

    print(f"\n开始训练:")
    print(f"  Epochs: {args.epochs if not args.test else '1 (test: 30 steps)'}")
    print(f"  Batch: {args.batch} × grad_acc {args.grad_acc}")
    print(f"  LR: {args.lr}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Alpha (soft weight): {args.alpha}")
    print("=" * 60)

    # 训练
    trainer_stats = trainer.train()

    # 显示训练统计
    if torch.cuda.is_available():
        used_mem = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2)
        print(f"\n训练时间: {trainer_stats.metrics['train_runtime']:.0f}s "
              f"({trainer_stats.metrics['train_runtime']/60:.1f} min)")
        print(f"峰值显存: {used_mem} GB")

    # 保存模型
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\n模型已保存: {output_dir}")

    # 记录配置
    config = {
        "model": args.model,
        "data": str(args.data),
        "epochs": args.epochs if not args.test else 1,
        "temperature": args.temperature,
        "alpha": args.alpha,
        "use_soft_labels": not args.hard_label,
        "max_steps": 30 if args.test else -1,
    }
    with open(output_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("=" * 60)
    print("训练完成！")
    print(f"  模型: {output_dir}")
    print(f"  配置: {output_dir}/train_config.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
