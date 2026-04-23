#!/usr/bin/env python3
"""
LoRA 合并脚本 - 将 adapter 合并到基础模型

合并后可以：
1. 用更少的显存推理
2. 避免 adapter 加载问题
3. 导出 GGUF 格式部署

Usage (WSL):
    python3 merge_lora.py --lora models/qwen3-4b-soft-full --output models/qwen3-4b-soft-merged
"""

import argparse
from pathlib import Path


def main():
    args = parse_args()

    print("=" * 60)
    print("LoRA Adapter 合并")
    print("=" * 60)

    from unsloth import FastLanguageModel

    print(f"\n加载基础模型: {args.base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=512,
        load_in_4bit=True,
    )

    print(f"加载 LoRA: {args.lora}")
    model.load_adapter(args.lora, adapter_name="soft_label")
    model.set_adapter("soft_label")

    print(f"\n合并到: {args.output}")
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # 合并保存 (16-bit)
    model.save_pretrained_merged(
        args.output,
        tokenizer,
        save_method="merged_16bit",
    )

    print(f"\n合并完成!")
    print(f"  模型: {args.output}")
    print(f"  格式: merged_16bit")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", type=str, required=True)
    parser.add_argument("--base-model", type=str, default="unsloth/Qwen3-4B-unsloth-bnb-4bit")
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()