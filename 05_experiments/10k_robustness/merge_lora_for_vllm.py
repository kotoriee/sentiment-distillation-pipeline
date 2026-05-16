#!/usr/bin/env python3
"""
合并 LoRA adapters 为 16bit merged model 用于 vLLM 部署

vLLM 不支持直接加载 LoRA adapters，需要先合并为完整模型。

Usage:
    python merge_lora_for_vllm.py --sft models/qwen35-9b-answer-first --output merged_sft_16bit
    python merge_lora_for_vllm.py --grpo outputs_grpo_rewardfix_v3/checkpoint-60 --output merged_grpo_16bit
"""

import argparse
from pathlib import Path

def merge_lora_to_16bit(lora_path: str, base_model: str, output_path: str):
    """使用 Unsloth 合并 LoRA 为 16bit 模型"""
    from unsloth import FastLanguageModel
    import torch

    print(f"加载基础模型: {base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=512,
        load_in_4bit=True,  # 先加载 4bit
    )

    print(f"加载 LoRA adapter: {lora_path}")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, lora_path)

    print(f"合并并保存到: {output_path}")
    # 合并为 16bit
    model = model.merge_and_unload()

    # 保存 merged 模型
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"合并完成: {output_path}")
    print(f"大小: {Path(output_path).stat().st_size / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", help="SFT LoRA 路径")
    parser.add_argument("--grpo", help="GRPO checkpoint 路径")
    parser.add_argument("--base-model", default="models/Qwen3.5-9B-ms/Qwen/Qwen3___5-9B", help="基础模型路径")
    parser.add_argument("--output", required=True, help="输出 merged 模型路径")
    args = parser.parse_args()

    lora_path = args.sft or args.grpo
    if not lora_path:
        raise ValueError("必须指定 --sft 或 --grpo")

    merge_lora_to_16bit(lora_path, args.base_model, args.output)


if __name__ == "__main__":
    main()