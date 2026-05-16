#!/usr/bin/env python3
"""
GRPO 10k 重新评估 - 使用修复后的 enable_thinking=False
"""

import os
import sys
import json
import subprocess

PROJECT_ROOT = "/mnt/d/kotoriee/llm/sentiment-distillation-pipeline"
os.chdir(f"{PROJECT_ROOT}/3_lora_training")

# WSL CUDA 环境
os.environ["LD_LIBRARY_PATH"] = "/home/kotoriee/miniconda3/lib/python3.13/site-packages/nvidia/cu13/lib"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTHONUNBUFFERED"] = "1"

# 运行评估
cmd = [
    "python", "-u", "eval_grpo.py",
    "--checkpoint", "outputs_grpo_9b_rewardfix_v3_nothink_continue_300/checkpoint-90",
    "--test-data", "../05_experiments/10k_robustness/data/eval_three_category_10k_fixed.json",
    "--base-model", "../models/Qwen3.5-9B-ms/Qwen/Qwen3___5-9B",
    "--batch-size", "8",
    "--max-new-tokens", "32",
]

print("Running GRPO 10k evaluation with enable_thinking=False fix...")
print(f"Command: {' '.join(cmd)}")
print()

subprocess.run(cmd, check=True)