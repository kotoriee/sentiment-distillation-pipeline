#!/bin/bash
# ==============================================================================
# 内存安全评估流程 - 合并LoRA + 逐条推理
# ==============================================================================

set -e

WSL_PROJECT="/mnt/d/0321/sentiment-distillation-pipeline"
MODEL="unsloth/Qwen3-4B-unsloth-bnb-4bit"
LORA_DIR="${WSL_PROJECT}/3_lora_training/models/qwen3-4b-soft-full"
MERGED_DIR="${WSL_PROJECT}/3_lora_training/models/qwen3-4b-soft-merged"
TEST_DATA="${WSL_PROJECT}/data/conversations/test_conversations.json"

# ==============================================================================
# Step 1: 合并 LoRA Adapter (可选，更稳定)
# ==============================================================================

echo ""
echo "=============================================================="
echo "Step 1: 合并 LoRA Adapter"
echo "=============================================================="

cd "${WSL_PROJECT}/4_evaluation"

python3 merge_lora.py \
    --lora "${LORA_DIR}" \
    --base-model "${MODEL}" \
    --output "${MERGED_DIR}"

echo "合并完成 ✓"

# ==============================================================================
# Step 2: 用合并模型评估 (显存更低)
# ==============================================================================

echo ""
echo "=============================================================="
echo "Step 2: 评估合并模型"
echo "=============================================================="

python3 << 'EVAL_SCRIPT'
import json
import gc
import time
from pathlib import Path
import torch
from tqdm import tqdm
from collections import Counter

from extract_output import extract_sentiment_auto

# 配置
MERGED_DIR = "/mnt/d/0321/sentiment-distillation-pipeline/3_lora_training/models/qwen3-4b-soft-merged"
TEST_DATA = "/mnt/d/0321/sentiment-distillation-pipeline/data/conversations/test_conversations.json"
OUTPUT_FILE = "/mnt/d/0321/sentiment-distillation-pipeline/6_experiments_results/soft_full_eval.json"

print("加载合并模型...")
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    MERGED_DIR,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR)

print(f"设备: {next(model.parameters()).device}")

# 加载测试数据
with open(TEST_DATA, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"测试数据: {len(data)} 条")

# 逐条推理
results = []
clear_interval = 20

print("\n开始推理...")
start_time = time.time()

for i, item in enumerate(tqdm(data, desc="推理")):
    # 构建 prompt
    if "conversations" in item:
        convs = item["conversations"]
        prompt = tokenizer.apply_chat_template(
            convs[:2], tokenize=False, add_generation_prompt=True
        )
    else:
        continue

    true_label = item.get("label", -1)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode
    input_len = inputs['input_ids'].shape[1]
    text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    # 解析
    pred_label, _, _ = extract_sentiment_auto(text)

    results.append({
        "true": true_label,
        "pred": pred_label,
        "correct": pred_label == true_label,
        "raw": text[:300],
    })

    # 定期清理显存
    if (i + 1) % clear_interval == 0:
        gc.collect()
        torch.cuda.empty_cache()

infer_time = time.time() - start_time

# 计算结果
valid = len([r for r in results if r["pred"] != -1])
correct = len([r for r in results if r["correct"]])
parse_errors = len([r for r in results if r["pred"] == -1])
accuracy = correct / valid * 100 if valid > 0 else 0

# 混淆矩阵
cm = Counter()
for r in results:
    if r['pred'] != -1:
        cm[(r['true'], r['pred'])] += 1

print(f"\n{'='*60}")
print("评估结果")
print(f"{'='*60}")
print(f"准确率: {accuracy:.2f}%")
print(f"总样本: {len(results)}")
print(f"正确: {correct}")
print(f"解析错误: {parse_errors}")
print(f"速度: {len(results)/infer_time:.2f} 条/秒")

print(f"\n各类召回率:")
for label, name in enumerate(['负面', '中性', '正面']):
    total = sum(cm[(label, p)] for p in [0, 1, 2])
    correct_l = cm[(label, label)]
    recall = correct_l / total * 100 if total > 0 else 0
    print(f"  {name}: {recall:.1f}% ({correct_l}/{total})")

print(f"\n对比硬标签基线:")
print(f"  硬标签准确率: 79.69%")
print(f"  硬标签中性召回: 72.1%")
print(f"  软标签准确率: {accuracy:.2f}%")

if accuracy > 79.69:
    print(f"\n  ✓ 软标签提升有效 (+{accuracy - 79.69:.2f}%)")
else:
    print(f"\n  ! 软标签未超越硬标签 ({accuracy - 79.69:.2f}%)")

# 保存结果
Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump({
        "accuracy": accuracy,
        "total": len(results),
        "correct": correct,
        "parse_errors": parse_errors,
        "speed": len(results)/infer_time,
        "results": results,
    }, f, ensure_ascii=False, indent=2)

print(f"\n结果已保存: {OUTPUT_FILE}")
EVAL_SCRIPT

echo ""
echo "=============================================================="
echo "评估完成"
echo "=============================================================="