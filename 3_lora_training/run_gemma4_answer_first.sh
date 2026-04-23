#!/bin/bash
# ==============================================================================
# Gemma 4 答案优先格式训练 - WSL 完整流程
# ==============================================================================

set -e

WSL_PROJECT="/mnt/d/0321/sentiment-distillation-pipeline"
DATA_DIR="${WSL_PROJECT}/data"

echo "=============================================================="
echo "Gemma 4 答案优先格式训练流程"
echo "=============================================================="

# ==============================================================================
# Step 1: 数据格式转换
# ==============================================================================

echo ""
echo "Step 1: 数据格式转换（答案优先）"
echo "=============================================================="

cd "${WSL_PROJECT}/3_lora_training"

# 转换训练数据
python3 convert_answer_first.py \
    --input "${DATA_DIR}/conversations/train_conversations.json" \
    --output "${DATA_DIR}/train_answer_first.json"

# 转换验证数据
python3 convert_answer_first.py \
    --input "${DATA_DIR}/conversations/val_conversations.json" \
    --output "${DATA_DIR}/val_answer_first.json"

# 转换测试数据
python3 convert_answer_first.py \
    --input "${DATA_DIR}/conversations/test_conversations.json" \
    --output "${DATA_DIR}/test_answer_first.json"

# 创建小批量测试集 (700条)
python3 << 'EOF'
import json
import random

with open("/mnt/d/0321/sentiment-distillation-pipeline/data/train_answer_first.json") as f:
    data = json.load(f)

# 分层采样
neg = [d for d in data if d['label'] == 0]
neu = [d for d in data if d['label'] == 1]
pos = [d for d in data if d['label'] == 2]

small = random.sample(neg, 233) + random.sample(neu, 233) + random.sample(pos, 234)

with open("/mnt/d/0321/sentiment-distillation-pipeline/data/train_700_answer_first.json", "w") as f:
    json.dump(small, f, ensure_ascii=False, indent=2)

print(f"Created train_700_answer_first.json: {len(small)} samples")
EOF

echo "数据转换完成 ✓"

# ==============================================================================
# Step 2: 小批量测试训练 (700条, 30步)
# ==============================================================================

echo ""
echo "Step 2: 小批量测试训练"
echo "=============================================================="

python3 train_gemma4_answer_first.py \
    --data "${DATA_DIR}/train_700_answer_first.json" \
    --output models/gemma4_test \
    --test

echo "测试训练完成 ✓"

# ==============================================================================
# Step 3: 全量训练 (7172条, 3epochs)
# ==============================================================================

echo ""
echo "Step 3: 全量训练"
echo "=============================================================="

python3 train_gemma4_answer_first.py \
    --data "${DATA_DIR}/train_answer_first.json" \
    --output models/gemma4-answer-first \
    --epochs 3 \
    --batch 1 \
    --grad-acc 16 \
    --lr 2e-5 \
    --temperature 2.0 \
    --alpha 0.5

echo "全量训练完成 ✓"

# ==============================================================================
# Step 4: 评估
# ==============================================================================

echo ""
echo "Step 4: 评估（答案优先格式）"
echo "=============================================================="

cd "${WSL_PROJECT}/4_evaluation"

python3 << 'EVAL_SCRIPT'
import json
import torch
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from unsloth import FastModel

MODEL = "/mnt/d/0321/sentiment-distillation-pipeline/3_lora_training/models/gemma4-answer-first"
DATA = "/mnt/d/0321/sentiment-distillation-pipeline/data/test_answer_first.json"
OUTPUT = "/mnt/d/0321/sentiment-distillation-pipeline/6_experiments_results/gemma4_answer_first_eval.json"

print("加载模型...")
model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=256,
    load_in_4bit=True,
)

print("加载测试数据...")
with open(DATA, encoding="utf-8") as f:
    test_data = json.load(f)

print(f"测试样本: {len(test_data)}")

# 简化 prompt（答案优先格式）
SYSTEM = "输出情感分类。格式: {\"sentiment\": 0/1/2}"

results = []
import re

for item in tqdm(test_data, desc="推理"):
    review = item["text"]
    true_label = item["label"]

    prompt = f'<bos><start_of_turn>user\n{SYSTEM}\n评论: {review}<end_of_turn>\n<start_of_turn>model\n'

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,  # 答案优先只需20字符
            do_sample=False,
            use_cache=True,
        )

    text = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)

    # 解析
    match = re.search(r'[0-2]', text)
    pred = int(match.group()) if match else -1

    results.append({
        "true": true_label,
        "pred": pred,
        "correct": pred == true_label,
        "raw": text[:50],
    })

# 统计
valid = len([r for r in results if r["pred"] != -1])
correct = len([r for r in results if r["correct"]])
accuracy = correct / valid * 100 if valid > 0 else 0

cm = Counter()
for r in results:
    if r['pred'] != -1:
        cm[(r['true'], r['pred'])] += 1

print(f"\n准确率: {accuracy:.2f}%")
print(f"正确: {correct}/{len(results)}")
print(f"解析错误: {len(results) - valid}")

print("\n各类召回率:")
for label, name in enumerate(['负面', '中性', '正面']):
    total = sum(cm.get((label, p), 0) for p in [0, 1, 2])
    correct_l = cm.get((label, label), 0)
    recall = correct_l / total * 100 if total > 0 else 0
    print(f"  {name}: {recall:.1f}%")

print(f"\n对比硬标签基线: 79.69%")
if accuracy > 79.69:
    print(f"  ✓ 超越硬标签 (+{accuracy - 79.69:.2f}%)")
else:
    print(f"  ! 未超越 ({accuracy - 79.69:.2f}%)")

# 保存
Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT, "w") as f:
    json.dump({
        "accuracy": accuracy,
        "total": len(results),
        "correct": correct,
        "results": results,
    }, f, ensure_ascii=False, indent=2)

print(f"\n结果保存: {OUTPUT}")
EVAL_SCRIPT

echo "评估完成 ✓"

echo ""
echo "=============================================================="
echo "完整流程执行完毕"
echo "=============================================================="