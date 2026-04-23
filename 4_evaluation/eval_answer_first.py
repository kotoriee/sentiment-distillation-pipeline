#!/usr/bin/env python3
"""
答案优先格式评估脚本

关键改进：
1. 使用与训练相同的 prompt 格式（ChatML）
2. max_new_tokens=20（答案在开头，只需少量 token）
3. 快速贪婪解码

预期速度: ~0.1秒/条，897条 ~2分钟

Usage (WSL):
    python3 eval_answer_first.py --model ../3_lora_training/models/qwen3-4b-answer-first
"""

import json
import argparse
import gc
import time
import re
from pathlib import Path
import torch
from tqdm import tqdm
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, default="../data/test_answer_first.json")
    parser.add_argument("--samples", type=int, default=897)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_sentiment(text: str) -> int:
    """解析答案优先格式输出"""
    # 清理特殊标记
    text = text.replace('<|channel>thought', '').replace('<channel|>', '')

    # 格式1: JSON {"sentiment": X}
    match = re.search(r'"sentiment":\s*([0-2])', text)
    if match:
        return int(match.group(1))

    # 格式2: 数字在开头（答案优先格式）
    text_clean = text.strip()
    if text_clean and text_clean[0] in '012':
        return int(text_clean[0])

    # 格式3: 取第一个 0/1/2
    numbers = re.findall(r'[0-2]', text)
    if numbers:
        return int(numbers[0])  # 第一个，因为答案在前

    return -1


def main():
    args = parse_args()

    print("=" * 60)
    print("答案优先格式评估")
    print("=" * 60)

    # 加载测试数据
    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)[:args.samples]

    print(f"\n测试数据: {len(data)} 条")

    # 检查数据格式
    if data[0].get("conversations"):
        asst_content = data[0]["conversations"][2]["content"]
        print(f"样本 assistant 内容: {asst_content[:80]}...")

    # 加载模型
    print(f"\n加载模型: {args.model}")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=512,  # 增大以避免截断长评论
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    print(f"  设备: {next(model.parameters()).device}")

    # 构建 prompts（使用 conversations 格式）
    prompts = []
    true_labels = []

    for item in data:
        # 使用前两个对话（system + user）作为输入
        conv = item["conversations"][:2]

        # ChatML 格式
        prompt = tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True
        )

        prompts.append(prompt)
        true_labels.append(item.get("label", -1))

    print(f"\nPrompt 示例:")
    print(f"  {prompts[0][:200]}...")

    print(f"\n开始推理 (batch={args.batch_size}, max_tokens=20, 贪婪解码)...")
    start_time = time.time()

    all_outputs = []

    for i in tqdm(range(0, len(prompts), args.batch_size), desc="推理"):
        batch = prompts[i:i + args.batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=512,  # 与模型 max_seq_length 一致
            padding=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        input_len = inputs['input_ids'].shape[1]
        for out in outputs:
            text = tokenizer.decode(out[input_len:], skip_special_tokens=True)
            all_outputs.append(text)

        if (i + args.batch_size) % 100 == 0:
            clear_memory()

    infer_time = time.time() - start_time
    speed = len(all_outputs) / infer_time

    print(f"\n推理完成: {infer_time:.1f}s ({speed:.2f} 条/秒)")

    # 解析结果
    correct = 0
    parse_errors = 0
    results = []

    for i, pred_text in enumerate(all_outputs):
        pred_label = extract_sentiment(pred_text)
        true_label = true_labels[i]

        if pred_label == -1:
            parse_errors += 1
        elif pred_label == true_label:
            correct += 1

        results.append({
            "true": true_label,
            "pred": pred_label,
            "correct": pred_label == true_label,
            "raw": pred_text[:50],
        })

    valid = len([r for r in results if r["pred"] != -1])
    accuracy = correct / valid * 100 if valid > 0 else 0

    # 混淆矩阵
    cm = Counter()
    for r in results:
        if r['pred'] != -1:
            cm[(r['true'], r['pred'])] += 1

    # 保存结果
    output_file = args.output or f"{args.model}/eval_answer_first.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": accuracy,
            "total": len(results),
            "correct": correct,
            "parse_errors": parse_errors,
            "speed": speed,
            "infer_time": infer_time,
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {output_file}")

    # 分析解析失败
    if parse_errors > 0:
        failures = [r for r in results if r['pred'] == -1][:5]
        print(f"\n解析失败样本 (前5条):")
        for i, r in enumerate(failures):
            print(f"  {i+1}. true={r['true']}, raw=\"{r['raw']}\"")

    print(f"\n{'='*60}")
    print("评估结果")
    print(f"{'='*60}")
    print(f"准确率: {accuracy:.2f}%")
    print(f"总样本: {len(results)}")
    print(f"正确: {correct}")
    print(f"解析错误: {parse_errors}")
    print(f"速度: {speed:.2f} 条/秒 ({infer_time:.1f}s)")

    print(f"\n混淆矩阵:")
    print("         预测")
    print("真实    Neg   Neu   Pos")
    for true_label in [0, 1, 2]:
        neg_pred = cm.get((true_label, 0), 0)
        neu_pred = cm.get((true_label, 1), 0)
        pos_pred = cm.get((true_label, 2), 0)
        name = ['Neg', 'Neu', 'Pos'][true_label]
        print(f"{name}     {neg_pred:4d}  {neu_pred:4d}  {pos_pred:4d}")

    print(f"\n各类召回率:")
    for label, name in enumerate(['负面', '中性', '正面']):
        total = sum(cm.get((label, p), 0) for p in [0, 1, 2])
        correct_l = cm.get((label, label), 0)
        recall = correct_l / total * 100 if total > 0 else 0
        print(f"  {name}: {recall:.1f}% ({correct_l}/{total})")

    # 对比基线
    print(f"\n{'='*60}")
    print("对比硬标签基线")
    print(f"{'='*60}")
    print(f"  硬标签准确率: 79.69%")
    print(f"  答案优先准确率: {accuracy:.2f}%")

    if accuracy > 79.69:
        print(f"\n  ✓ 超越硬标签 (+{accuracy - 79.69:.2f}%)")
    else:
        print(f"\n  ! 未超越 ({accuracy - 79.69:.2f}%)")

    del model
    del tokenizer
    clear_memory()

    print(f"\n{'='*60}")
    print("评估完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()