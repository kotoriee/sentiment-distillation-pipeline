#!/usr/bin/env python3
"""
动态温度对比实验评估脚本

对比固定温度 vs 动态温度模型的性能差异

Usage (WSL):
    python3 eval_dynamic_temp_comparison.py \
        --fixed ../3_lora_training/models/dynamic_temp_comparison/fixed_temp \
        --adaptive ../3_lora_training/models/dynamic_temp_comparison/adaptive_temp \
        --output ../6_experiments_results/dynamic_temp_comparison.json
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
    parser.add_argument("--fixed", type=str, required=True, help="固定温度模型路径")
    parser.add_argument("--adaptive", type=str, required=True, help="动态温度模型路径")
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
    text = text.replace('<|channel>thought', '').replace('<channel|>', '')

    # 格式1: JSON {"sentiment": X}
    match = re.search(r'"sentiment":\s*([0-2])', text)
    if match:
        return int(match.group(1))

    # 格式2: 数字在开头
    text_clean = text.strip()
    if text_clean and text_clean[0] in '012':
        return int(text_clean[0])

    # 格式3: 取第一个 0/1/2
    numbers = re.findall(r'[0-2]', text)
    if numbers:
        return int(numbers[0])

    return -1


def evaluate_model(model_path: str, data: list, tokenizer, batch_size: int = 8) -> dict:
    """评估单个模型"""

    print(f"\n加载模型: {model_path}")
    from unsloth import FastLanguageModel

    model, tok = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=512,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    # 构建 prompts
    prompts = []
    true_labels = []

    for item in data:
        conv = item["conversations"][:2]
        prompt = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
        true_labels.append(item.get("label", -1))

    print(f"开始推理 (batch={batch_size}, max_tokens=20)...")
    start_time = time.time()

    all_outputs = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="推理"):
        batch = prompts[i:i + batch_size]

        inputs = tok(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                use_cache=True,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )

        input_len = inputs['input_ids'].shape[1]
        for out in outputs:
            text = tok.decode(out[input_len:], skip_special_tokens=True)
            all_outputs.append(text)

        if (i + batch_size) % 100 == 0:
            clear_memory()

    infer_time = time.time() - start_time
    speed = len(all_outputs) / infer_time

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

    # 各类召回率
    recalls = {}
    for label, name in enumerate(['negative', 'neutral', 'positive']):
        total = sum(cm.get((label, p), 0) for p in [0, 1, 2])
        correct_l = cm.get((label, label), 0)
        recalls[name] = correct_l / total * 100 if total > 0 else 0

    # 清理
    del model
    del tok
    clear_memory()

    return {
        "accuracy": accuracy,
        "total": len(results),
        "correct": correct,
        "parse_errors": parse_errors,
        "speed": speed,
        "infer_time": infer_time,
        "recalls": recalls,
        "confusion_matrix": {str(k): v for k, v in cm.items()},
        "results": results[:100],  # 只保存前100条样本详情
    }


def main():
    args = parse_args()

    print("=" * 60)
    print("动态温度对比实验评估")
    print("=" * 60)

    # 加载测试数据
    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)[:args.samples]

    print(f"测试数据: {len(data)} 条")

    # 加载 tokenizer（共用）
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.fixed, trust_remote_code=True)

    # 评估固定温度模型
    print("\n" + "-" * 40)
    print("评估固定温度模型 (T=2.0)")
    print("-" * 40)
    fixed_results = evaluate_model(args.fixed, data, tokenizer, args.batch_size)

    # 评估动态温度模型
    print("\n" + "-" * 40)
    print("评估动态温度模型 (自适应)")
    print("-" * 40)
    adaptive_results = evaluate_model(args.adaptive, data, tokenizer, args.batch_size)

    # 对比结果
    comparison = {
        "fixed_temp": {
            "model": args.fixed,
            **fixed_results,
        },
        "adaptive_temp": {
            "model": args.adaptive,
            **adaptive_results,
        },
        "comparison": {
            "accuracy_diff": adaptive_results["accuracy"] - fixed_results["accuracy"],
            "parse_error_diff": adaptive_results["parse_errors"] - fixed_results["parse_errors"],
            "speed_diff": adaptive_results["speed"] - fixed_results["speed"],
            "neutral_recall_diff": adaptive_results["recalls"]["neutral"] - fixed_results["recalls"]["neutral"],
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 保存结果
    output_file = args.output or "dynamic_temp_comparison.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    # 输出对比表格
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)
    print(f"\n{'指标':<20} {'固定温度':<12} {'动态温度':<12} {'差异':<10}")
    print("-" * 54)
    print(f"{'准确率':<20} {fixed_results['accuracy']:.2f}%      {adaptive_results['accuracy']:.2f}%      {comparison['comparison']['accuracy_diff']:+.2f}%")
    print(f"{'解析错误':<20} {fixed_results['parse_errors']:<12} {adaptive_results['parse_errors']:<12} {comparison['comparison']['parse_error_diff']:+d}")
    print(f"{'推理速度':<20} {fixed_results['speed']:.2f}/s     {adaptive_results['speed']:.2f}/s     {comparison['comparison']['speed_diff']:+.2f}/s")
    print(f"{'负面召回':<20} {fixed_results['recalls']['negative']:.1f}%     {adaptive_results['recalls']['negative']:.1f}%     {adaptive_results['recalls']['negative'] - fixed_results['recalls']['negative']:+.1f}%")
    print(f"{'中性召回':<20} {fixed_results['recalls']['neutral']:.1f}%     {adaptive_results['recalls']['neutral']:.1f}%     {comparison['comparison']['neutral_recall_diff']:+.1f}%")
    print(f"{'正面召回':<20} {fixed_results['recalls']['positive']:.1f}%     {adaptive_results['recalls']['positive']:.1f}%     {adaptive_results['recalls']['positive'] - fixed_results['recalls']['positive']:+.1f}%")

    # 判断胜负
    if comparison['comparison']['accuracy_diff'] > 0:
        print(f"\n✓ 动态温度模型胜出 (+{comparison['comparison']['accuracy_diff']:.2f}%)")
    else:
        print(f"\n✗ 固定温度模型胜出 (+{-comparison['comparison']['accuracy_diff']:.2f}%)")

    print(f"\n结果已保存: {output_file}")


if __name__ == "__main__":
    main()