#!/usr/bin/env python3
"""
快速评估脚本 - Unsloth 推理优化版

优化点：
1. FastLanguageModel.for_inference() 加速
2. 贪婪解码 (temperature=0, do_sample=False)
3. max_new_tokens=64 (实际输出仅需50字符)
4. 批量推理 (batch_size=4)

Usage (WSL):
    python3 eval_fast.py --model ../3_lora_training/models/qwen3-4b-soft-full
"""

import json
import argparse
import gc
import time
from pathlib import Path
import torch
from tqdm import tqdm
from collections import Counter

# 导入解析模块（需要同目录有 extract_output.py）
from extract_output import extract_sentiment_auto


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--base-model", type=str, default="unsloth/Qwen3-4B-unsloth-bnb-4bit")
    parser.add_argument("--data", type=str, default="../data/conversations/test_conversations.json")
    parser.add_argument("--samples", type=int, default=897)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def clear_memory():
    """清理显存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()

    print("=" * 60)
    print("快速评估 - Unsloth 优化版")
    print("=" * 60)

    # 加载测试数据
    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)[:args.samples]

    print(f"\n测试数据: {len(data)} 条")

    # 加载模型 - 使用 unsloth 加速
    print(f"\n加载模型...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,  # 直接加载训练好的模型
        max_seq_length=512,
        load_in_4bit=True,
    )

    # 启用推理加速
    FastLanguageModel.for_inference(model)

    print(f"  设备: {next(model.parameters()).device}")

    # 构建 prompts
    prompts = []
    true_labels = []

    for item in data:
        if "conversations" in item:
            convs = item["conversations"]
            prompt = tokenizer.apply_chat_template(
                convs[:2], tokenize=False, add_generation_prompt=True
            )
        elif "text" in item:
            system_msg = "Output JSON: {\"sentiment\": 0/1/2}"
            user_msg = f"Review: {item['text']}"
            prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
        else:
            continue

        prompts.append(prompt)
        true_labels.append(item.get("label", -1))

    print(f"\n开始批量推理 (batch={args.batch_size}, max_tokens={args.max_tokens})...")
    print("使用贪婪解码加速...")
    start_time = time.time()

    all_outputs = []

    # 批量推理
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="推理"):
        batch_prompts = prompts[i:i + args.batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # 生成 - 贪婪解码（最快）
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=False,  # 贪婪解码，最快
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # 解码
        input_len = inputs['input_ids'].shape[1]
        for out in outputs:
            text = tokenizer.decode(out[input_len:], skip_special_tokens=True)
            all_outputs.append(text)

        # 定期清理（每100条）
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
        sentiment, _, _ = extract_sentiment_auto(pred_text)
        pred_label = sentiment
        true_label = true_labels[i]

        if pred_label == -1:
            parse_errors += 1
        elif pred_label == true_label:
            correct += 1

        results.append({
            "true": true_label,
            "pred": pred_label,
            "correct": pred_label == true_label,
            "raw": pred_text[:200],
        })

    # 计算准确率
    valid = len([r for r in results if r["pred"] != -1])
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
    print(f"速度: {speed:.2f} 条/秒")

    print(f"\n各类召回率:")
    for label, name in enumerate(['负面', '中性', '正面']):
        total = sum(cm[(label, p)] for p in [0, 1, 2])
        correct_l = cm[(label, label)]
        recall = correct_l / total * 100 if total > 0 else 0
        print(f"  {name}: {recall:.1f}% ({correct_l}/{total})")

    # 保存结果
    output_file = args.output or f"{args.model}/eval_fast.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": accuracy,
            "total": len(results),
            "correct": correct,
            "parse_errors": parse_errors,
            "speed": speed,
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {output_file}")

    # 对比基线
    print(f"\n{'='*60}")
    print("对比硬标签基线")
    print(f"{'='*60}")
    print(f"  硬标签准确率: 79.69%")
    print(f"  硬标签中性召回: 72.1%")
    print(f"  软标签准确率: {accuracy:.2f}%")

    neu_total = sum(cm[(1, p)] for p in [0, 1, 2])
    neu_correct = cm[(1, 1)]
    neu_recall = neu_correct / neu_total * 100 if neu_total > 0 else 0
    print(f"  软标签中性召回: {neu_recall:.1f}%")

    if accuracy > 79.69:
        print(f"\n  ✓ 软标签提升有效 (+{accuracy - 79.69:.2f}%)")
    else:
        print(f"\n  ! 未超越硬标签 ({accuracy - 79.69:.2f}%)")

    # 清理
    del model
    del tokenizer
    clear_memory()

    print(f"\n{'='*60}")
    print("评估完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()