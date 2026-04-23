#!/usr/bin/env python3
"""
内存安全评估脚本 - 逐条推理 + 显存清理

特点：
1. batch_size=1 逐条推理
2. 定期清理显存
3. 进度保存 + 断点续传
4. max_new_tokens=256 (实际输出约150字符)

Usage (WSL):
    python3 eval_safe.py --model models/qwen3-4b-soft-full
"""

import json
import argparse
import gc
import time
from pathlib import Path
import torch
from tqdm import tqdm

from extract_output import extract_sentiment_auto


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--base-model", type=str, default="unsloth/Qwen3-4B-unsloth-bnb-4bit")
    parser.add_argument("--data", type=str, default="../data/conversations/test_conversations.json")
    parser.add_argument("--samples", type=int, default=897)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="续传: 指定已有的结果文件路径")
    return parser.parse_args()


def clear_memory():
    """清理显存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main():
    args = parse_args()

    print("=" * 60)
    print("内存安全评估 (逐条推理)")
    print("=" * 60)

    # 加载测试数据
    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)[:args.samples]

    print(f"\n测试数据: {len(data)} 条")

    # 断点续传检查
    resume_results = []
    start_idx = 0
    if args.resume:
        with open(args.resume, "r", encoding="utf-8") as f:
            resume_data = json.load(f)
            resume_results = resume_data.get("results", [])
            start_idx = len(resume_results)
            print(f"续传: 从第 {start_idx} 条开始")

    # 加载模型
    print(f"\n加载模型...")
    print(f"  Base: {args.base_model}")
    print(f"  LoRA: {args.model}")

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=512,
        load_in_4bit=True,
    )

    # 加载 LoRA
    model = FastLanguageModel.get_peft_model(model)
    model.load_adapter(args.model, adapter_name="soft_label")
    model.set_adapter("soft_label")
    FastLanguageModel.for_inference(model)

    print(f"  设备: {next(model.parameters()).device}")

    # 显存状态
    if torch.cuda.is_available():
        mem_before = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  显存已用: {mem_before:.2f} GB")

    # 构建 prompts
    prompts = []
    true_labels = []

    for item in data[start_idx:]:
        if "conversations" in item:
            convs = item["conversations"]
            prompt = tokenizer.apply_chat_template(
                convs[:2], tokenize=False, add_generation_prompt=True
            )
        elif "text" in item:
            system_msg = "You are a professional e-commerce review sentiment analysis expert. Output a JSON object with sentiment (0/1/2), confidence (0-1), and rationale."
            user_msg = f"Review: {item['text']}"
            prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
        else:
            continue

        prompts.append(prompt)
        true_labels.append(item.get("label", -1))

    print(f"\n开始推理 (逐条, max_tokens={args.max_tokens})...")
    start_time = time.time()

    all_outputs = []
    clear_interval = 50  # 每50条清理一次显存

    for i, prompt in enumerate(tqdm(prompts, desc="推理")):
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                temperature=0.1,
                do_sample=True,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # Decode
        input_len = inputs['input_ids'].shape[1]
        text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        all_outputs.append(text)

        # 定期清理显存
        if (i + 1) % clear_interval == 0:
            clear_memory()
            if torch.cuda.is_available():
                mem_now = torch.cuda.max_memory_allocated() / 1024**3
                tqdm.write(f"  [显存] {mem_now:.2f} GB @ {i+1}/{len(prompts)}")

    # 最终清理
    clear_memory()

    infer_time = time.time() - start_time
    speed = len(all_outputs) / infer_time

    print(f"\n推理完成: {infer_time:.1f}s ({speed:.2f} 条/秒)")

    # 解析结果
    correct = 0
    parse_errors = 0
    results = resume_results.copy()

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
            "raw": pred_text[:300],
        })

    # 计算准确率
    valid = len([r for r in results if r["pred"] != -1])
    accuracy = correct / valid * 100 if valid > 0 else 0

    # 计算各类召回率
    from collections import Counter
    cm = Counter()
    for r in results:
        if r['pred'] != -1:
            cm[(r['true'], r['pred'])] += 1

    print(f"\n{'='*60}")
    print("评估结果")
    print(f"{'='*60}")
    print(f"总样本: {len(results)}")
    print(f"解析错误: {parse_errors}")
    print(f"正确: {correct}")
    print(f"准确率: {accuracy:.2f}%")
    print(f"推理速度: {speed:.2f} 条/秒")

    print(f"\n各类召回率:")
    for label in [0, 1, 2]:
        total = sum(cm[(label, p)] for p in [0, 1, 2])
        correct_l = cm[(label, label)]
        recall = correct_l / total * 100 if total > 0 else 0
        name = ['负面', '中性', '正面'][label]
        print(f"  {name}: {recall:.1f}% ({correct_l}/{total})")

    # 保存结果
    output_file = args.output or f"{args.model}/eval_safe.json"
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

    # 对比硬标签基线
    print(f"\n{'='*60}")
    print("对比硬标签基线")
    print(f"{'='*60}")
    print(f"  硬标签准确率: 79.69%")
    print(f"  硬标签中性召回: 72.1%")
    print(f"  软标签准确率: {accuracy:.2f}%")

    # 计算中性召回率
    neu_total = sum(cm[(1, p)] for p in [0, 1, 2])
    neu_correct = cm[(1, 1)]
    neu_recall = neu_correct / neu_total * 100 if neu_total > 0 else 0
    print(f"  软标签中性召回: {neu_recall:.1f}%")

    if accuracy > 79.69:
        print(f"\n  ✓ 软标签提升有效 (+{accuracy - 79.69:.2f}%)")
    else:
        print(f"\n  ! 软标签未超越硬标签 ({accuracy - 79.69:.2f}%)")

    # 释放模型
    del model
    del tokenizer
    clear_memory()

    print(f"\n{'='*60}")
    print("评估完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()