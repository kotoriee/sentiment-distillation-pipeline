#!/usr/bin/env python3
"""
修复解析失败样本 - 仅对失败样本使用完整 CoT 评估

策略：
1. 加载已完成的 eval_ultra_fast.json 结果
2. 筛选解析失败的样本 (pred=-1)
3. 对这些样本用完整 CoT prompt 重新推理
4. 合并结果并重新计算准确率

预期：
- 87条失败样本 → ~87 * 25秒 = ~35分钟
- 其他810条保留原结果
"""

import json
import argparse
import gc
import time
from pathlib import Path
import torch
from tqdm import tqdm
from collections import Counter
import re

# CoT 系统提示词（训练时的完整格式）
COT_SYSTEM = """You are a professional e-commerce review sentiment analysis expert.

## Task
Analyze the review and output:
1. Sentiment classification (negative/neutral/positive)
2. Reasoning chain explaining your analysis

## Analysis Process (REQUIRED)
Follow these steps:
1. Signal Detection: Identify positive/negative keywords
2. Context Analysis: Check rating vs text alignment
3. Intensity Calibration: Determine sentiment strength
4. Final Verdict: Summarize and classify

## Output Format (JSON)
{
    "sentiment": 0/1/2,
    "confidence": 0.0-1.0,
    "rationale": "Your reasoning chain"
}"""


def extract_sentiment_full(text: str) -> int:
    """解析完整输出（支持 CoT 格式）"""
    # 清理特殊标记
    text = text.replace('<|channel>thought', '').replace('<channel|>', '')

    # 格式1: JSON sentiment
    match = re.search(r'"sentiment":\s*([0-2])', text)
    if match:
        return int(match.group(1))

    # 格式2: 数字在末尾
    numbers = re.findall(r'[0-2]', text)
    if numbers:
        return int(numbers[-1])

    return -1


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", type=str, required=True, help="已有的评估结果文件")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, default="../data/test.json")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("修复解析失败样本 - 完整 CoT 评估")
    print("=" * 60)

    # 加载已有结果
    print(f"\n加载已有结果: {args.eval_file}")
    with open(args.eval_file, encoding="utf-8") as f:
        existing_results = json.load(f)

    results = existing_results["results"]

    # 筛选失败样本
    failures = [r for r in results if r["pred"] == -1]
    success = [r for r in results if r["pred"] != -1]

    print(f"  总样本: {len(results)}")
    print(f"  成功: {len(success)}")
    print(f"  失败: {len(failures)} ← 需要重新评估")

    if len(failures) == 0:
        print("\n没有解析失败样本，无需修复")
        return

    # 加载原始测试数据（获取完整评论文本）
    print(f"\n加载测试数据: {args.data}")
    with open(args.data, encoding="utf-8") as f:
        test_data = json.load(f)

    # 加载模型
    print(f"\n加载模型: {args.model}")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=512,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    print(f"  设备: {next(model.parameters()).device}")

    # 对失败样本用完整 CoT 重新推理
    print(f"\n开始完整 CoT 推理 ({len(failures)} 条失败样本)...")
    print("  max_tokens=512, 预估 ~25秒/条")
    start_time = time.time()

    fixed_results = []

    for i, fail in enumerate(tqdm(failures, desc="修复推理")):
        # 获取原始评论
        idx = results.index(fail)
        item = test_data[idx]

        # 获取评论文本
        if "text" in item:
            review = item["text"]
        else:
            # 从 conversations 提取
            user_msg = item["conversations"][1]["content"]
            if "Review:" in user_msg:
                review = user_msg.split("Review:")[-1].strip()
            else:
                review = user_msg

        # 构建完整 CoT prompt
        prompt = f"<|im_start|>system\n{COT_SYSTEM}<|im_end|>\n<|im_start|>user\n评论: {review}<|im_end|>\n<|im_start|>assistant\n"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        input_len = inputs['input_ids'].shape[1]
        text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

        # 解析
        pred = extract_sentiment_full(text)
        true_label = fail["true"]

        fixed_results.append({
            "true": true_label,
            "pred": pred,
            "correct": pred == true_label,
            "raw": text[:300],
            "fixed": True,  # 标记为修复样本
        })

        # 每10条清理显存
        if (i + 1) % 10 == 0:
            clear_memory()

    infer_time = time.time() - start_time
    speed = len(fixed_results) / infer_time

    print(f"\n修复推理完成: {infer_time:.1f}s ({speed:.2f} 条/秒)")

    # 合并结果
    print("\n合并结果...")
    all_results = success + fixed_results

    # 重新统计
    correct = len([r for r in all_results if r["correct"]])
    parse_errors = len([r for r in all_results if r["pred"] == -1])
    total = len(all_results)
    accuracy = correct / (total - parse_errors) * 100 if (total - parse_errors) > 0 else 0

    # 混淆矩阵
    cm = Counter()
    for r in all_results:
        if r["pred"] != -1:
            cm[(r["true"], r["pred"])] += 1

    print(f"\n{'='*60}")
    print("修复后结果")
    print(f"{'='*60}")
    print(f"准确率: {accuracy:.2f}% (原: {existing_results['accuracy']:.2f}%)")
    print(f"提升: {accuracy - existing_results['accuracy']:.2f}%")
    print(f"正确: {correct}/{total}")
    print(f"解析错误: {parse_errors} (原: {len(failures)})")

    print(f"\n混淆矩阵:")
    print("         预测")
    print("真实    Neg   Neu   Pos")
    for true_label in [0, 1, 2]:
        neg = cm.get((true_label, 0), 0)
        neu = cm.get((true_label, 1), 0)
        pos = cm.get((true_label, 2), 0)
        name = ['Neg', 'Neu', 'Pos'][true_label]
        print(f"{name}     {neg:4d}  {neu:4d}  {pos:4d}")

    print(f"\n各类召回率:")
    for label, name in enumerate(['负面', '中性', '正面']):
        total_l = sum(cm.get((label, p), 0) for p in [0, 1, 2])
        correct_l = cm.get((label, label), 0)
        recall = correct_l / total_l * 100 if total_l > 0 else 0
        print(f"  {name}: {recall:.1f}% ({correct_l}/{total_l})")

    # 保存结果
    output_file = args.output or args.eval_file.replace(".json", "_fixed.json")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": accuracy,
            "total": total,
            "correct": correct,
            "parse_errors": parse_errors,
            "fixed_samples": len(fixed_results),
            "original_accuracy": existing_results["accuracy"],
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {output_file}")

    # 对比基线
    print(f"\n{'='*60}")
    print("对比硬标签基线")
    print(f"{'='*60}")
    print(f"  硬标签准确率: 79.69%")
    print(f"  修复后准确率: {accuracy:.2f}%")

    if accuracy > 79.69:
        print(f"\n  ✓ 超越硬标签 (+{accuracy - 79.69:.2f}%)")
    else:
        print(f"\n  ! 未超越 ({accuracy - 79.69:.2f}%)")

    del model
    del tokenizer
    clear_memory()

    print(f"\n{'='*60}")
    print("修复完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()