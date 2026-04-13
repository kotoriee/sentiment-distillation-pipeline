#!/usr/bin/env python3
"""
Gemma 4 批量评估 - 直接推理 LoRA 模型（无需合并）(E2B 版本)

Usage:
    python3 eval_unsloth.py --model models/gemma4-e2b-rationale --data data/test_conversations.json --samples 50
"""

import json
import argparse
import re
import time
from pathlib import Path
import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Gemma 4 批量评估")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to LoRA adapters")
    parser.add_argument("--base-model", type=str,
                        default="unsloth/gemma-4-E2B-it",
                        help="Gemma 4 E2B base model")
    parser.add_argument("--data", type=str,
                        default="data/test_conversations.json",
                        help="Test data path (conversations format)")
    parser.add_argument("--samples", type=int, default=500,
                        help="Number of samples to evaluate")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for inference")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Max tokens (CoT + JSON)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file")
    return parser.parse_args()


def extract_sentiment(text: str) -> int:
    """从 Gemma 4 输出提取 sentiment

    Gemma 4 thinking 输出格式:
    <|channel>thought\n[reasoning]<channel|>\n{JSON}<turn|>
    """
    # 格式1: <|channel>thought 之后找 JSON
    if "<|channel>thought" in text:
        after = text.split("<|channel>thought", 1)[1]
        # 找 <channel|> 后的 JSON
        if "<channel|>" in after:
            json_part = after.split("<channel|>", 1)[1]
            match = re.search(r'"sentiment":\s*([0-2])', json_part)
            if match:
                return int(match.group(1))
        # 如果没有 <channel|>，直接在后面找
        match = re.search(r'"sentiment":\s*([0-2])', after)
        if match:
            return int(match.group(1))

    # 格式2: 之后找 JSON（兼容 Qwen3）
    if "</think>" in text:
        after = text.split("</think>", 1)[1]
        match = re.search(r'"sentiment":\s*([0-2])', after)
        if match:
            return int(match.group(1))

    # 格式3: </thinking> 之后找 JSON（兼容之前格式）
    if "</thinking>" in text:
        after = text.split("</thinking>", 1)[1]
        match = re.search(r'"sentiment":\s*([0-2])', after)
        if match:
            return int(match.group(1))

    # 格式4: 直接在文本中找 JSON（fallback）
    match = re.search(r'"sentiment":\s*([0-2])', text)
    if match:
        return int(match.group(1))

    return -1


def main():
    args = parse_args()

    # 加载依赖
    from unsloth import FastModel  # Gemma 4 使用 FastModel
    from unsloth.chat_templates import get_chat_template

    print("=" * 60)
    print("Gemma 4 E2B Batch Evaluation")
    print("=" * 60)

    # 加载测试数据
    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)[:args.samples]

    print(f"\n加载测试数据: {len(data)} 条")

    # 加载模型（使用 FastModel API）
    print(f"\n加载模型: {args.base_model}")
    print(f"加载 LoRA: {args.model}")

    model, tokenizer = FastModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=512,
        load_in_4bit=True,
    )

    # 应用 chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma-4-thinking",
    )

    # 加载 LoRA adapter
    model = FastModel.get_peft_model(model)
    model.load_adapter(args.model, adapter_name="gemma4")
    model.set_adapter("gemma4")
    FastModel.for_inference(model)

    print(f"模型设备: {next(model.parameters()).device}")

    # 构建 prompts
    prompts = []
    true_labels = []

    for item in data:
        # 支持两种格式：conversations 或原始(text+label)
        if "conversations" in item:
            convs = item["conversations"]
            # 使用 Gemma 4 thinking chat template
            prompt = tokenizer.apply_chat_template(
                convs[:2],  # system + user
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,  # 启用思考模式
            )
        elif "text" in item:
            # 原始格式，手动构建 conversations
            messages = [
                {"role": "system", "content": "You are a professional e-commerce review sentiment analysis expert. Output a JSON object with sentiment (0/1/2), confidence (0-1), and rationale."},
                {"role": "user", "content": f"Review: {item['text']}"}
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        else:
            continue

        prompts.append(prompt)
        true_labels.append(item.get("label", -1))

    print(f"\n开始批量推理 (batch_size={args.batch_size})...")
    print(f"推理参数: temp=1.0, top_p=0.95, top_k=64 (Gemma 4 推荐)")
    start_time = time.time()

    all_outputs = []
    for i in tqdm(range(0, len(prompts), args.batch_size)):
        batch = prompts[i:i + args.batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                temperature=1.0,     # Gemma 4 推荐
                top_p=0.95,          # Gemma 4 推荐
                top_k=64,            # Gemma 4 推荐
                do_sample=True,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        input_len = inputs['input_ids'].shape[1]
        for out in outputs:
            text = tokenizer.decode(out[input_len:], skip_special_tokens=True)
            all_outputs.append(text)

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
            "raw": pred_text[:500],  # 保存更长以观察完整输出
        })

    valid = len([r for r in results if r["pred"] != -1])
    accuracy = correct / valid * 100 if valid > 0 else 0

    print(f"\n{'='*60}")
    print("评估结果")
    print(f"{'='*60}")
    print(f"总样本: {len(results)}")
    print(f"解析错误: {parse_errors}")
    print(f"正确: {correct}")
    print(f"准确率: {accuracy:.2f}%")
    print(f"推理速度: {speed:.2f} 条/秒")

    # 保存结果
    output_file = args.output or f"{args.model}/eval_unsloth.json"
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

    # 错误示例
    errors = [r for r in results if not r["correct"] and r["pred"] != -1][:3]
    if errors:
        print(f"\n错误示例:")
        for i, e in enumerate(errors):
            print(f"{i+1}. True={e['true']}, Pred={e['pred']}")
            print(f"   输出: {e['raw'][:100]}...")


if __name__ == "__main__":
    main()