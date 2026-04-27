#!/usr/bin/env python3
"""
评估 LoRA 微调模型在答案优先测试集上的表现。
支持 Gemma4 E4B 和 Qwen3.5-9B。

参考: 4_evaluation/eval_answer_first.py

Usage:
    python evaluate_lora.py --model ../models/gemma4-e4b-answer-first --base unsloth/gemma-4-E4B-it --data ../data/test_answer_first.json --gemma
    python evaluate_lora.py --model ../models/qwen35-9b-answer-first --data ../data/test_answer_first.json --qwen
"""

import os
import json
import argparse
import gc
import time
import re
import numpy as np
from pathlib import Path
from collections import Counter
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to LoRA adapter")
    parser.add_argument("--base", default=None, help="Base model name (for Gemma4)")
    parser.add_argument("--data", required=True, help="Test data JSON")
    parser.add_argument("--gemma", action="store_true", help="Use Gemma4 (FastModel)")
    parser.add_argument("--qwen", action="store_true", help="Use Qwen3.5 (FastLanguageModel)")
    parser.add_argument("--batch-size", type=int, default=8, help="Generation batch size")
    parser.add_argument("--max-new-tokens", type=int, default=1, help="Max tokens to generate (1=fast, 20=full output)")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Model max sequence length")
    parser.add_argument("--max-input-length", type=int, default=512, help="Tokenizer input truncation length")
    parser.add_argument("--subset", type=int, default=0, help="Only evaluate first N samples (0=all)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    return parser.parse_args()


def extract_sentiment(text: str) -> int:
    """从生成文本中提取情感标签 (0, 1, 2)"""
    text = text.replace('<|channel>thought', '').replace('<channel|>', '')

    # JSON format: {"sentiment": X}
    match = re.search(r'"sentiment":\s*([0-2])', text)
    if match:
        return int(match.group(1))

    # First digit at start
    text_clean = text.strip()
    if text_clean and text_clean[0] in '012':
        return int(text_clean[0])

    # First 0/1/2 anywhere
    numbers = re.findall(r'[0-2]', text)
    if numbers:
        return int(numbers[0])

    return -1


def load_gemma4(model_path, base_model, max_seq_length):
    """Load Gemma4 model with LoRA adapter"""
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template

    print(f"Loading model: {model_path}")
    model, processor = FastModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    processor = get_chat_template(processor, chat_template="gemma-4-thinking")
    tokenizer = processor.tokenizer
    FastModel.for_inference(model)

    return model, tokenizer


def load_qwen(model_path, max_seq_length):
    """Load Qwen3.5 model with LoRA adapter directly"""
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    print(f"Loading model: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen3")
    return model, tokenizer


def run_eval(model, tokenizer, data, batch_size, max_new_tokens, max_input_length, is_gemma=False, model_name=''):
    """Batched generation evaluation"""
    print(f"\nEvaluating {len(data)} samples (batch_size={batch_size}, max_new_tokens={max_new_tokens})...")

    prompts = []
    true_labels = []

    for item in data:
        conv = item['conversations']
        # Use system + user as prompt
        prompt_conv = conv[:2]
        prompt = tokenizer.apply_chat_template(
            prompt_conv, tokenize=False, add_generation_prompt=True,
            enable_thinking=is_gemma
        )
        prompts.append(prompt)
        true_labels.append(item['label'])

    print(f"\nPrompt sample: {prompts[0][:200]}...")
    if hasattr(tokenizer, "truncation_side"):
        tokenizer.truncation_side = "left"

    all_outputs = []
    latencies = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="推理"):
        batch = prompts[i:i + batch_size]

        inputs = tokenizer(
            text=batch,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
            padding=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)

        input_len = inputs['input_ids'].shape[1]
        for out in outputs:
            text = tokenizer.decode(out[input_len:], skip_special_tokens=True)
            all_outputs.append(text)

        if (i + batch_size) % 100 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    infer_time = sum(latencies) / 1000
    speed = len(all_outputs) / infer_time
    print(f"\n推理完成: {infer_time:.1f}s ({speed:.2f} 条/秒)")

    # Parse results
    results = []
    for i, pred_text in enumerate(all_outputs):
        pred_label = extract_sentiment(pred_text)
        true_label = true_labels[i]
        results.append({
            "index": i,
            "true": true_label,
            "pred": pred_label,
            "raw": pred_text,
        })

    valid = [r for r in results if r["pred"] != -1]
    parse_errors = len(results) - len(valid)
    correct = sum(1 for r in valid if r["pred"] == r["true"])

    valid_preds = [r["pred"] for r in valid]
    valid_labels = [r["true"] for r in valid]

    accuracy = accuracy_score(valid_labels, valid_preds) if valid else 0
    f1_macro = f1_score(valid_labels, valid_preds, average='macro', zero_division=0) if valid else 0
    try:
        f1_per_class = f1_score(valid_labels, valid_preds, average=None, labels=[0, 1, 2], zero_division=0) if valid else [0, 0, 0]
    except Exception:
        f1_per_class = [0, 0, 0]
    try:
        report = classification_report(valid_labels, valid_preds, labels=[0, 1, 2], digits=4, zero_division=0) if valid else "N/A"
    except Exception:
        report = "N/A"
    try:
        cm = confusion_matrix(valid_labels, valid_preds, labels=[0, 1, 2]) if valid else np.zeros((3, 3), dtype=int)
    except Exception:
        cm = np.zeros((3, 3), dtype=int)

    # Latency stats
    n_batches = len(latencies)
    total_time_ms = sum(latencies)

    eval_results = {
        'model': model_name,
        'total_samples': len(results),
        'parse_failures': parse_errors,
        'valid_predictions': len(valid),
        'correct': correct,
        'accuracy': round(accuracy, 4),
        'f1_macro': round(f1_macro, 4),
        'f1_class_0': round(f1_per_class[0], 4),
        'f1_class_1': round(f1_per_class[1], 4),
        'f1_class_2': round(f1_per_class[2], 4),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'infer_time': round(infer_time, 2),
        'speed': round(speed, 2),
        'latency': {
            'batch_size': batch_size,
            'num_batches': n_batches,
            'total_time_ms': round(total_time_ms, 2),
            'total_time_min': round(total_time_ms / 60000, 2),
            'avg_batch_latency_ms': round(np.mean(latencies), 2),
            'avg_sample_latency_ms': round(np.mean(latencies) / batch_size, 2),
            'p50_batch_latency_ms': round(np.percentile(latencies, 50), 2),
            'p95_batch_latency_ms': round(np.percentile(latencies, 95), 2),
            'throughput_samples_per_sec': round(speed, 2),
        },
    }

    return eval_results, results


def main():
    args = parse_args()

    with open(args.data, encoding='utf-8') as f:
        data = json.load(f)

    if args.subset > 0:
        data = data[:args.subset]
        print(f"Using subset: {len(data)} samples")

    print("=" * 60)
    print("LoRA 模型评估")
    print("=" * 60)

    if args.gemma:
        model, tokenizer = load_gemma4(args.model, args.base, args.max_seq_length)
        results, raw_results = run_eval(model, tokenizer, data, args.batch_size, args.max_new_tokens, args.max_input_length, is_gemma=True, model_name=args.model)
    elif args.qwen:
        model, tokenizer = load_qwen(args.model, args.max_seq_length)
        results, raw_results = run_eval(model, tokenizer, data, args.batch_size, args.max_new_tokens, args.max_input_length, model_name=args.model)
    else:
        print("Error: specify --gemma or --qwen")
        return

    # Print results
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)

    print(f"\n模型: {results['model']}")
    print(f"测试样本: {results['total_samples']}")
    print(f"解析失败: {results['parse_failures']}")
    print(f"有效预测: {results['valid_predictions']}")
    print(f"正确: {results['correct']}")
    print(f"\nAccuracy:  {results['accuracy']:.4f}")
    print(f"Macro-F1:  {results['f1_macro']:.4f}")
    print(f"F1 (class 0): {results['f1_class_0']:.4f}")
    print(f"F1 (class 1): {results['f1_class_1']:.4f}")
    print(f"F1 (class 2): {results['f1_class_2']:.4f}")

    print(f"\nClassification Report:")
    print(results['classification_report'])

    print(f"\nConfusion Matrix:")
    cm = results['confusion_matrix']
    print("  Pred:  0    1    2")
    labels = ['Neg', 'Neu', 'Pos']
    for i, row in enumerate(cm):
        print(f"  True {labels[i]}: {row}")

    print(f"\nSpeed & Latency (batch_size={results['latency']['batch_size']}):")
    print(f"  Total time: {results['latency']['total_time_min']:.1f} min")
    print(f"  Avg batch: {results['latency']['avg_batch_latency_ms']:.1f} ms")
    print(f"  Avg sample: {results['latency']['avg_sample_latency_ms']:.1f} ms")
    print(f"  P50 batch: {results['latency']['p50_batch_latency_ms']:.1f} ms")
    print(f"  P95 batch: {results['latency']['p95_batch_latency_ms']:.1f} ms")
    print(f"  Throughput: {results['latency']['throughput_samples_per_sec']:.1f} samples/sec")

    # Save results
    output_dir = Path(args.model)
    model_name = Path(args.model).name
    output_file = Path(args.output) if args.output else output_dir / f"{model_name}_eval_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({"metrics": results, "predictions": raw_results}, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nResults saved: {output_file}")


if __name__ == "__main__":
    main()
