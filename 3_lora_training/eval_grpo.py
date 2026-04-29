#!/usr/bin/env python3
"""Evaluate GRPO-trained Qwen3.5-4B model on test set."""
import os
import json
import re
import time
import argparse
import numpy as np
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm

from unsloth import FastModel

def extract_sentiment(text: str) -> int:
    match = re.search(r'"sentiment":\s*([0-2])', text)
    if match:
        return int(match.group(1))
    text_clean = text.strip()
    if text_clean and text_clean[0] in '012':
        return int(text_clean[0])
    numbers = re.findall(r'[0-2]', text)
    if numbers:
        return int(numbers[0])
    return -1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs_sentiment_grpo/checkpoint-100")
    parser.add_argument("--test-data", default="../data/test_answer_first.json")
    parser.add_argument("--base-model", default="unsloth/Qwen3.5-4B")
    parser.add_argument("--subset", type=int, default=0, help="Evaluate first N samples (0=all)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--max-seq-length", type=int, default=512)
    args = parser.parse_args()

    # Load test data
    with open(args.test_data, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    if args.subset > 0:
        test_data = test_data[:args.subset]
    print(f"Test data: {len(test_data)} samples")

    # Load model with LoRA adapter
    print(f"Loading base model: {args.base_model}")
    print(f"Loading LoRA from: {args.checkpoint}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        gpu_memory_utilization=0.9,
    )
    # Load LoRA weights from checkpoint
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.checkpoint)
    print("Model loaded.")

    FastModel.for_inference(model)

    # Prepare prompts
    prompts = []
    true_labels = []
    for item in test_data:
        conv = item['conversations']
        prompt_conv = conv[:2]  # system + user
        prompt = tokenizer.apply_chat_template(
            prompt_conv, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)
        true_labels.append(item['label'])

    # Batch generation
    print(f"\nEvaluating (batch_size={args.batch_size}, max_new_tokens={args.max_new_tokens})...")
    all_outputs = []
    latencies = []

    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Generating"):
        batch = prompts[i:i + args.batch_size]
        inputs = tokenizer(
            text=batch,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_seq_length - args.max_new_tokens,
            padding=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
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

    infer_time = sum(latencies) / 1000
    speed = len(all_outputs) / infer_time
    print(f"\nInference complete: {infer_time:.1f}s ({speed:.2f} samples/sec)")

    # Parse results
    results = []
    for i, pred_text in enumerate(all_outputs):
        pred_label = extract_sentiment(pred_text)
        results.append({
            "index": i,
            "true": true_labels[i],
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
        f1_per_class = f1_score(valid_labels, valid_preds, average=None, labels=[0, 1, 2], zero_division=0)
    except Exception:
        f1_per_class = [0, 0, 0]
    report = classification_report(valid_labels, valid_preds, labels=[0, 1, 2], digits=4, zero_division=0)
    cm = confusion_matrix(valid_labels, valid_preds, labels=[0, 1, 2])

    print("\n" + "=" * 60)
    print("GRPO 评估结果")
    print("=" * 60)
    print(f"\n测试样本: {len(results)}")
    print(f"解析失败: {parse_errors}")
    print(f"有效预测: {len(valid)}")
    print(f"正确: {correct}")
    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Macro-F1:  {f1_macro:.4f}")
    print(f"F1 (class 0): {f1_per_class[0]:.4f}")
    print(f"F1 (class 1): {f1_per_class[1]:.4f}")
    print(f"F1 (class 2): {f1_per_class[2]:.4f}")
    print(f"\nClassification Report:")
    print(report)
    print(f"\nConfusion Matrix:")
    print("  Pred:  0    1    2")
    labels_map = ['Neg', 'Neu', 'Pos']
    for i, row in enumerate(cm):
        print(f"  True {labels_map[i]}: {row}")
    print(f"\nThroughput: {speed:.1f} samples/sec")

    # Save results
    output_dir = Path(args.checkpoint)
    output_file = output_dir / "grpo_eval_results.json"
    eval_data = {
        "metrics": {
            "total_samples": len(results),
            "parse_failures": parse_errors,
            "valid_predictions": len(valid),
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_class_0": round(f1_per_class[0], 4),
            "f1_class_1": round(f1_per_class[1], 4),
            "f1_class_2": round(f1_per_class[2], 4),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "speed": round(speed, 2),
        },
        "predictions": results,
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {output_file}")

if __name__ == "__main__":
    main()
