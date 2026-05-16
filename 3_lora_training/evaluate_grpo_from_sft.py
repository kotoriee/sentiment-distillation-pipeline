#!/usr/bin/env python3
"""Evaluate GRPO LoRA trained on top of a merged SFT LoRA model."""
import argparse
import gc
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from unsloth.chat_templates import get_chat_template


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="/mnt/d/kotoriee/llm/sentiment-distillation-pipeline/models/Qwen3.5-9B-ms/Qwen/Qwen3___5-9B")
    parser.add_argument("--sft-lora", default="/mnt/d/kotoriee/llm/sentiment-distillation-pipeline/3_lora_training/models/qwen35-9b-answer-first")
    parser.add_argument("--grpo-lora", default=None)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-input-length", type=int, default=512)
    return parser.parse_args()


def extract_sentiment(text: str) -> int:
    match = re.search(r'"sentiment"\s*:\s*([0-2])', text)
    if match:
        return int(match.group(1))
    text_clean = text.strip()
    if text_clean and text_clean[0] in "012":
        return int(text_clean[0])
    numbers = re.findall(r"[0-2]", text)
    if numbers:
        return int(numbers[0])
    return -1


def main():
    args = parse_args()

    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    if args.subset > 0:
        data = data[:args.subset]

    print(f"Test data: {len(data)} samples")
    print(f"Base model: {args.base_model}")
    print(f"SFT LoRA: {args.sft_lora}")
    print(f"GRPO LoRA: {args.grpo_lora or 'none'}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer = get_chat_template(tokenizer, chat_template="qwen3")
    tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "truncation_side"):
        tokenizer.truncation_side = "left"
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.sft_lora)
    model = model.merge_and_unload()
    if args.grpo_lora:
        model = PeftModel.from_pretrained(model, args.grpo_lora)
    model.eval()

    prompts = []
    labels = []
    for item in data:
        prompt = tokenizer.apply_chat_template(
            item["conversations"][:2],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
        labels.append(item["label"])

    outputs_text = []
    latencies = []
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Generating"):
        batch = prompts[i:i + args.batch_size]
        inputs = tokenizer(
            text=batch,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_length,
            padding=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

        input_len = inputs["input_ids"].shape[1]
        for row in generated:
            outputs_text.append(tokenizer.decode(row[input_len:], skip_special_tokens=True))

        if (i + args.batch_size) % 100 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    predictions = []
    for i, raw in enumerate(outputs_text):
        predictions.append({
            "index": i,
            "true": labels[i],
            "pred": extract_sentiment(raw),
            "raw": raw,
        })

    valid = [p for p in predictions if p["pred"] != -1]
    y_true = [p["true"] for p in valid]
    y_pred = [p["pred"] for p in valid]
    f1_per_class = f1_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0) if valid else [0, 0, 0]
    infer_time = sum(latencies) / 1000
    speed = len(predictions) / infer_time if infer_time else 0

    metrics = {
        "model": args.grpo_lora or args.sft_lora,
        "base_model": args.base_model,
        "sft_lora": args.sft_lora,
        "total_samples": len(predictions),
        "parse_failures": len(predictions) - len(valid),
        "valid_predictions": len(valid),
        "correct": sum(1 for p in valid if p["pred"] == p["true"]),
        "accuracy": round(accuracy_score(y_true, y_pred), 4) if valid else 0,
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4) if valid else 0,
        "f1_class_0": round(float(f1_per_class[0]), 4),
        "f1_class_1": round(float(f1_per_class[1]), 4),
        "f1_class_2": round(float(f1_per_class[2]), 4),
        "classification_report": classification_report(y_true, y_pred, labels=[0, 1, 2], digits=4, zero_division=0) if valid else "N/A",
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist() if valid else np.zeros((3, 3), dtype=int).tolist(),
        "infer_time": round(infer_time, 2),
        "speed": round(speed, 2),
    }

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "predictions": predictions}, f, indent=2, ensure_ascii=False)
    print(f"Results saved: {output}")


if __name__ == "__main__":
    main()
