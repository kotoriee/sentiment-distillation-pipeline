#!/usr/bin/env python3
"""
NVIDIA NIM API 情感分析评估
批量调用多个大模型 API，评估在测试集上的表现。

Usage:
    export NVIDIA_API_KEY=nvapi-xxx
    python eval_nvidia_api.py --model deepseek-ai/deepseek-v4-flash --subset 50
    python eval_nvidia_api.py --all --workers 10
"""

import os
import json
import time
import re
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

DATA_PATH = Path(__file__).parent.parent / "data" / "test_answer_first.json"
OUTPUT_DIR = Path(__file__).parent / "nvidia_api_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = "You are a sentiment classifier for e-commerce reviews."

USER_PROMPT = """Classify the sentiment of this e-commerce review.

FIRST, output a single line in this exact format:
Sentiment: N
Where N is 0 (negative), 1 (neutral), or 2 (positive).

Then optionally provide a brief explanation.

Review: {text}

Rules:
- Negative (0): complaints, returns, terrible quality, waste of money
- Neutral (1): mixed feelings, ok but not great, expected better
- Positive (2): love it, works perfectly, great value

Examples:
"Broke after one week, terrible!" → Sentiment: 0
"Does the job but nothing special" → Sentiment: 1
"Absolutely love this! Best purchase!" → Sentiment: 2

Output:"""


def get_client():
    key = os.environ.get("NVIDIA_API_KEY", "")
    if not key:
        raise ValueError("请设置 NVIDIA_API_KEY 环境变量: export NVIDIA_API_KEY=nvapi-xxx")
    return OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=key)


def extract_sentiment(text):
    """从输出中提取情感标签"""
    # Match "Sentiment: N" pattern (primary)
    match = re.search(r'Sentiment:\s*([0-2])', text)
    if match:
        return int(match.group(1))
    # Match JSON {"sentiment": N}
    match = re.search(r'"sentiment"\s*:\s*([0-2])', text)
    if match:
        return int(match.group(1))
    # First 0/1/2 digit anywhere
    match = re.search(r'[0-2]', text)
    if match:
        return int(match.group())
    return -1


def predict_single(client, text, model, temperature=0.3, max_retries=3):
    """单次 API 调用"""
    for attempt in range(max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT.format(text=text[:2000])}
                ],
                temperature=temperature,
                max_tokens=512,
                top_p=0.95,
            )
            content = completion.choices[0].message.content
            pred = extract_sentiment(content)
            return pred, content[:80]
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "Too Many Requests" in error_msg:
                wait = min(2 ** (attempt + 1), 60)
                time.sleep(wait)
            elif attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                return -1, f"ERR: {error_msg[:60]}"


def evaluate_model(client, model_name, data, workers=10, rate_limit=0.1, subset=0):
    """评估单个模型 - 使用并发加速"""
    if subset > 0:
        data = data[:subset]

    print(f"\n{'='*60}")
    print(f"评估模型: {model_name}")
    print(f"样本数: {len(data)} | 并发: {workers}")
    print(f"{'='*60}\n")

    # Sequential execution with rate limiting
    print(f"  Running sequentially with {rate_limit}s between requests...")
    import sys
    results = []
    errors = 0
    for i, item in enumerate(data):
        try:
            pred, raw = predict_single(client, item['text'], model_name)
            if pred == -1:
                errors += 1
            true_label = item['label']
            results.append({
                "id": item.get('id', ''),
                "pred": pred,
                "true": true_label,
                "raw": raw,
                "correct": pred == true_label if pred != -1 else False
            })
            if (i + 1) % 50 == 0 or (i + 1) == len(data):
                print(f"  [{i+1}/{len(data)}] done (errors={errors})")
                sys.stdout.flush()
            time.sleep(rate_limit)
        except Exception as e:
            errors += 1

    # 计算指标
    valid = [r for r in results if r["pred"] != -1]
    correct = sum(1 for r in valid if r["pred"] == r["true"])

    y_true = [r["true"] for r in valid]
    y_pred = [r["pred"] for r in valid]

    if valid:
        accuracy = correct / len(valid)
        # Per-class F1
        labels = [0, 1, 2]
        cm = Counter()
        for t, p in zip(y_true, y_pred):
            cm[(t, p)] += 1

        f1_per_class = {}
        for l in labels:
            tp = cm.get((l, l), 0)
            fp = sum(cm.get((p, l), 0) for p in labels) - tp
            fn = sum(cm.get((l, p), 0) for p in labels) - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            f1_per_class[l] = f1

        f1_macro = sum(f1_per_class.values()) / len(labels)
    else:
        accuracy = 0
        f1_macro = 0
        f1_per_class = {0: 0, 1: 0, 2: 0}

    return {
        "model": model_name,
        "total": len(data),
        "valid": len(valid),
        "errors": errors,
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_class_0": round(f1_per_class.get(0, 0), 4),
        "f1_class_1": round(f1_per_class.get(1, 0), 4),
        "f1_class_2": round(f1_per_class.get(2, 0), 4),
        "confusion_matrix": [[cm.get((t, p), 0) for p in labels] for t in labels] if valid else [[0]*3]*3,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="", help="单个模型 ID")
    parser.add_argument("--all", action="store_true", help="评估所有模型")
    parser.add_argument("--workers", type=int, default=5, help="并发线程数")
    parser.add_argument("--subset", type=int, default=0, help="只评估前 N 条")
    parser.add_argument("--rate-limit", type=float, default=0.3, help="请求间隔(秒)")
    args = parser.parse_args()

    # 加载测试数据
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    print(f"测试数据: {len(data)} 条")

    client = get_client()

    if args.all:
        models = [
            "deepseek-ai/deepseek-v4-flash",
            "qwen/qwen2.5-72b-instruct",
            "meta/llama-3.1-70b-instruct",
            "meta/llama-3.3-70b-instruct",
            "nvidia/nemotron-4-340b-instruct",
            "mistralai/mistral-large-2-instruct",
            "google/gemma-2-27b-it",
            "google/gemma-3-12b-it",
        ]
    else:
        models = [args.model]

    all_results = {}

    for model_name in models:
        result = evaluate_model(client, model_name, data, args.workers, args.rate_limit, args.subset)
        all_results[model_name] = result

        # 打印结果
        print(f"\n{'='*60}")
        print(f"模型: {model_name}")
        print(f"{'='*60}")
        print(f"准确率:  {result['accuracy']:.4f}")
        print(f"Macro-F1: {result['f1_macro']:.4f}")
        print(f"F1 Neg: {result['f1_class_0']:.4f}")
        print(f"F1 Neu: {result['f1_class_1']:.4f}")
        print(f"F1 Pos: {result['f1_class_2']:.4f}")
        print(f"有效/总数: {result['valid']}/{result['total']} (错误: {result['errors']})")

        if result['confusion_matrix']:
            print(f"\n混淆矩阵:")
            print("  Pred:  0    1    2")
            for i, row in enumerate(result['confusion_matrix']):
                print(f"  True {i}:   {row[0]:4d} {row[1]:4d} {row[2]:4d}")
        print()

    # 汇总对比
    print(f"\n{'='*60}")
    print("模型对比汇总")
    print(f"{'='*60}")
    print(f"{'Model':<45} {'Accuracy':>10} {'F1-macro':>10} {'F1-Neg':>9} {'F1-Neu':>9} {'F1-Pos':>9}")
    print("-" * 100)
    for m, r in all_results.items():
        short = m.split("/")[-1][:44]
        print(f"{short:<45} {r['accuracy']:>10.4f} {r['f1_macro']:>10.4f} "
              f"{r['f1_class_0']:>9.4f} {r['f1_class_1']:>9.4f} {r['f1_class_2']:>9.4f}")

    # 保存结果
    output_file = OUTPUT_DIR / "nvidia_api_comparison.json"
    # 移除详细结果以减小文件大小
    summary = {}
    for m, r in all_results.items():
        sr = {k: v for k, v in r.items() if k != "results"}
        summary[m] = sr

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n结果已保存: {output_file}")


if __name__ == "__main__":
    main()
