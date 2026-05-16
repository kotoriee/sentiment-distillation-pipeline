#!/usr/bin/env python3
"""
NVIDIA NIM API 情感分析评估 - 批量模式 + 断点续行
独立脚本，无模块依赖问题。

Usage:
    export NVIDIA_API_KEY=nvapi-xxx
    python eval_nvidia_api.py --model deepseek-ai/deepseek-v4-pro --batch-size 20
    python eval_nvidia_api.py --model deepseek-ai/deepseek-v4-pro --resume checkpoint.jsonl --batch-size 20
"""

import os
import json
import time
import re
import ast
import argparse
import sys
import numpy as np
from pathlib import Path
from collections import Counter
from threading import Lock
from openai import OpenAI
from tqdm import tqdm

DATA_PATH = Path(__file__).parent.parent / "data" / "test_answer_first.json"
OUTPUT_DIR = Path(__file__).parent / "nvidia_api_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = "You are a sentiment classifier for e-commerce reviews."

USER_PROMPT_BATCH = """Classify the sentiment of {count} e-commerce reviews.

For each review, output ONLY a JSON array of objects with format:
[{{"idx": 0, "sentiment": N}}, {{"idx": 1, "sentiment": N}}, ...]

Where N is 0 (negative), 1 (neutral), or 2 (positive).
No explanation, no markdown, just the JSON array.

Rules:
- Negative (0): complaints, returns, terrible quality, waste of money
- Neutral (1): mixed feelings, ok but not great, expected better
- Positive (2): love it, works perfectly, great value

{reviews}

Output:"""


def get_client():
    key = os.environ.get("NVIDIA_API_KEY", "")
    if not key:
        raise ValueError("请设置 NVIDIA_API_KEY 环境变量")
    return OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=key)


def parse_batch_response(text, count):
    # Strategy 1: regex for array
    match = re.search(r'\[[\s\S]*\]', text)
    if not match:
        return [(-1, "NO_JSON")] * count
    raw_array = match.group()

    # Strategy 2: direct JSON
    try:
        items = json.loads(raw_array)
        if isinstance(items, list):
            return _process_items(items, count)
    except Exception:
        pass

    # Strategy 3: ast.literal_eval
    try:
        items = ast.literal_eval(raw_array)
        if isinstance(items, list):
            return _process_items(items, count)
    except Exception:
        pass

    # Strategy 4: fix unquoted keys
    fixed = re.sub(r'([{,]\s*)(\w+)\s*:', lambda m: m.group(1) + '"' + m.group(2) + '":', raw_array)
    try:
        items = json.loads(fixed)
        if isinstance(items, list):
            return _process_items(items, count)
    except Exception:
        pass

    return [(-1, "PARSE_FAIL")] * count


def _process_items(items, count):
    result = {}
    for item in items:
        if isinstance(item, dict) and "idx" in item and "sentiment" in item:
            idx = item["idx"]
            sent = item["sentiment"]
            if isinstance(sent, int) and sent in (0, 1, 2) and 0 <= idx < count:
                result[idx] = (sent, "OK")
            else:
                result[idx] = (-1, f"BAD:{sent}")
    final = []
    for i in range(count):
        final.append(result.get(i, (-1, f"MISS_{i}")))
    return final


def predict_batch(client, reviews, model, temperature=0.3, max_retries=5):
    reviews_text = ""
    for i, (text, _) in enumerate(reviews):
        reviews_text += f"--- Review {i} ---\n{text[:500]}\n\n"

    prompt = USER_PROMPT_BATCH.format(count=len(reviews), reviews=reviews_text)

    for attempt in range(max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=512,
                top_p=0.95,
            )
            content = completion.choices[0].message.content
            results = parse_batch_response(content, len(reviews))
            return results, content[:200]
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "Too Many Requests" in error_msg:
                wait = min(2 ** (attempt + 2), 120)
                print(f"  [429] attempt {attempt+1}, waiting {wait}s")
                sys.stdout.flush()
                time.sleep(wait)
            elif attempt < max_retries:
                wait = 2 ** attempt * 5
                print(f"  [ERR] attempt {attempt+1}: {error_msg[:80]}, waiting {wait}s")
                sys.stdout.flush()
                time.sleep(wait)
            else:
                return [(-1, f"ERR:{error_msg[:40]}")] * len(reviews), f"ERR: {error_msg[:60]}"


def load_checkpoint(path):
    results = {}
    if path and Path(path).exists():
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    results[entry["idx"]] = entry
        print(f"  加载 checkpoint: {len(results)} 条已有结果")
    return results


def save_checkpoint(path, idx, pred, true_label, raw, correct):
    if not path:
        return
    entry = {"idx": idx, "pred": pred, "true": true_label, "raw": raw, "correct": correct}
    with open(path, 'a') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def extract_sentiment(text):
    match = re.search(r'Sentiment:\s*([0-2])', text)
    if match:
        return int(match.group(1))
    match = re.search(r'"sentiment"\s*:\s*([0-2])', text)
    if match:
        return int(match.group(1))
    match = re.search(r'[0-2]', text)
    if match:
        return int(match.group())
    return -1


def predict_single(client, text, model, temperature=0.3, max_retries=3):
    for attempt in range(max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Classify sentiment:\n{text[:2000]}\nOutput Sentiment: N where N is 0, 1, or 2."}
                ],
                temperature=temperature,
                max_tokens=64,
                top_p=0.95,
            )
            content = completion.choices[0].message.content
            return extract_sentiment(content), content[:80]
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "Too Many Requests" in error_msg:
                time.sleep(min(2 ** (attempt + 1), 60))
            elif attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                return -1, f"ERR: {error_msg[:60]}"


def compute_metrics(results, total):
    valid = [r for r in results if r is not None and r["pred"] != -1]
    correct = sum(1 for r in valid if r["pred"] == r["true"])
    y_true = [r["true"] for r in valid]
    y_pred = [r["pred"] for r in valid]

    if not valid:
        return {"accuracy": 0, "f1_macro": 0, "f1_class_0": 0, "f1_class_1": 0, "f1_class_2": 0,
                "confusion_matrix": [[0]*3]*3, "valid": 0, "errors": sum(1 for r in results if r and r["pred"] == -1)}

    accuracy = correct / len(valid)
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

    return {
        "accuracy": round(accuracy, 4), "f1_macro": round(f1_macro, 4),
        "f1_class_0": round(f1_per_class.get(0, 0), 4),
        "f1_class_1": round(f1_per_class.get(1, 0), 4),
        "f1_class_2": round(f1_per_class.get(2, 0), 4),
        "confusion_matrix": [[cm.get((t, p), 0) for p in labels] for t in labels],
        "valid": len(valid), "errors": sum(1 for r in results if r and r["pred"] == -1),
    }


def evaluate_model(client, model_name, data, rate_limit=1.5, subset=0,
                   checkpoint_path=None, resume_results=None, batch_size=1):
    if subset > 0:
        data = data[:subset]

    total = len(data)
    skip_count = len(resume_results) if resume_results else 0
    remaining = total - skip_count
    mode = f"batch({batch_size})" if batch_size > 1 else "single"

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Samples: {total} | Have: {skip_count} | Remaining: {remaining}")
    print(f"Mode: {mode} | Rate limit: {rate_limit}s")
    print(f"{'='*60}\n")

    results = [None] * total
    if resume_results:
        for idx, entry in resume_results.items():
            if idx < total:
                results[idx] = entry

    completed = skip_count
    last_save = 0
    save_interval = 10
    pending_indices = [i for i in range(total) if results[i] is None]
    start_time = time.perf_counter()

    if batch_size > 1:
        batches = []
        for i in range(0, len(pending_indices), batch_size):
            batches.append(pending_indices[i:i + batch_size])

        for batch_indices in tqdm(batches, desc="Batch"):
            try:
                items = [(data[idx]['text'], data[idx]['label']) for idx in batch_indices]
                batch_results, raw = predict_batch(client, items, model_name)

                for i, idx in enumerate(batch_indices):
                    pred, parse_status = batch_results[i] if i < len(batch_results) else (-1, "OOB")
                    true_label = data[idx]['label']
                    is_error = pred == -1
                    correct = pred == true_label if not is_error else False
                    results[idx] = {
                        "pred": pred, "true": true_label,
                        "raw": f"[batch] {parse_status}", "correct": correct
                    }
                    save_checkpoint(checkpoint_path, idx, pred, true_label, f"[batch] {parse_status}", correct)
            except KeyboardInterrupt:
                print(f"\n  Interrupted at [{completed}/{total}]")
                sys.stdout.flush()
                break
            except Exception as e:
                print(f"\n  [EXCEPTION] {e}, marking batch as error")
                sys.stdout.flush()
                for idx in batch_indices:
                    results[idx] = {
                        "pred": -1, "true": data[idx]['label'],
                        "raw": f"[batch] EXC:{str(e)[:30]}", "correct": False
                    }

            completed += len(batch_indices)
            if completed - last_save >= save_interval or completed >= total:
                elapsed = (time.perf_counter() - start_time) / 60
                done = completed - skip_count
                eta = (remaining - done) * (elapsed / max(done, 1))
                errs = sum(1 for r in results if r and r["pred"] == -1)
                print(f"  [{completed}/{total}] done (errors={errs}, ETA={eta:.0f}min)")
                sys.stdout.flush()
                last_save = completed

            time.sleep(rate_limit)
    else:
        for idx in tqdm(pending_indices, desc="Single"):
            pred, raw = predict_single(client, data[idx]['text'], model_name)
            is_error = pred == -1
            correct = pred == data[idx]['label'] if not is_error else False
            results[idx] = {"pred": pred, "true": data[idx]['label'], "raw": raw, "correct": correct}
            save_checkpoint(checkpoint_path, idx, pred, data[idx]['label'], raw, correct)
            completed += 1
            if completed - last_save >= save_interval:
                elapsed = (time.perf_counter() - start_time) / 60
                done = completed - skip_count
                eta = (remaining - done) * (elapsed / max(done, 1))
                errs = sum(1 for r in results if r and r["pred"] == -1)
                print(f"  [{completed}/{total}] done (errors={errs}, ETA={eta:.0f}min)")
                sys.stdout.flush()
                last_save = completed
            time.sleep(rate_limit)

    metrics = compute_metrics(results, total)
    metrics["model"] = model_name
    metrics["total"] = total
    metrics["results"] = results
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="", help="Model ID")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--rate-limit", type=float, default=1.5)
    parser.add_argument("--batch-size", type=int, default=1, help="Reviews per API call (1=single)")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Test data: {len(data)} items")

    client = get_client()

    if args.all:
        models = ["deepseek-ai/deepseek-v4-flash", "qwen/qwen2.5-72b-instruct"]
    else:
        models = [args.model]

    all_results = {}
    for model_name in models:
        checkpoint = args.checkpoint or f"{model_name.replace('/', '_')}_checkpoint.jsonl"
        resume = load_checkpoint(args.resume or checkpoint)

        result = evaluate_model(client, model_name, data, args.rate_limit, args.subset,
                                checkpoint_path=checkpoint, resume_results=resume,
                                batch_size=args.batch_size)
        all_results[model_name] = result

        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        print(f"Accuracy:  {result['accuracy']:.4f}")
        print(f"Macro-F1:  {result['f1_macro']:.4f}")
        print(f"F1 Neg: {result['f1_class_0']:.4f}")
        print(f"F1 Neu: {result['f1_class_1']:.4f}")
        print(f"F1 Pos: {result['f1_class_2']:.4f}")
        print(f"Valid/Total: {result['valid']}/{result['total']} (errors: {result['errors']})")

        if result['confusion_matrix']:
            print(f"\nConfusion Matrix:")
            print("  Pred:  0    1    2")
            for i, row in enumerate(result['confusion_matrix']):
                print(f"  True {i}:   {row[0]:4d} {row[1]:4d} {row[2]:4d}")

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Model':<40} {'Accuracy':>10} {'F1-macro':>10}")
    for m, r in all_results.items():
        short = m.split("/")[-1][:39]
        print(f"{short:<40} {r['accuracy']:>10.4f} {r['f1_macro']:>10.4f}")

    output_file = OUTPUT_DIR / "nvidia_api_comparison.json"
    summary = {m: {k: v for k, v in r.items() if k != "results"} for m, r in all_results.items()}
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
