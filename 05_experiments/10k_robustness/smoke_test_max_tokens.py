#!/usr/bin/env python3
"""
Smoke 测试：验证 max_new_tokens=1 加速可行性

目的：
- 验证 answer_first 格式在 max_new_tokens=1 下是否能正确输出
- 对比 SFT 和 GRPO 在不同 max_new_tokens 配置下的 parse_success_rate

决策规则：
- parse_success_rate >= 90% → 采用极速模式（max_new_tokens=1）
- parse_success_rate < 90% → 使用完整模式（max_new_tokens=20）

Usage:
    cd 05_experiments/10k_robustness
    python smoke_test_max_tokens.py
"""

import json
import re
import time
import gc
from pathlib import Path
import torch
from collections import Counter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "test_answer_first.json"
SFT_MODEL_PATH = PROJECT_ROOT / "3_lora_training" / "models" / "qwen35-9b-answer-first"
GRPO_MODEL_PATH = PROJECT_ROOT / "3_lora_training" / "outputs_grpo_9b_rewardfix_v3_nothink_continue_300" / "checkpoint-90"
OUTPUT_DIR = Path(__file__).parent / "results"
SMOKE_SUBSET = 100


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


def load_test_data(path: Path, n_samples: int = 100):
    """加载测试数据并抽取样本"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 抽取前 n_samples 条（确保各类别均衡）
    data = data[:n_samples]

    # 构建 prompts
    prompts = []
    true_labels = []

    for item in data:
        conv = item['conversations']
        prompt_conv = conv[:2]  # system + user
        prompt = tokenizer.apply_chat_template(
            prompt_conv, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)
        true_labels.append(item['label'])

    return prompts, true_labels, data


def evaluate_model(model, tokenizer, prompts, true_labels, max_new_tokens: int, batch_size: int = 8):
    """评估模型"""
    print(f"\n  配置: max_new_tokens={max_new_tokens}, batch_size={batch_size}")

    all_outputs = []
    latencies = []

    for i in tqdm(range(0, len(prompts), batch_size), desc=f"推理 (max_new={max_new_tokens})"):
        batch = prompts[i:i + batch_size]

        inputs = tokenizer(
            text=batch,
            return_tensors="pt",
            truncation=True,
            max_length=512,
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

        # 定期清理显存
        if (i + batch_size) % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 解析结果
    results = []
    parse_errors = 0
    correct = 0

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
            "raw": pred_text[:50],
        })

    valid = len(results) - parse_errors
    parse_success_rate = valid / len(results) * 100 if results else 0
    accuracy = correct / valid * 100 if valid > 0 else 0

    infer_time = sum(latencies) / 1000
    throughput = len(all_outputs) / infer_time
    avg_latency_ms = sum(latencies) / len(latencies) / batch_size

    return {
        "max_new_tokens": max_new_tokens,
        "total_samples": len(results),
        "valid_predictions": valid,
        "parse_errors": parse_errors,
        "parse_success_rate": round(parse_success_rate, 2),
        "accuracy": round(accuracy, 2),
        "correct": correct,
        "infer_time_sec": round(infer_time, 2),
        "throughput_samples_per_sec": round(throughput, 2),
        "avg_latency_ms_per_sample": round(avg_latency_ms, 2),
    }


def run_smoke_test():
    """运行 Smoke 测试"""
    global tokenizer  # 用于 load_test_data

    print("=" * 60)
    print("Smoke 测试：验证 max_new_tokens=1 加速可行性")
    print("=" * 60)

    # 加载测试数据
    print(f"\n加载测试数据: {DATA_PATH}")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)[:SMOKE_SUBSET]
    print(f"  样本数: {len(data)}")

    # 构建 prompts
    prompts = []
    true_labels = []
    for item in data:
        conv = item['conversations']
        prompt_conv = conv[:2]
        prompts.append(None)  # placeholder, tokenizer 需要后加载
        true_labels.append(item['label'])

    results = {}

    # ==================== SFT 测试 ====================
    print("\n" + "=" * 60)
    print("测试 SFT 模型")
    print("=" * 60)

    from unsloth import FastLanguageModel
    from peft import PeftModel

    print(f"\n加载 SFT 模型 (base + adapter):")
    print(f"  Base: unsloth/Qwen3.5-9B-unsloth-bnb-4bit")
    print(f"  Adapter: {SFT_MODEL_PATH}")

    # SFT: 加载 base model + LoRA adapter
    sft_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3.5-9B-unsloth-bnb-4bit",
        max_seq_length=512,
        load_in_4bit=True,
    )
    sft_model = PeftModel.from_pretrained(sft_model, str(SFT_MODEL_PATH))
    FastLanguageModel.for_inference(sft_model)

    # 构建 prompts（此时 tokenizer 已加载）
    sft_prompts = []
    for item in data:
        conv = item['conversations']
        prompt_conv = conv[:2]
        prompt = tokenizer.apply_chat_template(
            prompt_conv, tokenize=False, add_generation_prompt=True
        )
        sft_prompts.append(prompt)

    # 测试 max_new_tokens=1
    sft_fast = evaluate_model(sft_model, tokenizer, sft_prompts, true_labels, max_new_tokens=1)
    print(f"\n  结果 (max_new_tokens=1):")
    print(f"    parse_success_rate: {sft_fast['parse_success_rate']}%")
    print(f"    accuracy: {sft_fast['accuracy']}%")
    print(f"    throughput: {sft_fast['throughput_samples_per_sec']} samples/s")
    print(f"    latency: {sft_fast['avg_latency_ms_per_sample']} ms/sample")

    # 测试 max_new_tokens=20
    sft_full = evaluate_model(sft_model, tokenizer, sft_prompts, true_labels, max_new_tokens=20)
    print(f"\n  结果 (max_new_tokens=20):")
    print(f"    parse_success_rate: {sft_full['parse_success_rate']}%")
    print(f"    accuracy: {sft_full['accuracy']}%")
    print(f"    throughput: {sft_full['throughput_samples_per_sec']} samples/s")
    print(f"    latency: {sft_full['avg_latency_ms_per_sample']} ms/sample")

    results['sft'] = {
        'model_path': str(SFT_MODEL_PATH),
        'max_new_tokens_1': sft_fast,
        'max_new_tokens_20': sft_full,
        'speedup_factor': round(sft_full['infer_time_sec'] / sft_fast['infer_time_sec'], 2) if sft_fast['infer_time_sec'] > 0 else 0,
    }

    # 清理显存
    del sft_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ==================== GRPO 测试 ====================
    print("\n" + "=" * 60)
    print("测试 GRPO 模型 (checkpoint-90)")
    print("=" * 60)

    print(f"\n加载 GRPO 模型: {GRPO_MODEL_PATH}")
    # GRPO 需要加载 base model + LoRA adapter
    grpo_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3.5-9B-unsloth-bnb-4bit",
        max_seq_length=512,
        load_in_4bit=True,
    )
    from peft import PeftModel
    grpo_model = PeftModel.from_pretrained(grpo_model, str(GRPO_MODEL_PATH))
    FastLanguageModel.for_inference(grpo_model)

    # 构建 prompts
    grpo_prompts = []
    for item in data:
        conv = item['conversations']
        prompt_conv = conv[:2]
        prompt = tokenizer.apply_chat_template(
            prompt_conv, tokenize=False, add_generation_prompt=True
        )
        grpo_prompts.append(prompt)

    # 测试 max_new_tokens=1
    grpo_fast = evaluate_model(grpo_model, tokenizer, grpo_prompts, true_labels, max_new_tokens=1)
    print(f"\n  结果 (max_new_tokens=1):")
    print(f"    parse_success_rate: {grpo_fast['parse_success_rate']}%")
    print(f"    accuracy: {grpo_fast['accuracy']}%")
    print(f"    throughput: {grpo_fast['throughput_samples_per_sec']} samples/s")
    print(f"    latency: {grpo_fast['avg_latency_ms_per_sample']} ms/sample")

    # 测试 max_new_tokens=20
    grpo_full = evaluate_model(grpo_model, tokenizer, grpo_prompts, true_labels, max_new_tokens=20)
    print(f"\n  结果 (max_new_tokens=20):")
    print(f"    parse_success_rate: {grpo_full['parse_success_rate']}%")
    print(f"    accuracy: {grpo_full['accuracy']}%")
    print(f"    throughput: {grpo_full['throughput_samples_per_sec']} samples/s")
    print(f"    latency: {grpo_full['avg_latency_ms_per_sample']} ms/sample")

    results['grpo'] = {
        'model_path': str(GRPO_MODEL_PATH),
        'max_new_tokens_1': grpo_fast,
        'max_new_tokens_20': grpo_full,
        'speedup_factor': round(grpo_full['infer_time_sec'] / grpo_fast['infer_time_sec'], 2) if grpo_fast['infer_time_sec'] > 0 else 0,
    }

    # 清理显存
    del grpo_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ==================== 决策建议 ====================
    print("\n" + "=" * 60)
    print("决策建议")
    print("=" * 60)

    sft_decision = "max_new_tokens=1" if sft_fast['parse_success_rate'] >= 90 else "max_new_tokens=20"
    grpo_decision = "max_new_tokens=1" if grpo_fast['parse_success_rate'] >= 90 else "max_new_tokens=20"

    print(f"\nSFT:")
    print(f"  parse_success_rate (max_new=1): {sft_fast['parse_success_rate']}%")
    print(f"  建议: {sft_decision}")

    print(f"\nGRPO:")
    print(f"  parse_success_rate (max_new=1): {grpo_fast['parse_success_rate']}%")
    print(f"  建议: {grpo_decision}")

    results['recommendations'] = {
        'sft': {
            'recommended_max_new_tokens': 1 if sft_fast['parse_success_rate'] >= 90 else 20,
            'reason': f"parse_success_rate={sft_fast['parse_success_rate']}%" + (" >= 90%" if sft_fast['parse_success_rate'] >= 90 else " < 90%"),
        },
        'grpo': {
            'recommended_max_new_tokens': 1 if grpo_fast['parse_success_rate'] >= 90 else 20,
            'reason': f"parse_success_rate={grpo_fast['parse_success_rate']}%" + (" >= 90%" if grpo_fast['parse_success_rate'] >= 90 else " < 90%"),
        },
    }

    # 保存结果
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "smoke_test_max_tokens.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {output_file}")

    return results


if __name__ == "__main__":
    run_smoke_test()