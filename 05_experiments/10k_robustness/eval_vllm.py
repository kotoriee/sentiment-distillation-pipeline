#!/usr/bin/env python3
"""
vLLM 推理加速评估脚本

vLLM 提供以下加速:
- PagedAttention: 内存效率优化
- Continuous batching: 动态批处理
- KV cache 优化: 高效缓存管理
- 预计吞吐提升: 10x+ (相比 Unsloth 4bit)

Usage:
    1. 先运行 merge_lora_for_vllm.py 生成 merged_16bit 模型
    2. 启动 vLLM server: vllm serve merged_model --port 8000
    3. 运行评估: python eval_vllm.py --data data/smoke_subset_100.json --port 8000
"""

import json
import time
import argparse
import requests
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """You are a professional e-commerce review sentiment analysis expert.

## Task
Analyze the review and output:
1. Sentiment classification (negative/neutral/positive)
2. Reasoning chain explaining your analysis

## Output Format (JSON)
{
    "sentiment": 0/1/2
}"""


def build_prompt(text: str) -> str:
    """构建 Qwen chat 格式 prompt"""
    return f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nReview: {text}<|im_end|>\n<|im_start|>assistant\n"


def extract_sentiment(output: str) -> int | None:
    """从模型输出中提取情感标签"""
    # 直接数字
    if output.strip() in ["0", "1", "2"]:
        return int(output.strip())

    # JSON 格式
    import re
    json_match = re.search(r'\{[^}]*"sentiment"[^}]*\}', output)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return data.get("sentiment")
        except:
            pass

    # 关键词匹配
    if "negative" in output.lower() or "0" in output:
        return 0
    if "neutral" in output.lower() or "1" in output:
        return 1
    if "positive" in output.lower() or "2" in output:
        return 2

    return None


def call_vllm_api(prompt: str, url: str, max_tokens: int = 32) -> str:
    """调用 vLLM OpenAI-compatible API"""
    payload = {
        "model": "merged_model",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
    }

    response = requests.post(f"{url}/completions", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()["choices"][0]["text"]


def evaluate_vllm(data_path: str, api_url: str, max_tokens: int = 32, batch_size: int = 8):
    """使用 vLLM API 评估"""
    print(f"加载数据: {data_path}")
    data = json.loads(Path(data_path).read_text())

    print(f"评估 {len(data)} 样本 (max_tokens={max_tokens})")
    print(f"API URL: {api_url}")

    predictions = []
    labels = []
    parse_failures = 0
    infer_time = 0

    # vLLM 批处理请求
    for i in tqdm(range(0, len(data), batch_size), desc="推理"):
        batch = data[i:i + batch_size]
        prompts = [build_prompt(d["text"]) for d in batch]

        # 计时
        t0 = time.perf_counter()

        # 批量请求 (vLLM 支持 batch completions)
        try:
            payload = {
                "model": "merged_model",
                "prompt": prompts,
                "max_tokens": max_tokens,
                "temperature": 0.0,
            }
            response = requests.post(f"{api_url}/completions", json=payload, timeout=60)
            outputs = [c["text"] for c in response.json()["choices"]]
        except Exception as e:
            print(f"批处理失败: {e}, 回退到单样本请求")
            outputs = []
            for prompt in prompts:
                try:
                    output = call_vllm_api(prompt, api_url, max_tokens)
                    outputs.append(output)
                except Exception as e2:
                    outputs.append("")
                    parse_failures += 1

        t1 = time.perf_counter()
        infer_time += (t1 - t0)

        # 解析输出
        for d, output in zip(batch, outputs):
            pred = extract_sentiment(output)
            if pred is None:
                parse_failures += 1
                predictions.append(-1)
            else:
                predictions.append(pred)
            labels.append(d["label"])

    # 计算指标
    valid_preds = [p for p, l in zip(predictions, labels) if p != -1]
    valid_labels = [l for p, l in zip(predictions, labels) if p != -1]

    if len(valid_preds) > 0:
        accuracy = accuracy_score(valid_labels, valid_preds)
        f1_macro = f1_score(valid_labels, valid_preds, average="macro")
        precision, recall, f1_per_class, _ = precision_recall_fscore_support(
            valid_labels, valid_preds, labels=[0, 1, 2], zero_division=0
        )
        cm = confusion_matrix(valid_labels, valid_preds, labels=[0, 1, 2])
    else:
        accuracy = 0
        f1_macro = 0
        f1_per_class = [0, 0, 0]
        cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    results = {
        "model": "vllm_merged_model",
        "total_samples": len(data),
        "parse_failures": parse_failures,
        "valid_predictions": len(valid_preds),
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_class_0": f1_per_class[0],
        "f1_class_1": f1_per_class[1],
        "f1_class_2": f1_per_class[2],
        "confusion_matrix": cm.tolist(),
        "infer_time_sec": infer_time,
        "throughput": len(data) / infer_time if infer_time > 0 else 0,
        "latency_ms_per_sample": (infer_time * 1000) / len(data) if len(data) > 0 else 0,
    }

    # 保存结果
    output_name = Path(data_path).stem + "_vllm.json"
    output_path = OUTPUT_DIR / output_name
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"保存结果: {output_path}")

    # 打印摘要
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"有效预测: {len(valid_preds)}/{len(data)}")
    print(f"解析失败: {parse_failures}")
    print(f"准确率: {accuracy:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    print(f"Neutral F1: {f1_per_class[1]:.4f}")
    print(f"吞吐: {results['throughput']:.2f} samples/sec")
    print(f"延迟: {results['latency_ms_per_sample']:.2f} ms/sample")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="数据文件路径")
    parser.add_argument("--url", default="http://localhost:8000/v1", help="vLLM API URL")
    parser.add_argument("--max-tokens", type=int, default=32, help="最大生成 tokens")
    parser.add_argument("--batch-size", type=int, default=16, help="批处理大小")
    args = parser.parse_args()

    evaluate_vllm(args.data, args.url, args.max_tokens, args.batch_size)


if __name__ == "__main__":
    main()