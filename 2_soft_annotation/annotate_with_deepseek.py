#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三分类软标注批量运行脚本 v2
模型: DeepSeek-R1-0528-Qwen3-8B (推理模型，支持 <think> 块)

用法:
  # 小批量质量验证（先跑这个）
  python run_3cls_annotation.py --batch-test 20

  # 全量标注
  python run_3cls_annotation.py

  # 指定 key
  python run_3cls_annotation.py --api-key sk-xxx

输出:
  data/processed/soft_labels_raw.jsonl   (含概率分布 + 推理文本)
  data/processed/train_3cls.json         (instruction/input/output 格式)
  data/processed/val_3cls.json
  data/processed/test_3cls.json
"""

import os
import sys
import json
import time
import re
import argparse
import threading
import numpy as np
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).parent.parent.parent

# ===== 提示词：精简版，适配 R1 推理模型 =====
SYSTEM_PROMPT = "You are a sentiment classifier for e-commerce reviews."

USER_PROMPT = """Classify the sentiment of this e-commerce review into negative/neutral/positive.
Output ONLY a JSON object — no explanation, no extra text.

Review: {text}

Rules:
- Mixed review (clear pros AND cons) → neutral highest (0.6-0.7)
- Strong criticism (terrible/waste/broke/hate/return) → negative > 0.85
- Mild issue (ok/expected better/not great) → neutral > negative
- Pure praise (love/amazing/perfect) → positive > 0.85
- Values must sum to 1.0; confidence = max(probabilities)

Examples:
Review: "Absolutely love this! Works perfectly, best purchase ever."
{{"positive":0.92,"neutral":0.06,"negative":0.02,"confidence":0.92}}

Review: "Broke after one week. Terrible quality, total waste of money. Never buying again."
{{"positive":0.02,"neutral":0.05,"negative":0.93,"confidence":0.93}}

Review: "Doesnt cover tattoos, doesnt stay. not worth the waste of time or money."
{{"positive":0.03,"neutral":0.07,"negative":0.90,"confidence":0.90}}

Review: "Love the color but the smell is overwhelming and left my hair dry."
{{"positive":0.15,"neutral":0.65,"negative":0.20,"confidence":0.65}}

Output:"""


def get_client(api_key=None):
    key = api_key or os.environ.get("SILICONFLOW_API_KEY", "")
    if not key:
        config_path = ROOT / "config" / "api_keys.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            key = cfg.get("siliconflow", {}).get("api_key", "")
    if not key:
        raise ValueError("未找到 API Key。请设置 SILICONFLOW_API_KEY 或传入 --api-key")
    print(f"SiliconFlow API，key 前缀: {key[:12]}...")
    return OpenAI(api_key=key, base_url="https://api.siliconflow.cn/v1")


def strip_thinking(content):
    """剥离 R1 推理模型的思考块，返回 (answer, thinking)"""
    # 提取思考内容（用于调试）
    think_match = re.search(
        r'<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>|<think>(.*?)</think>',
        content, re.DOTALL
    )
    thinking = ""
    if think_match:
        thinking = (think_match.group(1) or think_match.group(2) or "").strip()

    # 剥离思考块
    answer = re.sub(r'<\|begin_of_thought\|>.*?<\|end_of_thought\|>', '', content, flags=re.DOTALL)
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
    # 剥离 solution 标签
    answer = re.sub(r'<\|begin_of_solution\|>|<\|end_of_solution\|>', '', answer)
    return answer.strip(), thinking


def call_api(client, text, model, temperature=0.6):
    """调用 API 获取三分类软标签，处理 R1 推理输出"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(text=text)}
            ],
            temperature=temperature,
            max_tokens=300,
        )
        raw_content = response.choices[0].message.content
        answer, thinking = strip_thinking(raw_content)

        # 提取 JSON
        json_match = re.search(r'\{[^{}]*\}', answer, re.DOTALL)
        if not json_match:
            # 尝试宽松匹配（有时模型输出嵌套 JSON）
            json_match = re.search(r'\{.*\}', answer, re.DOTALL)
        if not json_match:
            return None

        result = json.loads(json_match.group())

        pos = float(result.get('positive', 0.33))
        neu = float(result.get('neutral', 0.33))
        neg = float(result.get('negative', 0.33))
        total = pos + neu + neg
        if total > 0:
            pos, neu, neg = pos/total, neu/total, neg/total
        else:
            pos, neu, neg = 0.33, 0.33, 0.34

        probs = [neg, neu, pos]  # index 0=negative, 1=neutral, 2=positive
        hard_label = int(np.argmax(probs))
        confidence = float(result.get('confidence', max(probs)))

        return {
            "probabilities": probs,
            "hard_label": hard_label,
            "confidence": confidence,
            "thinking_preview": thinking[:200] if thinking else ""
        }
    except Exception as e:
        return None


def extract_review(text):
    """从 conversation 格式提取评论文本，在 <|im_end|> 处截断"""
    # 优先在 <|im_end|> 前截断（无论是否有换行）
    match = re.search(r'Review: (.*?)(?=<\|im_end\|>)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # 降级：截到字符串末尾
    match = re.search(r'Review: (.+)', text, re.DOTALL)
    return match.group(1).strip() if match else None


def load_progress(progress_path):
    if progress_path.exists():
        with open(progress_path) as f:
            done = set(json.load(f).get("completed_ids", []))
        print(f"断点续传：已处理 {len(done)} 条")
        return done
    return set()


def save_progress(progress_path, completed_ids):
    with open(progress_path, 'w') as f:
        json.dump({"completed_ids": list(completed_ids), "count": len(completed_ids)}, f)


def run_batch_test(client, records, model, temperature, n=20):
    """小批量质量验证，打印详细结果供人工审查"""
    print(f"\n{'='*60}")
    print(f"小批量质量验证 (前 {n} 条)")
    print(f"模型: {model} | temperature: {temperature}")
    print(f"{'='*60}\n")

    label_names = {0: "NEG", 1: "NEU", 2: "POS"}
    samples = records[:n]
    success = 0

    for i, item in enumerate(samples):
        result = call_api(client, item["text"], model, temperature)
        if result:
            success += 1
            probs = result["probabilities"]
            label = result["hard_label"]
            conf = result["confidence"]
            print(f"[{i+1:02d}] {label_names[label]} (conf={conf:.2f})")
            print(f"     neg={probs[0]:.2f} neu={probs[1]:.2f} pos={probs[2]:.2f}")
            # 显示截断后的评论（检查是否有 <|im_ 污染）
            review_preview = item['text'][:120]
            has_contamination = '<|im_' in item['text']
            print(f"     评论{'[污染!]' if has_contamination else ''}: {review_preview}{'...' if len(item['text'])>120 else ''}")
            if result["thinking_preview"]:
                print(f"     思考: {result['thinking_preview'][:100]}...")
            print()
        else:
            print(f"[{i+1:02d}] *** API 调用失败 ***")
            print(f"     评论: {item['text'][:80]}...")
            print()
        time.sleep(60 / 1000)  # 1000 RPM

    # 统计
    print(f"\n成功率: {success}/{n}")
    print("\n确认质量后，移除 --batch-test 参数运行全量标注")
    return success


def main():
    parser = argparse.ArgumentParser(description="三分类软标注 v2 (R1 推理模型)")
    parser.add_argument("--api-key", default="", help="SiliconFlow API Key")
    parser.add_argument("--model", default="Pro/deepseek-ai/DeepSeek-V3.2", help="模型名称")
    parser.add_argument("--temperature", type=float, default=0.6, help="温度（R1 Distill 用 0.6，V3 用 0.3）")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--batch-test", type=int, default=0, metavar="N",
                        help="小批量测试：只跑前 N 条并打印详情（质量验证用）")
    parser.add_argument("--max-n", type=int, default=0, help="每个 split 最多处理 N 条（0=全部）")
    parser.add_argument("--rate-limit", type=float, default=0.06, help="请求间隔秒数（默认 0.06s = 1000 RPM）")
    parser.add_argument("--workers", type=int, default=20, help="并发线程数（默认 20）")
    parser.add_argument("--save-interval", type=int, default=50, help="每 N 条保存一次进度")
    args = parser.parse_args()

    client = get_client(args.api_key)
    processed_dir = ROOT / "data" / "processed"
    raw_output = processed_dir / "soft_labels_raw.jsonl"
    progress_path = processed_dir / "annotation_3cls_progress.json"

    # 收集所有评论
    all_records = []
    for split in args.splits:
        split_path = processed_dir / f"{split}.json"
        if not split_path.exists():
            print(f"[跳过] {split}.json 不存在")
            continue
        with open(split_path) as f:
            data = json.load(f)
        for i, item in enumerate(data):
            review = extract_review(item['text'])
            if review:
                all_records.append({"id": f"{split}_{i}", "split": split, "text": review})
        print(f"{split}.json: {len(data)} 条")

    # ===== 小批量测试模式 =====
    if args.batch_test > 0:
        run_batch_test(client, all_records, args.model, args.temperature, args.batch_test)
        return

    # ===== 全量标注模式 =====
    completed_ids = load_progress(progress_path)
    todo = [r for r in all_records if r["id"] not in completed_ids]
    if args.max_n > 0:
        todo = todo[:args.max_n]

    print(f"\n总计: {len(all_records)} 条 | 待标注: {len(todo)} 条")
    print(f"模型: {args.model} | temperature: {args.temperature} | max_tokens: 300")
    print(f"并发: {args.workers} 线程 | 速率上限: {1/args.rate_limit:.0f} RPM\n")

    if not todo:
        print("所有记录已标注完成！生成格式化文件中...")
    else:
        write_lock = threading.Lock()
        errors = 0
        pbar = tqdm(total=len(todo), desc="三分类软标注")

        def process_item(item):
            result = call_api(client, item["text"], args.model, args.temperature)
            time.sleep(args.rate_limit)
            return item, result

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_item, item): item for item in todo}
            for future in as_completed(futures):
                item, result = future.result()
                with write_lock:
                    if result:
                        record = {
                            "id": item["id"],
                            "split": item["split"],
                            "text": item["text"],
                            "probabilities": result["probabilities"],
                            "hard_label": result["hard_label"],
                            "confidence": result["confidence"],
                            "model": args.model
                        }
                        with open(raw_output, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(record, ensure_ascii=False) + '\n')
                        completed_ids.add(item["id"])
                        probs = result["probabilities"]
                        pbar.set_postfix(
                            label=result["hard_label"],
                            conf=f"{result['confidence']:.2f}",
                            neg=f"{probs[0]:.2f}", neu=f"{probs[1]:.2f}", pos=f"{probs[2]:.2f}"
                        )
                    else:
                        errors += 1
                        pbar.set_postfix(errors=errors)
                    pbar.update(1)
                    if len(completed_ids) % args.save_interval == 0:
                        save_progress(progress_path, completed_ids)

        pbar.close()
        save_progress(progress_path, completed_ids)
        print(f"\n标注完成！成功: {len(completed_ids)}, 失败: {errors}")

    # ===== 格式化输出 =====
    if not raw_output.exists():
        print("尚无标注结果，跳过格式化")
        return

    print("\n生成训练格式文件...")
    results_by_split = {"train": [], "val": [], "test": []}

    with open(raw_output) as f:
        for line in f:
            r = json.loads(line)
            entry = {
                "instruction": "Classify the sentiment of this e-commerce review. Output only: 0 (negative), 1 (neutral), or 2 (positive).",
                "input": r["text"],
                "output": str(r["hard_label"]),
                "soft_labels": r["probabilities"],
                "confidence": r["confidence"]
            }
            results_by_split.get(r.get("split", "train"), results_by_split["train"]).append(entry)

    label_names = {0: "negative", 1: "neutral", 2: "positive"}
    for split in ["train", "val", "test"]:
        items = results_by_split[split]
        if not items:
            continue
        p = processed_dir / f"{split}_3cls.json"
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

        dist = {0: 0, 1: 0, 2: 0}
        for e in items:
            dist[int(e["output"])] += 1
        total = len(items)
        dist_str = " | ".join(f"{label_names[k]}:{dist[k]}({dist[k]/total*100:.0f}%)" for k in [0,1,2])
        print(f"  {split}_3cls.json: {total} 条 → {dist_str}")

    # 主训练文件
    train_items = results_by_split["train"]
    if train_items:
        main_path = processed_dir / "train_3cls.json"
        with open(main_path, 'w', encoding='utf-8') as f:
            json.dump(train_items, f, ensure_ascii=False, indent=2)
        print(f"\ntrain_3cls.json 保存完成: {len(train_items)} 条")


if __name__ == "__main__":
    main()
