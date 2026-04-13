#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 软标注脚本 - 概率分布输出
遵循最佳实践：
1. 输出概率分布而非硬标签
2. Temperature=0.3 保证稳定性
3. 多次采样取平均降低方差
4. 置信度筛选质量控制
"""

import os
import sys
import json
import time
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from collections import Counter

# 软标注提示词 - 要求输出概率分布
# 关键优化：明确混合情感 = 高 neutral 概率
SOFT_LABEL_PROMPT = '''你是一位专业的电商评论情感分析专家。

## 核心规则 ⭐

1. **混合情感 = 中性 (neutral > 0.5)**
   - 如果评论同时包含明显的优点和缺点
   - 例如："头发很软但味道难闻" → 有优点有缺点 → neutral 应该最高

2. **轻度不满 = 中性**
   - 只是小问题，不是强烈抱怨
   - 例如："磁力不够强" → 小问题 → neutral 应该较高

3. **强烈情感 = 对应类别**
   - 纯粹好评 → positive > 0.8
   - 纯粹差评 → negative > 0.8

## 分类标准

### 负面 (Negative) - probability > 0.7
- 强烈不满、退货、差评、浪费钱
- 纯粹批评，没有任何优点
- 词: terrible, awful, worst, hate, waste, garbage

### 中性 (Neutral) - probability > 0.5 ⭐ 关键
- **混合情感**: 既有优点也有缺点
- **轻度不满**: 小问题，非致命缺陷
- **中性词**: ok, 一般, 还可以, 凑合, 还行

### 正面 (Positive) - probability > 0.7
- 明显满意、推荐、喜欢
- 纯粹好评，没有缺点
- 词: love, great, amazing, excellent, perfect, best

## 任务

对以下评论进行情感分析，输出概率分布。

评论: "{text}"

**重要**:
- 如果评论有优缺点，neutral 必须 > 0.5
- 三个概率之和必须为 1

输出格式 (JSON):
{{
  "positive": 0.xx,
  "neutral": 0.xx,
  "negative": 0.xx,
  "confidence": 0.xx
}}'''


def get_client():
    """获取 API 客户端"""
    sf_key = os.environ.get("SILICONFLOW_API_KEY", "")
    if not sf_key:
        config_path = Path(__file__).parent.parent.parent / "config" / "api_keys.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                sf_key = config.get("siliconflow", {}).get("api_key", "")

    if sf_key:
        print("使用 SiliconFlow API")
        return OpenAI(
            api_key=sf_key,
            base_url="https://api.siliconflow.cn/v1"
        )

    raise ValueError("未找到有效的 API Key")


def generate_soft_label(client, text, model="Pro/deepseek-ai/DeepSeek-V3.2", temperature=0.3):
    """
    生成软标签（概率分布）

    Returns:
        dict: {
            "probabilities": [p_neg, p_neu, p_pos],
            "confidence": float,
            "reasoning": str,
            "hard_label": int  # argmax of probabilities
        }
    """
    prompt = SOFT_LABEL_PROMPT.format(text=text)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是电商评论情感分析专家。重要规则：混合情感(有优有缺)=高neutral概率。轻度不满也是中性。"},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=1000,
        )

        content = response.choices[0].message.content

        # 提取 JSON
        json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            # 尝试直接解析
            result = json.loads(content)

        # 解析概率
        pos = result.get('positive', 0.33)
        neu = result.get('neutral', 0.33)
        neg = result.get('negative', 0.33)

        # 归一化
        total = pos + neu + neg
        if total > 0:
            pos, neu, neg = pos/total, neu/total, neg/total
        else:
            pos, neu, neg = 0.33, 0.33, 0.34

        probabilities = [neg, neu, pos]  # [negative, neutral, positive]
        hard_label = int(np.argmax(probabilities))
        confidence = result.get('confidence', float(max(probabilities)))
        reasoning = result.get('reasoning', '')

        return {
            "probabilities": probabilities,
            "confidence": confidence,
            "reasoning": reasoning,
            "hard_label": hard_label,
            "raw_response": content
        }

    except Exception as e:
        print(f"API 错误: {e}")
        return None


def multi_sample_soft_label(client, text, model, n_samples=3, temperature=0.3):
    """
    多次采样取平均，降低方差
    """
    all_probs = []

    for _ in range(n_samples):
        result = generate_soft_label(client, text, model, temperature)
        if result:
            all_probs.append(result['probabilities'])

    if not all_probs:
        return None

    # 平均概率
    avg_probs = np.mean(all_probs, axis=0).tolist()

    return {
        "probabilities": avg_probs,
        "confidence": float(max(avg_probs)),
        "hard_label": int(np.argmax(avg_probs)),
        "n_samples": len(all_probs)
    }


def calibrate_neutral(probabilities, threshold=0.6):
    """
    校准中性类别
    当三分类概率都接近时才保留 Neutral

    Args:
        probabilities: [p_neg, p_neu, p_pos]
        threshold: 最大概率阈值

    Returns:
        (adjusted_label, is_uncertain)
    """
    max_prob = max(probabilities)

    if max_prob < threshold:
        # 低确定性，可能是中性
        # 检查是否三分类概率接近
        sorted_probs = sorted(probabilities, reverse=True)
        if sorted_probs[0] - sorted_probs[1] < 0.2:
            # 概率接近，倾向中性
            return 1, True
        else:
            return int(np.argmax(probabilities)), True
    else:
        return int(np.argmax(probabilities)), False


def test_soft_labeling():
    """测试软标注"""
    print("=" * 60)
    print("测试 LLM 软标注 (概率分布)")
    print("=" * 60)

    client = get_client()

    # 测试样本
    test_samples = [
        # 中性评论 - 之前错误分类的
        ("Super soft hair but it smells so bad. I ordered 3 colors and it just smells disgusting", 1),
        ("ok shampoo but doesnt really moisten the hair like it says", 1),
        ("Magnet is not very strong. Needed extra mounting grip.", 1),
        ("Did not make a difference on my skin. Applied well. Not sticky.", 1),
        # 正面
        ("This product is amazing! Best purchase ever!", 2),
        ("Great product, works perfectly. Highly recommend!", 2),
        # 负面
        ("Terrible quality, waste of money. Never buying again!", 0),
        ("Completely useless. Broke after one use.", 0),
    ]

    correct = 0
    total = 0
    neutral_correct = 0
    neutral_total = 0

    for text, expected in test_samples:
        print(f"\n{'='*50}")
        print(f"评论: {text[:60]}...")

        # 多次采样
        result = multi_sample_soft_label(client, text, "Pro/deepseek-ai/DeepSeek-V3.2", n_samples=1)

        if result:
            probs = result['probabilities']
            pred = result['hard_label']
            conf = result['confidence']

            is_correct = pred == expected
            correct += int(is_correct)
            total += 1

            if expected == 1:
                neutral_total += 1
                neutral_correct += int(is_correct)

            # 校准中性
            adj_label, is_uncertain = calibrate_neutral(probs)

            print(f"  概率分布:")
            print(f"    Negative: {probs[0]:.2f}")
            print(f"    Neutral:  {probs[1]:.2f}")
            print(f"    Positive: {probs[2]:.2f}")
            print(f"  硬标签: {pred} (校准后: {adj_label}) {'✓' if is_correct else '✗'}")
            print(f"  置信度: {conf:.2f} {'(不确定)' if is_uncertain else ''}")
            print(f"  期望: {expected}")
        else:
            print("  [X] 生成失败")

    print(f"\n{'='*60}")
    print(f"总体准确率: {correct}/{total} = {correct/total*100:.1f}%")
    if neutral_total > 0:
        print(f"Neutral 准确率: {neutral_correct}/{neutral_total} = {neutral_correct/neutral_total*100:.1f}%")

    return correct, total


def generate_soft_label_dataset(
    input_path,
    output_path,
    num_samples=100,
    model="Pro/deepseek-ai/DeepSeek-V3.2",
    n_samples_per_item=1
):
    """
    生成软标签数据集
    """
    print("=" * 60)
    print("LLM 软标注数据生成")
    print(f"模型: {model}")
    print(f"每条采样次数: {n_samples_per_item}")
    print("=" * 60)

    # 加载数据
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    print(f"加载数据: {len(data)} 条")

    # 采样（确保各类别均衡）
    by_label = {0: [], 1: [], 2: []}
    for item in data:
        label = item.get('sentiment_label', item.get('sentiment', 1))
        by_label[label].append(item)

    # 每个类别采样
    samples_per_class = num_samples // 3
    sampled = []
    for label, items in by_label.items():
        sampled.extend(items[:samples_per_class])

    print(f"采样: {len(sampled)} 条 (每类 {samples_per_class} 条)")

    # 初始化客户端
    client = get_client()

    # 生成
    results = []
    errors = 0

    for item in tqdm(sampled, desc="生成软标签"):
        text = item.get('original_text', item.get('text', ''))
        gt_label = item.get('sentiment_label', item.get('sentiment', 1))

        result = multi_sample_soft_label(client, text, model, n_samples_per_item)

        if result:
            # 校准中性
            adj_label, is_uncertain = calibrate_neutral(result['probabilities'])

            results.append({
                "id": item.get('id', ''),
                "text": text,
                "ground_truth_label": gt_label,
                "probabilities": result['probabilities'],
                "hard_label": result['hard_label'],
                "calibrated_label": adj_label,
                "confidence": result['confidence'],
                "is_uncertain": is_uncertain,
                "model": model
            })
        else:
            errors += 1

        time.sleep(0.5)

    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"\n生成完成: {len(results)} 条")
    print(f"错误: {errors} 条")
    print(f"保存到: {output_path}")

    # 统计
    correct_hard = sum(1 for r in results if r['hard_label'] == r['ground_truth_label'])
    correct_calib = sum(1 for r in results if r['calibrated_label'] == r['ground_truth_label'])

    print(f"\n硬标签准确率: {correct_hard}/{len(results)} = {correct_hard/len(results)*100:.1f}%")
    print(f"校准后准确率: {correct_calib}/{len(results)} = {correct_calib/len(results)*100:.1f}%")

    # 各类别准确率
    for label in [0, 1, 2]:
        items = [r for r in results if r['ground_truth_label'] == label]
        if items:
            acc = sum(1 for r in items if r['calibrated_label'] == label) / len(items) * 100
            names = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
            print(f"{names[label]} 准确率: {acc:.1f}%")

    # 置信度分布
    high_conf = sum(1 for r in results if r['confidence'] > 0.8)
    mid_conf = sum(1 for r in results if 0.6 <= r['confidence'] <= 0.8)
    low_conf = sum(1 for r in results if r['confidence'] < 0.6)

    print(f"\n置信度分布:")
    print(f"  高 (>0.8): {high_conf} ({high_conf/len(results)*100:.1f}%)")
    print(f"  中 (0.6-0.8): {mid_conf} ({mid_conf/len(results)*100:.1f}%)")
    print(f"  低 (<0.6): {low_conf} ({low_conf/len(results)*100:.1f}%)")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM 软标注")
    parser.add_argument("--test", action="store_true", help="测试软标注")
    parser.add_argument("--input", default="data/raw/amazon_train.jsonl", help="输入数据路径")
    parser.add_argument("--output", default="data/processed/soft_labels.jsonl", help="输出路径")
    parser.add_argument("--num", type=int, default=100, help="生成样本数")
    parser.add_argument("--model", default="Pro/deepseek-ai/DeepSeek-V3.2", help="模型名称")
    parser.add_argument("--samples", type=int, default=1, help="每条采样次数")

    args = parser.parse_args()

    if args.test:
        test_soft_labeling()
    else:
        generate_soft_label_dataset(
            args.input,
            args.output,
            args.num,
            args.model,
            args.samples
        )