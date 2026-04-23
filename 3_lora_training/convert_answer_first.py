#!/usr/bin/env python3
"""
数据格式转换 - 答案优先格式

将现有 "先CoT后答案" 格式转换为 "先答案后CoT" 格式：

原格式：
<|channel>thought
推理过程...（500+字符）
<channel|>
{"sentiment": 0, "confidence": 0.85}

新格式：
{"sentiment": 0}
<|channel>thought
推理过程...（可选，可截断）
<channel|>

优点：
- 答案立即可得（推理仅需20字符）
- CoT 可截断不影响结果
- 生产环境可用

Usage:
    python convert_answer_first.py --input data/conversations/train_conversations.json --output data/train_answer_first.json
"""

import json
import argparse
import re
from pathlib import Path
from tqdm import tqdm


def convert_assistant_content(content: str) -> str:
    """将 assistant 输出从 "先CoT后答案" 转换为 "先答案后CoT"""

    # 提取 sentiment 值
    match = re.search(r'"sentiment":\s*([0-2])', content)
    if not match:
        return content  # 无法解析，保留原格式

    sentiment = match.group(1)

    # 提取 confidence（可选）
    conf_match = re.search(r'"confidence":\s*([0-9.]+)', content)
    confidence = conf_match.group(1) if conf_match else "0.85"

    # 提取 CoT 部分
    # 格式: <|channel>thought\n{cot}<channel|>
    cot_match = re.search(r'<\|channel>thought\n(.*?)<channel\|>', content, re.DOTALL)
    cot = cot_match.group(1) if cot_match else ""

    # 构建新格式：答案优先
    # 仅输出 sentiment（推理速度最快）
    new_content = f'{{"sentiment": {sentiment}}}'

    # 如果需要保留 CoT（可选，可截断）
    if cot:
        new_content += f'\n<|channel>thought\n{cot}<channel|>'

    return new_content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--keep-cot", action="store_true", default=True,
                        help="保留 CoT 在答案后（可截断）")
    parser.add_argument("--no-cot", action="store_true",
                        help="移除 CoT，仅保留答案（最快推理）")
    args = parser.parse_args()

    if args.no_cot:
        args.keep_cot = False

    print("=" * 60)
    print("数据格式转换 - 答案优先")
    print("=" * 60)

    # 加载原数据
    print(f"\n加载: {args.input}")
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    print(f"样本数: {len(data)}")

    # 转换
    converted = []
    stats = {"original_len": [], "new_len": [], "failed": 0}

    for item in tqdm(data, desc="转换"):
        new_item = item.copy()

        if "conversations" in item:
            conv = item["conversations"]
            # 转换 assistant 内容
            assistant_content = conv[2]["content"]
            new_content = convert_assistant_content(assistant_content)

            if args.no_cot:
                # 移除 CoT
                new_content = re.sub(r'\n<\|channel>thought.*$', '', new_content)

            stats["original_len"].append(len(assistant_content))
            stats["new_len"].append(len(new_content))

            if new_content == assistant_content:
                stats["failed"] += 1

            # 更新 assistant 内容
            new_item["conversations"] = conv.copy()
            new_item["conversations"][2] = {
                "role": "assistant",
                "content": new_content
            }

        converted.append(new_item)

    # 统计
    import statistics
    print(f"\n转换统计:")
    print(f"  成功: {len(data) - stats['failed']}")
    print(f"  失败: {stats['failed']}")
    print(f"  原平均长度: {statistics.mean(stats['original_len']):.1f} 字符")
    print(f"  新平均长度: {statistics.mean(stats['new_len']):.1f} 字符")
    print(f"  长度缩减: {(1 - statistics.mean(stats['new_len'])/statistics.mean(stats['original_len']))*100:.1f}%")

    # 显示示例
    print(f"\n转换示例:")
    print("原格式:")
    print(f"  {data[0]['conversations'][2]['content'][:200]}...")
    print()
    print("新格式:")
    print(f"  {converted[0]['conversations'][2]['content'][:100]}...")

    # 保存
    print(f"\n保存: {args.output}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"\n完成! 共 {len(converted)} 条")


if __name__ == "__main__":
    main()