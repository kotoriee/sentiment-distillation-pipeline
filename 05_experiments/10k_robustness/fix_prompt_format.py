#!/usr/bin/env python3
"""
修复 10k 数据的 system prompt 格式
将简化版 prompt 替换为与训练数据一致的完整版本（包含 Analysis Process）
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = Path(__file__).parent / "data"

# 完整版 system prompt（与训练数据一致）
CORRECT_SYSTEM_PROMPT = """You are a professional e-commerce review sentiment analysis expert.

## Task
Analyze the review and output:
1. Sentiment classification (negative/neutral/positive)
2. Reasoning chain explaining your analysis

## Analysis Process (REQUIRED)
Follow these steps:
1. Signal Detection: Identify positive/negative keywords
2. Context Analysis: Check rating vs text alignment
3. Intensity Calibration: Determine sentiment strength
4. Final Verdict: Summarize and classify

## Output Format (JSON)
{
    "sentiment": 0/1/2,
    "confidence": 0.0-1.0,
    "rationale": "Your reasoning chain"
}"""

# 简化版 system prompt（需要替换）
OLD_SYSTEM_PROMPT = """You are a professional e-commerce review sentiment analysis expert.

## Task
Analyze the review and output:
1. Sentiment classification (negative/neutral/positive)
2. Reasoning chain explaining your analysis

## Output Format (JSON)
{
    "sentiment": 0/1/2
}"""


def fix_prompt(data_file: str, output_file: str):
    """修复数据文件的 system prompt"""
    input_path = DATA_DIR / data_file
    output_path = DATA_DIR / output_file

    print(f"加载: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"总样本: {len(data)}")

    fixed_count = 0
    for item in data:
        convs = item.get('conversations', [])
        if convs and convs[0].get('role') == 'system':
            old_content = convs[0].get('content', '')
            # 检查是否需要修复（缺少 Analysis Process）
            if 'Analysis Process' not in old_content:
                convs[0]['content'] = CORRECT_SYSTEM_PROMPT
                fixed_count += 1

    print(f"修复样本: {fixed_count}")

    # 保存修复后的数据
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"保存: {output_path}")

    return fixed_count


def main():
    print("=" * 60)
    print("修复 10k 数据 prompt 格式")
    print("=" * 60)

    # 修复三品类数据
    fixed = fix_prompt("eval_three_category_10k.json", "eval_three_category_10k_fixed.json")

    # 验证修复结果
    output_path = DATA_DIR / "eval_three_category_10k_fixed.json"
    with open(output_path, 'r', encoding='utf-8') as f:
        fixed_data = json.load(f)

    # 检查第一条样本
    first_conv = fixed_data[0]['conversations'][0]
    has_analysis = 'Analysis Process' in first_conv['content']

    print("\n验证:")
    print(f"  第一条样本包含 Analysis Process: {has_analysis}")
    print(f"  总修复数: {fixed}")

    if fixed > 0 and has_analysis:
        print("\n修复成功！")
    else:
        print("\n无需修复或修复失败")


if __name__ == "__main__":
    main()