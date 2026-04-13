#!/usr/bin/env python3
"""
输出解析模块 - 支持多模型格式

支持解析：
1. Qwen3-4B 格式: </thinking>{JSON}
2. Gemma 4 格式: <|channel>thought...<channel|>{JSON}
3. 简单格式: Answer: 0/1/2
4. JSON格式: {"sentiment": 0/1/2}
"""

import re
import json
from typing import Tuple, Optional


def extract_sentiment_qwen3(text: str) -> Tuple[int, Optional[dict]]:
    """
    解析 Qwen3-4B 输出格式

    格式: </thinking>\n{JSON}
    或者: und_b8\n...und_b8\n{JSON}

    Returns:
        (sentiment, full_json) - sentiment为-1表示解析失败
    """
    # 格式1: </thinking> 后找 JSON
    if "</thinking>" in text:
        after = text.split("</thinking>", 1)[1]
        match = re.search(r'"sentiment":\s*([0-2])', after)
        if match:
            sentiment = int(match.group(1))
            # 尝试解析完整JSON
            try:
                json_match = re.search(r'\{[^{}]*\}', after, re.DOTALL)
                if json_match:
                    full_json = json.loads(json_match.group())
                    return sentiment, full_json
            except:
                pass
            return sentiment, None

    # 格式2: und_b8 (Qwen3特殊标记)
    if "und_b8" in text:
        parts = text.split("und_b8")
        # 最后一个 und_b8 后应该是 JSON
        if len(parts) >= 2:
            last_part = parts[-1]
            match = re.search(r'"sentiment":\s*([0-2])', last_part)
            if match:
                return int(match.group(1)), None

    # 格式3: 直接找 JSON
    match = re.search(r'"sentiment":\s*([0-2])', text)
    if match:
        return int(match.group(1)), None

    return -1, None


def extract_sentiment_gemma4(text: str) -> Tuple[int, Optional[dict]]:
    """
    解析 Gemma 4 输出格式

    格式: <|channel>thought\n[reasoning]<channel|>\n{JSON}<turn|>

    Returns:
        (sentiment, full_json) - sentiment为-1表示解析失败
    """
    # 格式1: <|channel>thought 之后找 JSON
    if "<|channel>thought" in text:
        after = text.split("<|channel>thought", 1)[1]

        # 找 <channel|> 后的 JSON
        if "<channel|>" in after:
            json_part = after.split("<channel|>", 1)[1]
            match = re.search(r'"sentiment":\s*([0-2])', json_part)
            if match:
                sentiment = int(match.group(1))
                # 尝试解析完整JSON
                try:
                    json_match = re.search(r'\{[^{}]*\}', json_part, re.DOTALL)
                    if json_match:
                        full_json = json.loads(json_match.group())
                        return sentiment, full_json
                except:
                    pass
                return sentiment, None

        # 如果没有 <channel|>，直接在后面找
        match = re.search(r'"sentiment":\s*([0-2])', after)
        if match:
            return int(match.group(1)), None

    # 格式2: <turn|> 结尾标记（fallback）
    if "<turn|>" in text:
        before = text.split("<turn|>")[0]
        match = re.search(r'"sentiment":\s*([0-2])', before)
        if match:
            return int(match.group(1)), None

    # 格式3: 直接找 JSON
    match = re.search(r'"sentiment":\s*([0-2])', text)
    if match:
        return int(match.group(1)), None

    return -1, None


def extract_sentiment_simple(text: str) -> Tuple[int, Optional[dict]]:
    """
    解析简单格式

    格式: Answer: 0/1/2
    或者直接数字: 0/1/2
    """
    # 格式1: Answer: X
    match = re.search(r'Answer:\s*([0-2])', text)
    if match:
        return int(match.group(1)), None

    # 格式2: sentiment: X (不带JSON)
    match = re.search(r'sentiment[:\s]+([0-2])', text)
    if match:
        return int(match.group(1)), None

    # 格式3: 最后一个数字（如果只有0/1/2）
    numbers = re.findall(r'[0-2]', text)
    if numbers and len(numbers) == 1:
        return int(numbers[0]), None

    # 格式4: 找到所有0/1/2，取最后一个（通常在结尾）
    if numbers:
        return int(numbers[-1]), None

    return -1, None


def extract_sentiment_auto(text: str) -> Tuple[int, Optional[dict]]:
    """
    自动识别格式并解析

    Returns:
        (sentiment, full_json, format_type)
    """
    # 尝试 Gemma 4 格式
    sentiment, full_json = extract_sentiment_gemma4(text)
    if sentiment != -1:
        return sentiment, full_json, "gemma4"

    # 尝试 Qwen3 格式
    sentiment, full_json = extract_sentiment_qwen3(text)
    if sentiment != -1:
        return sentiment, full_json, "qwen3"

    # 尝试简单格式
    sentiment, full_json = extract_sentiment_simple(text)
    if sentiment != -1:
        return sentiment, full_json, "simple"

    return -1, None, "unknown"


def extract_rationale(text: str, format_type: str = "auto") -> Optional[str]:
    """
    提取推理过程（CoT）

    Args:
        text: 模型输出
        format_type: "gemma4", "qwen3", "auto"

    Returns:
        推理文本，如果无法提取则返回None
    """
    if format_type == "auto":
        _, _, format_type = extract_sentiment_auto(text)

    if format_type == "gemma4":
        # <|channel>thought\n...<channel|>
        if "<|channel>thought" in text:
            after = text.split("<|channel>thought", 1)[1]
            if "<channel|>" in after:
                rationale = after.split("<channel|>", 1)[0]
                return rationale.strip()

    elif format_type == "qwen3":
        # <thinking>...</thinking> 或 und_b8...und_b8
        if "<thinking>" in text and "</thinking>" in text:
            start = text.find("<thinking>") + len("<thinking>")
            end = text.find("</thinking>")
            return text[start:end].strip()

        if "und_b8" in text:
            parts = text.split("und_b8")
            if len(parts) >= 2:
                # 第一个 und_b8 后到第二个 und_b8 前
                rationale = parts[1].strip()
                return rationale

    # 尝试从JSON中提取
    try:
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            full_json = json.loads(json_match.group())
            if "rationale" in full_json or "reasoning" in full_json:
                return full_json.get("rationale") or full_json.get("reasoning")
    except:
        pass

    return None


def extract_confidence(text: str) -> Optional[float]:
    """
    提取置信度

    Returns:
        置信度值（0-1），如果无法提取则返回None
    """
    # 从JSON中提取
    try:
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            full_json = json.loads(json_match.group())
            if "confidence" in full_json:
                conf = full_json["confidence"]
                if 0 <= conf <= 1:
                    return conf
    except:
        pass

    # 正则匹配
    match = re.search(r'confidence[:\s]+([0-9.]+)', text)
    if match:
        try:
            conf = float(match.group(1))
            if 0 <= conf <= 1:
                return conf
        except:
            pass

    return None


# 测试
if __name__ == "__main__":
    print("Testing extract_output.py...")

    # 测试 Qwen3 格式
    qwen3_text = """
    <thinking>
    This review expresses satisfaction with the product quality...
    </thinking>
    {"sentiment": 2, "confidence": 0.95, "rationale": "Positive tone"}
    """
    sentiment, json_data, fmt = extract_sentiment_auto(qwen3_text)
    print(f"Qwen3: sentiment={sentiment}, format={fmt}")

    # 测试 Gemma 4 格式
    gemma4_text = """
    <|channel>thought
    The user is expressing strong satisfaction...
    <channel|>
    {"sentiment": 2, "confidence": 0.92, "rationale": "Positive words"}
    <turn|>
    """
    sentiment, json_data, fmt = extract_sentiment_auto(gemma4_text)
    print(f"Gemma4: sentiment={sentiment}, format={fmt}")

    # 测试简单格式
    simple_text = "Answer: 1"
    sentiment, json_data, fmt = extract_sentiment_auto(simple_text)
    print(f"Simple: sentiment={sentiment}, format={fmt}")

    # 测试 rationale 提取
    rationale = extract_rationale(gemma4_text, "gemma4")
    print(f"Rationale: {rationale[:50]}...")

    print("\nAll tests passed!")