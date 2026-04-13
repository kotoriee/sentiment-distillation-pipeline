"""
Gemma 4 数据预处理脚本 - Rationale Distillation

功能：
1. 数据质量检查
2. 划分 train/val/test (80/10/10)
3. 转换为 conversations 格式（Gemma 4 thinking channel）
4. 创建小批量测试集

Usage:
    python preprocess_rationale_data.py
"""

import json
from collections import Counter
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

# ============================================================
# 配置
# ============================================================
# 输入：复用 rationale_distillation_9k 的原始数据
INPUT_DIR = Path("d:/0321/workspaces/rationale_distillation_9k/data")
OUTPUT_DIR = Path("d:/0321/gamma4/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# System Prompt - 不手动添加 <|think|>，由 tokenizer 的 enable_thinking=True 自动处理
SYSTEM_PROMPT = """You are a professional e-commerce review sentiment analysis expert.

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


# ============================================================
# Step 1: 数据质量检查
# ============================================================
def check_data_quality(data: list) -> dict:
    """检查数据完整性，统计分布"""
    print("\n" + "="*60)
    print("Step 1: 数据质量检查")
    print("="*60)

    stats = {}

    # 总数据量
    stats["total"] = len(data)
    print(f"总数据量: {stats['total']}")

    # 标签分布
    label_dist = Counter(d["label"] for d in data)
    stats["labels"] = dict(label_dist)
    print(f"标签分布:")
    print(f"  负面(0): {label_dist[0]}")
    print(f"  中性(1): {label_dist[1]}")
    print(f"  正面(2): {label_dist[2]}")

    # 品类分布
    cat_dist = Counter(d.get("category", "unknown") for d in data)
    stats["categories"] = dict(cat_dist)
    print(f"品类分布: {dict(cat_dist)}")

    # 缺失字段检查
    missing_cot = sum(1 for d in data if not d.get("cot"))
    missing_soft = sum(1 for d in data if not d.get("soft_labels"))
    missing_label = sum(1 for d in data if d.get("label") is None)
    stats["missing"] = {"cot": missing_cot, "soft_labels": missing_soft, "label": missing_label}
    print(f"缺失字段:")
    print(f"  cot: {missing_cot}")
    print(f"  soft_labels: {missing_soft}")
    print(f"  label: {missing_label}")

    # CoT 平均长度
    cot_lengths = [len(d.get("cot", "")) for d in data]
    stats["cot_avg_length"] = sum(cot_lengths) / len(cot_lengths) if cot_lengths else 0
    print(f"CoT 平均长度: {stats['cot_avg_length']:.1f} 字符")

    # 检查数据质量
    is_valid = (missing_cot == 0 and missing_soft == 0 and missing_label == 0)
    stats["is_valid"] = is_valid
    print(f"\n数据质量: {'通过' if is_valid else '有缺失字段'}")

    return stats


# ============================================================
# Step 2: 划分 train/val/test（如果需要）
# ============================================================
def split_dataset(data: list) -> tuple:
    """按label分层划分 train/val/test"""
    print("\n" + "="*60)
    print("Step 2: 划分 train/val/test (80/10/10)")
    print("="*60)

    # 第一次划分: train(80%) vs temp(20%)
    train_data, temp_data = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=[d["label"] for d in data]
    )

    # 第二次划分: val(10%) vs test(10%)
    val_data, test_data = train_test_split(
        temp_data,
        test_size=0.5,
        random_state=42,
        stratify=[d["label"] for d in temp_data]
    )

    print(f"划分结果:")
    print(f"  train: {len(train_data)} 条")
    print(f"  val: {len(val_data)} 条")
    print(f"  test: {len(test_data)} 条")

    # 检查各划分的标签分布
    for name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        dist = Counter(d["label"] for d in split_data)
        print(f"  {name} 标签分布: Neg={dist[0]}, Neu={dist[1]}, Pos={dist[2]}")

    return train_data, val_data, test_data


# ============================================================
# Step 3: 转换为 conversations 格式（Gemma 4）
# ============================================================
def convert_to_conversations(record: dict) -> dict:
    """将单条数据转换为 conversations 格式（Gemma 4 thinking channel）"""
    user_content = f"Review: {record['text']}"

    # Gemma 4 格式：使用 <|channel>thought 标签
    # 格式: <|channel>thought\n{cot}<channel|>\n{JSON}
    cot_text = record.get("cot", "")
    assistant_output = {
        "sentiment": record["label"],
        "confidence": record.get("confidence", max(record.get("soft_labels", [0.33, 0.33, 0.34]))),
        "rationale": cot_text
    }
    json_output = json.dumps(assistant_output, ensure_ascii=False)
    assistant_content = f"<|channel>thought\n{cot_text}<channel|>\n{json_output}"

    return {
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ],
        "text": record["text"],
        "label": record["label"],
        "soft_labels": record.get("soft_labels", [0.33, 0.33, 0.34]),
        "category": record.get("category", "unknown")
    }


def convert_splits(train_data: list, val_data: list, test_data: list) -> tuple:
    """转换三个数据集"""
    print("\n" + "="*60)
    print("Step 3: 转换为 conversations 格式（Gemma 4）")
    print("="*60)

    train_conv = [convert_to_conversations(d) for d in train_data]
    val_conv = [convert_to_conversations(d) for d in val_data]
    test_conv = [convert_to_conversations(d) for d in test_data]

    print(f"转换完成:")
    print(f"  train_conversations: {len(train_conv)} 条")
    print(f"  val_conversations: {len(val_conv)} 条")
    print(f"  test_conversations: {len(test_conv)} 条")

    return train_conv, val_conv, test_conv


# ============================================================
# Step 4: 创建小批量测试集
# ============================================================
def create_small_batch(train_conv: list, size: int = 700) -> list:
    """创建小批量训练集用于快速测试"""
    print("\n" + "="*60)
    print(f"Step 4: 创建小批量测试集 ({size} 条)")
    print("="*60)

    # 按label分层采样
    neg_samples = [d for d in train_conv if d["label"] == 0]
    neu_samples = [d for d in train_conv if d["label"] == 1]
    pos_samples = [d for d in train_conv if d["label"] == 2]

    per_class = size // 3
    remainder = size % 3

    small_batch = []
    small_batch.extend(random.sample(neg_samples, min(per_class + (remainder > 0), len(neg_samples))))
    small_batch.extend(random.sample(neu_samples, min(per_class + (remainder > 1), len(neu_samples))))
    small_batch.extend(random.sample(pos_samples, min(per_class, len(pos_samples))))

    # 检查分布
    dist = Counter(d["label"] for d in small_batch)
    print(f"小批量测试集:")
    print(f"  总数: {len(small_batch)} 条")
    print(f"  标签分布: Neg={dist[0]}, Neu={dist[1]}, Pos={dist[2]}")

    return small_batch


# ============================================================
# Step 5: 格式验证
# ============================================================
def validate_format(data: list, name: str) -> bool:
    """验证 conversations 格式正确性（Gemma 4 格式）"""
    print(f"\n{name} 格式验证:")

    errors = []
    for i, d in enumerate(data[:5]):  # 检查前5条
        convs = d.get("conversations", [])

        # 检查结构
        if len(convs) != 3:
            errors.append(f"  [{i}] conversations 长度 != 3")
            continue

        if convs[0]["role"] != "system":
            errors.append(f"  [{i}] 第一条 role != system")
        if convs[1]["role"] != "user":
            errors.append(f"  [{i}] 第二条 role != user")
        if convs[2]["role"] != "assistant":
            errors.append(f"  [{i}] 第三条 role != assistant")

        # 注意：<|think|> 由 tokenizer 的 enable_thinking=True 自动添加，不在数据中存储

        # 检查 assistant content 格式
        assistant_content = convs[2]["content"]
        if "<|channel>thought" not in assistant_content:
            errors.append(f"  [{i}] assistant 缺少 <|channel>thought")
        if "<channel|>" not in assistant_content:
            errors.append(f"  [{i}] assistant 缺少 <channel|>")

        # 尝试解析 JSON 部分
        try:
            json_part = assistant_content.split("<channel|>", 1)[1]
            assistant_json = json.loads(json_part)
            if "sentiment" not in assistant_json:
                errors.append(f"  [{i}] JSON 缺少 sentiment")
        except (IndexError, json.JSONDecodeError):
            errors.append(f"  [{i}] JSON 解析失败")

    if errors:
        print("发现错误:")
        for e in errors:
            print(e)
        return False
    else:
        print("格式验证通过 [OK]")
        return True


def show_sample(data: list, idx: int = 0):
    """显示单条样本"""
    print(f"\n样本 [{idx}]:")
    d = data[idx]
    print(f"  text: {d['text'][:100]}...")
    print(f"  label: {d['label']}")
    print(f"  soft_labels: {d['soft_labels']}")
    print(f"  conversations:")
    for conv in d["conversations"]:
        content_preview = conv["content"][:100] + "..." if len(conv["content"]) > 100 else conv["content"]
        print(f"    [{conv['role']}]: {content_preview}")


# ============================================================
# Main
# ============================================================
def main():
    print("="*60)
    print("Gemma 4 数据预处理 - Rationale Distillation")
    print("="*60)

    # 读取数据（复用 rationale_distillation_9k 的原始数据）
    train_path = INPUT_DIR / "train.json"
    val_path = INPUT_DIR / "val.json"
    test_path = INPUT_DIR / "test.json"

    print(f"\n读取数据:")
    print(f"  train: {train_path}")
    print(f"  val: {val_path}")
    print(f"  test: {test_path}")

    # 加载三个数据集
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"数据量: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    # Step 1: 数据质量检查
    all_data = train_data + val_data + test_data
    stats = check_data_quality(all_data)

    # Step 3: 转换格式（Gemma 4）
    train_conv, val_conv, test_conv = convert_splits(train_data, val_data, test_data)

    # 保存 conversations 格式
    print("\n保存 conversations 格式数据...")
    for name, split_data in [("train", train_conv), ("val", val_conv), ("test", test_conv)]:
        output_path = OUTPUT_DIR / f"{name}_conversations.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"  {output_path}: {len(split_data)} 条")

    # Step 4: 创建小批量测试集
    small_batch = create_small_batch(train_conv, size=700)
    small_batch_path = OUTPUT_DIR / "train_700.json"
    with open(small_batch_path, "w", encoding="utf-8") as f:
        json.dump(small_batch, f, ensure_ascii=False, indent=2)
    print(f"  {small_batch_path}: {len(small_batch)} 条")

    # Step 5: 格式验证
    print("\n" + "="*60)
    print("Step 5: 格式验证")
    print("="*60)

    validate_format(train_conv, "train_conversations")
    validate_format(val_conv, "val_conversations")
    validate_format(test_conv, "test_conversations")

    # 显示样本
    print("\n" + "="*60)
    print("样本预览")
    print("="*60)
    show_sample(train_conv, idx=0)

    # 总结
    print("\n" + "="*60)
    print("预处理完成！")
    print("="*60)
    print(f"输出文件:")
    print(f"  {OUTPUT_DIR / 'train_conversations.json'}")
    print(f"  {OUTPUT_DIR / 'val_conversations.json'}")
    print(f"  {OUTPUT_DIR / 'test_conversations.json'}")
    print(f"  {OUTPUT_DIR / 'train_700.json'} (小批量测试集)")
    print("\n下一步: 运行训练脚本")
    print("  python train_soft_v2.py --data data/train_700.json --test")


if __name__ == "__main__":
    main()