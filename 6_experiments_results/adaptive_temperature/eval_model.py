"""
动态温度模型评估脚本
对比固定温度基线和自适应温度模型
"""

import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path


def load_model(base_model_path, adapter_path=None):
    """加载基础模型和可选的LoRA适配器"""
    print(f"加载模型: {base_model_path}")

    # 配置4-bit量化
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    if adapter_path:
        print(f"加载适配器: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    return model, tokenizer


def load_val_data(data_path):
    """加载验证数据并标准化字段名"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 适配不同的数据格式
    if isinstance(data, dict) and 'data' in data:
        data = data['data']

    # 标准化字段名
    for item in data:
        # 标签字段: 'label', 'labels', 或 'output'
        if 'label' not in item:
            if 'labels' in item:
                item['label'] = item['labels']
            elif 'output' in item:
                item['label'] = item['output']
        # 文本字段: 'text' 或 'input'
        if 'text' not in item and 'input' in item:
            item['text'] = item['input']

    return data


def evaluate_model(model, tokenizer, data, max_samples=None):
    """评估模型性能"""
    model.eval()

    if max_samples:
        data = data[:max_samples]

    predictions = []
    labels = []
    confidences = []

    print(f"\n评估 {len(data)} 个样本...")

    with torch.no_grad():
        for item in tqdm(data):
            # 构建prompt - 使用标准化后的字段
            instruction = item.get('instruction',
                "Classify the sentiment of this e-commerce review. Output only: 0 (negative), 1 (neutral), or 2 (positive).")
            text = item.get('text', '')  # 数据加载时已标准化
            prompt = f"{instruction}\n{text}\nSentiment:"

            # 编码
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(model.device)

            # 推理
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :3]  # 取最后token的前3个logits

            # 计算概率
            probs = F.softmax(logits, dim=-1)
            pred = probs.argmax(dim=-1).item()
            conf = probs.max().item()

            predictions.append(pred)
            # 标准化后的数据一定有 'label' 字段
            labels.append(int(item['label']))
            confidences.append(conf)

    # 计算指标
    predictions = np.array(predictions)
    labels = np.array(labels)
    confidences = np.array(confidences)

    accuracy = (predictions == labels).mean()
    avg_confidence = confidences.mean()

    print(f"\n{'='*60}")
    print("评估结果")
    print(f"{'='*60}")
    print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"平均置信度: {avg_confidence:.4f}")
    print(f"置信度-准确率差距: {abs(avg_confidence - accuracy):.4f}")

    print(f"\n分类报告:")
    print(classification_report(labels, predictions,
                               target_names=['Negative', 'Neutral', 'Positive'],
                               digits=4))

    print(f"混淆矩阵:")
    cm = confusion_matrix(labels, predictions)
    print(cm)

    # 计算ECE
    ece = compute_ece(confidences, predictions == labels)
    print(f"\nECE (期望校准误差): {ece:.4f}")

    return {
        'accuracy': accuracy,
        'avg_confidence': avg_confidence,
        'ece': ece,
        'predictions': predictions.tolist(),
        'labels': labels.tolist()
    }


def compute_ece(confidences, accuracies, n_bins=10):
    """计算期望校准误差"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)

        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            bin_weight = mask.sum() / len(confidences)
            ece += bin_weight * abs(bin_acc - bin_conf)

    return ece


def compare_with_baseline(adaptive_results, baseline_accuracy=0.8650):
    """对比基线结果"""
    print(f"\n{'='*60}")
    print("与基线对比")
    print(f"{'='*60}")
    print(f"基线准确率 (固定T=2.0): {baseline_accuracy:.4f}")
    print(f"动态温度准确率: {adaptive_results['accuracy']:.4f}")

    diff = adaptive_results['accuracy'] - baseline_accuracy
    if diff > 0:
        print(f"提升: +{diff:.4f} (+{diff*100:.2f}%)")
    else:
        print(f"下降: {diff:.4f} ({diff*100:.2f}%)")

    # 检查是否达到目标
    if adaptive_results['accuracy'] >= 0.875:
        print("✓ 达到目标 (≥87.5%)")
    else:
        print(f"✗ 未达到目标 (差距: {0.875 - adaptive_results['accuracy']:.4f})")


def main():
    parser = argparse.ArgumentParser(description='评估动态温度模型')
    parser.add_argument('--model_path', required=True, help='模型路径 (适配器目录)')
    parser.add_argument('--base_model', default='unsloth/Qwen3-4B-unsloth-bnb-4bit',
                       help='基础模型')
    parser.add_argument('--val_data', required=True, help='验证数据路径')
    parser.add_argument('--max_samples', type=int, default=None, help='最大评估样本数')
    parser.add_argument('--output', default='eval_results.json', help='输出结果文件')
    args = parser.parse_args()

    print("="*60)
    print("动态温度模型评估")
    print("="*60)

    # 加载模型
    model, tokenizer = load_model(args.base_model, args.model_path)

    # 加载数据
    data = load_val_data(args.val_data)

    # 评估
    results = evaluate_model(model, tokenizer, data, args.max_samples)

    # 对比基线
    compare_with_baseline(results)

    # 保存结果
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {args.output}")


if __name__ == '__main__':
    main()
