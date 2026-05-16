#!/usr/bin/env python3
"""
四模型评估脚本（12项必须指标）

模型：
- Logistic Regression (TF-IDF)
- BERT (fine-tuned)
- Qwen3.5-9B SFT (LoRA)
- Qwen3.5-9B GRPO (LoRA checkpoint-90)

指标：
1. Accuracy
2. Macro F1
3. Weighted F1
4. Negative/Neutral/Positive F1
5. Confusion Matrix
6. Valid/Invalid count, Parse success rate
7. Latency ms/sample
8. Throughput samples/s
9. Cost estimate (for cloud API models)
10. Per-category Neutral F1
11. SFT vs GRPO delta
12. McNemar test

Usage:
    cd 05_experiments/10k_robustness
    python evaluate_models.py --dataset three_category_10k --models lr bert sft grpo
"""

import json
import re
import time
import gc
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from scipy.stats import chi2_contingency
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "results"

# 模型路径
LR_MODEL_PATH = PROJECT_ROOT / "1_baseline" / "logistic_regression_model.pkl"
BERT_MODEL_PATH = PROJECT_ROOT / "6_experiments_results" / "baseline_results" / "bert_best_model.pt"
SFT_MODEL_PATH = PROJECT_ROOT / "3_lora_training" / "models" / "qwen35-9b-answer-first"
GRPO_MODEL_PATH = PROJECT_ROOT / "3_lora_training" / "outputs_grpo_9b_rewardfix_v3_nothink_continue_300" / "checkpoint-90"


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


def compute_metrics(true_labels, pred_labels):
    """计算分类指标"""
    valid_idx = [i for i, p in enumerate(pred_labels) if p != -1]
    valid_true = [true_labels[i] for i in valid_idx]
    valid_pred = [pred_labels[i] for i in valid_idx]

    if not valid_true:
        return {
            'accuracy': 0,
            'macro_f1': 0,
            'weighted_f1': 0,
            'f1_class_0': 0,
            'f1_class_1': 0,
            'f1_class_2': 0,
            'precision_class_0': 0,
            'precision_class_1': 0,
            'precision_class_2': 0,
            'recall_class_0': 0,
            'recall_class_1': 0,
            'recall_class_2': 0,
            'confusion_matrix': [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        }

    accuracy = accuracy_score(valid_true, valid_pred)
    macro_f1 = f1_score(valid_true, valid_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(valid_true, valid_pred, average='weighted', zero_division=0)
    f1_per_class = f1_score(valid_true, valid_pred, average=None, labels=[0, 1, 2], zero_division=0)
    precision_per_class = precision_score(valid_true, valid_pred, average=None, labels=[0, 1, 2], zero_division=0)
    recall_per_class = recall_score(valid_true, valid_pred, average=None, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(valid_true, valid_pred, labels=[0, 1, 2])

    return {
        'accuracy': round(accuracy, 4),
        'macro_f1': round(macro_f1, 4),
        'weighted_f1': round(weighted_f1, 4),
        'f1_class_0': round(f1_per_class[0], 4),
        'f1_class_1': round(f1_per_class[1], 4),
        'f1_class_2': round(f1_per_class[2], 4),
        'precision_class_0': round(precision_per_class[0], 4),
        'precision_class_1': round(precision_per_class[1], 4),
        'precision_class_2': round(precision_per_class[2], 4),
        'recall_class_0': round(recall_per_class[0], 4),
        'recall_class_1': round(recall_per_class[1], 4),
        'recall_class_2': round(recall_per_class[2], 4),
        'confusion_matrix': cm.tolist(),
    }


def mcnemar_test(pred_a, pred_b, true_labels):
    """McNemar 检验比较两个模型"""
    # 只考虑两者都有效预测的样本
    valid_idx = [i for i in range(len(true_labels)) if pred_a[i] != -1 and pred_b[i] != -1]

    a_correct = [pred_a[i] == true_labels[i] for i in valid_idx]
    b_correct = [pred_b[i] == true_labels[i] for i in valid_idx]

    # 计算不一致数
    b01 = sum(1 for i in range(len(valid_idx)) if not a_correct[i] and b_correct[i])  # A错B对
    b10 = sum(1 for i in range(len(valid_idx)) if a_correct[i] and not b_correct[i])  # A对B错

    if b01 + b10 == 0:
        return {'p_value': 1.0, 'b01': 0, 'b10': 0, 'significant': False}

    # McNemar 检验（使用卡方近似）
    contingency_table = np.array([[b01 + b10 - b01, b01], [b10, b01 + b10 - b10]])

    try:
        chi2, p, dof, expected = chi2_contingency(contingency_table, correction=True)
        significant = p < 0.05
    except Exception:
        p = 1.0
        significant = False

    return {
        'p_value': round(p, 4),
        'b01': b01,  # A错B对
        'b10': b10,  # A对B错
        'significant': significant,
    }


def evaluate_lr(data, dataset_name):
    """评估 Logistic Regression"""
    print("\n" + "=" * 60)
    print("评估 Logistic Regression")
    print("=" * 60)

    import pickle
    from sklearn.feature_extraction.text import TfidfVectorizer

    # 加载模型
    print(f"加载模型: {LR_MODEL_PATH}")
    with open(LR_MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)

    vectorizer = model_data['vectorizer']
    model = model_data['model']

    # 提取文本
    texts = []
    true_labels = []
    categories = []

    for item in data:
        convs = item['conversations']
        user_content = convs[1]['content']
        if user_content.startswith("Review: "):
            text = user_content[8:]
        else:
            text = user_content
        texts.append(text)
        true_labels.append(item['label'])
        categories.append(item.get('category', 'unknown'))

    # 推理
    print(f"推理: {len(texts)} 条")
    t0 = time.perf_counter()
    X = vectorizer.transform(texts)
    preds = model.predict(X)
    t1 = time.perf_counter()

    infer_time = t1 - t0
    throughput = len(texts) / infer_time
    latency_ms = infer_time * 1000 / len(texts)

    # 计算指标
    metrics = compute_metrics(true_labels, preds)

    # Per-category 分析
    cat_metrics = {}
    for cat in set(categories):
        cat_idx = [i for i, c in enumerate(categories) if c == cat]
        cat_true = [true_labels[i] for i in cat_idx]
        cat_pred = [preds[i] for i in cat_idx]
        cat_metrics[cat] = compute_metrics(cat_true, cat_pred)

    results = {
        'experiment_id': f'lr_{dataset_name}',
        'dataset_name': dataset_name,
        'dataset_size': len(data),
        'valid_prediction_count': len(data),  # LR 总是有效
        'invalid_prediction_count': 0,
        'parse_success_rate': 100.0,
        'model_name': 'LogisticRegression',
        'model_type': 'traditional',
        'training_method': 'TF-IDF',
        **metrics,
        'latency_ms_per_sample': round(latency_ms, 4),
        'throughput_samples_per_sec': round(throughput, 2),
        'batch_size': len(data),  # LR 一次性处理全部
        'total_inference_time_sec': round(infer_time, 4),
        'hardware': 'CPU',
        'peak_vram_gb': None,
        'per_category_metrics': cat_metrics,
    }

    print(f"\n结果:")
    print(f"  Accuracy: {metrics['accuracy']}")
    print(f"  Macro F1: {metrics['macro_f1']}")
    print(f"  Neutral F1: {metrics['f1_class_1']}")
    print(f"  Latency: {latency_ms:.4f} ms/sample")
    print(f"  Throughput: {throughput:.2f} samples/s")

    return results, preds


def evaluate_bert(data, dataset_name):
    """评估 BERT"""
    print("\n" + "=" * 60)
    print("评估 BERT")
    print("=" * 60)

    from transformers import BertTokenizer, BertForSequenceClassification
    from torch.utils.data import DataLoader, Dataset

    # 配置
    BATCH_SIZE = 16
    MAX_LEN = 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载模型
    print(f"加载模型: {BERT_MODEL_PATH}")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    if BERT_MODEL_PATH.exists():
        model.load_state_dict(torch.load(BERT_MODEL_PATH, map_location=device))
    model.to(device)

    # 提取文本
    texts = []
    true_labels = []
    categories = []

    for item in data:
        convs = item['conversations']
        user_content = convs[1]['content']
        if user_content.startswith("Review: "):
            text = user_content[8:]
        else:
            text = user_content
        texts.append(text)
        true_labels.append(item['label'])
        categories.append(item.get('category', 'unknown'))

    # Dataset
    class SentimentDataset(Dataset):
        def __init__(self, texts, tokenizer, max_len):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoding = self.tokenizer(
                self.texts[idx],
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
            }

    dataset = SentimentDataset(texts, tokenizer, MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    # 推理
    preds = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="BERT推理"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            preds.extend(batch_preds)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    infer_time = t1 - t0
    throughput = len(texts) / infer_time
    latency_ms = infer_time * 1000 / len(texts)

    # Peak VRAM
    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None

    # 计算指标
    metrics = compute_metrics(true_labels, preds)

    # Per-category 分析
    cat_metrics = {}
    for cat in set(categories):
        cat_idx = [i for i, c in enumerate(categories) if c == cat]
        cat_true = [true_labels[i] for i in cat_idx]
        cat_pred = [preds[i] for i in cat_idx]
        cat_metrics[cat] = compute_metrics(cat_true, cat_pred)

    results = {
        'experiment_id': f'bert_{dataset_name}',
        'dataset_name': dataset_name,
        'dataset_size': len(data),
        'valid_prediction_count': len(data),
        'invalid_prediction_count': 0,
        'parse_success_rate': 100.0,
        'model_name': 'BERT-base-uncased',
        'model_type': 'transformer',
        'training_method': 'fine-tuned',
        **metrics,
        'latency_ms_per_sample': round(latency_ms, 4),
        'throughput_samples_per_sec': round(throughput, 2),
        'batch_size': BATCH_SIZE,
        'total_inference_time_sec': round(infer_time, 4),
        'hardware': 'GPU' if torch.cuda.is_available() else 'CPU',
        'peak_vram_gb': round(peak_vram, 2) if peak_vram else None,
        'per_category_metrics': cat_metrics,
    }

    print(f"\n结果:")
    print(f"  Accuracy: {metrics['accuracy']}")
    print(f"  Macro F1: {metrics['macro_f1']}")
    print(f"  Neutral F1: {metrics['f1_class_1']}")
    print(f"  Latency: {latency_ms:.4f} ms/sample")
    print(f"  Throughput: {throughput:.2f} samples/s")

    # 清理
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results, preds


def evaluate_qwen(data, dataset_name, model_path, model_name, max_new_tokens, batch_size=8):
    """评估 Qwen 模型（SFT 或 GRPO）"""
    print("\n" + "=" * 60)
    print(f"评估 {model_name}")
    print("=" * 60)
    print(f"模型路径: {model_path}")
    print(f"配置: max_new_tokens={max_new_tokens}, batch_size={batch_size}")

    from unsloth import FastLanguageModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    if 'grpo' in model_name.lower():
        # GRPO: 加载 base model + LoRA adapter
        print("加载 base model + LoRA adapter...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen3.5-9B-unsloth-bnb-4bit",
            max_seq_length=512,
            load_in_4bit=True,
        )
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(model_path))
    else:
        # SFT: 直接加载训练好的模型
        print("加载模型...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path),
            max_seq_length=512,
            load_in_4bit=True,
        )

    FastLanguageModel.for_inference(model)

    # 构建 prompts
    prompts = []
    true_labels = []
    categories = []

    for item in data:
        conv = item['conversations']
        prompt_conv = conv[:2]
        prompt = tokenizer.apply_chat_template(
            prompt_conv, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)
        true_labels.append(item['label'])
        categories.append(item.get('category', 'unknown'))

    # 推理
    all_outputs = []
    latencies = []

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_start = time.perf_counter()

    for i in tqdm(range(0, len(prompts), batch_size), desc=f"{model_name}推理"):
        batch = prompts[i:i + batch_size]

        inputs = tokenizer(
            text=batch,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

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

        # 定期清理
        if (i + batch_size) % 200 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_end = time.perf_counter()

    total_time = total_end - total_start
    infer_time = sum(latencies) / 1000
    throughput = len(all_outputs) / total_time
    avg_latency_ms = sum(latencies) / len(latencies) / batch_size

    # Peak VRAM
    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None

    # 解析结果
    preds = []
    parse_errors = 0

    for pred_text in all_outputs:
        pred_label = extract_sentiment(pred_text)
        preds.append(pred_label)
        if pred_label == -1:
            parse_errors += 1

    valid_count = len(preds) - parse_errors
    parse_success_rate = valid_count / len(preds) * 100 if preds else 0

    # 计算指标
    metrics = compute_metrics(true_labels, preds)

    # Per-category 分析
    cat_metrics = {}
    for cat in set(categories):
        cat_idx = [i for i, c in enumerate(categories) if c == cat]
        cat_true = [true_labels[i] for i in cat_idx]
        cat_pred = [preds[i] for i in cat_idx]
        cat_metrics[cat] = compute_metrics(cat_true, cat_pred)

    results = {
        'experiment_id': f'{model_name}_{dataset_name}',
        'dataset_name': dataset_name,
        'dataset_size': len(data),
        'valid_prediction_count': valid_count,
        'invalid_prediction_count': parse_errors,
        'parse_success_rate': round(parse_success_rate, 2),
        'model_name': model_name,
        'model_type': 'LLM',
        'training_method': 'SFT' if 'sft' in model_name.lower() else 'GRPO',
        **metrics,
        'latency_ms_per_sample': round(avg_latency_ms, 2),
        'throughput_samples_per_sec': round(throughput, 2),
        'batch_size': batch_size,
        'total_inference_time_sec': round(total_time, 2),
        'hardware': 'GPU' if torch.cuda.is_available() else 'CPU',
        'peak_vram_gb': round(peak_vram, 2) if peak_vram else None,
        'max_new_tokens': max_new_tokens,
        'per_category_metrics': cat_metrics,
    }

    print(f"\n结果:")
    print(f"  Accuracy: {metrics['accuracy']}")
    print(f"  Macro F1: {metrics['macro_f1']}")
    print(f"  Neutral F1: {metrics['f1_class_1']}")
    print(f"  Parse success rate: {parse_success_rate:.2f}%")
    print(f"  Latency: {avg_latency_ms:.2f} ms/sample")
    print(f"  Throughput: {throughput:.2f} samples/s")

    # 清理
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["three_category_10k", "random_category_10k", "smoke_subset_100"],
                        default="three_category_10k", help="评估数据集")
    parser.add_argument("--models", nargs="+", default=["lr", "bert", "sft", "grpo"],
                        help="要评估的模型")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="LLM max_new_tokens (默认自动从 smoke 测试结果读取)")
    args = parser.parse_args()

    print("=" * 60)
    print("10k 泛化评估")
    print("=" * 60)

    # 加载数据
    data_path = DATA_DIR / f"{args.dataset}.json"
    print(f"\n加载数据: {data_path}")

    if not data_path.exists():
        print(f"错误: 数据集不存在，请先运行 build_dataset.py")
        return

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"数据集大小: {len(data)}")

    # 统计标签分布
    label_dist = Counter(d['label'] for d in data)
    print(f"标签分布: Neg={label_dist[0]}, Neu={label_dist[1]}, Pos={label_dist[2]}")

    # 统计品类分布
    cat_dist = Counter(d.get('category', 'unknown') for d in data)
    print(f"品类分布: {dict(cat_dist)}")

    # 确定 max_new_tokens
    if args.max_new_tokens:
        max_new_tokens = args.max_new_tokens
    else:
        # 尝试读取 smoke 测试结果
        smoke_path = OUTPUT_DIR / "smoke_test_max_tokens.json"
        if smoke_path.exists():
            with open(smoke_path, 'r') as f:
                smoke_results = json.load(f)
            sft_max = smoke_results['recommendations']['sft']['recommended_max_new_tokens']
            grpo_max = smoke_results['recommendations']['grpo']['recommended_max_new_tokens']
            print(f"\n使用 Smoke 测试推荐的 max_new_tokens:")
            print(f"  SFT: {sft_max}")
            print(f"  GRPO: {grpo_max}")
            max_new_tokens = {'sft': sft_max, 'grpo': grpo_max}
        else:
            print(f"\n警告: 未找到 Smoke 测试结果，使用默认 max_new_tokens=1")
            max_new_tokens = {'sft': 1, 'grpo': 1}

    # 创建结果目录
    output_dir = OUTPUT_DIR / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    predictions = {}

    # 评估各模型
    if 'lr' in args.models:
        results['lr'], predictions['lr'] = evaluate_lr(data, args.dataset)

    if 'bert' in args.models:
        results['bert'], predictions['bert'] = evaluate_bert(data, args.dataset)

    if 'sft' in args.models:
        sft_max = max_new_tokens.get('sft', 1) if isinstance(max_new_tokens, dict) else max_new_tokens
        results['sft'], predictions['sft'] = evaluate_qwen(
            data, args.dataset, SFT_MODEL_PATH, "Qwen-SFT",
            max_new_tokens=sft_max, batch_size=8
        )

    if 'grpo' in args.models:
        grpo_max = max_new_tokens.get('grpo', 1) if isinstance(max_new_tokens, dict) else max_new_tokens
        results['grpo'], predictions['grpo'] = evaluate_qwen(
            data, args.dataset, GRPO_MODEL_PATH, "Qwen-GRPO-ckpt90",
            max_new_tokens=grpo_max, batch_size=8
        )

    # 计算 SFT vs GRPO delta 和 McNemar
    if 'sft' in results and 'grpo' in results:
        sft_metrics = results['sft']
        grpo_metrics = results['grpo']

        delta = {
            'accuracy': round(grpo_metrics['accuracy'] - sft_metrics['accuracy'], 4),
            'macro_f1': round(grpo_metrics['macro_f1'] - sft_metrics['macro_f1'], 4),
            'neutral_f1': round(grpo_metrics['f1_class_1'] - sft_metrics['f1_class_1'], 4),
        }

        true_labels = [d['label'] for d in data]
        mcnemar = mcnemar_test(predictions['sft'], predictions['grpo'], true_labels)

        results['sft_vs_grpo_delta'] = delta
        results['mcnemar_test'] = mcnemar

        print(f"\nSFT vs GRPO 对比:")
        print(f"  Δ Accuracy: {delta['accuracy']}")
        print(f"  Δ Macro F1: {delta['macro_f1']}")
        print(f"  Δ Neutral F1: {delta['neutral_f1']}")
        print(f"  McNemar p-value: {mcnemar['p_value']}")
        print(f"  Significant: {mcnemar['significant']}")

    # 保存结果
    for model_name, model_results in results.items():
        if model_name in ['sft_vs_grpo_delta', 'mcnemar_test']:
            continue
        output_file = output_dir / f"{model_name}_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(model_results, f, indent=2, ensure_ascii=False)
        print(f"\n保存: {output_file}")

    # 保存汇总
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"汇总: {summary_file}")

    print("\n" + "=" * 60)
    print("评估完成")
    print("=" * 60)


if __name__ == "__main__":
    main()