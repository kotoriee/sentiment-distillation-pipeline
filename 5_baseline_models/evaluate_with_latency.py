"""
带延迟统计的模型评估脚本
"""

import pandas as pd
import numpy as np
from pathlib import Path
import torch
import json
import time
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))

DATA_DIR = Path("experiments/denoising_setup")
OUTPUT_DIR = Path("experiments/denoising_setup/results_latency")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_data(split, use_original_label=False):
    """加载数据"""
    if split in ['train', 'val']:
        filepath = DATA_DIR / f"{split}_cleaned.csv"
        df = pd.read_csv(filepath)
        return df['text'].tolist(), df['label'].tolist()
    else:  # test
        filepath = DATA_DIR / "test_original.csv"
        df = pd.read_csv(filepath)
        if use_original_label:
            return df['text'].tolist(), df['label'].tolist()
        else:
            return df['text'].tolist(), df['cleaned_label'].tolist()


def evaluate_with_latency(model, data_loader, device, model_name="Model"):
    """
    评估模型并测量推理延迟
    返回: predictions, true_labels, latency_stats
    """
    model.eval()
    predictions = []
    true_labels = []
    latencies = []  # 存储每个batch的延迟

    print(f"\n[{model_name}] 评估中并测量延迟...")

    # 预热 (warmup)
    print("  Warmup...")
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= 5:  # 预热5个batch
                break
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

    # 正式测试
    print("  正式测试...")
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # 测量推理时间
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            # 计算延迟
            batch_latency = (end_time - start_time) * 1000  # 转换为毫秒
            latencies.append(batch_latency)

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # 计算延迟统计
    batch_size = data_loader.batch_size
    latency_stats = {
        'batch_size': batch_size,
        'num_batches': len(latencies),
        'total_samples': len(predictions),
        'total_time_ms': sum(latencies),
        'avg_batch_latency_ms': np.mean(latencies),
        'std_batch_latency_ms': np.std(latencies),
        'avg_sample_latency_ms': np.mean(latencies) / batch_size,
        'std_sample_latency_ms': np.std(latencies) / batch_size,
        'min_latency_ms': np.min(latencies),
        'max_latency_ms': np.max(latencies),
        'p50_latency_ms': np.percentile(latencies, 50),
        'p95_latency_ms': np.percentile(latencies, 95),
        'p99_latency_ms': np.percentile(latencies, 99),
        'throughput_samples_per_sec': len(predictions) / (sum(latencies) / 1000)
    }

    return np.array(predictions), np.array(true_labels), latency_stats


def evaluate_bert_with_latency():
    """评估BERT并测量延迟"""
    print("="*60)
    print("BERT 延迟测试")
    print("="*60)

    # 加载数据
    print("\n加载数据...")
    test_texts, test_labels = load_data('test', use_original_label=True)
    print(f"  Test: {len(test_texts)} 条")

    # 加载模型
    print("\n加载 BERT 模型...")
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # 加载训练好的权重
    model_path = DATA_DIR / "results" / "bert_best_model.pt"
    if not model_path.exists():
        model_path = DATA_DIR / "results_optimized" / "bert_optimized_best_model.pt"

    if model_path.exists():
        print(f"  加载权重: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("  警告: 未找到训练好的权重，使用预训练权重")

    model = model.to(device)

    # 测试不同batch size的延迟
    results = {}
    MAX_LEN = 256

    for batch_size in [1, 8, 16]:
        print(f"\n{'='*60}")
        print(f"Batch Size = {batch_size}")
        print('='*60)

        test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, MAX_LEN)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        test_pred, test_true, latency_stats = evaluate_with_latency(
            model, test_loader, device, f"BERT-bs{batch_size}"
        )

        # 计算性能指标
        test_acc = accuracy_score(test_true, test_pred)
        test_f1_macro = f1_score(test_true, test_pred, average='macro')

        print(f"\n  性能指标:")
        print(f"    Accuracy: {test_acc:.4f}")
        print(f"    Macro-F1: {test_f1_macro:.4f}")

        print(f"\n  延迟统计 (Batch Size={batch_size}):")
        print(f"    总样本数: {latency_stats['total_samples']}")
        print(f"    总时间: {latency_stats['total_time_ms']:.2f} ms")
        print(f"    平均批次延迟: {latency_stats['avg_batch_latency_ms']:.2f} ± {latency_stats['std_batch_latency_ms']:.2f} ms")
        print(f"    平均单样本延迟: {latency_stats['avg_sample_latency_ms']:.2f} ± {latency_stats['std_sample_latency_ms']:.2f} ms")
        print(f"    P50延迟: {latency_stats['p50_latency_ms']:.2f} ms")
        print(f"    P95延迟: {latency_stats['p95_latency_ms']:.2f} ms")
        print(f"    P99延迟: {latency_stats['p99_latency_ms']:.2f} ms")
        print(f"    吞吐量: {latency_stats['throughput_samples_per_sec']:.2f} samples/sec")

        results[f'batch_{batch_size}'] = {
            'accuracy': float(test_acc),
            'macro_f1': float(test_f1_macro),
            'latency': latency_stats
        }

    # 保存结果
    output = {
        'model': 'BERT (bert-base-uncased)',
        'device': str(device),
        'max_len': MAX_LEN,
        'results': results
    }

    with open(OUTPUT_DIR / "bert_latency_results.json", 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "="*60)
    print(f"BERT延迟测试完成! 结果保存: {OUTPUT_DIR / 'bert_latency_results.json'}")
    print("="*60)

    return output


def evaluate_svm_with_latency():
    """评估SVM并测量延迟"""
    from svm_classifier import SVMSentimentClassifier
    import pickle

    print("="*60)
    print("SVM 延迟测试")
    print("="*60)

    # 加载数据
    print("\n加载数据...")
    test_texts, test_labels = load_data('test', use_original_label=True)
    print(f"  Test: {len(test_texts)} 条")

    # 加载模型
    print("\n加载 SVM 模型...")
    model_path = DATA_DIR / "results" / "svm_model.pkl"

    if not model_path.exists():
        print(f"  错误: 模型文件不存在 {model_path}")
        return None

    with open(model_path, 'rb') as f:
        data = pickle.load(f)
        vectorizer = data['vectorizer']
        classifier = data['classifier']

    print(f"  模型加载完成")

    # 测试延迟
    print("\n测量推理延迟...")

    # 预热
    print("  Warmup...")
    _ = vectorizer.transform(test_texts[:10])
    _ = classifier.predict(_)

    # 正式测试 - 批量预测
    latencies = []
    batch_sizes = [1, 10, 100, len(test_texts)]

    for bs in batch_sizes:
        if bs > len(test_texts):
            continue

        print(f"\n  Batch Size = {bs}")

        # 分batch测试
        num_batches = max(1, len(test_texts) // bs)
        batch_latencies = []

        for i in range(num_batches):
            start_idx = i * bs
            end_idx = min((i + 1) * bs, len(test_texts))
            batch_texts = test_texts[start_idx:end_idx]

            start_time = time.perf_counter()

            # SVM推理流程：vectorizer -> classifier
            X = vectorizer.transform(batch_texts)
            _ = classifier.predict(X)

            end_time = time.perf_counter()

            batch_latency = (end_time - start_time) * 1000
            batch_latencies.append(batch_latency)

        avg_latency = np.mean(batch_latencies)
        std_latency = np.std(batch_latencies)
        avg_per_sample = avg_latency / bs

        print(f"    平均批次延迟: {avg_latency:.2f} ± {std_latency:.2f} ms")
        print(f"    平均单样本延迟: {avg_per_sample:.2f} ms")

    # 全量测试（用于计算准确率）
    print("\n全量预测...")
    start_time = time.perf_counter()
    X_test = vectorizer.transform(test_texts)
    test_pred = classifier.predict(X_test)
    end_time = time.perf_counter()

    total_time_ms = (end_time - start_time) * 1000
    avg_latency_ms = total_time_ms / len(test_texts)

    test_acc = accuracy_score(test_labels, test_pred)
    test_f1_macro = f1_score(test_labels, test_pred, average='macro')

    print(f"\n  性能指标:")
    print(f"    Accuracy: {test_acc:.4f}")
    print(f"    Macro-F1: {test_f1_macro:.4f}")

    print(f"\n  延迟统计:")
    print(f"    总样本数: {len(test_texts)}")
    print(f"    总时间: {total_time_ms:.2f} ms")
    print(f"    平均单样本延迟: {avg_latency_ms:.2f} ms")
    print(f"    吞吐量: {len(test_texts) / (total_time_ms / 1000):.2f} samples/sec")

    # 保存结果
    output = {
        'model': 'SVM (TF-IDF)',
        'total_samples': len(test_texts),
        'total_time_ms': float(total_time_ms),
        'avg_latency_ms': float(avg_latency_ms),
        'throughput_samples_per_sec': len(test_texts) / (total_time_ms / 1000),
        'accuracy': float(test_acc),
        'macro_f1': float(test_f1_macro)
    }

    with open(OUTPUT_DIR / "svm_latency_results.json", 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "="*60)
    print(f"SVM延迟测试完成! 结果保存: {OUTPUT_DIR / 'svm_latency_results.json'}")
    print("="*60)

    return output


def main():
    print("="*60)
    print("模型推理延迟测试")
    print("="*60)

    # 运行BERT延迟测试
    bert_results = evaluate_bert_with_latency()

    print("\n\n")

    # 运行SVM延迟测试
    svm_results = evaluate_svm_with_latency()

    # 生成对比报告
    print("\n\n" + "="*60)
    print("延迟对比总结")
    print("="*60)

    if bert_results and 'batch_1' in bert_results['results']:
        bert_latency = bert_results['results']['batch_1']['latency']['avg_sample_latency_ms']
        print(f"\nBERT (batch_size=1):")
        print(f"  单样本延迟: {bert_latency:.2f} ms")
        print(f"  吞吐量: {bert_results['results']['batch_1']['latency']['throughput_samples_per_sec']:.2f} samples/sec")

    if svm_results:
        print(f"\nSVM (TF-IDF):")
        print(f"  单样本延迟: {svm_results['avg_latency_ms']:.2f} ms")
        print(f"  吞吐量: {svm_results['throughput_samples_per_sec']:.2f} samples/sec")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
