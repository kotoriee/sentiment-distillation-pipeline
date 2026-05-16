#!/usr/bin/env python3
"""
汇总云端API评估结果
"""

import json
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score


def evaluate_jsonl(path):
    preds, trues = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            preds.append(data['pred'])
            trues.append(data['true'])

    acc = accuracy_score(trues, preds)
    f1_macro = f1_score(trues, preds, average='macro')
    f1_per_class = f1_score(trues, preds, average=None)

    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_class_0': f1_per_class[0],
        'f1_class_1': f1_per_class[1],
        'f1_class_2': f1_per_class[2],
        'samples': len(trues)
    }


def main():
    results = {}

    # Evaluate checkpoint jsonl files
    for jsonl_file in Path('4_evaluation').glob('*.jsonl'):
        metrics = evaluate_jsonl(jsonl_file)
        model_name = jsonl_file.stem.replace('_checkpoint', '')
        results[model_name] = metrics
        print(f'{model_name}: {metrics["samples"]} samples, acc={metrics["accuracy"]:.4f}, f1={metrics["f1_macro"]:.4f}, f1_neu={metrics["f1_class_1"]:.4f}')

    # Read nvidia api comparison
    nvidia_path = Path('4_evaluation/nvidia_api_results/nvidia_api_comparison.json')
    if nvidia_path.exists():
        with open(nvidia_path, 'r', encoding='utf-8') as f:
            nvidia_data = json.load(f)
            for model, metrics in nvidia_data.items():
                results[model] = metrics
                print(f'{model}: {metrics["valid"]} valid, acc={metrics["accuracy"]:.4f}, f1={metrics["f1_macro"]:.4f}, f1_neu={metrics["f1_class_1"]:.4f}')

    # Save summary
    output_path = Path('4_evaluation/api_results_summary.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\nSummary saved to: {output_path}')


if __name__ == '__main__':
    main()