"""
优化版 SVM 训练 - 使用 Sentence-BERT 替代 TF-IDF
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import pickle

sys.path.insert(0, str(Path(__file__).parent))

DATA_DIR = Path("experiments/denoising_setup")
OUTPUT_DIR = Path("experiments/denoising_setup/results_optimized")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class SVMSentimentClassifierOptimized:
    """使用 Sentence-BERT 的 SVM 分类器"""

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Args:
            model_name: Sentence-BERT 模型名称
                - 'all-MiniLM-L6-v2': 轻量版，384维 (推荐，平衡速度和效果)
                - 'all-mpnet-base-v2': 效果更好，768维 (更慢)
        """
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        print(f"Loading Sentence-BERT model: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        self.classifier = None
        print(f"Model loaded. Embedding dimension: {self.encoder.get_sentence_embedding_dimension()}")

    def get_embeddings(self, texts, batch_size=32, show_progress=True):
        """获取文本的 Sentence-BERT embedding"""
        if show_progress:
            print(f"  Encoding {len(texts)} texts with {self.model_name}...")

        embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings

    def fit(self, texts, labels, C=1.0, kernel='rbf', gamma='scale', class_weight='balanced'):
        """训练 SVM 分类器"""
        # 获取 Sentence-BERT embeddings
        X = self.get_embeddings(texts)

        print(f"\nTraining SVM (C={C}, kernel={kernel}, class_weight={class_weight})...")
        self.classifier = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            class_weight=class_weight,
            probability=True,
            random_state=42
        )
        self.classifier.fit(X, labels)
        print("Training completed")

        # 返回训练集上的表现
        train_pred = self.classifier.predict(X)
        train_acc = accuracy_score(labels, train_pred)
        train_f1 = f1_score(labels, train_pred, average='macro')
        print(f"  Train Accuracy: {train_acc:.4f}, Train Macro-F1: {train_f1:.4f}")

        return self

    def predict(self, texts):
        """预测标签"""
        X = self.get_embeddings(texts, show_progress=False)
        return self.classifier.predict(X)

    def predict_proba(self, texts):
        """预测概率"""
        X = self.get_embeddings(texts, show_progress=False)
        return self.classifier.predict_proba(X)

    def save(self, filepath):
        """保存模型"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'encoder_name': self.model_name,
                'classifier': self.classifier
            }, f)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """加载模型"""
        from sentence_transformers import SentenceTransformer

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.model_name = data['encoder_name']
        self.encoder = SentenceTransformer(self.model_name)
        self.classifier = data['classifier']
        print(f"Model loaded from {filepath}")
        return self


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


def hyperparameter_search(train_texts, train_labels, val_texts, val_labels):
    """超参数搜索"""
    print("\n" + "="*60)
    print("超参数搜索 (Sent-BERT + SVM)")
    print("="*60)

    from sentence_transformers import SentenceTransformer

    # 先编码一次，重复使用
    print("Encoding texts with Sentence-BERT...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    X_train = encoder.encode(train_texts, show_progress_bar=True)
    X_val = encoder.encode(val_texts, show_progress_bar=True)

    # 搜索空间
    C_values = [0.1, 1.0, 10.0]
    kernel_values = ['rbf', 'linear']
    gamma_values = ['scale', 'auto', 0.001, 0.01]

    best_f1 = 0
    best_params = {}

    for C in C_values:
        for kernel in kernel_values:
            for gamma in gamma_values:
                clf = SVC(
                    C=C,
                    kernel=kernel,
                    gamma=gamma,
                    class_weight='balanced',
                    random_state=42
                )
                clf.fit(X_train, train_labels)
                val_pred = clf.predict(X_val)
                val_f1 = f1_score(val_labels, val_pred, average='macro')

                print(f"  C={C:4.1f}, kernel={kernel:7s}, gamma={str(gamma):6s} -> Val F1: {val_f1:.4f}")

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_params = {'C': C, 'kernel': kernel, 'gamma': gamma}

    print(f"\n最佳参数: {best_params}, Val F1: {best_f1:.4f}")
    return best_params, encoder


def main():
    print("="*60)
    print("SVM 优化版 (Sentence-BERT) - 数据降噪组设置")
    print("训练: 清洗标签 | 测试: 原始硬标签")
    print("="*60)

    # 配置
    USE_HYPERPARAM_SEARCH = True  # 是否进行超参数搜索
    SENTENCE_BERT_MODEL = 'all-MiniLM-L6-v2'  # 轻量版Sentence-BERT

    # 加载数据
    print("\n[1/3] 加载数据...")
    train_texts, train_labels = load_data('train')
    val_texts, val_labels = load_data('val')
    test_texts, test_labels = load_data('test', use_original_label=True)

    print(f"  Train: {len(train_texts)} 条")
    print(f"  Val:   {len(val_texts)} 条")
    print(f"  Test:  {len(test_texts)} 条 (原始硬标签)")

    # 训练
    print("\n[2/3] 训练 SVM 模型...")

    if USE_HYPERPARAM_SEARCH:
        # 使用超参数搜索
        best_params, encoder = hyperparameter_search(
            train_texts, train_labels,
            val_texts, val_labels
        )

        # 使用最佳参数训练最终模型
        print("\n使用最佳参数训练最终模型...")
        X_train = encoder.encode(train_texts, show_progress_bar=True)
        clf = SVC(
            C=best_params['C'],
            kernel=best_params['kernel'],
            gamma=best_params['gamma'],
            class_weight='balanced',
            probability=True,
            random_state=42
        )
        clf.fit(X_train, train_labels)

        # 包装成优化版分类器
        svm_clf = SVMSentimentClassifierOptimized(SENTENCE_BERT_MODEL)
        svm_clf.encoder = encoder
        svm_clf.classifier = clf
    else:
        # 直接使用默认参数
        svm_clf = SVMSentimentClassifierOptimized(SENTENCE_BERT_MODEL)
        svm_clf.fit(
            train_texts,
            train_labels,
            C=1.0,
            kernel='rbf',
            class_weight='balanced'
        )

    print("  训练完成")

    # 测试
    print("\n[3/3] 测试集评估...")
    test_pred = svm_clf.predict(test_texts)

    test_acc = accuracy_score(test_labels, test_pred)
    test_f1_macro = f1_score(test_labels, test_pred, average='macro')
    test_f1_weighted = f1_score(test_labels, test_pred, average='weighted')

    print(f"\n  Test Accuracy: {test_acc:.4f}")
    print(f"  Test Macro-F1: {test_f1_macro:.4f}")
    print(f"  Test Weighted-F1: {test_f1_weighted:.4f}")

    print("\n  Classification Report:")
    print(classification_report(test_labels, test_pred,
                               target_names=['Negative', 'Neutral', 'Positive']))

    print("\n  Confusion Matrix:")
    cm = confusion_matrix(test_labels, test_pred)
    print(f"    True\\Pred  Neg  Neu  Pos")
    for i, row in enumerate(cm):
        label = ['Neg', 'Neu', 'Pos'][i]
        print(f"    {label}        {row[0]:3d}  {row[1]:3d}  {row[2]:3d}")

    # 保存结果
    results = {
        'model': f'SVM (Sentence-BERT: {SENTENCE_BERT_MODEL})',
        'config': {
            'sentence_bert_model': SENTENCE_BERT_MODEL,
            'use_hyperparam_search': USE_HYPERPARAM_SEARCH,
            'best_params': best_params if USE_HYPERPARAM_SEARCH else 'default'
        },
        'test_accuracy': float(test_acc),
        'test_macro_f1': float(test_f1_macro),
        'test_weighted_f1': float(test_f1_weighted)
    }

    with open(OUTPUT_DIR / "svm_optimized_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # 保存模型
    svm_clf.save(str(OUTPUT_DIR / "svm_optimized_model.pkl"))

    print("\n" + "="*60)
    print("SVM 优化版评估完成!")
    print(f"结果保存: {OUTPUT_DIR}")
    print("="*60)

    return results


if __name__ == "__main__":
    results = main()
    print(f"\n结果: {results}")
