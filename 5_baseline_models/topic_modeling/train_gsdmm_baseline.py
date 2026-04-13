"""
GSDMM 情感分类基线

方法：
1. 用GSDMM对评论进行聚类（无监督）
2. 基于聚类中的情感关键词，将聚类映射到情感标签
3. 评估映射后的分类性能

这是论文中的第三种传统NLP基线方法。
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
import sys
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import re

sys.path.insert(0, str(Path(__file__).parent))
from gsdmm_model import GSDMMModel, GSDMMConfig

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

DATA_DIR = Path("experiments/denoising_setup")
OUTPUT_DIR = Path("experiments/denoising_setup/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 情感种子词（用于映射聚类到情感标签）
SENTIMENT_SEEDS = {
    'negative': [
        'bad', 'terrible', 'awful', 'worst', 'horrible', 'hate', 'disappointed',
        'poor', 'waste', 'useless', 'broken', 'defective', 'return', 'refund',
        'problem', 'issue', 'fail', 'error', 'cheap', 'flimsy', 'junk'
    ],
    'neutral': [
        'okay', 'average', 'fine', 'acceptable', 'standard', 'normal',
        'decent', 'moderate', 'regular', 'usual', 'typical', 'common',
        'neither', 'nor', 'however', 'but', 'although', 'while'
    ],
    'positive': [
        'good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'best',
        'awesome', 'fantastic', 'wonderful', 'happy', 'satisfied', 'recommend',
        'quality', 'nice', 'beautiful', 'comfortable', 'easy', 'fast', 'quick'
    ]
}


def preprocess_text(text: str) -> List[str]:
    """文本预处理：分词、去停用词"""
    if not isinstance(text, str):
        return []

    # 小写化
    text = text.lower()
    # 去除特殊字符
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 分词
    tokens = word_tokenize(text)
    # 去停用词和短词
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    return tokens


def assign_cluster_sentiment(
    cluster_idx: int,
    topic_words: List[Tuple[str, float]],
    texts_in_cluster: List[str]
) -> int:
    """
    基于主题词和聚类中文本，为聚类分配情感标签

    Args:
        cluster_idx: 聚类索引
        topic_words: 聚类的主题词 [(word, prob), ...]
        texts_in_cluster: 聚类中的所有文本

    Returns:
        情感标签: 0=Negative, 1=Neutral, 2=Positive
    """
    # 方法1：基于种子词匹配
    topic_word_set = {w for w, _ in topic_words}

    neg_score = len(topic_word_set & set(SENTIMENT_SEEDS['negative']))
    neu_score = len(topic_word_set & set(SENTIMENT_SEEDS['neutral']))
    pos_score = len(topic_word_set & set(SENTIMENT_SEEDS['positive']))

    # 方法2：基于文本中的情感词频率
    for text in texts_in_cluster[:50]:  # 抽样检查前50条
        tokens = set(preprocess_text(text))
        neg_score += len(tokens & set(SENTIMENT_SEEDS['negative'])) * 0.1
        pos_score += len(tokens & set(SENTIMENT_SEEDS['positive'])) * 0.1

    scores = [neg_score, neu_score, pos_score]
    return scores.index(max(scores))


def train_gsdmm_baseline():
    """训练GSDMM基线模型"""
    print("="*70)
    print("GSDMM 情感分类基线")
    print("="*70)

    # 加载数据
    print("\n[1/4] 加载数据...")
    train_df = pd.read_csv(DATA_DIR / "train_cleaned.csv")
    test_df = pd.read_csv(DATA_DIR / "test_original.csv")

    train_texts = train_df['text'].tolist()
    train_labels = train_df['label'].tolist()
    test_texts = test_df['text'].tolist()
    test_labels = test_df['label'].tolist()

    print(f"  Train: {len(train_texts)} 条")
    print(f"  Test: {len(test_texts)} 条")

    # 预处理
    print("\n[2/4] 预处理文本...")
    train_tokens = [preprocess_text(t) for t in train_texts]
    test_tokens = [preprocess_text(t) for t in test_texts]

    # 过滤空文档
    train_tokens = [t for t in train_tokens if t]
    test_tokens = [t for t in test_tokens if t]

    print(f"  有效训练样本: {len(train_tokens)}")
    print(f"  有效测试样本: {len(test_tokens)}")

    # 训练GSDMM
    print("\n[3/4] 训练GSDMM模型...")
    config = GSDMMConfig(
        K=20,           # 初始聚类数
        alpha=0.1,      # 文档-聚类先验
        beta=0.1,       # 聚类-词先验
        n_iter=100,     # 迭代次数
        random_state=42
    )

    model = GSDMMModel(config)
    model.fit(train_tokens, verbose=True)

    # 获取活跃聚类
    active_clusters = model.get_active_clusters()
    print(f"\n  活跃聚类数: {len(active_clusters)}/{config.K}")

    # 为每个聚类分配情感标签
    print("\n[4/4] 映射聚类到情感标签...")

    # 获取训练集的聚类分配
    train_cluster_assignments = model.doc_cluster_assignments

    # 为每个聚类确定情感标签
    cluster_to_sentiment = {}

    for cluster_idx in active_clusters:
        # 获取聚类的主题词
        topic_words = model.get_topic_words(cluster_idx, top_n=20)

        # 获取聚类中的文本
        texts_in_cluster = [
            train_texts[i] for i, c in enumerate(train_cluster_assignments)
            if c == cluster_idx
        ]

        # 分配情感标签
        sentiment = assign_cluster_sentiment(cluster_idx, topic_words, texts_in_cluster)
        cluster_to_sentiment[cluster_idx] = sentiment

        print(f"\n  聚类 {cluster_idx} -> 情感 {['Negative', 'Neutral', 'Positive'][sentiment]}")
        print(f"    主题词: {', '.join([w for w, _ in topic_words[:10]])}")
        print(f"    样本数: {len(texts_in_cluster)}")

    # 预测测试集
    print("\n[5/5] 测试集预测...")
    test_predictions = model.predict(test_tokens)
    test_pred_labels = [cluster_to_sentiment.get(p, 1) for p in test_predictions]  # 默认Neutral

    # 评估
    test_acc = accuracy_score(test_labels, test_pred_labels)
    test_f1_macro = f1_score(test_labels, test_pred_labels, average='macro')
    test_f1_weighted = f1_score(test_labels, test_pred_labels, average='weighted')

    print(f"\n  测试结果:")
    print(f"    Accuracy: {test_acc:.4f}")
    print(f"    Macro-F1: {test_f1_macro:.4f}")
    print(f"    Weighted-F1: {test_f1_weighted:.4f}")

    print("\n  Classification Report:")
    print(classification_report(test_labels, test_pred_labels,
                               target_names=['Negative', 'Neutral', 'Positive']))

    # 混淆矩阵
    print("\n  Confusion Matrix:")
    cm = confusion_matrix(test_labels, test_pred_labels)
    print(f"    True\\Pred  Neg  Neu  Pos")
    for i, row in enumerate(cm):
        label = ['Neg', 'Neu', 'Pos'][i]
        print(f"    {label}        {row[0]:3d}  {row[1]:3d}  {row[2]:3d}")

    # 保存结果
    results = {
        'model': 'GSDMM (Topic Clustering + Sentiment Mapping)',
        'config': {
            'K': config.K,
            'alpha': config.alpha,
            'beta': config.beta,
            'n_iter': config.n_iter,
            'active_clusters': len(active_clusters)
        },
        'cluster_to_sentiment': cluster_to_sentiment,
        'test_accuracy': float(test_acc),
        'test_macro_f1': float(test_f1_macro),
        'test_weighted_f1': float(test_f1_weighted),
        'confusion_matrix': cm.tolist()
    }

    with open(OUTPUT_DIR / "gsdmm_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # 保存模型
    with open(OUTPUT_DIR / "gsdmm_model.pkl", 'wb') as f:
        pickle.dump({
            'model': model,
            'cluster_to_sentiment': cluster_to_sentiment,
            'config': config
        }, f)

    print("\n" + "="*70)
    print("GSDMM基线评估完成!")
    print(f"结果保存: {OUTPUT_DIR}")
    print("="*70)

    return results


if __name__ == "__main__":
    # 确保NLTK数据已下载
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

    results = train_gsdmm_baseline()
    print(f"\n最终结果: {results}")
