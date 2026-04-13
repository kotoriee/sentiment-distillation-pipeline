"""
GSDMM 主题模型 (gsdmm_model.py)

Gibbs Sampling Dirichlet Multinomial Mixture Model for Short Text Clustering

这是论文的核心学术贡献之一：针对电商评论短文本，GSDMM 比 LDA 更稳定。

参考文献:
- Yin, J., & Wang, J. (2014). A Dirichlet Multinomial Mixture Model-based Approach
  for Short Text Clustering. KDD 2014.
- Cheng, E., et al. (2025). An Enhanced Model-based Approach for Short Text Clustering.
  IEEE TKDE.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field
from collections import defaultdict
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GSDMMConfig:
    """
    GSDMM 模型配置参数

    参数说明:
        K: 初始聚类数（最大聚类数），GSDMM会自动推断最优聚类数
        alpha: 文档-聚类先验参数（Dirichlet参数）
        beta: 聚类-词先验参数（Dirichlet参数）
        n_iter: Gibbs采样迭代次数
        random_state: 随机种子
    """
    K: int = 15  # 初始最大聚类数
    alpha: float = 0.1  # 文档-聚类先验
    beta: float = 0.1  # 聚类-词先验
    n_iter: int = 50  # 采样迭代次数
    random_state: int = 42


class GSDMMModel:
    """
    GSDMM - Gibbs Sampling Dirichlet Multinomial Mixture Model

    专为短文本聚类设计的主题模型，核心优势：
    1. 自动推断聚类数（不需要预设K）
    2. 对短文本（<50词）效果优于LDA
    3. 收敛速度快，稳定性好

    算法原理（基于 Yin & Wang 2014）：
    - 假设：每篇文档只属于一个主题（与LDA的混合主题不同）
    - 使用 Dirichlet Multinomial Mixture 建模
    - 通过 Gibbs Sampling 进行推断
    """

    def __init__(self, config: Optional[GSDMMConfig] = None):
        """
        初始化 GSDMM 模型

        Args:
            config: 模型配置，如果为None则使用默认配置
        """
        self.config = config or GSDMMConfig()

        # 设置随机种子
        np.random.seed(self.config.random_state)

        # 模型状态（在fit后初始化）
        self.vocab: Dict[str, int] = {}  # 词汇表：词 -> 索引
        self.vocab_reverse: Dict[int, str] = {}  # 反向词汇表：索引 -> 词
        self.K: int = self.config.K  # 当前聚类数

        # 聚类统计信息
        self.cluster_doc_count: np.ndarray = None  # 每个聚类的文档数
        self.cluster_word_count: np.ndarray = None  # 每个聚类的总词数
        self.cluster_word_dist: np.ndarray = None  # 每个聚类的词分布 (K x V)

        # 文档聚类分配
        self.doc_cluster_assignments: np.ndarray = None

        # 训练后的结果
        self.is_fitted = False
        self.cluster_word_distribution: Dict[int, Dict[str, float]] = {}  # 每个聚类的词概率分布
        self.cluster_topic_words: Dict[int, List[Tuple[str, float]]] = {}  # 每个聚类的Top词

    def _build_vocab(self, documents: List[List[str]]) -> None:
        """
        构建词汇表

        Args:
            documents: 分词后的文档列表，每个文档是词列表
        """
        logger.info("Building vocabulary...")

        word_set = set()
        for doc in documents:
            word_set.update(doc)

        self.vocab = {word: idx for idx, word in enumerate(sorted(word_set))}
        self.vocab_reverse = {idx: word for word, idx in self.vocab.items()}

        logger.info(f"Vocabulary size: {len(self.vocab)}")

    def _initialize_clusters(self, documents: List[List[str]]) -> None:
        """
        初始化聚类分配

        Args:
            documents: 分词后的文档列表
        """
        n_docs = len(documents)
        V = len(self.vocab)

        # 初始化聚类统计
        self.cluster_doc_count = np.zeros(self.K, dtype=np.int32)
        self.cluster_word_count = np.zeros(self.K, dtype=np.int32)
        self.cluster_word_dist = np.zeros((self.K, V), dtype=np.int32)

        # 随机初始化文档聚类分配
        self.doc_cluster_assignments = np.random.randint(0, self.K, size=n_docs)

        # 统计初始聚类信息
        for doc_idx, doc in enumerate(documents):
            cluster_idx = self.doc_cluster_assignments[doc_idx]

            self.cluster_doc_count[cluster_idx] += 1
            self.cluster_word_count[cluster_idx] += len(doc)

            for word in doc:
                word_idx = self.vocab[word]
                self.cluster_word_dist[cluster_idx, word_idx] += 1

        logger.info(f"Initialized {self.K} clusters")

    def _sample_cluster(self, doc: List[str], exclude_cluster: int) -> int:
        """
        为文档采样新的聚类分配

        这是 GSDMM 的核心算法：计算文档分配到每个聚类的概率，
        然后进行采样。

        采样公式（基于 Yin & Wang 2014）:
        P(z_d=k | z_{-d}, w) ∝
            (n_k^{-d} + α) / (N - 1 + K·α) ×
            Γ(∑_v n_kv^{-d} + V·β) / Γ(∑_v n_kv^{-d} + len(d) + V·β) ×
            ∏_{v∈d} Γ(n_kv^{-d} + n_dv + β) / Γ(n_kv^{-d} + β)

        简化计算（取对数，避免数值溢出）:
        log P(z_d=k) = log(n_k + α) - log(N - 1 + K·α)
                     + ∑_{v∈d} log(n_kv + n_dv + β) - log(n_kv + β)
                     + V·β·log(∑_v n_kv + V·β) - (∑_v n_kv + len(d) + V·β)·log(∑_v n_kv + V·β)

        Args:
            doc: 文档（词列表）
            exclude_cluster: 要排除的当前聚类（即文档当前所属聚类）

        Returns:
            采样得到的新聚类索引
        """
        # 计算每个聚类的对数概率
        log_probs = np.zeros(self.K)

        N = np.sum(self.cluster_doc_count)  # 总文档数
        V = len(self.vocab)

        for k in range(self.K):
            # 第一部分：聚类先验概率
            # P(z=k) ∝ (n_k + α)
            log_probs[k] = np.log(self.cluster_doc_count[k] + self.config.alpha + 1e-10)

            # 第二部分：词分布概率
            # P(w|z=k) ∝ ∏_{v∈doc} (n_kv + β) / (∑_v n_kv + V·β)
            for word in doc:
                word_idx = self.vocab[word]
                numerator = self.cluster_word_dist[k, word_idx] + self.config.beta
                denominator = self.cluster_word_count[k] + V * self.config.beta

                if denominator > 0:
                    log_probs[k] += np.log(numerator / denominator + 1e-10)

        # 归一化为概率（使用 log-sum-exp 技巧避免数值溢出）
        log_probs -= np.max(log_probs)  # 减去最大值提高数值稳定性
        probs = np.exp(log_probs)
        probs /= np.sum(probs)

        # 采样
        new_cluster = np.random.choice(self.K, p=probs)

        return new_cluster

    def fit(self, documents: List[List[str]], verbose: bool = True) -> 'GSDMMModel':
        """
        训练 GSDMM 模型

        使用 Gibbs Sampling 算法进行推断。

        Args:
            documents: 分词后的文档列表，每个文档是词列表
            verbose: 是否输出训练过程信息

        Returns:
            self: 训练后的模型
        """
        logger.info("Starting GSDMM training...")

        # 1. 构建词汇表
        self._build_vocab(documents)

        # 2. 初始化聚类
        self._initialize_clusters(documents)

        n_docs = len(documents)

        # 3. Gibbs Sampling 迭代
        for iteration in range(self.config.n_iter):
            # 对每篇文档重新采样聚类分配
            for doc_idx, doc in enumerate(documents):
                if not doc:  # 跳过空文档
                    continue

                # 获取当前聚类
                old_cluster = self.doc_cluster_assignments[doc_idx]

                # 从当前聚类中移除文档
                self.cluster_doc_count[old_cluster] -= 1
                self.cluster_word_count[old_cluster] -= len(doc)
                for word in doc:
                    word_idx = self.vocab[word]
                    self.cluster_word_dist[old_cluster, word_idx] -= 1

                # 采样新聚类
                new_cluster = self._sample_cluster(doc, old_cluster)

                # 将文档分配到新聚类
                self.doc_cluster_assignments[doc_idx] = new_cluster
                self.cluster_doc_count[new_cluster] += 1
                self.cluster_word_count[new_cluster] += len(doc)
                for word in doc:
                    word_idx = self.vocab[word]
                    self.cluster_word_dist[new_cluster, word_idx] += 1

            # 输出进度
            if verbose and (iteration + 1) % 10 == 0:
                active_clusters = np.sum(self.cluster_doc_count > 0)
                logger.info(
                    f"Iteration {iteration + 1}/{self.config.n_iter}: "
                    f"Active clusters = {active_clusters}"
                )

        # 4. 提取聚类主题词
        self._extract_topic_words()

        self.is_fitted = True
        logger.info("GSDMM training completed!")

        return self

    def _extract_topic_words(self, top_n: int = 10) -> None:
        """
        提取每个聚类的代表性词（主题词）

        对于每个聚类，计算每个词在该聚类中的概率分布，
        并保存Top-N词作为主题代表。

        Args:
            top_n: 每个聚类保留的Top词数量
        """
        logger.info("Extracting topic words...")

        V = len(self.vocab)

        for cluster_idx in range(self.K):
            # 跳过空聚类
            if self.cluster_doc_count[cluster_idx] == 0:
                continue

            # 计算词概率分布
            word_probs = {}
            total_words = self.cluster_word_count[cluster_idx]

            for word, word_idx in self.vocab.items():
                word_count = self.cluster_word_dist[cluster_idx, word_idx]
                if word_count > 0:
                    word_prob = (word_count + self.config.beta) / (total_words + V * self.config.beta)
                    word_probs[word] = word_prob

            # 排序并保存Top-N词
            sorted_words = sorted(word_probs.items(), key=lambda x: x[1], reverse=True)[:top_n]

            self.cluster_topic_words[cluster_idx] = sorted_words
            self.cluster_word_distribution[cluster_idx] = word_probs

    def predict(self, documents: List[List[str]]) -> np.ndarray:
        """
        预测新文档的聚类分配

        Args:
            documents: 分词后的文档列表

        Returns:
            聚类分配数组，shape = (n_docs,)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction!")

        predictions = []
        for doc in documents:
            if not doc:
                predictions.append(-1)  # 空文档
                continue

            # 为文档采样聚类（使用训练好的统计信息）
            cluster = self._sample_cluster(doc, exclude_cluster=-1)
            predictions.append(cluster)

        return np.array(predictions)

    def get_topic_words(self, cluster_idx: int, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        获取指定聚类的主题词

        Args:
            cluster_idx: 聚类索引
            top_n: 返回的Top词数量

        Returns:
            主题词列表，格式为 [(word, probability), ...]
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first!")

        if cluster_idx not in self.cluster_topic_words:
            return []

        return self.cluster_topic_words[cluster_idx][:top_n]

    def get_active_clusters(self) -> List[int]:
        """
        获取活跃聚类列表（文档数>0的聚类）

        Returns:
            活跃聚类索引列表
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first!")

        return [idx for idx in range(self.K) if self.cluster_doc_count[idx] > 0]

    def get_cluster_distribution(self) -> Dict[int, int]:
        """
        获取聚类分布（每个聚类的文档数）

        Returns:
            字典：{cluster_idx: doc_count}
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first!")

        distribution = {}
        for cluster_idx in self.get_active_clusters():
            distribution[cluster_idx] = int(self.cluster_doc_count[cluster_idx])

        return distribution

    def print_topics(self, top_n: int = 10) -> None:
        """
        打印所有主题

        Args:
            top_n: 每个主题显示的词数量
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first!")

        print("\n" + "=" * 60)
        print("GSDMM Topics")
        print("=" * 60)

        for cluster_idx in sorted(self.get_active_clusters()):
            topic_words = self.get_topic_words(cluster_idx, top_n)
            doc_count = self.cluster_doc_count[cluster_idx]

            words_str = ", ".join([f"{word}({prob:.4f})" for word, prob in topic_words])

            print(f"\nTopic {cluster_idx} ({doc_count} docs): {words_str}")

        print("\n" + "=" * 60)


# ==================== 辅助函数 ====================

def compute_coherence_score(
    model: GSDMMModel,
    documents: List[List[str]],
    top_n: int = 10
) -> float:
    """
    计算 GSDMM 模型的主题一致性得分

    使用 C_v 指标评估主题质量。

    Args:
        model: 训练好的 GSDMM 模型
        documents: 文档列表（用于计算词共现）
        top_n: 每个主题使用的词数量

    Returns:
        平均主题一致性得分
    """
    try:
        from gensim.models import CoherenceModel
        from gensim import corpora

        # 提取主题词
        topics = []
        for cluster_idx in model.get_active_clusters():
            topic_words = model.get_topic_words(cluster_idx, top_n)
            words = [word for word, prob in topic_words]
            if words:
                topics.append(words)

        if not topics:
            return 0.0

        # 使用 Gensim 计算 C_v
        dictionary = corpora.Dictionary(documents)
        coherence_model = CoherenceModel(
            topics=topics,
            texts=documents,
            dictionary=dictionary,
            coherence='c_v'
        )

        return coherence_model.get_coherence()

    except ImportError:
        logger.warning("Gensim not available, returning 0.0 for coherence")
        return 0.0


if __name__ == "__main__":
    # 示例用法
    print("GSDMM Model Implementation")
    print("=" * 60)

    # 创建模拟数据
    sample_docs = [
        ["手机", "电池", "太差", "续航", "不行"],
        ["手机", "屏幕", "很好", "清晰", "满意"],
        ["物流", "很快", "服务", "好"],
        ["物流", "慢", "差评", "等", "很久"],
        ["屏幕", "显示", "效果", "棒", "清晰"],
    ]

    # 训练模型
    config = GSDMMConfig(K=10, n_iter=20)
    model = GSDMMModel(config)
    model.fit(sample_docs)

    # 打印主题
    model.print_topics(top_n=5)

    # 获取聚类分布
    distribution = model.get_cluster_distribution()
    print(f"\nCluster distribution: {distribution}")

    print("\nGSDMM model test completed!")