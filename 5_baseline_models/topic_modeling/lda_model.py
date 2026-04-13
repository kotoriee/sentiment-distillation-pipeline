"""
LDA 主题模型 (lda_model.py)

传统 NLP 基线方法，使用 LDA (Latent Dirichlet Allocation) 进行主题建模。
对应论文中的 Traditional NLP Baseline。
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import pickle

from gensim import corpora, models
from gensim.models import CoherenceModel
import jieba
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 确保 NLTK 数据已下载
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


@dataclass
class LDAConfig:
    """LDA 模型配置"""
    num_topics: int = 15  # 主题数量
    passes: int = 15  # 训练轮数
    iterations: int = 100  # 每轮迭代次数
    alpha: str = 'auto'  # 文档-主题先验
    eta: str = 'auto'  # 主题-词先验
    random_state: int = 42
    language: str = 'en'  # 'en', 'zh', 'ru'


class LDATopicModel:
    """
    LDA 主题模型 - 传统 NLP 基线
    
    使用 Gensim 实现 LDA 进行主题建模。
    """
    
    def __init__(self, config: Optional[LDAConfig] = None):
        self.config = config or LDAConfig()
        self.dictionary: Optional[corpora.Dictionary] = None
        self.lda_model: Optional[models.LdaModel] = None
        self.corpus: Optional[List] = None
        self.is_fitted = False
        
        # 初始化预处理器
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = self._load_stopwords()
    
    def _load_stopwords(self) -> set:
        """加载停用词"""
        lang = self.config.language
        if lang == 'en':
            return set(stopwords.words('english'))
        elif lang == 'zh':
            return set(['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '中', '为', '来', '个', '而', '及', '与', '或', '但'])
        elif lang == 'ru':
            return set(['и', 'в', 'не', 'на', 'я', 'быть', 'он', 'с', 'что', 'а', 'по', 'это', 'она', 'к', 'но', 'мы', 'как', 'из', 'у', 'то', 'за', 'свой', 'ее', 'мочь', 'весь', 'тот', 'этот', 'так', 'его', 'ее', 'их'])
        return set()
    
    def _preprocess(self, text: str) -> List[str]:
        """预处理单条文本，返回词列表"""
        if not text:
            return []
        
        text = text.lower().strip()
        
        if self.config.language == 'en':
            # 英文：分词、词形还原、去停用词
            tokens = nltk.word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens if t.isalpha()]
            tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
            return tokens
        
        elif self.config.language == 'zh':
            # 中文：jieba 分词、去停用词
            tokens = jieba.lcut(text)
            tokens = [t for t in tokens if t not in self.stop_words and len(t.strip()) > 1]
            return tokens
        
        elif self.config.language == 'ru':
            # 俄文：简化处理
            return text.split()
        
        return text.split()
    
    def preprocess_texts(self, texts: List[str]) -> List[List[str]]:
        """批量预处理文本"""
        return [self._preprocess(t) for t in texts]
    
    def fit(self, texts: List[str]) -> 'LDATopicModel':
        """
        训练 LDA 模型
        
        Args:
            texts: 原始文本列表
        """
        # 预处理
        processed_texts = self.preprocess_texts(texts)
        
        # 创建词典
        self.dictionary = corpora.Dictionary(processed_texts)
        self.dictionary.filter_extremes(no_below=2, no_above=0.9)
        
        # 创建语料库 (词袋表示)
        self.corpus = [self.dictionary.doc2bow(text) for text in processed_texts]
        
        # 训练 LDA 模型
        self.lda_model = models.LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.config.num_topics,
            passes=self.config.passes,
            iterations=self.config.iterations,
            alpha=self.config.alpha,
            eta=self.config.eta,
            random_state=self.config.random_state
        )
        
        self.is_fitted = True
        return self
    
    def get_topics(self, num_words: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """
        获取所有主题的词分布
        
        Returns:
            {topic_id: [(word, probability), ...]}
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        topics = {}
        for topic_id in range(self.config.num_topics):
            topics[topic_id] = self.lda_model.show_topic(topic_id, num_words)
        return topics
    
    def get_document_topics(self, text: str) -> List[Tuple[int, float]]:
        """
        获取单条文本的主题分布
        
        Returns:
            [(topic_id, probability), ...]
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        processed = self._preprocess(text)
        bow = self.dictionary.doc2bow(processed)
        return self.lda_model.get_document_topics(bow)
    
    def get_dominant_topic(self, text: str) -> Tuple[int, float]:
        """
        获取文本的主导主题
        
        Returns:
            (topic_id, probability)
        """
        topics = self.get_document_topics(text)
        if not topics:
            return (-1, 0.0)
        return max(topics, key=lambda x: x[1])
    
    def evaluate_coherence(self, texts: List[str], coherence_type: str = 'c_v') -> float:
        """
        评估主题一致性
        
        Args:
            texts: 原始文本列表
            coherence_type: 'c_v', 'u_mass', 'c_uci', 'c_npmi'
        
        Returns:
            一致性分数
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        processed_texts = self.preprocess_texts(texts)
        
        coherence_model = CoherenceModel(
            model=self.lda_model,
            texts=processed_texts,
            dictionary=self.dictionary,
            coherence=coherence_type
        )
        
        return coherence_model.get_coherence()
    
    def evaluate_perplexity(self) -> float:
        """
        评估困惑度 (Perplexity)
        
        越低越好
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        return self.lda_model.log_perplexity(self.corpus)
    
    def get_topic_summary(self) -> Dict[str, Any]:
        """
        获取主题模型摘要
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        topics = self.get_topics(num_words=5)
        topic_summaries = {}
        
        for topic_id, words in topics.items():
            topic_summaries[topic_id] = {
                'top_words': [w for w, _ in words],
                'word_probs': {w: p for w, p in words}
            }
        
        return {
            'num_topics': self.config.num_topics,
            'num_terms': len(self.dictionary),
            'topics': topic_summaries
        }
    
    def save(self, filepath: str):
        """保存模型"""
        self.lda_model.save(filepath + '.lda')
        self.dictionary.save(filepath + '.dict')
        with open(filepath + '.config', 'wb') as f:
            pickle.dump(self.config, f)
    
    def load(self, filepath: str):
        """加载模型"""
        self.lda_model = models.LdaModel.load(filepath + '.lda')
        self.dictionary = corpora.Dictionary.load(filepath + '.dict')
        with open(filepath + '.config', 'rb') as f:
            self.config = pickle.load(f)
        self.is_fitted = True
        return self
