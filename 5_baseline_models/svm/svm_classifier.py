"""
SVM 情感分类器 (svm_classifier.py)

传统 NLP 基线方法，使用 SVM + TF-IDF 进行情感分类。
对应论文中的 Traditional NLP Baseline。
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import pickle
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
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
class SVMConfig:
    """SVM 分类器配置"""
    max_features: int = 10000
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    C: float = 1.0
    kernel: str = 'rbf'
    gamma: str = 'scale'
    class_weight: str = 'balanced'
    use_grid_search: bool = False
    language: str = 'en'


class TextPreprocessor:
    """文本预处理器"""
    
    def __init__(self, language: str = 'en'):
        self.language = language
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = self._load_stopwords()
    
    def _load_stopwords(self) -> set:
        """加载停用词"""
        if self.language == 'en':
            return set(stopwords.words('english'))
        elif self.language == 'zh':
            return set(['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'])
        elif self.language == 'ru':
            return set(['и', 'в', 'не', 'на', 'я', 'быть', 'он', 'с', 'что', 'а', 'по', 'это', 'она', 'к', 'но', 'мы', 'как', 'из', 'у', 'то', 'за', 'свой', 'ее', 'мочь'])
        return set()
    
    def preprocess(self, text: str) -> str:
        """预处理文本"""
        if not text:
            return ""
        
        text = text.lower().strip()
        
        if self.language == 'en':
            tokens = nltk.word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens if t.isalpha()]
            tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
            return ' '.join(tokens)
        
        elif self.language == 'zh':
            tokens = jieba.lcut(text)
            tokens = [t for t in tokens if t not in self.stop_words and len(t.strip()) > 0]
            return ' '.join(tokens)
        
        elif self.language == 'ru':
            return text
        
        return text


class SVMSentimentClassifier:
    """SVM 情感分类器 - 传统 NLP 基线"""
    
    def __init__(self, config: Optional[SVMConfig] = None):
        self.config = config or SVMConfig()
        self.preprocessor = TextPreprocessor(self.config.language)
        self.pipeline: Optional[Pipeline] = None
        self.is_fitted = False
    
    def _build_pipeline(self) -> Pipeline:
        """构建 SVM + TF-IDF 管道"""
        return Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=self.config.max_features,
                ngram_range=self.config.ngram_range,
                min_df=self.config.min_df,
                max_df=self.config.max_df,
                sublinear_tf=True
            )),
            ('svm', SVC(
                C=self.config.C,
                kernel=self.config.kernel,
                gamma=self.config.gamma,
                class_weight=self.config.class_weight,
                probability=True,
                random_state=42
            ))
        ])
    
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """批量预处理文本"""
        return [self.preprocessor.preprocess(t) for t in texts]
    
    def fit(self, texts: List[str], labels: List[int]) -> 'SVMSentimentClassifier':
        """训练模型"""
        processed_texts = self.preprocess_texts(texts)
        self.pipeline = self._build_pipeline()
        
        if self.config.use_grid_search:
            param_grid = {
                'svm__C': [0.1, 1, 10],
                'svm__kernel': ['rbf', 'linear'],
                'svm__gamma': ['scale', 'auto']
            }
            self.pipeline = GridSearchCV(
                self.pipeline, param_grid, 
                cv=3, scoring='f1_macro', n_jobs=-1
            )
        
        self.pipeline.fit(processed_texts, labels)
        self.is_fitted = True
        return self
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """预测标签"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        processed_texts = self.preprocess_texts(texts)
        return self.pipeline.predict(processed_texts)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """预测概率"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        processed_texts = self.preprocess_texts(texts)
        return self.pipeline.predict_proba(processed_texts)
    
    def evaluate(self, texts: List[str], labels: List[int]) -> Dict[str, Any]:
        """评估模型性能"""
        predictions = self.predict(texts)
        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1_macro': f1_score(labels, predictions, average='macro'),
            'f1_weighted': f1_score(labels, predictions, average='weighted'),
            'per_class_f1': f1_score(labels, predictions, average=None).tolist(),
            'classification_report': classification_report(
                labels, predictions, 
                target_names=['Negative', 'Neutral', 'Positive']
            )
        }
    
    def save(self, filepath: str):
        """保存模型"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'pipeline': self.pipeline,
                'config': self.config,
                'is_fitted': self.is_fitted
            }, f)
    
    def load(self, filepath: str):
        """加载模型"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.pipeline = data['pipeline']
            self.config = data['config']
            self.is_fitted = data['is_fitted']
        return self
