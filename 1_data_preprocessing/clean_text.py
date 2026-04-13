"""
数据层 - 双流清洗架构

实现两种清洗流程：
1. NLP 深度流: 去停用词、标点、词形还原 (用于 SVM/LDA)
2. LLM 轻度流: 仅去 URL/HTML, 保留完整语法 (用于 Qwen/GPT-4)

遵循 CLAUDE.md 约束：
- 中文分词: jieba (强制)
- 英文分词: nltk (强制)
- 俄文处理: natasha (强制)
"""

import re
import html
from typing import List, Dict, Set, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== 多语言分词库导入 ====================

# 中文: jieba
try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    logger.warning("jieba not installed. Chinese tokenization will be degraded.")

# 英文: nltk
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer
    from nltk.corpus import stopwords as nltk_stopwords
    NLTK_AVAILABLE = True

    # 尝试下载必要的 NLTK 数据
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logger.info("Downloading NLTK punkt...")
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        logger.info("Downloading NLTK stopwords...")
        nltk.download('stopwords', quiet=True)

    _stemmer = PorterStemmer()
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("nltk not installed. English tokenization will be degraded.")

# 俄文: natasha (强制)
try:
    from natasha import (
        Segmenter,
        MorphVocab,
        NewsEmbedding,
        NewsMorphTagger,
        Doc
    )
    NATASHA_AVAILABLE = True

    # 初始化 natasha 组件
    _segmenter = Segmenter()
    _morph_vocab = MorphVocab()
    _emb = NewsEmbedding()
    _morph_tagger = NewsMorphTagger(_emb)
except ImportError:
    NATASHA_AVAILABLE = False
    logger.error("natasha not installed! Russian processing will fail.")

from .schema import RawRecord, ProcessedRecord, classify_length


# ==================== 正则表达式预编译 ====================

URL_PATTERN = re.compile(
    r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?',
    re.IGNORECASE
)
HTML_PATTERN = re.compile(r'<[^>]+>')
EMAIL_PATTERN = re.compile(r'\S+@\S+\.\S+')
WHITESPACE_PATTERN = re.compile(r'\s+')


# ==================== 停用词管理 ====================

class StopwordManager:
    """多语言停用词管理器"""

    def __init__(self):
        self._stopwords: Dict[str, Set[str]] = {}
        self._load_stopwords()

    def _load_stopwords(self):
        """加载各语言停用词"""
        # 中文停用词 (jieba 有内置，但我们需要自定义)
        self._stopwords['zh'] = self._get_zh_stopwords()

        # 英文停用词 (优先使用 nltk，否则使用内置)
        if NLTK_AVAILABLE:
            try:
                self._stopwords['en'] = set(nltk_stopwords.words('english'))
            except:
                self._stopwords['en'] = self._get_en_stopwords()
        else:
            self._stopwords['en'] = self._get_en_stopwords()

        # 俄文停用词 (natasha 没有内置停用词，使用内置)
        self._stopwords['ru'] = self._get_ru_stopwords()

    def _get_zh_stopwords(self) -> Set[str]:
        """中文停用词"""
        return {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
            '你', '会', '着', '没有', '看', '好', '自己', '这', '那',
            '这个', '那个', '这些', '那些', '之', '与', '及', '等',
            '可以', '就是', '但是', '如果', '因为', '所以', '然而',
            # 电商评论常见停用词
            '商品', '产品', '卖家', '买家', '快递', '物流', '包装',
            '已经', '非常', '真的', '感觉', '觉得', '东西', '卖家'
        }

    def _get_en_stopwords(self) -> Set[str]:
        """英文停用词 (备用)"""
        return {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into',
            'through', 'during', 'before', 'after', 'above', 'below',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
            'its', 'our', 'their', 'what', 'which', 'who', 'when', 'where',
            'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
            'own', 'same', 'so', 'than', 'too', 'very', 'just',
            # 电商领域
            'product', 'item', 'seller', 'shipping', 'package', 'order'
        }

    def _get_ru_stopwords(self) -> Set[str]:
        """俄文停用词"""
        return {
            'и', 'в', 'не', 'на', 'я', 'быть', 'он', 'с', 'что', 'а',
            'по', 'это', 'она', 'к', 'но', 'мы', 'как', 'из', 'у',
            'то', 'за', 'свой', 'ее', 'так', 'который', 'весь',
            'год', 'от', 'такой', 'для', 'ты', 'же', 'все', 'тот',
            'мочь', 'вы', 'человек', 'его', 'сказать', 'этот',
            # 电商领域
            'продукт', 'товар', 'продавец', 'доставка', 'упаковка',
            'заказ', 'магазин', 'покупка'
        }

    def get_stopwords(self, language: str) -> Set[str]:
        """获取指定语言的停用词"""
        return self._stopwords.get(language, set())


# 全局停用词管理器
_stopword_manager = StopwordManager()


# ==================== LLM 轻度清洗 ====================

def clean_for_llm(text: str) -> str:
    """
    轻度清洗 - 保留完整语法和标点（用于 LLM 输入）

    处理步骤：
    1. HTML 实体解码
    2. 去除 URL
    3. 去除邮箱
    4. 去除 HTML 标签
    5. 规范化空白
    """
    if not text:
        return ""

    text = html.unescape(text)
    text = URL_PATTERN.sub(' ', text)
    text = EMAIL_PATTERN.sub(' ', text)
    text = HTML_PATTERN.sub(' ', text)
    text = WHITESPACE_PATTERN.sub(' ', text)

    return text.strip()


# ==================== 多语言分词函数 ====================

def tokenize_chinese(text: str) -> List[str]:
    """
    中文分词 - 强制使用 jieba
    """
    if not text:
        return []

    if JIEBA_AVAILABLE:
        # 使用 jieba 精确模式
        tokens = list(jieba.cut(text, cut_all=False))
        # 过滤空字符串和单字符（通常是标点）
        return [t.strip() for t in tokens if t.strip() and len(t.strip()) > 1]
    else:
        # 降级方案：按字符分割
        logger.warning("jieba not available, using character-level fallback")
        return [c for c in text if '\u4e00' <= c <= '\u9fff']


def tokenize_english(text: str, apply_stemming: bool = False) -> List[str]:
    """
    英文分词 - 强制使用 nltk

    Args:
        text: 输入文本
        apply_stemming: 是否应用 Porter Stemmer
    """
    if not text:
        return []

    if NLTK_AVAILABLE:
        # 使用 nltk word_tokenize
        tokens = word_tokenize(text.lower())
        # 过滤非字母词
        tokens = [t for t in tokens if t.isalpha()]

        if apply_stemming:
            tokens = [_stemmer.stem(t) for t in tokens]

        return tokens
    else:
        # 降级方案：简单正则
        logger.warning("nltk not available, using regex fallback")
        return re.findall(r'\b[a-zA-Z]+\b', text.lower())


def tokenize_russian(text: str, apply_lemmatization: bool = True) -> List[str]:
    """
    俄文分词 - 强制使用 natasha

    Args:
        text: 输入文本
        apply_lemmatization: 是否应用词形还原（lemma）
    """
    if not text:
        return []

    if not NATASHA_AVAILABLE:
        raise ImportError(
            "natasha is required for Russian tokenization! "
            "Install with: pip install natasha"
        )

    # 使用 natasha 进行分词和词形还原
    doc = Doc(text)
    doc.segment(_segmenter)
    doc.tag_morph(_morph_tagger)

    tokens = []
    for token in doc.tokens:
        # 只保留俄文字符的词
        if re.match(r'^[\u0400-\u04ff]+$', token.text):
            if apply_lemmatization:
                # 词形还原
                token.lemmatize(_morph_vocab)
                lemma = token.lemma if token.lemma else token.text.lower()
                tokens.append(lemma)
            else:
                tokens.append(token.text.lower())

    return tokens


# ==================== NLP 深度清洗 ====================

def clean_for_nlp(
    text: str,
    language: str,
    remove_stopwords: bool = True,
    apply_stemming: bool = False
) -> str:
    """
    深度清洗 - 去停用词、标点、词形还原（用于传统 NLP）

    Args:
        text: 原始文本
        language: 语言代码 (zh/en/ru)
        remove_stopwords: 是否去除停用词
        apply_stemming: 是否应用词干提取/词形还原

    Returns:
        空格分隔的清洗后词序列
    """
    if not text:
        return ""

    # 先进行轻度清洗
    text = clean_for_llm(text)

    # 按语言分词
    if language == "zh":
        tokens = tokenize_chinese(text)
    elif language == "en":
        tokens = tokenize_english(text, apply_stemming=apply_stemming)
    elif language == "ru":
        tokens = tokenize_russian(text, apply_lemmatization=apply_stemming)
    else:
        # 默认按空格分词
        tokens = text.lower().split()

    # 去除停用词
    if remove_stopwords:
        stopwords = _stopword_manager.get_stopwords(language)
        tokens = [t for t in tokens if t not in stopwords and len(t) > 1]

    return ' '.join(tokens)


# ==================== 主要处理函数 ====================

def process_record(
    raw: RawRecord,
    remove_stopwords: bool = True,
    apply_stemming: bool = False
) -> ProcessedRecord:
    """
    处理单个 RawRecord，生成 ProcessedRecord。
    """
    # 双流清洗
    text_for_llm = clean_for_llm(raw.original_text)
    text_for_nlp = clean_for_nlp(
        raw.original_text,
        raw.language,
        remove_stopwords=remove_stopwords,
        apply_stemming=apply_stemming
    )

    # 统计特征
    word_count = len(text_for_nlp.split()) if text_for_nlp else 0
    char_count = len(text_for_llm)
    length_category = classify_length(word_count)

    # 构建 ProcessedRecord
    processed_data = raw.model_dump()
    processed_data.update({
        'text_for_nlp': text_for_nlp,
        'text_for_llm': text_for_llm,
        'word_count': word_count,
        'char_count': char_count,
        'length_category': length_category,
    })

    return ProcessedRecord(**processed_data)


def process_batch(
    records: List[RawRecord],
    remove_stopwords: bool = True,
    apply_stemming: bool = False,
    show_progress: bool = True
) -> List[ProcessedRecord]:
    """
    批量处理记录。
    """
    processed = []
    total = len(records)

    for i, record in enumerate(records):
        try:
            proc = process_record(record, remove_stopwords, apply_stemming)
            processed.append(proc)
        except Exception as e:
            logger.warning(f"Error processing record {record.id}: {e}")
            continue

        if show_progress and (i + 1) % 1000 == 0:
            logger.info(f"Processed {i + 1}/{total} records")

    if show_progress:
        logger.info(f"Completed: {len(processed)}/{total} records processed")

    return processed


# ==================== 测试 ====================

if __name__ == "__main__":
    print("Testing preprocessor with CLAUDE.md constraints...\n")

    # 测试中文 (jieba)
    print("Test 1: Chinese tokenization (jieba)")
    text_zh = "这个手机真的很好用，快递速度很快！"
    tokens_zh = tokenize_chinese(text_zh)
    print(f"  Input:  {text_zh}")
    print(f"  Tokens: {tokens_zh}")
    print(f"  Lib:    {'jieba' if JIEBA_AVAILABLE else 'fallback'}")
    print()

    # 测试英文 (nltk)
    print("Test 2: English tokenization (nltk)")
    text_en = "This product is amazing! Fast shipping."
    tokens_en = tokenize_english(text_en, apply_stemming=True)
    print(f"  Input:  {text_en}")
    print(f"  Tokens: {tokens_en}")
    print(f"  Lib:    {'nltk' if NLTK_AVAILABLE else 'fallback'}")
    print()

    # 测试俄文 (natasha)
    if NATASHA_AVAILABLE:
        print("Test 3: Russian tokenization (natasha)")
        text_ru = "Этот товар очень хороший, доставка быстрая!"
        tokens_ru = tokenize_russian(text_ru, apply_lemmatization=True)
        print(f"  Input:  {text_ru}")
        print(f"  Tokens: {tokens_ru}")
        print(f"  Note:   Lemmatization applied (хороший -> хороший)")
        print()
    else:
        print("Test 3: Russian skipped (natasha not installed)")
        print()

    # 测试双流清洗
    print("Test 4: Dual-stream cleaning")
    from .schema import RawRecord

    test_record = RawRecord(
        language="en",
        source="amazon_hf",
        original_text="This product is AMAZING!!! Check https://example.com",
        sentiment_label=2,
        rating=5
    )

    processed = process_record(test_record)
    print(f"  Original: {test_record.original_text}")
    print(f"  For LLM:  {processed.text_for_llm}")
    print(f"  For NLP:  {processed.text_for_nlp}")
    print(f"  Words:    {processed.word_count}")
    print()

    print("✓ All preprocessor tests completed!")
