"""
数据层 - 多源数据加载器

支持：
1. Hugging Face datasets 流式加载 (Amazon Reviews)
2. 本地 CSV/JSONL 加载 (Ozon 俄文数据)
"""

import json
import csv
from typing import List, Optional, Iterator
from pathlib import Path
import logging

try:
    from datasets import load_dataset, Dataset, IterableDataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logging.warning("HuggingFace datasets not installed. HF loading will not work.")

from .schema import RawRecord, RawRecordBatch

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 常量配置 ====================

# Hugging Face 数据集配置
HF_DATASETS = {
    # McAuley-Lab/Amazon-Reviews-2023 (推荐)
    "mcauley_2023": {
        "repo_id": "McAuley-Lab/Amazon-Reviews-2023",
        "config_name": "raw_review_{category}",  # 如 All_Beauty, Electronics
        "text_column": "text",
        "rating_column": "rating",
        "product_column": "parent_asin",
        "category_column": "category",
        "trust_remote_code": True,
    },
    # 旧版 amazon_reviews_multi (已过时，不建议使用)
    "amazon_reviews_multi": {
        "repo_id": "defunct-datasets/amazon_reviews_multi",
        "config_name": "{language}",  # 需要替换为 zh/en/ja 等
        "text_column": "review_body",
        "rating_column": "stars",
        "product_column": "product_id",
        "category_column": "product_category",
    }
}

# 语言代码映射
LANGUAGE_MAP = {
    "zh": "zh",
    "en": "en",
    "ru": "ru",  # Amazon multi 可能不支持俄文，需要本地数据
}

# 星级到情感标签的映射
# 1-2星 -> 负面(0), 3星 -> 中性(1), 4-5星 -> 正面(2)
RATING_TO_LABEL = {
    1: 0, 2: 0,   # 负面
    3: 1,         # 中性
    4: 2, 5: 2    # 正面
}


# ==================== 核心加载函数 ====================

def fetch_hf_dataset(
    language: str,
    n_samples: int = 20000,
    split: str = "train",
    dataset_name: str = "amazon_reviews_multi",
    shuffle: bool = True,
    seed: int = 42
) -> RawRecordBatch:
    """
    从 Hugging Face 流式加载数据集。

    Args:
        language: 语言代码 ("zh", "en", etc.)
        n_samples: 加载样本数
        split: 数据集划分 ("train", "test", "validation")
        dataset_name: 数据集名称
        shuffle: 是否随机打乱
        seed: 随机种子

    Returns:
        List[RawRecord]: 原始记录列表

    Raises:
        ImportError: 如果未安装 datasets 库
        ValueError: 如果语言不支持
    """
    if not HF_AVAILABLE:
        raise ImportError(
            "HuggingFace datasets library is required. "
            "Install with: pip install datasets"
        )

    if dataset_name not in HF_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    config = HF_DATASETS[dataset_name]

    # 替换配置中的语言占位符
    config_name = config["config_name"].format(language=language)

    logger.info(f"Loading {dataset_name}/{config_name} ({n_samples} samples)...")

    try:
        # 流式加载，节省内存
        dataset = load_dataset(
            config["repo_id"],
            config_name,
            split=split,
            streaming=True
        )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # 如果需要打乱，使用 shuffle
    if shuffle:
        dataset = dataset.shuffle(seed=seed, buffer_size=10000)

    # 截取前 n_samples
    dataset = dataset.take(n_samples)

    # 转换为 RawRecord
    records = []
    for item in dataset:
        try:
            record = _convert_hf_to_rawrecord(item, language, config)
            if record:
                records.append(record)
        except Exception as e:
            logger.warning(f"Skipping invalid record: {e}")
            continue

    logger.info(f"Loaded {len(records)} valid records")
    return records


def load_mcauley_2023(
    category: str = "All_Beauty",
    n_samples: int = 10000,
    split: str = "full"
) -> RawRecordBatch:
    """
    从 McAuley-Lab/Amazon-Reviews-2023 加载数据。

    推荐的数据集，包含高质量英文评论。
    完整类别列表：https://amazon-reviews-2023.github.io/

    Args:
        category: 商品类别（如 "All_Beauty", "Electronics", "Books"）
        n_samples: 加载样本数
        split: 数据集划分（"full" 为完整数据集）

    Returns:
        List[RawRecord]: 原始记录列表
    """
    if not HF_AVAILABLE:
        raise ImportError(
            "HuggingFace datasets library is required. "
            "Install with: pip install datasets"
        )

    logger.info(f"Loading McAuley-Lab/Amazon-Reviews-2023/{category}...")

    try:
        from datasets import load_dataset

        # 数据集配置名称格式：raw_review_<Category>
        config_name = f"raw_review_{category}"

        # 加载数据集
        dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            config_name,
            split=split,
            trust_remote_code=True
        )

        # 限制样本数
        if n_samples:
            dataset = dataset.select(range(min(n_samples, len(dataset))))

        records = []
        for item in dataset:
            # 提取文本和评分
            text = item.get("text", "").strip()
            rating = item.get("rating", None)

            if not text or rating is None:
                continue

            # 映射评分到情感标签
            rating_int = int(float(rating))
            label = RATING_TO_LABEL.get(rating_int, 1)  # 默认中性

            record = RawRecord(
                language="en",
                source="amazon_mcauley",
                original_text=text,
                sentiment_label=label,
                rating=rating_int,
                product_id=item.get("parent_asin"),
                category=category
            )
            records.append(record)

        logger.info(f"Loaded {len(records)} records from McAuley-Lab dataset")
        return records

    except Exception as e:
        logger.error(f"Failed to load McAuley-Lab dataset: {e}")
        raise


def load_local_csv(
    file_path: str,
    language: str = "ru",
    text_column: str = "text",
    label_column: Optional[str] = "label",
    rating_column: Optional[str] = "rating",
    n_samples: Optional[int] = None
) -> RawRecordBatch:
    """
    从本地 CSV 加载数据（主要用于俄文 Ozon 数据）。

    Args:
        file_path: CSV 文件路径
        language: 语言代码
        text_column: 文本列名
        label_column: 标签列名 (0/1/2)
        rating_column: 星级列名 (1-5)，如果提供则优先使用
        n_samples: 加载样本数限制

    Returns:
        List[RawRecord]: 原始记录列表
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Loading local CSV: {file_path}")

    records = []
    count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            if n_samples and count >= n_samples:
                break

            try:
                # 获取文本
                text = row.get(text_column, "").strip()
                if not text:
                    continue

                # 获取标签（优先使用 label_column，否则从 rating 计算）
                if label_column and label_column in row:
                    label = int(row[label_column])
                elif rating_column and rating_column in row:
                    rating = int(float(row[rating_column]))
                    label = map_rating_to_label(rating)
                else:
                    logger.warning(f"No label found for row {count}, skipping")
                    continue

                # 验证标签
                if label not in [0, 1, 2]:
                    logger.warning(f"Invalid label {label}, skipping")
                    continue

                # 获取 rating（如果存在且有效）
                rating = None
                if rating_column and rating_column in row and row[rating_column]:
                    try:
                        rating_val = int(float(row[rating_column]))
                        if 1 <= rating_val <= 5:
                            rating = rating_val
                    except (ValueError, TypeError):
                        pass

                record = RawRecord(
                    language=language,  # type: ignore
                    source="ozon_local",  # type: ignore
                    original_text=text,
                    sentiment_label=label,  # type: ignore
                    rating=rating,
                    product_id=row.get("product_id"),
                    category=row.get("category")
                )
                records.append(record)
                count += 1

            except Exception as e:
                logger.warning(f"Error parsing row {count}: {e}")
                continue

    logger.info(f"Loaded {len(records)} records from CSV")
    return records


def load_local_jsonl(
    file_path: str,
    n_samples: Optional[int] = None
) -> RawRecordBatch:
    """
    从本地 JSONL 加载已处理的 RawRecord。

    Args:
        file_path: JSONL 文件路径
        n_samples: 加载样本数限制

    Returns:
        List[RawRecord]: 原始记录列表
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Loading JSONL: {file_path}")

    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if n_samples and i >= n_samples:
                break
            try:
                data = json.loads(line.strip())
                record = RawRecord(**data)
                records.append(record)
            except Exception as e:
                logger.warning(f"Error parsing line {i}: {e}")
                continue

    logger.info(f"Loaded {len(records)} records from JSONL")
    return records


# ==================== 辅助函数 ====================

def map_rating_to_label(rating: int) -> int:
    """
    将星级评分映射为情感标签。

    Args:
        rating: 星级 (1-5)

    Returns:
        0=负面, 1=中性, 2=正面

    Raises:
        ValueError: 如果评分不在 1-5 范围内
    """
    if rating not in RATING_TO_LABEL:
        raise ValueError(f"Rating must be 1-5, got {rating}")
    return RATING_TO_LABEL[rating]


def _convert_hf_to_rawrecord(
    item: dict,
    language: str,
    config: dict
) -> Optional[RawRecord]:
    """
    将 Hugging Face 数据项转换为 RawRecord。
    """
    # 提取文本
    text = item.get(config["text_column"], "").strip()
    if not text:
        return None

    # 提取评分
    rating = item.get(config["rating_column"])
    if rating is None:
        return None

    try:
        rating = int(float(rating))
        label = map_rating_to_label(rating)
    except (ValueError, TypeError):
        return None

    # 提取其他字段
    product_id = item.get(config.get("product_column", ""))
    category = item.get(config.get("category_column", ""))

    return RawRecord(
        language=language,  # type: ignore
        source="amazon_hf",  # type: ignore
        original_text=text,
        sentiment_label=label,  # type: ignore
        rating=rating,
        product_id=str(product_id) if product_id else None,
        category=str(category) if category else None
    )


# ==================== 批量加载接口 ====================

def load_multilingual_dataset(
    languages: List[str] = ["zh", "en"],
    samples_per_lang: int = 20000,
    local_ru_path: Optional[str] = None
) -> RawRecordBatch:
    """
    加载多语言数据集（中英从 HF，俄文从本地）。

    Args:
        languages: 要加载的语言列表
        samples_per_lang: 每种语言的样本数
        local_ru_path: 俄文本地数据路径（如果包含 "ru"）

    Returns:
        List[RawRecord]: 合并后的原始记录列表
    """
    all_records = []

    for lang in languages:
        if lang == "ru":
            # 俄文从本地加载
            if local_ru_path:
                records = load_local_csv(local_ru_path, language="ru", n_samples=samples_per_lang)
                all_records.extend(records)
            else:
                logger.warning("Russian data path not provided, skipping")
        else:
            # 其他语言从 HF 加载
            try:
                records = fetch_hf_dataset(lang, n_samples=samples_per_lang)
                all_records.extend(records)
            except Exception as e:
                logger.error(f"Failed to load {lang} dataset: {e}")
                continue

    logger.info(f"Total loaded: {len(all_records)} records across {len(languages)} languages")
    return all_records


# ==================== 测试 ====================

if __name__ == "__main__":
    print("Testing data loader...\n")

    # 测试 1: 模拟 RawRecord 创建
    print("Test 1: Creating mock records")
    mock_records = [
        RawRecord(
            language="zh",
            source="amazon_hf",
            original_text="这个手机很好用",
            sentiment_label=2,
            rating=5
        ),
        RawRecord(
            language="en",
            source="amazon_hf",
            original_text="This product is terrible",
            sentiment_label=0,
            rating=1
        )
    ]
    print(f"Created {len(mock_records)} mock records")

    # 测试 2: 评分映射
    print("\nTest 2: Rating to label mapping")
    for rating in [1, 2, 3, 4, 5]:
        label = map_rating_to_label(rating)
        print(f"  {rating} stars -> label {label}")

    # 测试 3: 尝试加载 HuggingFace 数据（如果可用）
    if HF_AVAILABLE:
        print("\nTest 3: Loading sample from HuggingFace (this may take a moment)...")
        try:
            # 只加载 5 条测试
            records = fetch_hf_dataset("en", n_samples=5)
            print(f"Successfully loaded {len(records)} records from HF")
            if records:
                print(f"Sample: {records[0].original_text[:50]}...")
                print(f"Label: {records[0].sentiment_label}")
        except Exception as e:
            print(f"HF loading test skipped (expected if offline): {e}")
    else:
        print("\nTest 3: Skipped (HuggingFace datasets not installed)")

    print("\n✓ Loader tests completed!")
