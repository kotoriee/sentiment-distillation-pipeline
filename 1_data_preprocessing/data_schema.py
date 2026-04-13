"""
数据层 - 全局数据契约 (Data Schema)

定义 RawRecord 和 ProcessedRecord，确保数据在 Pipeline 中的类型安全。
"""

from typing import List, Optional, Literal
try:
    from pydantic import BaseModel, Field, field_validator
except ImportError:
    from pydantic import BaseModel, Field, validator as field_validator
from datetime import datetime
import uuid


class RawRecord(BaseModel):
    """
    原始统一数据契约

    从各种数据源（HF datasets, 本地 CSV）加载后的标准化格式。
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()),
                    description="全局唯一 UUID")
    language: Literal["zh", "en", "ru"] = Field(
        ..., description="语言代码: 中文/英文/俄文")
    source: Literal["amazon_hf", "amazon_mcauley", "ozon_local", "yelp_hf", "mock_data"] = Field(
        ..., description="数据来源")
    original_text: str = Field(..., min_length=1,
                               description="原始评论文本")
    sentiment_label: Literal[0, 1, 2] = Field(
        ..., description="情感标签: 0=负面, 1=中性, 2=正面")
    rating: Optional[int] = Field(
        default=None, ge=1, le=5,
        description="原始星级评分 (1-5)")
    product_id: Optional[str] = Field(
        default=None, description="商品ID (用于文档池化)")
    category: Optional[str] = Field(
        default=None, description="商品类别")
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="记录创建时间")

    @field_validator('original_text')
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        """确保文本不为空或仅空白字符"""
        if not v or not v.strip():
            raise ValueError("original_text cannot be empty")
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "language": "zh",
                "source": "amazon_hf",
                "original_text": "这个手机电池太差了，一天要充三次",
                "sentiment_label": 0,
                "rating": 2,
                "product_id": "B08N5WRWNW",
                "category": "Electronics"
            }
        }


class ProcessedRecord(RawRecord):
    """
    清洗与增强后数据契约

    继承 RawRecord 的所有字段，增加清洗后的文本和元数据。
    """
    # 双流清洗结果
    text_for_nlp: str = Field(
        ..., description="深度清洗文本（去停用词、标点、词形还原）")
    text_for_llm: str = Field(
        ..., description="轻度清洗文本（仅去 URL/HTML，保留完整语法）")

    # 文本统计特征
    word_count: int = Field(
        ..., ge=0, description="分词后有效词数")
    char_count: int = Field(
        ..., ge=0, description="字符数")
    length_category: Literal["short", "medium", "long"] = Field(
        ..., description="长度分类")

    # 可选：Agent 蒸馏数据
    soft_label: Optional[List[float]] = Field(
        default=None, description="蒸馏后的概率分布 [p_neg, p_neu, p_pos]")
    rationale: Optional[str] = Field(
        default=None, description="云端 Agent 提取的情感推导思维链 (CoT)")

    # 处理元数据
    processed_at: datetime = Field(
        default_factory=datetime.now,
        description="处理时间")
    preprocessor_version: str = Field(
        default="1.0.0", description="预处理器版本")

    @field_validator('soft_label')
    @classmethod
    def validate_soft_label(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        """验证软标签是合法的概率分布"""
        if v is None:
            return v
        if len(v) != 3:
            raise ValueError("soft_label must have exactly 3 elements [p_neg, p_neu, p_pos]")
        if not all(0 <= p <= 1 for p in v):
            raise ValueError("all probabilities must be in [0, 1]")
        if abs(sum(v) - 1.0) > 1e-6:
            raise ValueError(f"probabilities must sum to 1, got {sum(v)}")
        return v

    @field_validator('length_category')
    @classmethod
    def validate_length_consistency(cls, v: str, info) -> str:
        """验证长度分类与 word_count 一致"""
        word_count = info.data.get('word_count')
        if word_count is not None:
            expected = classify_length(word_count)
            if v != expected:
                raise ValueError(
                    f"length_category '{v}' inconsistent with word_count {word_count} "
                    f"(expected '{expected}')"
                )
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "language": "zh",
                "source": "amazon_hf",
                "original_text": "这个手机电池太差了，一天要充三次",
                "sentiment_label": 0,
                "rating": 2,
                "product_id": "B08N5WRWNW",
                "category": "Electronics",
                "text_for_nlp": "手机 电池 太差 一天 充 三次",
                "text_for_llm": "这个手机电池太差了，一天要充三次",
                "word_count": 6,
                "char_count": 16,
                "length_category": "short",
                "soft_label": [0.85, 0.10, 0.05],
                "rationale": "用户抱怨电池续航差，'太差'表达强烈负面情绪"
            }
        }


def classify_length(word_count: int,
                   short_threshold: int = 15,
                   long_threshold: int = 50) -> Literal["short", "medium", "long"]:
    """
    根据词数对文本进行长度分类。

    Args:
        word_count: 词数
        short_threshold: 短文本阈值（默认15词）
        long_threshold: 长文本阈值（默认50词）

    Returns:
        "short" | "medium" | "long"
    """
    if word_count <= short_threshold:
        return "short"
    elif word_count <= long_threshold:
        return "medium"
    else:
        return "long"


# 便捷类型别名
RawRecordBatch = List[RawRecord]
ProcessedRecordBatch = List[ProcessedRecord]


if __name__ == "__main__":
    # 测试数据契约
    print("Testing RawRecord...")
    raw = RawRecord(
        language="zh",
        source="amazon_hf",
        original_text="  这个手机电池太差了！  ",
        sentiment_label=0,
        rating=2
    )
    print(f"RawRecord created: {raw.id}")
    print(f"Text after validation: '{raw.original_text}'")

    print("\nTesting ProcessedRecord...")
    processed = ProcessedRecord(
        **raw.model_dump(),
        text_for_nlp="手机 电池 太差",
        text_for_llm="这个手机电池太差了！",
        word_count=3,
        char_count=9,
        length_category="short"
    )
    print(f"ProcessedRecord created successfully")
    print(f"Length category: {processed.length_category}")

    # 测试软标签验证
    print("\nTesting soft_label validation...")
    try:
        invalid = ProcessedRecord(
            **raw.model_dump(),
            text_for_nlp="test",
            text_for_llm="test",
            word_count=1,
            char_count=4,
            length_category="short",
            soft_label=[0.5, 0.5]  # 错误的：只有2个元素
        )
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("\n✓ All schema tests passed!")
