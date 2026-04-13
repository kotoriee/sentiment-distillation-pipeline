"""
Multilingual Prompt Templates for Sentiment Classification

Provides language-specific prompts for sentiment analysis using local LLMs.
Supports Chinese (zh), English (en), and Russian (ru).
"""

from typing import Dict, List, Literal

# Language code type
Language = Literal["zh", "en", "ru"]

# Sentiment label mapping
SENTIMENT_LABELS = {
    "zh": {0: "负面", 1: "中性", 2: "正面"},
    "en": {0: "negative", 1: "neutral", 2: "positive"},
    "ru": {0: "негативный", 1: "нейтральный", 2: "позитивный"},
}

# System prompts for each language
SYSTEM_PROMPTS: Dict[Language, str] = {
    "zh": """你是一位专业的电商评论情感分析专家。你的任务是分析用户评论的情感倾向。

请仔细阅读评论，判断其情感为：负面(0)、中性(1)或正面(2)。

分析要求：
1. 考虑评论的整体语气和用词
2. 注意反讽和夸张表达
3. 综合考虑产品质量、服务、物流等多个维度
4. 给出简短的推理过程

重要：你必须在回复内容中直接输出JSON格式，不要只输出思考过程。

输出格式（JSON）：
{
    "sentiment": 0/1/2,
    "confidence": 0.0-1.0,
    "rationale": "简短的推理说明"
}""",

    "en": """You are a professional e-commerce review sentiment analysis expert. Your task is to analyze the sentiment of user reviews.

Please read the review carefully and determine if the sentiment is: negative(0), neutral(1), or positive(2).

Analysis requirements:
1. Consider the overall tone and word choice of the review
2. Pay attention to sarcasm and exaggeration
3. Consider multiple dimensions: product quality, service, shipping, etc.
4. Provide a brief reasoning process

IMPORTANT: You must output the JSON format directly in your response content, not just in your thinking process.

Output format (JSON):
{
    "sentiment": 0/1/2,
    "confidence": 0.0-1.0,
    "rationale": "Brief reasoning explanation"
}""",

    "ru": """Вы являетесь экспертом по анализу тональности отзывов электронной коммерции. Ваша задача - анализировать тональность отзывов пользователей.

Пожалуйста, внимательно прочитайте отзыв и определите, является ли тональность: негативной(0), нейтральной(1) или позитивной(2).

Требования к анализу:
1. Учитывайте общий тон и выбор слов в отзыве
2. Обращайте внимание на сарказм и преувеличение
3. Учитывайте несколько аспектов: качество продукта, обслуживание, доставка и т.д.
4. Предоставьте краткое обоснование

Формат вывода (JSON):
{
    "sentiment": 0/1/2,
    "confidence": 0.0-1.0,
    "rationale": "Краткое объяснение reasoning"
}""",
}

# User prompt templates for each language
USER_PROMPT_TEMPLATES: Dict[Language, str] = {
    "zh": """请分析以下电商评论的情感倾向：

评论内容：
"{text}"

请输出 JSON 格式的分析结果。""",

    "en": """Please analyze the sentiment of the following e-commerce review:

Review content:
"{text}"

Please output the analysis result in JSON format.""",

    "ru": """Пожалуйста, проанализируйте тональность следующего отзыва:

Содержание отзыва:
"{text}"

Пожалуйста, выведите результат анализа в формате JSON.""",
}


def get_system_prompt(language: Language) -> str:
    """
    Get system prompt for specified language.

    Args:
        language: Language code ("zh", "en", or "ru")

    Returns:
        System prompt string

    Raises:
        ValueError: If language is not supported
    """
    if language not in SYSTEM_PROMPTS:
        raise ValueError(f"Unsupported language: {language}. Supported: {list(SYSTEM_PROMPTS.keys())}")
    return SYSTEM_PROMPTS[language]


def get_sentiment_prompt(text: str, language: Language) -> str:
    """
    Get user prompt for sentiment classification.

    Args:
        text: Review text to analyze
        language: Language code ("zh", "en", or "ru")

    Returns:
        Formatted prompt string
    """
    if language not in USER_PROMPT_TEMPLATES:
        raise ValueError(f"Unsupported language: {language}")

    template = USER_PROMPT_TEMPLATES[language]
    return template.format(text=text)


def format_chat_messages(
    text: str,
    language: Language,
    few_shot_examples: List[Dict] = None
) -> List[Dict[str, str]]:
    """
    Format messages for chat completion API.

    Args:
        text: Review text to analyze
        language: Language code
        few_shot_examples: Optional few-shot examples as list of dicts with "input" and "output"

    Returns:
        List of message dicts with "role" and "content"
    """
    messages = [{"role": "system", "content": get_system_prompt(language)}]

    # Add few-shot examples if provided
    if few_shot_examples:
        for example in few_shot_examples:
            user_content = get_sentiment_prompt(example["input"], language)
            assistant_content = example["output"]

            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": assistant_content})

    # Add the actual query
    messages.append({"role": "user", "content": get_sentiment_prompt(text, language)})

    return messages


# Few-shot examples for each language (for improved accuracy)
FEW_SHOT_EXAMPLES: Dict[Language, List[Dict]] = {
    "zh": [
        {
            "input": "这个产品太棒了，质量非常好，物流也很快！",
            "output": '{"sentiment": 2, "confidence": 0.95, "rationale": "用户使用了积极的词汇如\"太棒了\"、\"非常好\"、\"很快\"，表达了高度满意"}'
        },
        {
            "input": "一般般吧，没有想象的那么好，但也还可以接受。",
            "output": '{"sentiment": 1, "confidence": 0.75, "rationale": "用户态度中性，既表达了失望（\"没有想象的那么好\"），也表示可接受"}'
        },
        {
            "input": "完全不值这个价，质量太差了，退货！",
            "output": '{"sentiment": 0, "confidence": 0.92, "rationale": "用户表达了强烈不满，使用了\"不值\"、\"太差\"等负面词汇，并提到退货"}'
        }
    ],
    "en": [
        {
            "input": "This product is amazing! Great quality and fast shipping.",
            "output": '{"sentiment": 2, "confidence": 0.96, "rationale": "User uses positive words like \"amazing\", \"Great\", expressing high satisfaction"}'
        },
        {
            "input": "It's okay, not as good as I expected but acceptable.",
            "output": '{"sentiment": 1, "confidence": 0.78, "rationale": "Neutral tone, mixed feelings - disappointment but also acceptance"}'
        },
        {
            "input": "Terrible quality, complete waste of money. Never buying again!",
            "output": '{"sentiment": 0, "confidence": 0.94, "rationale": "Strong negative sentiment with words like \"Terrible\", \"waste\", and \"Never buying again\""}'
        }
    ],
    "ru": [
        {
            "input": "Отличный товар! Качество на высоте, доставка быстрая.",
            "output": '{"sentiment": 2, "confidence": 0.95, "rationale": "Пользователь использует позитивные слова \"Отличный\", \"высоте\", выражая высокую удовлетворенность"}'
        },
        {
            "input": "Нормально, но не так хорошо, как ожидалось. В целом сойдет.",
            "output": '{"sentiment": 1, "confidence": 0.76, "rationale": "Нейтральный тон, смешанные чувства - разочарование, но и принятие"}'
        },
        {
            "input": "Ужасное качество, полное разочарование. Деньги на ветер!",
            "output": '{"sentiment": 0, "confidence": 0.93, "rationale": "Сильный негатив с словами \"Ужасное\", \"разочарование\", \"Деньги на ветер\""}'
        }
    ]
}


def get_few_shot_examples(language: Language, n_examples: int = 3) -> List[Dict]:
    """
    Get few-shot examples for specified language.

    Args:
        language: Language code
        n_examples: Number of examples to return (max available)

    Returns:
        List of example dicts
    """
    examples = FEW_SHOT_EXAMPLES.get(language, [])
    return examples[:min(n_examples, len(examples))]


# Enhanced CoT prompt for distillation (with detailed neutral criteria)
COT_DISTILL_PROMPT = """You are an expert e-commerce review sentiment analyzer. Analyze the sentiment carefully.

## Sentiment Definitions:

**Negative (0)**: Clear dissatisfaction, complaints, or negative experiences.
- Keywords: terrible, bad, waste, broken, disappointed, refund, poor quality
- Tone: Frustrated, angry, regretful

**Positive (2)**: Clear satisfaction, praise, or positive experiences.
- Keywords: great, amazing, love, perfect, excellent, recommend, best
- Tone: Happy, enthusiastic, satisfied

**Neutral (1)**: Mixed feelings, factual statements, or mild opinions. IMPORTANT: This is the most common misclassification!
- **Mixed reviews**: Both positive and negative aspects mentioned (e.g., "Good product but shipping was slow")
- **Factual statements**: Pure description without emotion (e.g., "The color is blue, size is medium")
- **Mild opinions**: Weak or qualified praise/criticism (e.g., "It's okay", "Not bad", "Works as expected")
- **Average ratings**: 3-star equivalent sentiment
- **"But" statements**: Usually neutral when pros and cons are balanced
- Keywords: okay, average, acceptable, decent, adequate, ordinary, mediocre, fair

## Neutral Classification Examples:
- "It works well but the price is a bit high" (pro + con = neutral)
- "The product is decent, nothing special" (mild opinion)
- "Received as described" (factual)
- "It's okay for the price" (qualified/weak positive)
- "Not great, not terrible" (balanced)
- "Does what it's supposed to do" (functional/factual)

## Analysis Steps:
1. Identify key phrases and emotional words
2. Check for mixed sentiments (both positive AND negative)
3. Evaluate intensity of opinion (strong vs mild)
4. Consider the overall tone and context
5. Decide: Is there clear positive OR negative sentiment? If neither is dominant → NEUTRAL

Review: "{text}"

Provide your analysis in JSON format:
{
    "sentiment": 0/1/2,
    "confidence": 0.0-1.0,
    "reasoning": "Step-by-step analysis explaining your decision, especially if neutral"
}"""


def get_cot_distill_prompt(text: str) -> str:
    """Get CoT prompt for distillation with enhanced neutral criteria."""
    return COT_DISTILL_PROMPT.format(text=text)
