# -*- coding: utf-8 -*-
"""
Qwen3.5-9B GRPO 情感分析训练 - 从 SFT checkpoint 开始

在 SFT 微调后的模型上继续 GRPO 强化学习：
1. 加载 base model + SFT LoRA adapter
2. 在其上训练新的 GRPO LoRA
3. 细粒度奖励函数 + KL penalty

支持两种奖励模式：
- 传统模式：correct/wrong/missing/format + length penalty
- Latency-Aware 模式：更严厉惩罚 + 早期正确奖励 + 短完整奖励 + 超长惩罚

Usage:
    # Smoke test (10 steps, 20 samples)
    python train_qwen35_9b_grpo_from_sft.py --smoke

    # Quick test (100 steps, 1000 samples)
    python train_qwen35_9b_grpo_from_sft.py --test

    # Full training (默认传统模式)
    python train_qwen35_9b_grpo_from_sft.py

    # Latency-Aware 奖励函数 (推荐)
    python train_qwen35_9b_grpo_from_sft.py --latency-aware \
        --correct-reward 10 --wrong-reward -10 --missing-reward -12 \
        --early-correct-K 10 --short-target-L 16 \
        --max-completion-length-cap 32

Reward 设计 (Latency-Aware 模式):
    R = 0
    if pred is None: R += -12
    else:
        if pred == gold: R += +10
        else: R += -10
    if format_valid: R += +1
    if pred == gold and first_label_token_pos <= K: R += +3  (early correct)
    if pred == gold and format_valid and output_tokens <= L_target: R += +2  (short complete)
    if output_tokens > L_target: R += -0.1 * min(output_tokens - L_target, 30)
"""

import os
os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")

import unsloth  # noqa: F401 - import before transformers/peft so Unsloth can patch them.
import json
import re
import argparse
import numpy as np
from datasets import Dataset

# ============== CLI 参数 ==============
parser = argparse.ArgumentParser(description="Qwen3.5-9B GRPO 从 SFT checkpoint 开始训练")
parser.add_argument("--data", default="/mnt/d/kotoriee/llm/sentiment-distillation-pipeline/data/train_answer_first.json", help="训练数据路径")
parser.add_argument("--test-data", default="/mnt/d/kotoriee/llm/sentiment-distillation-pipeline/data/test_answer_first.json", help="测试数据路径")
parser.add_argument("--base-model", default="/mnt/d/kotoriee/llm/sentiment-distillation-pipeline/models/Qwen3.5-9B-ms/Qwen/Qwen3___5-9B", help="Base model 路径")
parser.add_argument("--sft-lora", default="/mnt/d/kotoriee/llm/sentiment-distillation-pipeline/3_lora_training/models/qwen35-9b-answer-first", help="SFT LoRA 路径")
parser.add_argument("--max-steps", type=int, default=1000)
parser.add_argument("--output-dir", default="outputs_grpo_9b_from_sft")
parser.add_argument("--lora-rank", type=int, default=32)
parser.add_argument("--correct-reward", type=float, default=4.0)
parser.add_argument("--wrong-reward", type=float, default=-2.0)
parser.add_argument("--missing-reward", type=float, default=-4.0)
parser.add_argument("--format-reward", type=float, default=1.0)
parser.add_argument("--beta", type=float, default=0.05)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--num-generations", type=int, default=8)
parser.add_argument("--learning-rate", type=float, default=5e-7)
parser.add_argument("--warmup-ratio", type=float, default=0.03)
parser.add_argument("--steps-per-generation", type=int, default=16)
parser.add_argument("--max-completion-length-cap", type=int, default=128)
parser.add_argument("--save-steps", type=int, default=0)
parser.add_argument("--smoke", action="store_true", help="smoke test 模式 (10步, 20条数据)")
parser.add_argument("--test", action="store_true", help="快速测试模式 (100步, 1000条数据)")
# Latency-aware reward parameters (用户提案)
parser.add_argument("--latency-aware", action="store_true", help="启用延迟感知奖励函数")
parser.add_argument("--early-correct-reward", type=float, default=3.0, help="早期正确奖励 (first token pos <= K)")
parser.add_argument("--short-complete-reward", type=float, default=2.0, help="短完整奖励 (tokens <= L_target)")
parser.add_argument("--extra-text-penalty-rate", type=float, default=0.1, help="超长惩罚率 (per token after L_target)")
parser.add_argument("--early-correct-K", type=int, default=10, help="早期正确阈值: first_label_token_pos <= K")
parser.add_argument("--short-target-L", type=int, default=16, help="短输出目标: output_tokens <= L_target")
parser.add_argument("--max-extra-penalty", type=int, default=30, help="最大超长惩罚 token 数")
args = parser.parse_args()

if args.smoke:
    args.max_steps = 10
elif args.test:
    args.max_steps = 100

# ============== 数据加载 ==============
with open(args.data, 'r', encoding='utf-8') as f:
    train_data = json.load(f)

if args.smoke:
    train_data = train_data[:20]
elif args.test:
    train_data = train_data[:1000]

print(f"训练数据: {len(train_data)} 条")

from unsloth import FastLanguageModel

max_seq_length = 4096
lora_rank = args.lora_rank

# Load the existing SFT LoRA through Unsloth. GRPOTrainer expects Unsloth's
# for_training / for_inference hooks, and saving this keeps a directly
# evaluable adapter instead of a second LoRA that depends on a merged base.
print(f"Loading SFT LoRA as trainable Unsloth model: {args.sft_lora}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.sft_lora,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=False,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.9,
)
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(tokenizer, chat_template="qwen3")
tokenizer.pad_token = tokenizer.eos_token
if hasattr(model, "print_trainable_parameters"):
    model.print_trainable_parameters()

# 启用梯度检查点
model.enable_input_require_grads()
model.gradient_checkpointing_enable()

model.train()

print("模型配置: continue training SFT LoRA with GRPO")

# ============== GRPO 格式设置 ==============

# The SFT model was trained with ANSWER-FIRST format:
# {"sentiment": N}
# <|channel>thought
# <reasoning text>
# So we align the GRPO prompt to match.
reasoning_start = "<|channel>thought"
solution_start = "{"
solution_end = "}"

system_prompt = """You are a professional e-commerce review sentiment analysis expert.

Analyze the review and classify the sentiment as:
- 0: Negative
- 1: Neutral
- 2: Positive

First output your answer as a JSON object: {"sentiment": 0/1/2}
Then provide your reasoning after the <|channel>thought tag."""

# ============== 数据格式化 ==============

def format_sentiment_data(item):
    conv = item['conversations']
    system_content = (
        conv[0]['content']
        + '\n\nIMPORTANT: Your response must start with {"sentiment": N}. '
        + 'Do not output <think> or any other text before the JSON.'
    )
    prompt_messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": conv[1]['content']},
    ]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    label = item['label']
    return {
        "prompt": prompt_text,
        "question": conv[1]['content'],
        "answer": str(label),
    }

formatted_data = [format_sentiment_data(item) for item in train_data]
dataset = Dataset.from_list(formatted_data)
print(f"GRPO 格式数据: {len(dataset)} 条")

# 按 prompt 长度过滤
def normalize_token_ids(tokens):
    while isinstance(tokens, list) and len(tokens) == 1 and isinstance(tokens[0], (str, list, tuple)):
        tokens = tokens[0]
    if isinstance(tokens, tuple):
        return list(tokens)
    return tokens


def prompt_to_tokens(prompt):
    if isinstance(prompt, str):
        return normalize_token_ids(tokenizer(text=prompt, add_special_tokens=False)["input_ids"])

    tokens = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=False,
    )
    if isinstance(tokens, dict):
        return tokens["input_ids"]
    input_ids = getattr(tokens, "input_ids", None)
    if input_ids is not None:
        return input_ids
    tokens = normalize_token_ids(tokens)
    if isinstance(tokens, str):
        return normalize_token_ids(tokenizer(text=tokens, add_special_tokens=False)["input_ids"])
    return tokens


tokenized_lengths = dataset.map(
    lambda x: {"tokens": prompt_to_tokens(x["prompt"])},
)
tokenized_lengths = tokenized_lengths.map(lambda x: {"L": len(x["tokens"])})
print(f"Sample prompt length: {tokenized_lengths['L'][0]}")
print(f"Sample prompt decoded: {tokenizer.decode(tokenized_lengths['tokens'][0])}")

maximum_length = int(np.quantile(tokenized_lengths["L"], 0.95))
print(f"Max prompt length (95th percentile) = {maximum_length}")

dataset = dataset.select(np.where(np.array(tokenized_lengths["L"]) <= maximum_length)[0])
del tokenized_lengths

max_prompt_length = maximum_length + 1
# Cap completion length aggressively. The reward focuses on answer-first JSON,
# so long chain-of-thought style completions are not useful for GRPO here.
max_completion_length = min(max_seq_length - max_prompt_length, args.max_completion_length_cap)
print(f"Max completion length (capped): {max_completion_length}")
print(
    "Reward config: "
    f"correct={args.correct_reward}, wrong={args.wrong_reward}, "
    f"missing={args.missing_reward}, format={args.format_reward}"
)

# ============== 奖励函数 ==============

# Correctness extraction must be strict enough to avoid rewarding incidental
# numbers in a long "Thinking Process". Format reward below is stricter still:
# it only rewards answer-first JSON followed by the reasoning tag.
MATCH_SENTIMENT_PATTERNS = [
    re.compile(r'\{\s*"sentiment"\s*:\s*([0-2])\s*\}'),
    re.compile(r'\b[Ss]entiment\s*[:=]\s*([0-2])\b'),
]
STRICT_ANSWER_FIRST_PATTERN = re.compile(
    r'^\s*\{\s*"sentiment"\s*:\s*([0-2])(?:\s*[,}])',
    re.DOTALL,
)

PRINTED_TIMES = 0
PRINT_EVERY_STEPS = 5 if args.smoke else 20


def extract_sentiment(text):
    """Try multiple patterns to extract sentiment value."""
    for pattern in MATCH_SENTIMENT_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(1)
    return None


def completion_to_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return completion.get("content", "")
    if isinstance(completion, (list, tuple)) and completion:
        return completion_to_text(completion[0])
    return ""


def sentiment_accuracy_reward(completions, answer, **kwargs):
    responses = [completion_to_text(completion) for completion in completions]
    extracted = [extract_sentiment(r) for r in responses]
    scores = []
    for guess, true_label in zip(extracted, answer):
        if guess is None:
            scores.append(args.missing_reward)
        elif guess == true_label:
            scores.append(args.correct_reward)
        else:
            scores.append(args.wrong_reward)
    return scores


def format_compliance_reward(completions, **kwargs):
    responses = [completion_to_text(completion) for completion in completions]
    scores = []
    for resp in responses:
        scores.append(args.format_reward if STRICT_ANSWER_FIRST_PATTERN.search(resp) else 0.0)
    return scores


def reasoning_quality_reward(completions, **kwargs):
    responses = [completion_to_text(completion) for completion in completions]
    scores = []
    for resp in responses:
        reasoning_match = re.search(
            rf"{re.escape(reasoning_start)}(.+?)$",
            resp, re.DOTALL
        )
        reasoning_text = reasoning_match.group(1).strip() if reasoning_match else ""
        reasoning_len = len(reasoning_text)

        if reasoning_len == 0:
            scores.append(-0.5)
        elif 30 <= reasoning_len <= 200:
            scores.append(1.0)
        elif 200 < reasoning_len <= 500:
            scores.append(0.0)
        else:
            scores.append(-0.5)
    return scores


def runaway_length_penalty(completions, **kwargs):
    """Penalty-only length guard. Do not reward short but wrong answers."""
    responses = [completion_to_text(completion) for completion in completions]
    scores = []
    for resp in responses:
        token_len = len(resp)  # char length as proxy
        if token_len < 800:
            scores.append(0.0)
        elif token_len < 1500:
            scores.append(-1.0)
        else:
            scores.append(-3.0)
    return scores


# ============== Latency-Aware 奖励函数 (用户提案) ==============

def find_first_label_token_position(text):
    """
    找到第一个 sentiment label token 的位置。
    对于 answer-first 格式 {"sentiment": 0/1/2}:
    返回该 token 在输出中的位置（从 0 开始）
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    sentiment_match = STRICT_ANSWER_FIRST_PATTERN.search(text)
    if not sentiment_match:
        for pattern in MATCH_SENTIMENT_PATTERNS:
            sentiment_match = pattern.search(text)
            if sentiment_match:
                break
    if not sentiment_match:
        return -1
    sentiment_value = sentiment_match.group(1)
    for i, token_id in enumerate(tokens):
        decoded_token = tokenizer.decode([token_id])
        if sentiment_value in decoded_token:
            return i
    return -1


def get_output_token_count(text):
    """获取输出 token 数量"""
    return len(tokenizer.encode(text, add_special_tokens=False))


def latency_aware_accuracy_reward(completions, answer, **kwargs):
    """
    延迟感知准确率奖励:
    R_accuracy =
        if pred is None: -12 (更严厉)
        elif pred == gold: +10 (更高奖励)
        else: -10
    """
    responses = [completion_to_text(completion) for completion in completions]
    extracted = [extract_sentiment(r) for r in responses]
    scores = []
    for guess, true_label in zip(extracted, answer):
        if guess is None:
            scores.append(args.missing_reward)
        elif guess == true_label:
            scores.append(args.correct_reward)
        else:
            scores.append(args.wrong_reward)
    return scores


def early_correct_reward(completions, answer, **kwargs):
    """
    早期正确奖励:
    R_early = +3 if pred == gold and first_label_token_pos <= K else 0
    """
    responses = [completion_to_text(completion) for completion in completions]
    extracted = [extract_sentiment(r) for r in responses]
    scores = []
    for resp, guess, true_label in zip(responses, extracted, answer):
        if guess is None or guess != true_label:
            scores.append(0.0)
            continue
        first_pos = find_first_label_token_position(resp)
        if first_pos >= 0 and first_pos <= args.early_correct_K:
            scores.append(args.early_correct_reward)
        else:
            scores.append(0.0)
    return scores


def short_complete_reward(completions, answer, **kwargs):
    """
    短完整奖励:
    R_short = +2 if pred == gold and format_valid and output_tokens <= L_target else 0
    """
    responses = [completion_to_text(completion) for completion in completions]
    extracted = [extract_sentiment(r) for r in responses]
    scores = []
    for resp, guess, true_label in zip(responses, extracted, answer):
        if guess is None or guess != true_label:
            scores.append(0.0)
            continue
        format_valid = STRICT_ANSWER_FIRST_PATTERN.search(resp) is not None
        if not format_valid:
            scores.append(0.0)
            continue
        token_count = get_output_token_count(resp)
        if token_count <= args.short_target_L:
            scores.append(args.short_complete_reward)
        else:
            scores.append(0.0)
    return scores


def extra_text_penalty(completions, **kwargs):
    """
    超长惩罚:
    R_penalty = -0.1 * min(output_tokens - L_target, 30)
    """
    responses = [completion_to_text(completion) for completion in completions]
    scores = []
    for resp in responses:
        token_count = get_output_token_count(resp)
        if token_count <= args.short_target_L:
            scores.append(0.0)
        else:
            excess = min(token_count - args.short_target_L, args.max_extra_penalty)
            scores.append(-args.extra_text_penalty_rate * excess)
    return scores


def print_sample_callback(prompts, completions, **kwargs):
    global PRINTED_TIMES
    if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
        questions = kwargs.get("question") or []
        question = questions[0] if questions else str(prompts[0])[:300]
        response = completion_to_text(completions[0])
        sentiment = extract_sentiment(response)
        true_label = kwargs.get('answer', ['?'])[0]

        if args.latency_aware:
            first_pos = find_first_label_token_position(response)
            token_count = get_output_token_count(response)
            format_valid = STRICT_ANSWER_FIRST_PATTERN.search(response) is not None

            # 计算各项奖励
            r_accuracy = 0
            r_early = 0
            r_short = 0

            if sentiment is None:
                r_accuracy = args.missing_reward
            elif sentiment == true_label:
                r_accuracy = args.correct_reward
                if first_pos >= 0 and first_pos <= args.early_correct_K:
                    r_early = args.early_correct_reward
                if format_valid and token_count <= args.short_target_L:
                    r_short = args.short_complete_reward
            else:
                r_accuracy = args.wrong_reward

            r_format = args.format_reward if format_valid else 0
            excess = max(0, min(token_count - args.short_target_L, args.max_extra_penalty))
            r_penalty = -args.extra_text_penalty_rate * excess

            total_reward = r_accuracy + r_format + r_early + r_short + r_penalty

            print(
                f"\n{'='*60}",
                f"\nReview: {question[:100]}...",
                f"\nTrue Label: {true_label}",
                f"\nExtracted: {sentiment if sentiment else 'None'}",
                f"\nFirst Label Pos: {first_pos} (K={args.early_correct_K})",
                f"\nToken Count: {token_count} (L={args.short_target_L})",
                f"\nFormat Valid: {format_valid}",
                f"\nRewards: acc={r_accuracy}, fmt={r_format}, early={r_early}, short={r_short}, penalty={r_penalty}",
                f"\nTotal Reward: {total_reward}",
                f"\nResponse preview: {response[:150]}...",
                f"\n{'='*60}"
            )
        else:
            print(
                f"\n{'='*60}",
                f"\nReview: {question[:100]}...",
                f"\nTrue Label: {true_label}",
                f"\nExtracted: {sentiment if sentiment else 'None'}",
                f"\nResponse length: {len(response)} chars",
                f"\nResponse preview: {response[:150]}...",
                f"\n{'='*60}"
            )
    PRINTED_TIMES += 1
    return [0.0] * len(completions)


# ============== 训练配置 ==============

from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    beta = args.beta,
    temperature = args.temperature,
    num_generations = args.num_generations,
    learning_rate = args.learning_rate,
    weight_decay = 0.01,
    warmup_ratio = args.warmup_ratio,
    lr_scheduler_type = "cosine",
    optim = "adamw_torch",
    per_device_train_batch_size = 1,
    steps_per_generation = args.steps_per_generation,
    max_steps = args.max_steps,
    save_steps = args.save_steps or max(5, args.max_steps // 10),
    max_prompt_length = max_prompt_length,
    max_completion_length = max_completion_length,
    logging_steps = 1,
    report_to = "none",
    output_dir = args.output_dir,
    bf16 = True,
)

# 根据 latency-aware 参数选择 reward 函数组合
if args.latency_aware:
    reward_funcs = [
        latency_aware_accuracy_reward,
        format_compliance_reward,
        early_correct_reward,
        short_complete_reward,
        extra_text_penalty,
        print_sample_callback,
    ]
    print("\n使用 Latency-Aware 奖励函数:")
    print(f"  correct_reward = {args.correct_reward}, wrong_reward = {args.wrong_reward}, missing_reward = {args.missing_reward}")
    print(f"  early_correct_reward = {args.early_correct_reward} (K={args.early_correct_K})")
    print(f"  short_complete_reward = {args.short_complete_reward} (L={args.short_target_L})")
    print(f"  extra_text_penalty_rate = {args.extra_text_penalty_rate}")
else:
    reward_funcs = [
        sentiment_accuracy_reward,
        format_compliance_reward,
        runaway_length_penalty,
        print_sample_callback,
    ]
    print("\n使用传统奖励函数:")
    print(f"  correct_reward = {args.correct_reward}, wrong_reward = {args.wrong_reward}")
    print(f"  missing_reward = {args.missing_reward}, format_reward = {args.format_reward}")

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = reward_funcs,
    args = training_args,
    train_dataset = dataset,
)

# ============== 开始训练 ==============

print("\n开始 GRPO 训练 (从 SFT checkpoint)...")
trainer.train()

# ============== 保存模型 ==============

model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
print(f"\nGRPO LoRA 保存到: {args.output_dir}")
print("\n训练完成！")
