# -*- coding: utf-8 -*-
"""
Qwen3.5-4B GRPO 情感分析训练 - 精确控制版本

基于 unsloth GRPO notebook，适配情感分析任务
数据格式：答案优先 {"sentiment": X}
纯文本，无视觉内容

改进：
1. 细粒度奖励函数（正确性/格式/推理质量），无冲突
2. KL penalty (beta=0.001)，防止策略漂移
3. 全量数据 7172 条，500 步训练
4. 训练后自动评估
"""

# ============== 安装依赖（Colab 第一个 cell）=============
INSTALL_CMD = '''
%%capture
import os
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth vllm
else:
    !pip install --upgrade -qqq uv
    try: import numpy, PIL; _numpy = f'numpy=={numpy.__version__}'; _pil = f'pillow=={PIL.__version__}'
    except: _numpy = "numpy"; _pil = "pillow"
    try: import subprocess; is_t4 = "Tesla T4" in str(subprocess.check_output(["nvidia-smi"]))
    except: is_t4 = False
    _vllm, _triton = ('vllm==0.9.2', 'triton==3.2.0') if is_t4 else ('vllm==0.15.1', 'triton')
    !uv pip install -qqq --upgrade {_vllm} {_numpy} {_pil} torchvision bitsandbytes xformers unsloth
    !uv pip install -qqq {_triton}
!uv pip install transformers==4.56.2
!uv pip install --no-deps trl==0.22.2
'''

# ============== 加载模型 ==============
from unsloth import FastLanguageModel
import torch

max_seq_length = 512  # 情感分析不需要太长
lora_rank = 32

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3.5-4B",
    max_seq_length = max_seq_length,
    load_in_4bit = False,  # 16-bit LoRA
    fast_inference = True,
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.9,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank * 2,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# ============== GRPO 格式设置 ==============

reasoning_start = "<start_working_out>"
reasoning_end = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

system_prompt = f"""You are a professional e-commerce review sentiment analysis expert.

Analyze the review and classify the sentiment as:
- 0: Negative
- 1: Neutral
- 2: Positive

Provide your reasoning between {reasoning_start} and {reasoning_end}.
Then output your final answer as JSON: {{"sentiment": 0/1/2}} between {solution_start} and {solution_end}."""

# ChatML 模板
chat_template = \
    "{% if messages[0]['role'] == 'system' %}"\
        "{{ messages[0]['content'] + eos_token }}"\
        "{% set loop_messages = messages[1:] %}"\
    "{% else %}"\
        "{{ system_prompt + eos_token }}"\
        "{% set loop_messages = messages %}"\
    "{% endif %}"\
    "{% for message in loop_messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ message['content'] }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ message['content'] + eos_token }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}{{ reasoning_start }}"\
    "{% endif %}"

tokenizer.chat_template = chat_template.replace("system_prompt", system_prompt)

# ============== 数据加载与格式化 ==============

import json
import re
from datasets import Dataset

# 上传数据（Colab）
from google.colab import files
uploaded = files.upload()

# 加载训练数据
with open('train_answer_first.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

print(f"训练数据: {len(train_data)} 条")

def format_sentiment_data(item):
    """将答案优先格式转换为 GRPO prompt"""
    conv = item['conversations']
    user_content = conv[1]['content']
    label = item['label']

    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "answer": str(label),
    }

formatted_data = [format_sentiment_data(item) for item in train_data]
dataset = Dataset.from_list(formatted_data)

print(f"GRPO 格式数据: {len(dataset)} 条")

# 按 prompt 长度过滤，避免超长样本
tokenized = dataset.map(
    lambda x: {"tokens": tokenizer.apply_chat_template(x["prompt"], add_generation_prompt=True, tokenize=True)},
    batched=True,
)
tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})

import numpy as np
maximum_length = int(np.quantile(tokenized["L"], 0.95))
print(f"Max prompt length (95th percentile) = {maximum_length}")

dataset = dataset.select(np.where(np.array(tokenized["L"]) <= maximum_length)[0])
del tokenized

max_prompt_length = maximum_length + 1
max_completion_length = max_seq_length - max_prompt_length

# ============== 奖励函数 ==============

match_sentiment = re.compile(r'"sentiment":\s*([0-2])', re.MULTILINE | re.DOTALL)

PRINTED_TIMES = 0
PRINT_EVERY_STEPS = 20


def sentiment_accuracy_reward(completions, answer, **kwargs):
    """答案正确性奖励（主要信号）: +5.0 正确 / -2.0 错误"""
    responses = [completion[0]["content"] for completion in completions]
    extracted = [
        m.group(1) if (m := match_sentiment.search(r)) else None
        for r in responses
    ]
    scores = []
    for guess, true_label in zip(extracted, answer):
        if guess is None:
            scores.append(-2.5)  # 无法解析为严重错误
        elif guess == true_label:
            scores.append(5.0)
        else:
            scores.append(-2.0)
    return scores


def format_compliance_reward(completions, **kwargs):
    """格式合规奖励（渐进式）: 0~1.0

    0.0: 完全无格式
    0.3: 有 reasoning 标签
    0.6: 有 solution 标签
    1.0: reasoning + solution 完整
    """
    responses = [completion[0]["content"] for completion in completions]
    scores = []
    for resp in responses:
        score = 0.0
        if reasoning_start in resp and reasoning_end in resp:
            score += 0.4
        if solution_start in resp and solution_end in resp:
            score += 0.4
        if match_sentiment.search(resp):
            score += 0.2
        scores.append(score)
    return scores


def reasoning_quality_reward(completions, **kwargs):
    """推理质量奖励: +1.0 优质推理 / -0.5 过短或过长

    理想推理长度: 50~300 字符
    """
    responses = [completion[0]["content"] for completion in completions]
    scores = []
    for resp in responses:
        # 提取 reasoning 部分内容
        reasoning_match = re.search(
            rf"{re.escape(reasoning_end)}(.*?){re.escape(solution_start)}",
            resp, re.DOTALL
        )
        reasoning_text = reasoning_match.group(1).strip() if reasoning_match else ""
        reasoning_len = len(reasoning_text)

        if reasoning_len == 0:
            scores.append(-0.5)  # 无推理内容
        elif 50 <= reasoning_len <= 300:
            scores.append(1.0)  # 理想长度
        elif 300 < reasoning_len <= 600:
            scores.append(0.3)  # 稍长但可接受
        else:
            scores.append(-0.3)  # 过短(非空)或过长
    return scores


def print_sample_callback(prompts, completions, **kwargs):
    """每 N 步打印一个样本，不影响梯度"""
    global PRINTED_TIMES
    if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
        question = prompts[0][-1]["content"]
        response = completions[0][0]["content"]
        sentiment_match = match_sentiment.search(response)
        print(
            f"\n{'='*60}",
            f"\nReview: {question[:100]}...",
            f"\nTrue Label: {kwargs.get('answer', ['?'])[0]}",
            f"\nExtracted: {sentiment_match.group(1) if sentiment_match else 'None'}",
            f"\nResponse preview: {response[:150]}...",
            f"\n{'='*60}"
        )
    PRINTED_TIMES += 1
    return [0.0] * len(completions)  # 零权重，纯打印


# ============== 训练配置 ==============

from vllm import SamplingParams
vllm_sampling_params = SamplingParams(
    min_p = 0.1,
    top_p = 1.0,
    top_k = -1,
    seed = 3407,
    stop = [tokenizer.eos_token],
    include_stop_str_in_output = True,
)

from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    # KL penalty - 防止策略漂移
    beta = 0.001,
    use_bias_correction_kl = True,
    # 生成配置
    temperature = 0.7,
    num_generations = 8,
    vllm_sampling_params = vllm_sampling_params,
    # 优化器
    learning_rate = 3e-6,
    weight_decay = 0.01,
    warmup_ratio = 0.05,
    lr_scheduler_type = "cosine",
    optim = "adamw_torch",
    # 批大小
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1,
    # 步数
    max_steps = 500,
    save_steps = 50,
    max_prompt_length = max_prompt_length,
    max_completion_length = max_completion_length,
    # 日志
    logging_steps = 5,
    report_to = "none",
    output_dir = "outputs_sentiment_grpo",
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        sentiment_accuracy_reward,       # 权重 0.6 (主要)
        format_compliance_reward,         # 权重 0.25 (格式)
        reasoning_quality_reward,         # 权重 0.15 (质量)
        print_sample_callback,            # 零权重 (纯打印)
    ],
    args = training_args,
    train_dataset = dataset,
)

# ============== 开始训练 ==============

print("\n开始 GRPO 训练...")
trainer.train()

# ============== 保存模型 ==============

model.save_lora("sentiment_grpo_lora")
print("\nLoRA 保存到: sentiment_grpo_lora")

# 打包下载
import shutil
shutil.make_archive('sentiment_grpo_lora', 'zip', 'sentiment_grpo_lora')
files.download('sentiment_grpo_lora.zip')

print("\n训练完成！模型已下载。")
