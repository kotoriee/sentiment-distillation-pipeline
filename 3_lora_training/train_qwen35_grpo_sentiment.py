# -*- coding: utf-8 -*-
"""
Qwen3.5-4B GRPO 情感分析训练 - 纯文本版本

基于 unsloth GRPO notebook，适配情感分析任务
数据格式：答案优先 {"sentiment": X}
纯文本，无视觉内容

关键改动：
1. 数据集改为情感分析数据
2. 奖励函数改为情感分类正确性
3. 格式改为答案优先 JSON 格式
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
    model_name = "unsloth/Qwen3.5-4B",  # 或 Qwen3-4B
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

# 情感分析专用格式
reasoning_start = "<start_working_out>"
reasoning_end = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

system_prompt = """You are a professional e-commerce review sentiment analysis expert.

Analyze the review and classify the sentiment as:
- 0: Negative
- 1: Neutral
- 2: Positive

Provide your reasoning between {reasoning_start} and {reasoning_end}.
Then output your final answer as JSON: {"sentiment": 0/1/2} between {solution_start} and {solution_end}."""

system_prompt = system_prompt.format(
    reasoning_start=reasoning_start,
    reasoning_end=reasoning_end,
    solution_start=solution_start,
    solution_end=solution_end,
)

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
import pandas as pd
from datasets import Dataset

# 上传数据
from google.colab import files
uploaded = files.upload()

# 加载训练数据
with open('train_answer_first.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

# 加载测试数据（用于奖励计算）
with open('test_answer_first.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

print(f"训练数据: {len(train_data)} 条")
print(f"测试数据: {len(test_data)} 条")

# 格式化数据为 GRPO 格式
def format_sentiment_data(item):
    """将答案优先格式转换为 GRPO 格式"""
    conv = item['conversations']
    user_content = conv[1]['content']  # Review
    label = item['label']

    # 提取原有 reasoning
    assistant_content = conv[2]['content']
    # 去掉 JSON 部分，只保留 reasoning
    reasoning = assistant_content.split('\n')[1] if '\n' in assistant_content else ""
    reasoning = reasoning.replace('<|channel>thought', '').replace('<channel|>', '').strip()

    # 构建新的 GRPO 格式响应
    response = f"{reasoning_start}{reasoning}{reasoning_end}" \
               f"{solution_start}{{\"sentiment\": {label}}}{solution_end}"

    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "answer": str(label),  # 真实标签用于奖励计算
    }

# 转换数据
formatted_data = [format_sentiment_data(item) for item in train_data[:1000]]  # 先用1000条测试
dataset = Dataset.from_list(formatted_data)

print(f"GRPO 格式数据: {len(dataset)} 条")

# ============== 奖励函数 ==============

# 匹配格式正则
solution_end_regex = r"</SOLUTION>[\s]{0,}" + \
    "(?:" + re.escape(tokenizer.eos_token) + ")?"

match_format = re.compile(
    rf"{reasoning_end}.*?"\
    rf"{solution_start}(.+?){solution_end_regex}"\
    rf"[\s]{{0,}}$",
    flags = re.MULTILINE | re.DOTALL
)

# 匹配 sentiment 数字
match_sentiment = re.compile(
    r'"sentiment":\s*([0-2])',
    flags = re.MULTILINE | re.DOTALL
)

def match_format_exactly(completions, **kwargs):
    """格式完全正确奖励 +3"""
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        if match_format.search(response) is not None:
            score += 3.0
        scores.append(score)
    return scores

def match_format_approximately(completions, **kwargs):
    """部分格式正确奖励"""
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        score += 0.5 if response.count(reasoning_end) == 1 else -1.0
        score += 0.5 if response.count(solution_start) == 1 else -1.0
        score += 0.5 if response.count(solution_end) == 1 else -1.0
        scores.append(score)
    return scores

def check_sentiment(prompts, completions, answer, **kwargs):
    """情感分类正确性奖励"""
    responses = [completion[0]["content"] for completion in completions]

    # 提取预测的 sentiment
    extracted_responses = [
        match_sentiment.search(r).group(1) if match_sentiment.search(r) is not None else None
        for r in responses
    ]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(-2.0)
            continue
        # 正确分类 +5
        if guess == true_answer:
            score += 5.0
        else:
            score -= 1.5  # 错误分类惩罚
        scores.append(score)
    return scores

global PRINTED_TIMES
PRINTED_TIMES = 0
PRINT_EVERY_STEPS = 10

def check_sentiment_printed(prompts, completions, answer, **kwargs):
    """打印输出的奖励函数"""
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        match_sentiment.search(r).group(1) if match_sentiment.search(r) is not None else None
        for r in responses
    ]

    global PRINTED_TIMES
    global PRINT_EVERY_STEPS
    if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
        print(
            '*'*20 + f"\nReview:\n{question[:100]}...",
            f"\nTrue Label: {answer[0]}",
            f"\nResponse:\n{responses[0][:200]}...",
            f"\nExtracted: {extracted_responses[0]}"
        )
    PRINTED_TIMES += 1

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(-2.5)
            continue
        scores.append(3.5 if guess == true_answer else -1.5)
    return scores

# ============== 训练配置 ==============

# 计算 max_prompt_length
tokenized = dataset.map(
    lambda x: {"tokens": tokenizer.apply_chat_template(x["prompt"], add_generation_prompt=True, tokenize=True)},
    batched=True,
)
tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})

import numpy as np
maximum_length = int(np.quantile(tokenized["L"], 0.9))
print("Max prompt length = ", maximum_length)

dataset = dataset.select(np.where(np.array(tokenized["L"]) <= maximum_length)[0])
del tokenized

max_prompt_length = maximum_length + 1
max_completion_length = max_seq_length - max_prompt_length

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
    vllm_sampling_params = vllm_sampling_params,
    temperature = 1.0,
    learning_rate = 5e-6,
    weight_decay = 0.001,
    warmup_ratio = 0.1,
    lr_scheduler_type = "linear",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1,
    num_generations = 4,
    max_prompt_length = max_prompt_length,
    max_completion_length = max_completion_length,
    max_steps = 100,  # 先测试100步
    save_steps = 100,
    report_to = "none",
    output_dir = "outputs_sentiment_grpo",
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        match_format_exactly,
        match_format_approximately,
        check_sentiment,
        check_sentiment_printed,
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