# -*- coding: utf-8 -*-
"""
Qwen3.5-4B 答案优先格式软标签蒸馏训练 - Google Colab 版本

基于本地已有的训练脚本和数据集，适配 Colab 免费 T4 GPU

数据格式：答案优先 {"sentiment": X}
训练方法：软标签蒸馏 (KL + SFT 混合损失)

使用方法：
1. 在 Colab 中运行此脚本
2. 上传 train_answer_first.json 数据文件
3. 训练完成后下载模型

预计训练时间：~2小时 (T4 GPU, 7172样本, 3 epochs)
"""

# ============== 安装依赖 ==============
# 在 Colab 第一个 cell 运行

import os, importlib.util

# Colab 安装命令（复制到第一个 cell）
INSTALL_CMD = '''
%%capture
import os, importlib.util
!pip install --upgrade -qqq uv
if importlib.util.find_spec("torch") is None or "COLAB_" in "".join(os.environ.keys()):
    try: import numpy, PIL; _numpy = f"numpy=={numpy.__version__}"; _pil = f"pillow=={PIL.__version__}"
    except: _numpy = "numpy"; _pil = "pillow"
    !uv pip install -qqq \
        "torch==2.8.0" "triton>=3.3.0" {_numpy} {_pil} torchvision bitsandbytes xformers==0.0.32.post2 \
        "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
        "unsloth[base] @ git+https://github.com/unslothai/unsloth"
elif importlib.util.find_spec("unsloth") is None:
    !uv pip install -qqq unsloth
!uv pip install --upgrade --no-deps tokenizers trl==0.22.2 unsloth unsloth_zoo
!uv pip install transformers==5.2.0
!uv pip install --no-build-isolation flash-linear-attention causal_conv1d==1.6.0
'''

# ============== 数据准备 ==============

# 数据文件上传命令（复制到第二个 cell）
DATA_UPLOAD_CMD = '''
# 上传训练数据
from google.colab import files
uploaded = files.upload()

# 保存上传的文件
import json
for fn in uploaded.keys():
    with open(fn, 'wb') as f:
        f.write(uploaded[fn])
    print(f'上传文件: {fn}')

# 验证数据
with open('train_answer_first.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f'数据量: {len(data)} 条')
print(f'格式检查: {data[0]["conversations"][2]["content"][:30]}...')
'''

# ============== 训练脚本主体 ==============

import torch
import torch.nn.functional as F
import json
import numpy as np
from pathlib import Path

MAX_SEQ_LENGTH = 512
LORA_RANK = 16
RANDOM_STATE = 3407

def train_qwen35_answer_first(
    train_data_path: str = "train_answer_first.json",
    output_dir: str = "qwen35-4b-answer-first",
    epochs: int = 3,
    temperature: float = 2.0,
    alpha: float = 0.5,
    use_dynamic_temp: bool = False,
):
    """
    Qwen3.5-4B 答案优先格式训练

    Args:
        train_data_path: 训练数据路径
        output_dir: 输出目录
        epochs: 训练轮数
        temperature: KL温度
        alpha: KL权重
        use_dynamic_temp: 是否使用动态温度策略
    """

    print("=" * 60)
    print("Qwen3.5-4B 答案优先格式软标签蒸馏训练")
    print("=" * 60)

    # 加载模型
    from unsloth import FastLanguageModel

    print("\n加载模型: Qwen/Qwen3.5-4B")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen3.5-4B",
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,  # Colab T4 用 16-bit
        load_in_16bit=True,
        use_gradient_checkpointing="unsloth",
    )

    # LoRA 配置
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_RANK,
        lora_dropout=0,
        bias="none",
        random_state=RANDOM_STATE,
    )

    print(f"\nLoRA 参数: r={LORA_RANK}, alpha={LORA_RANK}")

    # 加载数据
    with open(train_data_path, encoding="utf-8") as f:
        data = json.load(f)

    print(f"数据: {len(data)} 条")

    # 动态温度策略
    def adaptive_temperature(confidence: float) -> float:
        if confidence > 0.9:
            return 1.5
        elif confidence > 0.6:
            return 2.0
        else:
            return min(2.5 + (0.6 - confidence) * 2, 3.0)

    # 预处理数据
    sentiment_ids = [
        tokenizer.encode('0', add_special_tokens=False)[0],
        tokenizer.encode('1', add_special_tokens=False)[0],
        tokenizer.encode('2', add_special_tokens=False)[0],
    ]
    print(f"Sentiment token IDs: {sentiment_ids}")

    records = []
    for item in data:
        conv = item['conversations']

        # ChatML 格式
        full_text = tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=False
        )
        prefix_text = tokenizer.apply_chat_template(
            conv[:-1], tokenize=False, add_generation_prompt=True
        )

        full_ids = tokenizer.encode(full_text, add_special_tokens=False)[:MAX_SEQ_LENGTH]
        prefix_len = min(len(tokenizer.encode(prefix_text, add_special_tokens=False)), len(full_ids))

        labels = [-100] * prefix_len + full_ids[prefix_len:]

        # 定位 sentiment token
        sentiment_pos = -1
        target_str = '"sentiment": '
        idx = full_text.find(target_str)
        if idx != -1:
            pos = len(tokenizer.encode(full_text[:idx + len(target_str)], add_special_tokens=False))
            if pos < len(full_ids) and full_ids[pos] in sentiment_ids:
                sentiment_pos = pos

        # 置信度
        soft_labels = item.get('soft_labels', [0.33, 0.33, 0.34])
        confidence = max(soft_labels)

        records.append({
            'input_ids': full_ids,
            'attention_mask': [1] * len(full_ids),
            'labels': labels,
            'soft_labels': soft_labels,
            'sentiment_pos': sentiment_pos,
            'confidence': confidence,
        })

    print(f"Sentiment 定位成功: {sum(1 for r in records if r['sentiment_pos'] != -1)}/{len(records)}")

    # 创建 Dataset
    from datasets import Dataset as HFDataset
    dataset = HFDataset.from_list(records)

    # DataCollator
    class DataCollator:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def __call__(self, features):
            soft_labels = [f.pop('soft_labels') for f in features]
            sentiment_pos = [f.pop('sentiment_pos') for f in features]
            confidence = [f.pop('confidence') for f in features]

            max_len = max(len(f['input_ids']) for f in features)
            pad_id = self.tokenizer.pad_token_id or 0

            input_ids = torch.tensor([
                f['input_ids'] + [pad_id] * (max_len - len(f['input_ids']))
                for f in features
            ], dtype=torch.long)

            attention_mask = torch.tensor([
                f['attention_mask'] + [0] * (max_len - len(f['attention_mask']))
                for f in features
            ], dtype=torch.long)

            labels = torch.tensor([
                f['labels'] + [-100] * (max_len - len(f['labels']))
                for f in features
            ], dtype=torch.long)

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'soft_labels': torch.tensor(soft_labels, dtype=torch.float32),
                'sentiment_pos': torch.tensor(sentiment_pos, dtype=torch.long),
                'confidence': torch.tensor(confidence, dtype=torch.float32),
            }

    # 自定义 compute_loss
    def compute_loss(model, inputs, return_outputs=False, **kwargs):
        soft_labels = inputs.pop("soft_labels", None)
        sentiment_pos = inputs.pop("sentiment_pos", None)
        labels = inputs.pop("labels", None)
        confidences = inputs.pop("confidence", None)

        outputs = model(**inputs)
        logits = outputs.logits

        # SFT Loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        if (shift_labels != -100).sum() == 0:
            sft_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        else:
            sft_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        # KL Loss
        kl_losses = []
        temps_used = []

        if soft_labels is not None and sentiment_pos is not None:
            soft_labels = soft_labels.to(logits.device)

            for b in range(logits.size(0)):
                pos = sentiment_pos[b].item()
                if pos == -1:
                    continue

                target_pos = pos - 1
                if target_pos < 0 or target_pos >= logits.size(1):
                    continue

                # 温度选择
                if use_dynamic_temp:
                    confidence = confidences[b].item() if confidences is not None else 0.5
                    temp = adaptive_temperature(confidence)
                else:
                    temp = temperature
                temps_used.append(temp)

                # KL 计算
                sent_logits = logits[b, target_pos, sentiment_ids].float()
                sent_logits = sent_logits / temp
                student_log_probs = F.log_softmax(sent_logits, dim=-1)
                teacher_probs = soft_labels[b]

                kl = F.kl_div(student_log_probs, teacher_probs, reduction='sum')
                kl = kl * (temp ** 2)
                kl_losses.append(kl)

        if kl_losses:
            kl_loss = torch.stack(kl_losses).mean()
            total_loss = alpha * kl_loss + (1 - alpha) * sft_loss
        else:
            kl_loss = torch.tensor(0.0, device=logits.device)
            total_loss = sft_loss

        # Debug 输出
        if not hasattr(compute_loss, '_step'):
            compute_loss._step = 0
        if compute_loss._step < 5:
            avg_temp = np.mean(temps_used) if temps_used else temperature
            print(f"[{compute_loss._step}] sft={sft_loss.item():.4f} kl={kl_loss.item():.4f} "
                  f"total={total_loss.item():.4f} avg_temp={avg_temp:.2f}")
            compute_loss._step += 1

        return (total_loss, outputs) if return_outputs else total_loss

    # 训练配置
    n_examples = len(data)
    total_steps = (n_examples // 16) * epochs  # batch=1, grad_acc=16
    warmup_steps = max(1, int(total_steps * 0.05))

    from transformers import TrainingArguments, Trainer

    training_args = TrainingArguments(
        output_dir=output_dir + "_checkpoints",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_steps=warmup_steps,
        num_train_epochs=epochs,
        learning_rate=2e-5,
        logging_steps=20,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=RANDOM_STATE,
        report_to="none",
        bf16=True,
        fp16=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollator(tokenizer),
    )

    # 替换 compute_loss
    trainer.compute_loss = compute_loss

    print(f"\n开始训练:")
    print(f"  Epochs: {epochs}")
    print(f"  Temperature: {temperature if not use_dynamic_temp else 'adaptive (1.5-3.0)'}")
    print(f"  Alpha: {alpha}")

    # 显示内存状态
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"\nGPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    # 开始训练
    trainer_stats = trainer.train()

    # 显示训练统计
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

    print(f"\n{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")

    # 保存模型
    print(f"\n保存模型到 {output_dir}...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 配置记录
    config = {
        "model": "Qwen/Qwen3.5-4B",
        "data": train_data_path,
        "epochs": epochs,
        "temperature": temperature if not use_dynamic_temp else "adaptive",
        "alpha": alpha,
        "format": "answer_first",
        "train_time_seconds": trainer_stats.metrics['train_runtime'],
    }
    with open(Path(output_dir) / "train_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n训练完成!")
    print(f"模型保存: {output_dir}")

    return trainer_stats


# ============== 下载模型 ==============

# 下载命令（复制到最后一个 cell）
DOWNLOAD_CMD = '''
# 下载模型到本地
from google.colab import files
import os

# 打包模型文件
import shutil
shutil.make_archive('qwen35-4b-answer-first', 'zip', 'qwen35-4b-answer-first')

# 下载
files.download('qwen35-4b-answer-first.zip')
print('模型已下载!')
'''

# ============== 主函数 ==============

if __name__ == "__main__":
    # 固定温度训练
    train_qwen35_answer_first(
        train_data_path="train_answer_first.json",
        output_dir="qwen35-4b-answer-first",
        epochs=3,
        temperature=2.0,
        alpha=0.5,
        use_dynamic_temp=False,
    )

    # 可选：动态温度训练
    # train_qwen35_answer_first(
    #     train_data_path="train_answer_first.json",
    #     output_dir="qwen35-4b-adaptive-temp",
    #     epochs=3,
    #     use_dynamic_temp=True,
    # )