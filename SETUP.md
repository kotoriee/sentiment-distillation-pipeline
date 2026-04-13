# 本地环境配置指南

## 适用场景

- 新电脑（干净环境）
- RTX 3070 8GB VRAM 或同等显卡
- Windows + WSL2 或 Linux

---

## 1. 基础环境准备

### 1.1 安装 WSL2 (Windows用户)

```powershell
# PowerShell (管理员)
wsl --install -d Ubuntu-22.04

# 重启后进入 WSL
wsl
```

### 1.2 安装 Miniconda

```bash
# 下载 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 安装
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# 初始化
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

# 添加到 bashrc
echo 'eval "$($HOME/miniconda3/bin/conda shell.bash hook)"' >> ~/.bashrc
source ~/.bashrc
```

### 1.3 创建专用虚拟环境

```bash
# 创建干净环境（关键：避免依赖冲突）
conda create -n sentiment python=3.10 -y

# 激活环境
conda activate sentiment

# 确认 Python 版本
python --version  # 应显示 Python 3.10.x
```

---

## 2. GPU 驱动验证

### 2.1 NVIDIA 驾动检查

```bash
# 检查驱动版本
nvidia-smi

# 预期输出：
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.xx      Driver Version: 535.xx      CUDA Version: 12.2     |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |                               |                      |               MIG M. |
# |===============================+======================+======================|
# |   0  RTX 3070      Off        | 00000000:01:00.0  On |                  N/A |
# | 30%   45C    P8    15W / 220W |    500MiB /  8192MiB |      4%      Default |
# +-------------------------------+----------------------+----------------------+
```

### 2.2 CUDA 版本确认

```bash
# 检查 CUDA 版本（需要 >= 11.8）
nvcc --version

# 如果未安装 nvcc，通过 nvidia-smi 查看支持的 CUDA 版本
# Driver Version 对应的 CUDA Version 即为支持的最高版本
```

---

## 3. Unsloth 安装（关键步骤）

### ⚠️ 重要提示

**不要使用 `pip install unsloth`**，会导致 transformers 版本冲突。

必须使用官方安装脚本：

```bash
# 官方安装脚本（解决所有依赖冲突）
curl -fsSL https://unsloth.ai/install.sh | sh

# 等待安装完成（约2-3分钟）
# 脚本会自动安装：
# - unsloth
# - transformers (正确版本)
# - torch
# - triton
# - 其他依赖
```

### 3.1 验证 Unsloth 安装

```bash
# 测试 FastLanguageModel（Qwen3-4B）
python -c "
from unsloth import FastLanguageModel
print('FastLanguageModel OK')
"

# 测试 FastModel（Gemma 4）
python -c "
from unsloth import FastModel
print('FastModel OK')
"

# 测试 chat templates
python -c "
from unsloth.chat_templates import get_chat_template
print('Chat templates OK')
"
```

---

## 4. 其他依赖安装

```bash
# 切换到项目目录后安装
pip install -r requirements.txt

# 或手动安装核心依赖
pip install datasets trl peft bitsandbytes
pip install jieba nltk natasha  # NLP预处理
pip install scikit-learn matplotlib seaborn  # 评估
```

---

## 5. 项目克隆与数据获取

### 5.1 克隆仓库

```bash
# 克隆到工作目录
cd ~/projects  # 或你的工作目录
git clone https://github.com/kotoriee/sentiment-distillation-pipeline.git

# 进入项目
cd sentiment-distillation-pipeline
```

### 5.2 验证数据完整性

```bash
# 检查数据文件
ls -la data/

# 预期输出：
# train.json      (~8.5MB, 7172条)
# val.json        (~1.1MB, 896条)
# test.json       (~1.1MB, 897条)
# train_700.json  (~1.9MB, 700条)
# conversations/  (~24MB, 3个文件)
```

### 5.3 数据质量检查

```bash
cd 1_data_preprocessing
python quality_check.py --data ../data/train.json --check all
```

---

## 6. GPU 显存验证

### 6.1 测试模型加载

**测试 Qwen3-4B（~5GB VRAM）**：

```bash
python -c "
import torch
from unsloth import FastLanguageModel

print('加载 Qwen3-4B...')
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='Qwen/Qwen2.5-4B-Instruct',
    max_seq_length=512,
    load_in_4bit=True,
)

# 检查显存
gpu_mem = torch.cuda.max_memory_reserved() / 1024**3
print(f'显存占用: {gpu_mem:.2f} GB')
print('Qwen3-4B 加载成功!')
"
```

**测试 Gemma 4 E2B（~4GB VRAM）**：

```bash
python -c "
import torch
from unsloth import FastModel

print('加载 Gemma 4 E2B...')
model, tokenizer = FastModel.from_pretrained(
    model_name='unsloth/gemma-4-E2B-it',
    max_seq_length=512,
    load_in_4bit=True,
)

# 检查显存
gpu_mem = torch.cuda.max_memory_reserved() / 1024**3
print(f'显存占用: {gpu_mem:.2f} GB')
print('Gemma 4 E2B 加载成功!')
"
```

---

## 7. 训练测试

### 7.1 小批量训练（Qwen3-4B）

```bash
cd 3_lora_training

# 测试模式（30步，约10分钟）
python train_soft_label.py --data ../data/train_700.json --test

# 观察输出：
# - Loss 应在合理范围（Qwen3: ~2-3）
# - 显存峰值应 < 8GB
# - 模型保存到 models/ 目录
```

### 7.2 小批量训练（Gemma 4 E2B）

```bash
# 测试模式（30步，约10分钟）
python train_gemma4.py --data ../data/conversations/train_conversations.json --test

# 注意：Gemma 4 的 Loss 在 13-15 范围是正常的（多模态特性）
```

---

## 8. 评估测试

```bash
cd 4_evaluation

# 评估模型（50样本测试）
python eval_model.py \
    --model ../3_lora_training/models/rationale-soft-v2 \
    --data ../data/test.json \
    --samples 50
```

---

## 9. 常见问题排查

### Q1: Unsloth 安装失败

```bash
# 检查 transformers 版本冲突
pip list | grep transformers

# 如果版本不对，手动修复
pip uninstall transformers
pip install transformers==4.45.0  # Unsloth兼容版本
```

### Q2: CUDA out of memory

```bash
# 降低 batch size
python train_soft_label.py --data ../data/train_700.json --test --batch 1 --grad-acc 8

# 或减小序列长度
# 修改脚本中 MAX_SEQ_LENGTH = 512 → 256
```

### Q3: 模型下载慢

```bash
# 使用 ModelScope 镜像（中国用户）
export HF_ENDPOINT=https://hf-mirror.com

# 或设置代理
export HF_ENDPOINT=https://huggingface.co
export HTTP_PROXY=http://127.0.0.1:7890
```

---

## 10. 完整安装清单

```
✅ WSL2 Ubuntu-22.04
✅ Miniconda3
✅ conda env: sentiment (Python 3.10)
✅ NVIDIA Driver >= 535
✅ CUDA >= 11.8
✅ Unsloth (官方脚本安装)
✅ torch >= 2.4.0
✅ transformers <= 5.5.0
✅ datasets, trl, peft, bitsandbytes
✅ jieba, nltk, natasha
✅ scikit-learn, matplotlib, seaborn
✅ 项目克隆完成
✅ 数据文件完整
✅ GPU测试通过
✅ 训练测试通过
```

---

## 快速启动脚本

保存为 `setup.sh`：

```bash
#!/bin/bash
# 快速环境配置脚本（新电脑）

# 1. 创建 conda 环境
conda create -n sentiment python=3.10 -y
conda activate sentiment

# 2. 安装 Unsloth
curl -fsSL https://unsloth.ai/install.sh | sh

# 3. 安装其他依赖
pip install datasets trl peft bitsandbytes
pip install jieba nltk natasha scikit-learn matplotlib seaborn

# 4. 克隆项目
git clone https://github.com/kotoriee/sentiment-distillation-pipeline.git
cd sentiment-distillation-pipeline

# 5. 验证安装
python -c "
import torch
from unsloth import FastLanguageModel, FastModel
print('✅ 环境配置完成')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# 6. 运行质量检查
cd 1_data_preprocessing
python quality_check.py --data ../data/train.json --check all

echo "🎉 安装完成！开始训练："
echo "cd 3_lora_training && python train_soft_label.py --data ../data/train_700.json --test"
```