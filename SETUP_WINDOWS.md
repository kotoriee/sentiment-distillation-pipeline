# Windows 原生环境配置指南

适用：Windows 10/11 + NVIDIA GPU + 新电脑干净环境

---

## 1. 安装 Anaconda/Miniconda

### 方式A：Anaconda（图形界面，推荐新手）

1. 下载：https://www.anaconda.com/download
2. 安装时勾选 "Add Anaconda to PATH"
3. 打开 **Anaconda Prompt**

### 方式B：Miniconda（轻量，推荐熟悉用户）

1. 下载：https://docs.conda.io/en/latest/miniconda.html
2. 安装后打开 CMD/PowerShell：
```powershell
%USERPROFILE%\miniconda3\Scripts\conda.exe init powershell
# 重启 PowerShell
```

---

## 2. 创建虚拟环境

```powershell
# Anaconda Prompt 或 PowerShell
conda create -n sentiment python=3.10 -y
conda activate sentiment
```

---

## 3. 验证 GPU 驱动

```powershell
# CMD 或 PowerShell
nvidia-smi
```

预期输出：
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx      Driver Version: 535.xx      CUDA Version: 12.2     |
| ...                                                                         |
|   0  RTX 3070      Off  | ...                                               |
+-----------------------------------------------------------------------------+
```

---

## 4. 安装 Unsloth（关键）

**方式A：官方脚本（推荐）**

```powershell
# PowerShell
curl -fsSL https://unsloth.ai/install.sh | sh
```

**方式B：pip 安装（Windows兼容）**

```powershell
pip install unsloth
pip install --no-deps transformers==4.45.0
pip install bitsandbytes trl peft datasets
```

---

## 5. 安装其他依赖

```powershell
# 克隆项目（或手动下载ZIP解压）
git clone https://github.com/kotoriee/sentiment-distillation-pipeline.git
cd sentiment-distillation-pipeline

# 安装依赖
pip install -r requirements.txt
```

---

## 6. 验证安装

```powershell
python -c "from unsloth import FastLanguageModel; print('FastLanguageModel OK')"
python -c "from unsloth import FastModel; print('FastModel OK')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

---

## 7. 测试训练

```powershell
cd 3_lora_training
python train_soft_label.py --data ../data/train_700.json --test
```

---

## Windows 特有注意事项

### 1. 长路径限制

Windows 默认限制路径长度260字符，可能影响模型文件：

```powershell
# 解决方法：启用长路径支持
# 以管理员打开 PowerShell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
    -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

### 2. 文件编码

编辑 Python 文件时使用 UTF-8 编码，避免 GBK 错误。

### 3. CUDA 版本匹配

检查 PyTorch CUDA 版本：

```python
import torch
print(torch.version.cuda)  # 应为 11.8 或 12.x
```

如果不匹配，重新安装 PyTorch：

```powershell
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 常见问题

### Q: `bitsandbytes` 安装失败

Windows 原生可能有问题，使用 4-bit 量化替代方案：

```powershell
pip install bitsandbytes-windows
# 或使用 load_in_8bit=True 替代 load_in_4bit=True
```

### Q: 模型下载慢

```powershell
# 使用镜像
set HF_ENDPOINT=https://hf-mirror.com

# 或设置代理
set HTTP_PROXY=http://127.0.0.1:7890
set HTTPS_PROXY=http://127.0.0.1:7890
```

---

## 快速启动脚本（Windows）

保存为 `setup_windows.bat`：

```batch
@echo off
echo === 创建 conda 环境 ===
conda create -n sentiment python=3.10 -y
conda activate sentiment

echo === 安装 Unsloth ===
pip install unsloth
pip install --no-deps transformers==4.45.0
pip install bitsandbytes trl peft datasets

echo === 安装其他依赖 ===
pip install jieba nltk natasha scikit-learn matplotlib seaborn

echo === 克隆项目 ===
git clone https://github.com/kotoriee/sentiment-distillation-pipeline.git
cd sentiment-distillation-pipeline

echo === 验证安装 ===
python -c "import torch; from unsloth import FastLanguageModel; print('OK')"

echo === 完成 ===
echo 运行训练: cd 3_lora_training && python train_soft_label.py --data ../data/train_700.json --test
pause
```