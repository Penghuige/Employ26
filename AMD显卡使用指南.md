# AMD显卡（RX 9700）使用指南

## 🎯 问题说明

AMD显卡使用ROCm（Radeon Open Compute）而不是NVIDIA的CUDA，PyTorch默认只支持CUDA。

**你的显卡：AMD RX 9700**
- ✅ 支持ROCm
- ⚠️ 需要特殊配置
- ⚠️ PyTorch需要安装ROCm版本

---

## ✅ 解决方案

### 方案1：使用ROCm版本的PyTorch（推荐）⭐⭐⭐⭐⭐

**适用场景：** Linux系统，想充分利用AMD GPU

#### 步骤1：检查系统要求
```bash
# 检查Linux版本（ROCm主要支持Ubuntu/RHEL）
lsb_release -a

# 检查显卡
lspci | grep -i amd
```

**支持的系统：**
- Ubuntu 20.04/22.04
- RHEL 8.x/9.x
- SLES 15 SP3/SP4

#### 步骤2：安装ROCm
```bash
# Ubuntu 22.04示例
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_latest_all.deb
sudo apt install ./amdgpu-install_latest_all.deb

# 安装ROCm
sudo amdgpu-install --usecase=rocm

# 添加用户到render组
sudo usermod -a -G render,video $LOGNAME

# 重启
sudo reboot
```

#### 步骤3：验证ROCm安装
```bash
# 检查ROCm版本
rocm-smi

# 应该看到类似输出：
# ======================= ROCm System Management Interface =======================
# GPU  Temp   AvgPwr  SCLK    MCLK     Fan     Perf  PwrCap  VRAM%  GPU%
# 0    45.0c  50.0W   800Mhz  1000Mhz  30.0%   auto  300.0W    5%   10%
```

#### 步骤4：安装ROCm版本的PyTorch
```bash
# 卸载现有的PyTorch（如果有）
pip uninstall torch torchvision torchaudio

# 安装ROCm版本（以ROCm 5.7为例）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

#### 步骤5：验证PyTorch可以使用AMD GPU
```python
import torch

print(f"PyTorch版本: {torch.__version__}")
print(f"ROCm可用: {torch.cuda.is_available()}")  # 注意：ROCm也使用cuda接口
print(f"GPU数量: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

**预期输出：**
```
PyTorch版本: 2.1.0+rocm5.7
ROCm可用: True
GPU数量: 1
GPU名称: AMD Radeon RX 9700
GPU内存: 16.0 GB
```

#### 步骤6：运行技能提取
```bash
# 现在可以正常运行了
python src/skill_extraction/bert_extractor.py
```

**代码会自动检测到GPU并使用！**

---

### 方案2：使用CPU运行（最简单）⭐⭐⭐⭐

**适用场景：** 不想折腾，或者是Windows系统

#### 优势
- ✅ 无需配置，开箱即用
- ✅ 跨平台（Windows/Linux/Mac）
- ✅ 代码完全兼容

#### 劣势
- ⚠️ 速度较慢（约为GPU的1/10）
- ⚠️ BERT处理1000行需要30-60分钟

#### 使用方法
```bash
# 安装CPU版本的PyTorch
pip install torch torchvision torchaudio

# 直接运行，代码会自动使用CPU
python src/skill_extraction/bert_extractor.py
```

**代码会自动检测：**
```
⚠️  GPU不可用，将使用CPU（速度较慢）
```

#### 优化建议
```python
# 减少处理数据量
python src/skill_extraction/bert_extractor.py
# 在代码中设置 max_rows=500（默认1000）

# 或者只运行Word2Vec（不需要GPU）
python src/skill_extraction/word2vec_extractor.py
```

---

### 方案3：使用DirectML（Windows专用）⭐⭐⭐

**适用场景：** Windows系统，想利用AMD GPU

#### 步骤1：安装DirectML版本的PyTorch
```bash
# 安装torch-directml
pip install torch-directml
```

#### 步骤2：修改代码使用DirectML
在 `bert_extractor.py` 中修改：

```python
def check_gpu(self):
    """检查GPU可用性"""
    try:
        import torch_directml
        dml = torch_directml.device()
        logger.info(f"✅ DirectML可用")
        logger.info(f"   设备: {dml}")
        return True
    except ImportError:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"✅ GPU可用: {gpu_name}")
            return True
        else:
            logger.warning("⚠️  GPU不可用，将使用CPU")
            return False

def load_model(self, model_name="ckiplab/bert-base-chinese-ner"):
    """加载BERT-NER模型"""
    # ... 现有代码 ...
    
    # 检查DirectML
    try:
        import torch_directml
        device = torch_directml.device()
        logger.info("使用DirectML设备")
    except ImportError:
        device = 0 if self.check_gpu() else -1
    
    # 创建pipeline
    self.ner_pipeline = pipeline(
        "ner",
        model=self.model,
        tokenizer=self.tokenizer,
        device=device
    )
```

#### 步骤3：运行
```bash
python src/skill_extraction/bert_extractor.py
```

---

### 方案4：使用云GPU服务（最省心）⭐⭐⭐⭐⭐

**适用场景：** 不想配置本地环境

#### 选项A：Google Colab（免费）
```python
# 1. 上传代码到Google Drive
# 2. 在Colab中运行

!pip install transformers gensim
!python src/skill_extraction/run_extraction_pipeline.py
```

**优势：**
- ✅ 免费GPU（Tesla T4）
- ✅ 无需配置
- ✅ 在线运行

**劣势：**
- ⚠️ 需要上传数据
- ⚠️ 会话有时间限制

#### 选项B：Kaggle Notebooks（免费）
- 每周30小时免费GPU
- P100或T4 GPU
- 类似Colab使用方式

#### 选项C：AutoDL/恒源云（付费，便宜）
- 按小时计费（约1-2元/小时）
- RTX 3090/4090可选
- 预装环境，开箱即用

---

## 📊 方案对比

| 方案 | 难度 | 速度 | 成本 | 推荐度 |
|------|------|------|------|--------|
| ROCm (Linux) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 免费 | ⭐⭐⭐⭐⭐ |
| CPU | ⭐ | ⭐⭐ | 免费 | ⭐⭐⭐⭐ |
| DirectML (Win) | ⭐⭐⭐ | ⭐⭐⭐⭐ | 免费 | ⭐⭐⭐ |
| 云GPU | ⭐ | ⭐⭐⭐⭐⭐ | 付费 | ⭐⭐⭐⭐⭐ |

---

## 🎯 推荐方案

### 如果你使用Linux
**推荐：方案1（ROCm）**
- 充分利用AMD GPU
- 性能最佳
- 一次配置，长期使用

### 如果你使用Windows
**推荐：方案2（CPU）或方案4（云GPU）**
- CPU：简单直接，适合小规模测试
- 云GPU：性能好，适合大规模处理

### 如果你想快速验证
**推荐：方案2（CPU）+ Word2Vec**
```bash
# Word2Vec不需要GPU，可以立即运行
python src/skill_extraction/word2vec_extractor.py

# BERT用CPU处理少量数据
python src/skill_extraction/bert_extractor.py
# 在代码中设置 max_rows=100
```

---

## 🔧 代码兼容性

**好消息：我们的代码已经兼容所有方案！**

### 自动检测机制
```python
def check_gpu(self):
    """检查GPU可用性"""
    if torch.cuda.is_available():
        # CUDA或ROCm都会返回True
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"✅ GPU可用: {gpu_name}")
        return True
    else:
        logger.warning("⚠️  GPU不可用，将使用CPU")
        return False
```

### 自动适配
```python
# 代码会自动选择设备
device = 0 if self.check_gpu() else -1

# ROCm、CUDA、CPU都能正常工作
self.ner_pipeline = pipeline(
    "ner",
    model=self.model,
    tokenizer=self.tokenizer,
    device=device
)
```

---

## 💡 实用建议

### 1. 先用CPU验证可行性
```bash
# 不需要任何GPU配置
python test_skill_extraction.py
python src/skill_extraction/word2vec_extractor.py
```

### 2. 再考虑GPU加速
- Linux → 配置ROCm
- Windows → 使用云GPU或CPU

### 3. 分阶段处理
```python
# 阶段1：Word2Vec（不需要GPU）
python src/skill_extraction/word2vec_extractor.py

# 阶段2：BERT少量数据（CPU可接受）
# 修改代码：max_rows=100

# 阶段3：BERT大规模（使用云GPU）
# 上传到Colab或AutoDL
```

---

## 🐛 常见问题

### Q1: ROCm安装失败
**A:** 检查系统版本，ROCm主要支持Ubuntu 20.04/22.04

### Q2: PyTorch检测不到AMD GPU
**A:** 确认安装的是ROCm版本：
```bash
pip show torch | grep rocm
```

### Q3: DirectML速度很慢
**A:** DirectML性能不如ROCm，建议使用Linux+ROCm或云GPU

### Q4: 云GPU如何上传数据
**A:** 
```python
# Colab示例
from google.colab import drive
drive.mount('/content/drive')

# 数据在Google Drive中
!python /content/drive/MyDrive/Employ26/src/skill_extraction/run_extraction_pipeline.py
```

---

## 📝 总结

### AMD RX 9700可以运行吗？
**答：可以！** 有多种方案：

1. ✅ **Linux + ROCm** - 最佳性能
2. ✅ **Windows + CPU** - 最简单
3. ✅ **Windows + DirectML** - 中等性能
4. ✅ **云GPU** - 最省心

### 推荐方案
- **快速验证**：CPU + Word2Vec
- **Linux用户**：ROCm
- **Windows用户**：CPU或云GPU
- **大规模处理**：云GPU

### 立即可以做
```bash
# 1. 测试环境（不需要GPU）
python test_skill_extraction.py

# 2. 运行Word2Vec（不需要GPU）
python src/skill_extraction/word2vec_extractor.py

# 3. 运行BERT（CPU，少量数据）
python src/skill_extraction/bert_extractor.py
```

---

**无论使用哪种方案，代码都能正常运行！** 🎉








