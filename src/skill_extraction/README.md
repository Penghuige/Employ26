# 技能提取模块使用说明

## 📋 模块概述

本模块实现了两种技能提取方法：
1. **Word2Vec** - 基于词向量的相似技能发现
2. **BERT-NER** - 基于命名实体识别的技能提取

## 📁 项目结构

```
src/skill_extraction/
├── __init__.py                      # 模块初始化
├── word2vec_extractor.py            # Word2Vec提取器
├── bert_extractor.py                # BERT-NER提取器
├── run_extraction_pipeline.py       # 主流程（整合两种方法）
└── README.md                        # 本文件

output/skill_extraction/             # 输出目录（自动创建）
├── word2vec_expanded_skills.txt     # Word2Vec扩展词典
├── word2vec_expanded_skills.json    # Word2Vec详细结果
├── word2vec_expansion_report.txt    # Word2Vec报告
├── bert_extracted_skills.txt        # BERT提取词典
├── bert_extracted_entities.json     # BERT详细结果
├── bert_skill_statistics.csv        # BERT技能统计
├── bert_extraction_report.txt       # BERT报告
├── final_expanded_skills.txt        # 最终合并词典
├── final_skills_detail.json         # 最终详细信息
└── final_extraction_report.txt      # 最终对比报告

models/                              # 模型目录（自动创建）
└── word2vec_skills.model            # Word2Vec训练的模型
```

## 🚀 使用方法

### 方法1：运行完整流水线（推荐）

```bash
# 进入项目目录
cd d:\pythonProject\leisure\Employ26

# 运行完整流水线（使用样本数据）
python src/skill_extraction/run_extraction_pipeline.py
```

**流程说明：**
1. Word2Vec训练并扩展技能（约5分钟）
2. BERT提取技能实体（约10-30分钟，取决于GPU）
3. 合并两种方法的结果
4. 生成最终词典和报告

### 方法2：单独运行Word2Vec

```bash
# 只运行Word2Vec
python src/skill_extraction/word2vec_extractor.py
```

**输出：**
- `output/skill_extraction/word2vec_expanded_skills.txt` - 扩展词典
- `output/skill_extraction/word2vec_expansion_report.txt` - 分析报告

### 方法3：单独运行BERT

```bash
# 只运行BERT（需要先安装transformers）
python src/skill_extraction/bert_extractor.py
```

**输出：**
- `output/skill_extraction/bert_extracted_skills.txt` - 提取词典
- `output/skill_extraction/bert_extraction_report.txt` - 分析报告

### 方法4：在代码中调用

```python
from src.skill_extraction import SkillExtractionPipeline

# 创建流水线
pipeline = SkillExtractionPipeline()

# 运行完整流程
all_skills = pipeline.run_full_pipeline(
    use_sample=True,   # 使用样本数据
    max_rows=1000      # BERT处理1000行
)

print(f"提取技能数: {len(all_skills)}")
```

## 📦 依赖安装

### 基础依赖（Word2Vec）
```bash
pip install pandas numpy gensim tqdm
```

### BERT依赖（可选，用于BERT-NER）
```bash
# CPU版本
pip install transformers torch

# GPU版本（推荐，需要CUDA）
pip install transformers
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## ⚙️ 配置参数

### Word2Vec参数

在 `word2vec_extractor.py` 中修改：

```python
# 训练参数
params = {
    'vector_size': 300,      # 词向量维度（100-300）
    'window': 5,             # 上下文窗口（3-10）
    'min_count': 5,          # 最小词频（3-10）
    'workers': 4,            # 并行线程数
    'epochs': 10,            # 训练轮数（5-20）
    'sg': 1,                 # 1=Skip-gram, 0=CBOW
}

# 扩展参数
topn = 20                    # 每个技能查找top-n相似词
threshold = 0.5              # 相似度阈值（0.3-0.7）
```

### BERT参数

在 `bert_extractor.py` 中修改：

```python
# 模型选择
model_name = "ckiplab/bert-base-chinese-ner"  # 中文NER模型

# 处理参数
max_length = 512             # 最大文本长度
batch_size = 100             # 批处理大小
score_threshold = 0.5        # 置信度阈值
```

### 流水线参数

在 `run_extraction_pipeline.py` 中修改：

```python
# 运行参数
use_sample = True            # True=样本数据, False=全量数据
max_rows = 1000              # BERT最大处理行数（快速测试用）
```

## 📊 输出说明

### 1. Word2Vec输出

**word2vec_expanded_skills.txt** - jieba词典格式
```
Python 20000 nz
Java 20000 nz
数据分析 20000 nz
...
```

**word2vec_expansion_report.txt** - 详细报告
```
Python:
  - pandas                  (相似度: 0.856)
  - numpy                   (相似度: 0.823)
  - sklearn                 (相似度: 0.791)
  ...
```

### 2. BERT输出

**bert_extracted_skills.txt** - jieba词典格式
```
Python 15000 nz
机器学习 12000 nz
数据分析 18000 nz
...
```

**bert_skill_statistics.csv** - 技能统计
```csv
技能,出现次数
Python,150
Java,120
数据分析,200
...
```

### 3. 最终输出

**final_expanded_skills.txt** - 合并后的最终词典
```
Python 20000 nz
Java 20000 nz
机器学习 20000 nz
...
```

**final_extraction_report.txt** - 对比报告
```
总技能数: 500 个

两种方法都发现: 120 个
仅Word2Vec发现: 180 个
仅BERT发现: 200 个

技能列表：
Python                         - seed, w2v(0.95), bert(150次)
Java                           - seed, w2v(0.92), bert(120次)
...
```

## 🔍 验证结果

### 检查Word2Vec结果
```bash
# 查看扩展了多少技能
wc -l output/skill_extraction/word2vec_expanded_skills.txt

# 查看报告
cat output/skill_extraction/word2vec_expansion_report.txt
```

### 检查BERT结果
```bash
# 查看提取了多少技能
wc -l output/skill_extraction/bert_extracted_skills.txt

# 查看统计
head -20 output/skill_extraction/bert_skill_statistics.csv
```

### 检查最终结果
```bash
# 查看最终技能数
wc -l output/skill_extraction/final_expanded_skills.txt

# 查看对比报告
cat output/skill_extraction/final_extraction_report.txt
```

## ⚠️ 注意事项

### 1. 数据要求
- 需要先运行NLP预处理（`src/nlp_analysis/text_preprocessing.py`）
- 确保 `output/nlp_processed/` 目录下有处理后的数据

### 2. 内存要求
- Word2Vec：约1-2GB内存
- BERT（CPU）：约4-8GB内存
- BERT（GPU）：约2-4GB显存

### 3. 运行时间
- Word2Vec训练：5-10分钟（样本数据）
- BERT提取（CPU）：30-60分钟（1000行）
- BERT提取（GPU）：5-10分钟（1000行）

### 4. GPU使用
- 代码会自动检测GPU
- 如果有GPU，BERT会自动使用
- 如果没有GPU，会使用CPU（较慢）

### 5. 模型下载
- BERT模型首次运行会自动下载（约400MB）
- 如果网络问题，可以手动下载后指定路径
- 下载地址：https://huggingface.co/ckiplab/bert-base-chinese-ner

## 🐛 常见问题

### Q1: 提示"未找到数据文件"
**A:** 需要先运行NLP预处理：
```bash
python src/nlp_analysis/text_preprocessing.py
```

### Q2: BERT模型下载失败
**A:** 使用镜像源或手动下载：
```bash
# 使用镜像
export HF_ENDPOINT=https://hf-mirror.com
python src/skill_extraction/bert_extractor.py
```

### Q3: 内存不足
**A:** 减少处理数据量：
```python
# 在代码中修改
max_rows = 500  # 减少到500行
```

### Q4: GPU未被使用
**A:** 检查PyTorch安装：
```python
import torch
print(torch.cuda.is_available())  # 应该返回True
```

### Q5: Word2Vec训练很慢
**A:** 减少训练轮数：
```python
epochs = 5  # 从10减少到5
```

## 📈 性能优化

### 1. 使用GPU加速BERT
```bash
# 安装GPU版本的PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 增加批处理大小
```python
batch_size = 200  # 如果显存足够，可以增加
```

### 3. 使用多线程
```python
workers = 8  # Word2Vec训练时使用更多线程
```

### 4. 缓存模型
```python
# Word2Vec模型训练一次后可以重复使用
train_new = False  # 加载已有模型
```

## 🔄 更新词典

提取完成后，更新主词典：

```bash
# 备份原词典
cp dicts/userdict_zh_recruitment.txt dicts/userdict_zh_recruitment.txt.bak

# 合并新词典
cat output/skill_extraction/final_expanded_skills.txt >> dicts/userdict_zh_recruitment.txt

# 去重
sort -u dicts/userdict_zh_recruitment.txt > dicts/userdict_zh_recruitment_new.txt
mv dicts/userdict_zh_recruitment_new.txt dicts/userdict_zh_recruitment.txt
```

## 📞 技术支持

如有问题，请查看：
1. 代码中的详细注释
2. 日志输出信息
3. 生成的报告文件

---

**祝使用顺利！** 🚀

