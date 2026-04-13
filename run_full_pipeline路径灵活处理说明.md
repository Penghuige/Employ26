# run_full_pipeline.py 路径灵活处理 - 完整说明

## ✅ 已修复的问题

### 问题：路径硬编码，无法灵活处理样本数据和总体数据

**修复前**：
- NLP输出路径固定为 `output/nlp_processed_full`
- 职业解析输入路径固定
- 数据整合路径固定
- 无法切换样本数据和总体数据

**修复后**：
- ✅ 支持 `--sample` 参数切换数据类型
- ✅ 自动调整所有模块的路径
- ✅ 灵活处理样本数据和总体数据
- ✅ 新增 `--skip-integration` 参数

---

## 📊 修复的文件

### 1. `run_full_pipeline.py`（主流程）

#### 新增参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--sample` | 使用样本数据模式 | False（全量数据） |
| `--skip-nlp` | 跳过NLP处理 | False |
| `--skip-parsing` | 跳过职业解析 | False |
| `--skip-integration` | 跳过数据整合 | False |
| `--input` | 原始数据目录 | data/ |

#### 路径自动切换

```python
# 根据 --sample 参数自动切换路径
data_type = '样本数据' if args.sample else '全量数据'
nlp_output_dir = 'output/nlp_processed' if args.sample else 'output/nlp_processed_full'
```

#### 参数传递

```python
# 阶段1：NLP处理
processor = process_full_data_nlp.FullDataNLPProcessor(
    input_dir=args.input,
    output_dir=nlp_output_dir  # 动态路径
)

# 阶段2：职业解析
parser_obj = parse_all_occupations.BatchOccupationParser(
    base_dir=base_dir,
    input_dir=nlp_output_dir  # 传递正确的输入路径
)

# 阶段3：数据整合
integrator = DataIntegrator(
    base_dir=base_dir,
    use_full_data=not args.sample  # 传递数据类型
)
```

---

### 2. `parse_all_occupations.py`（职业解析）

#### 新增参数支持

```python
def __init__(self, base_dir=None, input_dir=None):
    """初始化
    
    Args:
        base_dir: 项目根目录
        input_dir: NLP处理后的数据目录（默认：output/nlp_processed_full/）
    """
    # 设置输入目录
    if input_dir is None:
        self.input_dir = base_dir / 'output' / 'nlp_processed_full'
    else:
        self.input_dir = Path(input_dir)
        if not self.input_dir.is_absolute():
            self.input_dir = base_dir / self.input_dir
```

#### 命令行参数

```bash
# 使用默认路径（全量数据）
python parse_all_occupations.py

# 使用自定义路径
python parse_all_occupations.py --input output/nlp_processed

# 使用样本数据
python parse_all_occupations.py --input output/nlp_processed
```

---

### 3. `integrate_occupation.py`（数据整合）

#### 已支持数据类型切换

```python
def __init__(self, base_dir=None, use_full_data=True):
    """初始化
    
    Args:
        base_dir: 项目根目录
        use_full_data: 是否使用全量数据（True=全量，False=样本）
    """
    # 根据数据类型设置路径
    if use_full_data:
        self.nlp_dir = base_dir / 'output' / 'nlp_processed_full'
    else:
        self.nlp_dir = base_dir / 'output' / 'nlp_processed'
```

---

## 🚀 使用方法

### 方法1：处理全量数据（默认）

```bash
# 完整流程
python run_full_pipeline.py

# 跳过已完成的步骤
python run_full_pipeline.py --skip-nlp --skip-parsing
```

**数据流**：
```
data/
  ↓ NLP处理
output/nlp_processed_full/
  ↓ 职业解析
output/job_title_parsing/
  ↓ 数据整合
output/integrated/
  ↓ 分析报告
output/reports/
```

---

### 方法2：处理样本数据

```bash
# 完整流程（样本数据）
python run_full_pipeline.py --sample

# 跳过已完成的步骤
python run_full_pipeline.py --sample --skip-nlp --skip-parsing
```

**数据流**：
```
data/
  ↓ NLP处理
output/nlp_processed/  ⭐ 样本数据路径
  ↓ 职业解析
output/job_title_parsing/
  ↓ 数据整合
output/integrated/
  ↓ 分析报告
output/reports/
```

---

### 方法3：自定义输入目录

```bash
# 使用自定义原始数据目录
python run_full_pipeline.py --input path/to/data/

# 样本数据 + 自定义目录
python run_full_pipeline.py --sample --input path/to/data/
```

---

### 方法4：单独运行各步骤

#### NLP处理
```bash
# 全量数据
python process_full_data_nlp.py

# 样本数据
python process_full_data_nlp.py --output output/nlp_processed
```

#### 职业解析
```bash
# 全量数据
python parse_all_occupations.py

# 样本数据
python parse_all_occupations.py --input output/nlp_processed
```

#### 数据整合
```bash
# 全量数据
python src/preprocessing/integrate_occupation.py

# 样本数据
python src/preprocessing/integrate_occupation.py --sample
```

#### 分析报告
```bash
# 分析（从integrated读取，不区分样本/全量）
python run_all_analysis.py
```

---

## 📁 目录结构对比

### 全量数据模式（默认）

```
项目根目录/
├── data/                          # 原始数据
├── output/
│   ├── nlp_processed_full/       # NLP处理后（全量）⭐
│   ├── nlp_reports/              # NLP处理报告
│   ├── job_title_parsing/        # 职业解析结果
│   ├── integrated/               # 整合数据
│   └── reports/                  # 分析报告
└── dicts/                         # 词典文件
```

### 样本数据模式

```
项目根目录/
├── data/                          # 原始数据
├── output/
│   ├── nlp_processed/            # NLP处理后（样本）⭐
│   ├── nlp_reports/              # NLP处理报告
│   ├── job_title_parsing/        # 职业解析结果
│   ├── integrated/               # 整合数据
│   └── reports/                  # 分析报告
└── dicts/                         # 词典文件
```

---

## 🔄 完整数据流

### 全量数据流程

```
原始数据 (data/)
    ↓ python run_full_pipeline.py
NLP处理 (output/nlp_processed_full/)
    ↓
职业解析 (output/job_title_parsing/)
    ↓
数据整合 (output/integrated/)
    ↓
分析报告 (output/reports/)
```

### 样本数据流程

```
原始数据 (data/)
    ↓ python run_full_pipeline.py --sample
NLP处理 (output/nlp_processed/)
    ↓
职业解析 (output/job_title_parsing/)
    ↓
数据整合 (output/integrated/)
    ↓
分析报告 (output/reports/)
```

---

## 💡 使用场景

### 场景1：开发测试（使用样本数据）

```bash
# 快速测试流程（样本数据，速度快）
python run_full_pipeline.py --sample
```

**优点**：
- 处理速度快（几分钟）
- 适合调试和测试
- 验证流程正确性

---

### 场景2：正式分析（使用全量数据）

```bash
# 完整分析（全量数据，结果准确）
python run_full_pipeline.py
```

**优点**：
- 数据完整
- 结果准确
- 适合正式报告

---

### 场景3：增量更新

```bash
# 只重新运行分析（跳过耗时的前期步骤）
python run_full_pipeline.py --skip-nlp --skip-parsing --skip-integration
```

**优点**：
- 节省时间
- 只更新分析结果
- 适合调整分析参数

---

## ⚠️ 注意事项

### 1. 路径一致性

确保整个流程使用相同的数据类型：
- ✅ 全流程使用全量数据
- ✅ 全流程使用样本数据
- ❌ 不要混用（会导致数据不匹配）

### 2. 跳过步骤的前提

使用 `--skip-*` 参数前，确保该步骤已经完成：
- `--skip-nlp`：确保 `output/nlp_processed_full/` 有数据
- `--skip-parsing`：确保 `output/job_title_parsing/` 有数据
- `--skip-integration`：确保 `output/integrated/` 有数据

### 3. 数据类型切换

如果从样本数据切换到全量数据（或反之），需要重新运行完整流程：

```bash
# 从样本切换到全量
python run_full_pipeline.py  # 不加 --sample

# 从全量切换到样本
python run_full_pipeline.py --sample
```

---

## 📝 修复总结

| 修复项 | 修复前 | 修复后 |
|--------|--------|--------|
| NLP输出路径 | 固定 `nlp_processed_full` | 根据 `--sample` 动态切换 |
| 职业解析输入 | 固定路径 | 支持 `--input` 参数 |
| 数据整合类型 | 固定全量数据 | 支持 `use_full_data` 参数 |
| 参数传递 | 无参数传递 | 完整的参数传递链 |
| 灵活性 | 低 | 高 |

---

## 🎯 推荐使用

### 开发阶段
```bash
# 使用样本数据快速测试
python run_full_pipeline.py --sample
```

### 生产阶段
```bash
# 使用全量数据正式分析
python run_full_pipeline.py
```

### 增量更新
```bash
# 只重新分析
python run_full_pipeline.py --skip-nlp --skip-parsing --skip-integration
```

---

**修复完成时间**：2026-03-09  
**状态**：✅ 完全支持样本数据和全量数据的灵活切换  
**影响文件**：
- ✅ `run_full_pipeline.py`
- ✅ `parse_all_occupations.py`
- ✅ `integrate_occupation.py`（之前已修复）
- ✅ `process_full_data_nlp.py`（之前已修复）
