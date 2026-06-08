# CLAUDE.md

> **最后更新**: 2026-06-08

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

广东省三大招聘网站（智联招聘、猎聘网、前程无忧）2022-2025年招聘数据的NLP分析项目，总数据量约5GB。

## 常用命令

### 运行完整分析流程（已归档）
```bash
python archive/process/run_all_analysis.py
```
主入口被移动到 `archive/process/`，当前活跃的开发工作集中在技能抽取和岗位匹配模块。

### 岗位匹配 CLI（活跃）
```bash
# 预处理职业大典
python -m src.job_title_parsing.cli preprocess-catalog

# 构建层级关键词词典
python -m src.job_title_parsing.cli build-hierarchy-dict

# 批量匹配岗位到职业细类
python -m src.job_title_parsing.cli match --jobs-table recruit.main.jobs_sample --progress

# 评估匹配结果
python -m src.job_title_parsing.cli evaluate --result-table recruit.main.job_match_results
```

### 技能抽取流水线（活跃）
```bash
# FlatSkillPipeline 主流程入口
python -m src.skill_extraction.occupation_skill_pipeline --help

# BGE 微调迭代流水线（已重命名 step_01 ~ step_07）
python -m src.bge.step_01_deduplicate
python -m src.bge.step_02_filter
# ... 依次执行至 step_07_iterate_pipeline
```

### RAG CLI
```bash
python -m src.rag.cli build       # 构建 FAISS 索引
python -m src.rag.cli query --title "Java开发工程师" --requirements "..."
python -m src.rag.cli judge --title "..." --requirements "..." --candidates-json "[...]"
```

### 运行测试
```bash
pytest src/tests/ -v
pytest src/tests/test_job_title_matching_fixes.py -v  # 单个测试文件
```

### 代码质量
```bash
black src/ tests/        # 格式化
flake8 src/ tests/       # lint
pytest --cov=src tests/  # 覆盖率
python -m compileall -q src  # 编译检查
```

### 环境
- **包管理**: conda（见 `.vscode/settings.json`）
- **Python**: 3.10+
- **依赖安装**: `pip install -r requirements.txt`

## 核心架构

### 数据层

- **PostgreSQL** 是唯一数据库。所有原始数据、中间处理结果和最终产出均存入 PG 表。**禁止使用 DuckDB**。
- 连接参数通过 `config/paths.py` 的 `ProjectPaths` 集中管理，支持 `EMPLOYDATA_PG_*` 环境变量覆盖：
  ```python
  from config.paths import get_project_paths
  paths = get_project_paths()
  conn_params = paths.pg_connection_params  # {host, port, dbname, user, password, schema}
  ```
- 默认连接参数：`localhost:5432`，数据库 `employ26`，用户 `postgres`。生产环境通过环境变量覆盖。
- 原始数据表在 `recruit.raw_data.*`，生产处理表在 `recruit.main.*`。
- **报告类文本**（`.txt`、`.md`、`.html`）写入 `output/reports/`，不存入数据库。
- **数据文件（~5GB）不提交到 git**，由 `.gitignore` 排除。

### 三大活跃子系统

**1. 岗位名称匹配 (`src/job_title_parsing/`)**

核心任务：将招聘岗位名称匹配到《中国职业分类大典》的职业细类代码。

- `cli.py` — 唯一入口，支持 `preprocess-catalog`、`build-hierarchy-dict`、`match`、`evaluate` 四个子命令
- `matching_pipeline.py` — 主匹配流程（`MatchPipeline`），串联 title cleaning → BM25/n-gram 召回 → 多信号打分 → TopK 候选输出
- `scoring.py` — 多信号加权打分（标题匹配、任务重叠、描述相似度、层级加分、别名加分、冲突惩罚）
- `title_cleaner.py` — 岗位名称清洗（去噪声短语、薪资模式、门店后缀等）
- `occupation_parser.py` — 基于最长后缀匹配 + 词典的职业核心词提取（270个核心词，准确率98%）
- `bm25_index.py` / `ngram_retrieval.py` — 两路召回（BM25 + character n-gram）
- `hierarchy_filter.py` — 基于关键词→大类词典的层级过滤
- `alias_builder.py` — 人工别名字典管理
- `match_utils.py` — YAML 配置加载（使用自实现的简单 YAML 解析器，不依赖 PyYAML）
- `matching_evaluator.py` — 匹配精度/召回率评估

匹配配置在 `config/default.yaml`，控制权重、阈值、噪声模式等。

**2. 技能抽取 (`src/skill_extraction/`)**

核心任务：从岗位描述中抽取标准化职业技能，构建"职业→技能"词典。

- `config.py` → `load_skill_extraction_config()` — 统一配置入口，从 `config/database.yaml` 读取所有表名/路径/模型配置，返回 `SkillExtractionConfig` dataclass
- `occupation_skill_pipeline.py` — `FlatSkillPipeline` 主流程，含采样、vLLM 推理、BGE 匹配、词典输出
- `bge_matcher.py` — BGE embedding + FAISS 进行技能到词典的语义匹配
- `context_classifier.py` — 上下文分类器（区分硬技能/软素质/职责描述）
- `merge_similar_skills.py` — 合并语义重复的技能条目
- `match_flat_skills_to_duckdb.py` — 将平面化技能词典匹配回 PostgreSQL 招聘数据
- `iterate_flat_skill_dictionary.py` / `iteration_rules.py` — 词典自我迭代和规则管理
- `llm_labeling_utils.py` — LLM 标注工具函数
- `context_labels.py` — 上下文标签定义
- `regression_eval.py` — 回归评估
- `llm_label_context_dataset.py` / `llm_label_regression_dataset.py` — LLM 标注数据集构建

> **注意**: `pipeline_v1.py` 和 `pipeline_v2.py` 已归档至 `archive/experiments/pipeline_versions/`。`history/` 目录已迁移至 `archive/skill_extraction_history/`。

技能抽取遵守 `.cursorrules` 中定义的严格规则：排除软素质词、空泛职责词、福利待遇词；优先高精度低噪声。

**3. LLM 基础设施 (`src/llm/`、`src/utils/llm_router.py`)**

- `src/utils/llm_router.py` — `LLMRouter` 统一的两级模型路由客户端（cheap/strong），从 `.env.local` 加载配置
- `src/llm/vllm_server.py` — vLLM 服务端封装（Windows 兼容的 FastAPI + SSE streaming + OpenAI 兼容接口）
- `src/llm/start_session.py` — 本地 vLLM 终端对话窗口

> **注意**: `src/llm/archive/` 已迁移至 `archive/llm_history/`。

### 其他模块

- **`src/bert/`** — BERT NER 训练/预测流水线（基于 chinese-roberta-wwm-ext 进行岗位描述技能实体识别）
- **`src/bge/`** — BGE embedding 微调与迭代流水线（`step_01_deduplicate` → `step_02_filter` → ... → `step_07_iterate_pipeline`：去重、过滤、微调、匹配、自动标注、阈值评估、迭代）
- **`src/rag/`** — 本地职业知识库 RAG（`OccupationRAG`）：职业大典 Excel → BGE 向量 → FAISS 检索 → DeepSeek 生成
- **`src/analysis/`** — 统计分析（职业薪资、行业趋势、时间趋势、学历分布 → 已归档）
- **`src/visualization/`** — pyecharts 词云生成
- **`src/preprocessing/`** — 数据预处理（数据清洗、样本提取、职业整合、JD 解析）
- **`src/nlp_analysis/`** — NLP 文本分析（分词预处理）
- **`src/tests/`** — 测试文件（文件名须以 `test_` 开头）

### 配置结构

| 文件 | 用途 |
|------|------|
| `config/default.yaml` | 匹配引擎参数（打分权重、噪声模式、别名映射） |
| `config/database.yaml` | **PostgreSQL 连接参数**、表名、模型路径、API 配置 |
| `config/paths.py` | **集中路径与连接管理**（`ProjectPaths` dataclass，支持环境变量覆盖） |
| `config/skill_dictionary_iteration.json` | 技能词典迭代参数 |
| `config/system_prompt.md` | 毒舌傲娇女仆 system prompt（与项目无关，仅测试用） |
| `.env.local` | LLM 配置（base_url、api_key、model 选择），不提交 git |
| `dicts/` | 所有词典/黑名单/同义词表（UTF-8，`#` 注释），代码不硬编码词汇 |

### 归档结构

| 目录 | 内容 |
|------|------|
| `archive/process/` | 历史主流程脚本（`run_all_analysis.py` 等） |
| `archive/scripts/` | 基准测试、评估、迁移工具脚本 |
| `archive/experiments/pipeline_versions/` | 旧版流水线（`pipeline_v1.py`、`pipeline_v2.py`） |
| `archive/experiments/penghui/` | penghui 实验脚本（RAG 训练、多维度验证） |
| `archive/skill_extraction_history/` | 技能抽取历史版本 |
| `archive/llm_history/` | LLM 历史版本（NER 标注、vLLM 启动器） |
| `archive/job_title_parsing_history/` | 岗位匹配历史版本 |
| `archive/docs/` | 历史文档和周报 |

## 路径管理

### 集中配置 (`config/paths.py`)

所有模型路径和 PostgreSQL 连接参数必须通过 `config/paths.py` 的 `ProjectPaths` dataclass 获取，**禁止代码内硬编码任何路径或连接字符串**。

```python
from config.paths import get_project_paths

paths = get_project_paths()

# PostgreSQL 连接
pg_params = paths.pg_connection_params
# {"host": "localhost", "port": 5432, "dbname": "employ26", "user": "postgres", ...}

# 模型路径
bge_model = paths.bge_model_path   # models/bge-base-zh-v1.5
qwen_model = paths.qwen_model_path # models/Qwen3-8B
```

### 环境变量覆盖

支持通过环境变量覆盖默认值，实现跨环境可移植：

| 环境变量 | 覆盖项 | 默认值 |
|----------|--------|--------|
| `EMPLOYDATA_PG_HOST` | PostgreSQL 主机 | `localhost` |
| `EMPLOYDATA_PG_PORT` | PostgreSQL 端口 | `5432` |
| `EMPLOYDATA_PG_DBNAME` | 数据库名 | `employ26` |
| `EMPLOYDATA_PG_USER` | 用户名 | `postgres` |
| `EMPLOYDATA_PG_PASSWORD` | 密码 | （空） |
| `EMPLOYDATA_PG_SCHEMA` | schema 前缀 | `recruit` |
| `EMPLOYDATA_BGE_MODEL_PATH` | BGE embedding 模型路径 | `models/bge-base-zh-v1.5` |
| `EMPLOYDATA_QWEN_MODEL_PATH` | Qwen LLM 模型路径 | `models/Qwen3-8B` |
| `EMPLOYDATA_BERT_MODEL_PATH` | BERT 模型路径 | `models/chinese-roberta-wwm-ext` |

### 导入规范

- **禁止** `sys.path.insert()` 和 `sys.path.append()` 运行时修改
- **使用** `python -m src.xxx` 从项目根目录运行脚本，确保模块路径正确
- **包内使用相对导入**：`from .config import X` 或 `from ..utils import Y`
- **跨包子模块**可使用绝对导入 `from src.rag.config import RAGConfig`，前提是运行入口在项目根目录
- **禁止**星号导入（`from module import *`），必须显式列出导入对象

## Python 注释、docstring 与类型提示

### Google 风格 docstring（强制）

所有公开接口（模块、类、函数、方法）必须包含 Google 风格 docstring：

```python
def parse_salary(salary_str: str) -> tuple[float | None, float | None]:
    """解析原始薪资字符串，提取月薪上下限。

    Args:
        salary_str: 原始薪资文本，如 "10-15K" 或 "8000-12000元/月"。

    Returns:
        一个 `(lower_bound, upper_bound)` 元组。无法解析时返回 `(None, None)`。

    Raises:
        ValueError: 启用严格模式且薪资格式无效时抛出。
    """
```

### 模块 docstring

每个 `.py` 文件必须以模块级 docstring 开头：

```python
"""模块功能简述。

详细说明模块职责、输入输出、用法示例和注意事项。
"""
```

### 类型提示

- 所有公开接口必须包含完整的类型提示（参数和返回值）
- 使用 `from __future__ import annotations` 启用延迟求值
- 复杂类型使用 `typing` 模块（`Optional`、`Union`、`List`、`Dict` 等）

### 注释语言

- **变量命名**: 英文（snake_case）
- **注释**: 中文
- **docstring**: 中文

## 代码风格与一致性

### 格式化

项目使用以下工具强制代码风格一致：

- **black** — 代码格式化（line-length=88）
- **isort** — import 排序（profile=black）
- **flake8** — 代码检查（max-line-length=88, extend-ignore=E203,W503）
- **mypy** — 类型检查（可选，python_version=3.10）

### 检查命令

```bash
# 编译检查（每次提交前必须通过）
python -m compileall -q src

# 格式检查
black --check src/
flake8 src/

# 自动修复
black src/
isort src/
```

### 命名规范

| 类型 | 规范 | 示例 |
|------|------|------|
| 模块/文件 | snake_case | `title_cleaner.py`、`step_01_deduplicate.py` |
| 类 | PascalCase | `MatchPipeline`、`NERDataset` |
| 函数/方法 | snake_case | `parse_salary()`、`load_config()` |
| 常量 | UPPER_SNAKE_CASE | `MAX_BATCH_SIZE`、`DEFAULT_DB_PATH` |
| 私有成员 | 前缀 `_` | `_resolve_local_model_path()` |

### 包目录结构

- 每个包目录必须包含 `__init__.py`
- `__init__.py` 可通过 `__all__` 显式暴露公共接口
- 归档目录不应包含 `__init__.py`，防止活跃代码误导入

## 关键设计原则

- **PostgreSQL 统一存储**: 除报告类文本外，所有源数据和中间处理结果存入 PostgreSQL。**禁止使用 DuckDB**
- **分块处理**: 5GB 数据必须分批读取，避免内存溢出
- **增量与缓存**: 中间结果缓存到磁盘（`output/skill_extraction/cache/`），支持断点续传
- **词典驱动**: 所有词汇列表（技能、停用词、黑名单、别名）必须存储在 `dicts/`，禁止代码内硬编码
- **数据库作为数据枢纽**: 输入输出均为 PostgreSQL 表，不使用临时 CSV
- **配置集中化**: 通过 `load_skill_extraction_config()`、`load_database_config()` 和 `get_project_paths()` 统一管理，上层脚本不应各自拼接路径或猜测表名
- **路径统一管理**: 所有路径和连接参数通过 `config/paths.py` 获取，支持环境变量覆盖。禁止硬编码 `D:\...` 等绝对路径
- **修改文件只允许使用 Write 和 Edit 工具**，禁止通过生成辅助 Python 脚本来修改其他文件
- **变量命名英文，注释中文**（`.cursorrules` 约定）

## 协作规范

### 分支策略

- `main` — 只接受 PR 合并，禁止直接 push
- `feature/xxx` — 新功能
- `fix/xxx` — Bug 修复
- `refactor/xxx` — 重构
- `experiment/xxx` — 实验性代码

### 提交信息

采用 Conventional Commits 格式：
```
feat: 添加批量匹配缓存
fix: 修复 title_cleaner 空字符串崩溃
refactor: 统一路径管理为 config/paths.py
```

### PR 门禁

PR 必须通过以下检查：
1. `python -m compileall -q src`
2. `black --check src/`
3. `flake8 src/`
4. `pytest -q`

### 归档规则

- 历史版本代码移入 `archive/` 目录
- 归档目录不需要 `__init__.py`（不作为可导入包）
- 归档时在 `archive/README.md` 记录归档原因、替代入口和责任人
- 实验脚本文件名包含版本号时，归档时在目录内添加说明文档
