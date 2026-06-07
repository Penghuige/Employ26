# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

广东省三大招聘网站（智联招聘、猎聘网、前程无忧）2022-2025年招聘数据的NLP分析项目，总数据量约5GB。

## 常用命令

### 运行完整分析流程（已归档）
```bash
python archive/run_all_analysis.py
```
主入口被移动到 `archive/`，当前活跃的开发工作集中在技能抽取和岗位匹配模块。

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
# v2 平面化词典流水线
python -m src.skill_extraction.pipeline_v2 --help
```

### 运行测试
```bash
pytest tests/ -v
pytest tests/test_job_title_matching_fixes.py -v  # 单个测试文件
```

### 代码质量
```bash
black src/ tests/      # 格式化
flake8 src/ tests/     # lint
pytest --cov=src tests/ # 覆盖率
```

### 环境
- **包管理**: conda（见 `.vscode/settings.json`）
- **Python**: 3.8+
- **依赖安装**: `pip install -r requirements.txt`

## 核心架构

### 数据层
- **DuckDB** 是主数据库（路径: `output/recruit.duckdb`）。所有流水线通过 DuckDB 读取和写入，不使用 CSV 中间文件。
- 原始数据表在 `recruit.raw_data.*`，生产处理表在 `recruit.main.*`。
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
- `evaluate_matching.py` / `matching_evaluator.py` — 匹配精度/召回率评估

匹配配置在 `config/default.yaml`，控制权重、阈值、噪声模式等。

**2. 技能抽取 (`src/skill_extraction/`)**

核心任务：从岗位描述中抽取标准化职业技能，构建"职业→技能"词典。

- `config.py` → `load_skill_extraction_config()` — 统一配置入口，从 `config/database.yaml` 读取所有表名/路径/模型配置，返回 `SkillExtractionConfig` dataclass
- `pipeline_v2.py` — 当前主流水线入口（平面化词典 v2）
- `occupation_skill_pipeline.py` — `FlatSkillPipeline` 主流程，含采样、vLLM 推理、BGE 匹配、词典输出
- `bge_matcher.py` — BGE embedding + FAISS 进行技能到词典的语义匹配
- `context_classifier.py` — 上下文分类器（区分硬技能/软素质/职责描述）
- `merge_similar_skills.py` — 合并语义重复的技能条目
- `match_flat_skills_to_duckdb.py` — 将平面化技能词典匹配回 DuckDB 招聘数据
- `iterate_flat_skill_dictionary.py` / `iteration_rules.py` — 词典自我迭代和规则管理
- `llm_labeling_utils.py` — LLM 标注工具函数
- `history/` — 历史版本的技能流水线实现（含 LLM 标注、词典导入导出等），仅作参考

技能抽取遵守 `.cursorrules` 中定义的严格规则：排除软素质词、空泛职责词、福利待遇词；优先高精度低噪声。

**3. LLM 基础设施 (`src/llm/`、`src/utils/llm_router.py`)**

- `src/utils/llm_router.py` — `LLMRouter` 统一的两级模型路由客户端（cheap/strong），从 `.env.local` 加载配置，支持 GPT-5 Responses API
- `src/llm/vllm_launcher.py` — Windows 兼容的 vLLM 服务启动器（FastAPI + SSE streaming + OpenAI 兼容接口）
- `src/llm/vllm_server.py` — vLLM 服务端封装
- `src/llm/history/` — 历史 NER 标注流水线（bio_converter, qwen3_extractor, batch_annotator）

### 其他模块

- **`src/bert/`** — BERT NER 训练/预测流水线（基于 chinese-roberta-wwm-ext 进行岗位描述技能实体识别）
- **`src/bge/`** — BGE embedding 微调与迭代流水线（D1-D7 步骤：去重、过滤、微调、匹配、自动标注、阈值评估、迭代）
- **`src/rag/`** — 本地职业知识库 RAG（`LocalOccupationRAG`）：职业大典 Excel → BGE 向量 → FAISS 检索 → Qwen3 生成
- **`src/analysis/`** — 统计分析（职业薪资、行业趋势、时间趋势、学历分布 → 已归档）
- **`src/visualization/`** — pyecharts 词云生成

### 配置结构

| 文件 | 用途 |
|------|------|
| `config/default.yaml` | 匹配引擎参数（打分权重、噪声模式、别名映射） |
| `config/database.yaml` | DuckDB 路径/表名、模型路径、API 密钥 |
| `config/skill_dictionary_iteration.json` | 技能词典迭代参数 |
| `config/system_prompt.md` | 毒舌傲娇女仆 system prompt（与项目无关，仅测试用） |
| `.env.local` | LLM 配置（base_url、api_key、model 选择），不提交 git |
| `dicts/` | 所有词典/黑名单/同义词表（UTF-8，`#` 注释），代码不硬编码词汇 |

### 关键设计原则

- **分块处理**: 5GB 数据必须用 DuckDB/chunk 读取，避免内存溢出
- **增量与缓存**: 中间结果缓存到磁盘（`output/skill_extraction/cache/`），支持断点续传
- **词典驱动**: 所有词汇列表（技能、停用词、黑名单、别名）必须存储在 `dicts/`，禁止代码内硬编码
- **DuckDB 作为数据枢纽**: 输入输出均为 DuckDB 表，不使用临时 CSV
- **配置集中化**: 通过 `load_skill_extraction_config()` 和 `load_database_config()` 统一管理，上层脚本不应各自拼接路径或猜测表名
- **修改文件只允许使用 Write 和 Edit 工具**，禁止通过生成辅助 Python 脚本来修改其他文件
- **变量命名英文，注释中文**（`.cursorrules` 约定）
