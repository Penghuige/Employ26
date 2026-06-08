# 广东省招聘数据NLP分析项目

> **最后更新**: 2026-06-08

基于广东省三大招聘网站（智联招聘、猎聘网、前程无忧）2022-2025年招聘数据的NLP分析项目。

## 快速开始

### 环境配置

```bash
# Python 3.10+
pip install -r requirements.txt

# PostgreSQL 连接（默认 localhost:5432，生产环境通过环境变量覆盖）
export EMPLOYDATA_PG_HOST=localhost
export EMPLOYDATA_PG_PORT=5432
export EMPLOYDATA_PG_DBNAME=employ26
export EMPLOYDATA_PG_USER=postgres
export EMPLOYDATA_PG_PASSWORD=your_password

# 模型路径（可选，有默认值）
export EMPLOYDATA_BGE_MODEL_PATH=models/bge-base-zh-v1.5
export EMPLOYDATA_QWEN_MODEL_PATH=models/Qwen3-8B
```

> **数据库**: 项目统一使用 **PostgreSQL**。除报告类文本（`.txt`、`.md`、`.html`）外，
> 所有源数据、中间处理结果和最终产出均存入 PG 表。详情见 [config/database.yaml](config/database.yaml)。

### 运行完整分析（已归档）

```bash
python archive/process/run_all_analysis.py
```

### 活跃功能入口

```bash
# 岗位匹配
python -m src.job_title_parsing.cli match --progress

# 技能抽取
python -m src.skill_extraction.occupation_skill_pipeline --help

# RAG 检索
python -m src.rag.cli query --title "Java开发工程师" --requirements "Spring..."

# BGE 微调流水线
python -m src.bge.step_01_deduplicate
```

### 查看结果

分析完成后，查看以下文件：

**可视化图表**（在浏览器中打开）:
- `output/reports/职业类别薪资分析图.html`
- `output/reports/行业景气度分析图.html`
- `output/reports/词云图.html`
- `output/reports/时间趋势图.html`

**文本报告**:
- `output/reports/职业类别薪资分析报告.txt`
- `output/reports/行业景气度分析报告.txt`
- `output/reports/薪资分析报告.txt`

## 核心功能

### 1. 职业类别分析
- 从岗位名称中提取职业核心词（准确率98%）
- 分析职业类别与薪资的关系
- 职业类别月度趋势分析
- 学历×职业类别交叉分析

### 2. 行业景气度分析
- 行业招聘量排行
- 城市×行业分布
- 行业月度趋势（观察景气度变化）

### 3. 薪资分析
- 职业类别薪资对比
- 学历、经验、城市对薪资的影响
- 月度薪资趋势

### 4. 文本分析
- jieba中文分词
- TF-IDF关键词提取
- 词云可视化

## 项目结构

```
Employ26/
├── src/                        # 活跃源代码
│   ├── job_title_parsing/      # 岗位名称匹配（核心）
│   ├── skill_extraction/       # 技能抽取（核心）
│   ├── rag/                    # RAG 知识库检索
│   ├── bert/                   # BERT NER 训练/推理
│   ├── bge/                    # BGE 微调迭代流水线 (step_01 ~ step_07)
│   ├── llm/                    # LLM 基础设施
│   ├── analysis/               # 统计分析
│   ├── preprocessing/          # 数据预处理
│   ├── nlp_analysis/           # NLP 文本分析
│   ├── visualization/          # 可视化
│   ├── utils/                  # 工具函数
│   └── tests/                  # 测试（文件名须以 test_ 开头）
├── config/                     # 配置文件
│   ├── default.yaml            # 匹配引擎参数
│   ├── database.yaml           # PostgreSQL 连接 / 模型路径
│   └── paths.py                # 集中路径与连接管理 (ProjectPaths)
├── archive/                    # 归档代码（不可被活跃代码依赖）
│   ├── process/                # 历史主流程
│   ├── scripts/                # 工具/基准测试脚本
│   ├── experiments/            # 实验代码
│   ├── docs/                   # 历史文档
│   ├── skill_extraction_history/
│   ├── llm_history/
│   └── job_title_parsing_history/
├── dicts/                      # 词典文件
├── output/                     # 输出结果
│   ├── reports/                # 分析报告（文本类，不入库）
│   └── skill_extraction/       # 技能抽取产物
└── README.md
```

## 数据存储规范

| 数据类型 | 存储方式 | 位置 |
|----------|----------|------|
| 原始招聘数据 | PostgreSQL | `recruit.raw_data.*` |
| 清洗后数据 | PostgreSQL | `recruit.main.*` |
| 匹配/分析结果 | PostgreSQL | `recruit.main.*` |
| 报告文本 | 文件系统 | `output/reports/` |
| 可视化图表 | 文件系统 | `output/reports/` |

**禁止使用 DuckDB**。数据库连接参数统一从 `config/paths.py` 的 `ProjectPaths.pg_connection_params` 获取。

## 依赖安装

```bash
pip install -r requirements.txt
```

## 数据字段

### 原始字段
- 发布时间、岗位名称、工作城市、薪资水平
- 学历要求、经验要求、岗位描述
- 公司名称、公司规模、公司行业

### 新增字段
- `occupation_core`: 职业核心词（如"工程师"）
- `core_category`: 职业类别（如"技术类"）
- `publish_month`: 发布月份（YYYY-MM）
- `city_clean`: 标准化城市名
- `industry_clean`: 标准化行业名
- `平均薪资`: 薪资平均值（元/月）

## 分析流程

1. **数据整合** - 添加职业类别和标准化字段
2. **职业类别薪资分析** - 分析职业与薪资关系
3. **行业景气度分析** - 分析行业招聘趋势
4. **时间趋势分析** - 分析月度变化
5. **词云生成** - 生成关键词词云
6. **基础薪资分析** - 多维度薪资对比

## 核心发现示例

### 职业类别薪资排行
1. 产品类 - 平均: 20,733元/月
2. 管理类 - 平均: 20,419元/月
3. 技术类 - 平均: 17,193元/月

### 行业招聘量排行
1. 电子技术 - 6,253个岗位
2. 电子/半导体 - 5,404个岗位
3. 计算机软件 - 5,362个岗位

## 注意事项

- 本项目使用1%样本数据进行分析
- 职业名称解析准确率达98%
- 所有时间维度统一按"月"处理
- 数据文件（~5GB）不提交到 git

## 开发规范

详见 [CLAUDE.md](CLAUDE.md)，核心规则摘要：

### 数据库
- **PostgreSQL 是唯一数据库，禁用 DuckDB**
- 连接参数通过 `config/paths.py` 获取，支持 `EMPLOYDATA_PG_*` 环境变量

### 路径管理
- **禁止硬编码绝对路径**（如 `D:\model\...`）
- 所有路径通过 `config/paths.py` 统一获取
- 支持 `EMPLOYDATA_*` 环境变量覆盖

### 代码风格
- **格式化**: black (line-length=88) + isort (profile=black)
- **Lint**: flake8 (max-line-length=88)
- **类型检查**: mypy (python_version=3.10)
- **命名**: 变量英文 snake_case，注释中文，类 PascalCase

### docstring 规范
- Google 风格 docstring（Args / Returns / Raises）
- 所有公开接口必须包含 docstring 和类型提示
- 每个 .py 文件必须以模块 docstring 开头

### 导入规范
- 禁止 `sys.path.insert()` 和星号导入
- 使用 `python -m src.xxx` 运行脚本
- 包内使用相对导入

### 提交规范
- Conventional Commits 格式
- PR 必须通过 `compileall + black --check + flake8 + pytest`

---

**项目状态**: ✅ 核心功能已完成  
**最后更新**: 2026-06-08
