# 招聘岗位到《中国职业分类大典》初步匹配系统

## 1. 项目简介

本模块用于将招聘岗位数据中的：
- 岗位名称
- 岗位描述

初步匹配到《中国职业分类大典》职业条目，输出：
- `clean_title`
- `top_k` 候选职业
- 每个候选的融合分数
- `top1` 匹配结果
- 可解释证据（如 `title` 命中、`tasks` 命中、层级命中等）
- 基础结构化特征（平台词、领域词、职能词、对象词、冲突词）

本版本仅实现：
- 数据预处理
- 规则清洗
- BM25 召回
- 字符 n-gram 补充检索
- 加权融合
- 层级过滤
- 基础结构化特征抽取
- 评估
- 命令行工具

**未实现任何大模型相关逻辑**，包括但不限于：Qwen、Prompt、LoRA、SFT、推理。

---

## 2. 数据格式说明

### 2.1 职业大典数据（DuckDB 主，CSV 辅）
至少包含字段：
- `code`
- `title`
- `desc`
- `tasks`
- `级别`
- `分类代码`
- `职业代码`
- `大类`
- `中类`
- `小类`
- `细类`

### 2.2 招聘岗位数据（DuckDB 主，CSV 辅）
至少包含字段：
- `岗位名称`
- `岗位描述`

可选字段：
- `job_id`
- `工作城市`
- `薪资水平`
- `公司行业`

---

## 3. 安装方式

建议 Python 3.10+。

```bash
pip install -r requirements.txt
```

本模块依赖项目现有依赖，核心只需要：
- `pandas`
- `jieba`
- `duckdb`（若需要保存到 DuckDB）

---

## 4. 运行示例（DuckDB 默认）

### 4.0 数据库配置统一位置

本模块默认的 DuckDB 路径、线程数、输入输出表名统一放在：`config/database.yaml`

当前包含：
- `database.duckdb_path`
- `database.duckdb_threads`
- `job_title_parsing.catalog_table`
- `job_title_parsing.catalog_preprocessed_table`
- `job_title_parsing.jobs_table`
- `job_title_parsing.match_result_table`

如果你不传 CLI 参数，默认就读取这里的配置。

### 4.1 预处理职业大典（默认从 DuckDB 读取）

```bash
python -m src.job_title_parsing.cli preprocess-catalog
```

默认等价于：
- 输入库：`output/recruit.duckdb`
- 输入表：`recruit.main.chinese_occupational_dictionary_joined`
- 输出表：`recruit.main.chinese_occupational_dictionary_joined_preprocessed`

显式指定示例：

```bash
python -m src.job_title_parsing.cli preprocess-catalog \
  --catalog-duckdb output/recruit.duckdb \
  --catalog-table recruit.main.chinese_occupational_dictionary_joined \
  --output-duckdb output/recruit.duckdb \
  --output-table recruit.main.chinese_occupational_dictionary_joined_preprocessed
```

### 4.2 批量匹配岗位（默认读写 DuckDB）

```bash
python -m src.job_title_parsing.cli match --jobs-table recruit.main.xxxx__sample --output-table recruit.main.job_match_results --job-title-col 岗位名称 --job-desc-col 岗位描述 --job-id-col job_id --top-k 5 --debug
```

说明：
- 职业大典默认读取：`recruit.main.chinese_occupational_dictionary_joined`
- 岗位默认读取：`recruit.main.jobs_sample`（建议你显式传 `--jobs-table`）
- 匹配结果默认写入：`recruit.main.job_match_results`
- 结果中会新增：`platform_terms`、`domain_terms`、`function_terms`、`object_terms`、`conflict_terms`

### 4.3 评估结果（默认从 DuckDB 读取）

```bash
python -m src.job_title_parsing.cli evaluate \
  --result-table recruit.main.job_match_results
```

如需 CSV 临时流程（仅小体量数据），再使用 `--catalog-csv` / `--jobs-csv` / `--result-csv`。

---

## 5. 模块说明

### `catalog_preprocessor.py`
职业大典预处理：
- 读取 CSV
- 清洗空白/全半角/异常换行
- 拆分 `tasks`
- 构造 `task_list`、`task_text_joined`、`hierarchy_text`
- 构造 `retrieval_title_text` / `retrieval_desc_text` / `retrieval_task_text`
- 默认 DuckDB 路径和表名读取 `config/database.yaml`
- 支持保存到 DuckDB

### `alias_builder.py`
职业别名构建：
- 读取 `alias_dict_example.json`
- 规则生成弱别名
- 支持口语岗位名映射（如 `导购 -> 商品营业员`）

### `title_cleaner.py`
岗位名称去噪：
- 去除招聘营销词
- 去除薪资模式
- 去除地点/门店尾巴
- 尽量保留职业核心词

### `jd_parser.py`
岗位描述解析：
- 生成 `jd_clean`
- 生成 `jd_sentences`
- 抽取 `core_task_sentences`
- 提取 `domain_keywords`

### `feature_extractor.py`
岗位结构化特征抽取：
- 从标题与 JD 提取平台词
- 提取领域词、职能词、对象词
- 提取高风险冲突词
- 为后续歧义消解和冲突惩罚提供底座

### `bm25_index.py`
双路 BM25 检索：
- `search_title(query, top_k)`
- `search_tasks(query, top_k)`

### `ngram_retrieval.py`
字符 n-gram 补充分数：
- 2-gram / 3-gram overlap
- 解决分词错误导致的漏召回

### `hierarchy_filter.py`
层级过滤：
- 通过关键词推断职业大类
- 若无法判断则不过滤

### `scoring.py`
融合打分：
- `title_bm25_score`
- `task_bm25_score`
- `desc_ngram_score`
- `alias_exact_match_bonus`
- `task_overlap_bonus`
- `hierarchy_match_bonus`

### `matching_pipeline.py`
主流程：
- 岗位名称去噪
- JD 解析
- 基础结构化特征抽取
- 层级过滤
- title/tasks 双路召回
- ngram 打分
- 分数融合
- 输出 top_k 候选

### `matching_evaluator.py`
评估：
- `top1 accuracy`
- `top3 recall`
- `top5 recall`
- `unmatched rate`

### `cli.py`
命令行入口。
- DuckDB 路径与相关表名默认读取 `config/database.yaml`
- `match` 结果会额外输出结构化特征列

---

## 6. 配置说明

主业务配置文件位置：`config/default.yaml`

数据库与表配置位置：`config/database.yaml`

默认权重：
- `title_weight: 0.40`
- `task_weight: 0.35`
- `desc_weight: 0.15`
- `hierarchy_weight: 0.10`
- `alias_bonus: 0.10`
- `task_overlap_bonus: 0.08`

还包含：
- BM25 参数
- n-gram 参数
- 岗位名称噪音词表
- 动作词表
- 领域关键词
- 层级关键词映射

---

## 7. 匹配策略说明

本系统当前采用以下思路：
1. 先做岗位名称去噪
2. 再从 JD 中抽出 `core_task_sentences`
3. 抽取基础结构化特征
4. 构建 `title` 索引和 `tasks` 索引
5. 用 `title/tasks` 双路召回
6. 用 `n-gram` 和 bonus 做重排
7. 输出 `top_k` 候选
8. 暂时不做 unmatched 判定模型，但保留空 title 的弱处理逻辑

---

## 8. 后续扩展方向

后续可以在当前规则+检索框架上继续增强：
- 引入更强的别名体系
- 更细粒度的层级分类器
- 增加歧义词消解
- 增加冲突惩罚
- 结合公司行业/城市/薪资做重排
- 增加 unmatched 判定
- 接入 Qwen 做候选重排与解释生成

**注意：当前版本没有实现任何 Qwen 代码。**
