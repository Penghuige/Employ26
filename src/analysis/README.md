# `src/analysis` 目录说明

当前 `src/analysis` 分成两条链路：

- 结构化统计主链路：直接读取 PostgreSQL `public.recruitment_jobs_normalized` 和 `public.skill_extraction_requirement_matches`，再导出常规统计报表。
- requirement text 第二阶段链路：直接读取 PostgreSQL `public.recruitment_jobs_normalized` 和 `public.job_description_parsed`，将 requirement text 抽取成可复用的约束事实层，再导出正式统计产物。

两条链路共用以下基础约定：

- 统一基础表：`public.recruitment_jobs_normalized`
- 统一轻量标准化字段：`publish_month`、`city_normalized`、`industry_normalized`、`company_size_normalized`
- 统一批次目录模式：`output/reports/{workflow}_{mm-dd}/`
- 统一运行清单：`run_manifest.json`
- 统一报告格式：Markdown

## 当前推荐入口

优先使用统一 CLI：

```bash
python -m src.analysis.cli structured run --with-excel
python -m src.analysis.cli requirements run
```

两条链路都支持 `--output-dir` 显式指定本次批次输出目录。

### 1. 结构化统计主链路

推荐命令：

```bash
python -m src.analysis.cli structured run
```

常用选项：

- `--with-excel`：最后运行 [`generate_excel_summary.py`](/d:/PythonProjects/Employ26/src/analysis/generate_excel_summary.py)
- `--skip-standardized`：跳过 [`generate_standardized_tables.py`](/d:/PythonProjects/Employ26/src/analysis/generate_standardized_tables.py)
- `--output-dir`：显式指定结构化统计批次输出目录

主输入：

- `public.recruitment_jobs_normalized`：招聘记录统一规范层
- `public.skill_extraction_requirement_matches`：职业匹配结果层，按 `recruitment_record_id` 回连招聘记录

单脚本调试顺序：

1. 先确保 [`backfill_recruitment_jobs_normalized.py`](/d:/PythonProjects/Employ26/src/data_pipeline/backfill_recruitment_jobs_normalized.py) 已回填统一规范层
2. 再确保 [`requirement_match_prep.py`](/d:/PythonProjects/Employ26/src/data_pipeline/requirement_match_prep.py) 已写入职业匹配结果层
3. 再运行 [`occupation_salary_analysis.py`](/d:/PythonProjects/Employ26/src/analysis/occupation_salary_analysis.py)
4. 再运行 [`education_distribution_analysis.py`](/d:/PythonProjects/Employ26/src/analysis/education_distribution_analysis.py)
5. 再运行 [`industry_trend_analysis.py`](/d:/PythonProjects/Employ26/src/analysis/industry_trend_analysis.py)
6. 如需交付层汇总，再运行 [`generate_standardized_tables.py`](/d:/PythonProjects/Employ26/src/analysis/generate_standardized_tables.py)
7. 如需最终汇总 Excel，再运行 [`generate_excel_summary.py`](/d:/PythonProjects/Employ26/src/analysis/generate_excel_summary.py)

### 2. requirement text 统计链路

推荐命令：

```bash
python -m src.analysis.cli requirements run
```

常用选项：

- `--output-dir`：显式指定 requirement text 批次输出目录

单脚本调试入口：

- [`requirement_text_analysis.py`](/d:/PythonProjects/Employ26/src/analysis/requirement_text_analysis.py)

依赖前置：

1. `public.recruitment_jobs_normalized` 已完成 sample 回填
2. `public.job_description_parsed` 已有岗位描述解析结果
3. `analysis_lexicon` schema 已建好，并存在唯一 `is_current = true` 的正式 release

相关脚本：

- [`backfill_recruitment_jobs_normalized.py`](/d:/PythonProjects/Employ26/src/data_pipeline/backfill_recruitment_jobs_normalized.py)
- [`analysis_lexicon.py`](/d:/PythonProjects/Employ26/src/db/analysis_lexicon.py)

## requirement text 第二阶段的正式口径

- 主输入：`public.recruitment_jobs_normalized` + `public.job_description_parsed`
- 主样本：`requirements_text` 非空的招聘记录
- 正式中间层：`public.requirement_constraint_facts`
- 主逻辑：`切分 -> 规范化 -> 约束抽取 -> 写入 PostgreSQL -> 聚合报表`
- 词汇资源：只读取 PostgreSQL `analysis_lexicon` 当前正式 release
- 规则资源：`analysis_lexicon.requirement_rules`
- 历史兼容：若 `job_description_parsed` 仍保留 `__source_table` / `__source_row_number` 旧字段，链路会自动映射后再与规范层回连

当前边界：

- `hard skill` 与 `soft skill` 相关词项暂只保留为探索性 hint，不作为当前正式研究结论
- 当前可正式使用的是 requirement 约束、模板噪声与招聘门槛强度统计，不是“稳定技能分类”统计
- 后续若要正式发布技能分类口径，需要单独补做更细的词典治理、歧义审查与标注验证

## 输出目录与文件

默认输出目录：

- 结构化统计：`output/reports/structured_analysis_{mm-dd}/`
- requirement text：`output/reports/req_analysis_{mm-dd}/`

固定产物：

- `run_manifest.json`
- `coverage_diagnostics.csv`
- `lexicon_summary.csv`
- `constraint_dimension_frequency.csv`
- `constraint_value_distribution.csv`
- `constraint_by_city_industry.csv`
- `template_noise_report.csv`
- `requirement_stringency_index.csv`
- `report.md`

命令示例：

```bash
python -m src.db.analysis_lexicon --ensure-schema
python -m src.db.analysis_lexicon --bootstrap-v1 --version v2_curated_requirement_analysis
python -m src.db.requirement_constraint_facts --ensure-schema
python -m src.data_pipeline.backfill_recruitment_jobs_normalized
python -m src.analysis.cli requirements run
```

结构化统计主链路的 Markdown 报告产物：

- `output/reports/structured_analysis_{mm-dd}/职业类别薪资分析报告.md`
- `output/reports/structured_analysis_{mm-dd}/学历需求分布分析报告.md`
- `output/reports/structured_analysis_{mm-dd}/行业景气度分析报告.md`
- `output/reports/structured_analysis_{mm-dd}/结构化维度补充分析报告.md`

新增规范 CSV 产物示例：

- `salary_by_occupation_month.csv`
- `salary_by_education_occupation.csv`
- `education_by_occupation_month.csv`
- `industry_monthly_jobs.csv`
- `experience_by_occupation.csv`
- `company_size_by_city_industry.csv`
- `city_occupation_demand.csv`

## 职业词典四表的当前建议

当前职业词典已经收敛到统一入口：

- `public.occ_dict_unified`：统一职业词典主表，包含职业叶子节点与分类骨架节点
- `public.occ_dict`：兼容 view
- `public.occ_dict_detailed`：兼容 view
- `public.occ_dict_pro`：兼容 view
- `public.occ_dict_class`：兼容 view

当前推荐：

- 职业匹配、检索、预处理默认读 `public.occ_dict_unified`
- 如需兼容旧脚本，可继续读 `public.occ_dict_detailed` / `public.occ_dict_pro`
- 分类骨架回查可读 `public.occ_dict_class`

## 历史 CSV 适配器说明

[`occupation_integration.py`](/d:/PythonProjects/Employ26/src/data_pipeline/occupation_integration.py) 只保留为历史兼容适配器，用于读取旧的职业解析 CSV 并生成 `output/integrated`。当前结构化统计主链路不再依赖它；如需新增统计字段，应优先补到 PostgreSQL 规范层或职业匹配/事实结果层，再由 [`structured_pg_source.py`](/d:/PythonProjects/Employ26/src/analysis/structured_pg_source.py) 暴露给分析脚本。
