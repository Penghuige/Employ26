# SRC 项目管理审计与优化报告

## 0. 审计范围与结论

本次审计基于上传的 `src.zip` 解压目录完成，覆盖源码结构、Python AST、基础 PEP 8 风格、导入关系、硬编码路径、归档候选、命名规范与协作流程。由于 `bert/llm_annotate.py` 存在语法错误，该文件无法进入 AST 级 docstring/type-hint 统计，需先修复后复扫。

### 总览指标

| 指标 | 数量 |
| --- | ---: |
| Python 源文件 | 134 |
| 目录数（不含 `__pycache__`） | 19 |
| `.pyc` 缓存文件 | 204 |
| 编译/语法错误 | 1 |
| 缺少模块 docstring | 28 |
| 缺少公开接口 docstring | 184 |
| 缺少公开接口类型提示 | 178 |
| 疑似硬编码路径命中 | 42 |
| `sys.path` 运行时修改 | 12 |
| 星号导入 | 1 |
| 超过 black 默认 88 列的行 | 988 |

### 严重程度分级

| 严重程度 | 判定标准 | 本项目主要问题 |
| --- | --- | --- |
| 致命 | 会导致项目无法编译、无法稳定运行或污染版本库基础状态 | `bert/llm_annotate.py` 语法错误；大量 `.pyc` 与 `__pycache__` 进入源码包 |
| 重要 | 影响多人协作、跨环境运行、CI、可维护性或接口稳定性 | 大量硬编码 `D:\...` 路径；公开接口文档与类型提示缺失；`sys.path` 注入；循环引用风险；重复方法定义 |
| 建议 | 不一定立即破坏运行，但会增加评审、迁移和长期维护成本 | 文件命名不统一；实验脚本混入 `tests/`；单文件过长；周报文件名编码异常 |

## 1. 优先级待办事项清单

| 优先级 | 严重程度 | 任务 | 文件/目录 | 可执行动作 |
| ---: | --- | --- | --- | --- |
| P0 | 致命 | 修复语法错误，恢复全项目可编译 | `bert/llm_annotate.py:293-323` | 将 `entities_to_bio()` docstring 结尾的 `""` 改为 `"""`，然后运行 `python -m compileall -q src` |
| P0 | 致命 | 清理缓存文件并加入忽略规则 | 全项目 `__pycache__/`, `*.pyc` | 执行 `Get-ChildItem -Recurse -Directory -Filter "__pycache__" \| Remove-Item -Recurse -Force`，并使用本报告附带 `.gitignore.recommended` |
| P1 | 重要 | 将模型、数据库、输出路径从源码迁移到配置层 | `bge/*`, `rag/config.py`, `skill_extraction/config.py`, `preprocessing/*`, `penghui/*` | 使用 `config/database.yaml` + 环境变量覆盖，见第 6 节代码片段 |
| P1 | 重要 | 为公开接口补齐 docstring 与类型提示 | 重点：`job_title_parsing/occupation_dictionary_pipeline.py`, `bert/bert_ner.py`, `analysis/*`, `utils/*` | 使用 `docstring_todo_templates.md` 分批补齐，先覆盖被 README 标记为主链路的模块 |
| P1 | 重要 | 去除 `sys.path.insert()` 与星号导入 | `bge/*`, `llm/*`, `rag/batch_rag_match.py`, `skill_extraction/llm_labeling_utils.py` | 统一用 `python -m src.xxx` 启动；显式导入所需对象 |
| P1 | 重要 | 处理潜在循环引用 | `skill_extraction/context_classifier.py`, `match_flat_skills_to_duckdb.py`, `regression_eval.py` | 将共享类型抽到 `skill_extraction/context_types.py` 或 `protocols.py` |
| P1 | 重要 | 合并重复方法定义 | `analysis/occupation_salary_analysis.py:220/289`, `:246/315` | 确认两版口径后合并；避免后定义静默覆盖前定义 |
| P2 | 建议 | 规范命名与阶段脚本 | `bge/D1_*.py` 至 `D7_*.py` | 用 `git mv` 改为 `step_01_deduplicate.py` 等蛇形命名 |
| P2 | 建议 | 归档实验和旧版代码 | `penghui/`, `llm/archive/`, `job_title_parsing/archive/`, `tests` 下非测试脚本 | 先执行 `archive_plan.ps1` 的 `-WhatIf` 预览，再由维护者确认移动 |
| P2 | 建议 | 拆分超长模块 | `skill_extraction/match_flat_skills_to_duckdb.py`, `merge_similar_skills.py`, `rag/batch_rag_match.py` | 按 CLI、配置、匹配器、报告输出、评测逻辑拆分 |

## 2. Python 注释、docstring 与类型提示审计

### 2.1 主要发现

- 缺少 module docstring 的文件：`__init__.py`, `analysis/__init__.py`, `bert/bert_ner.py`, `bert/train_bert.py`, `bge/D1_quchong.py`, `bge/D2_filter.py`, `bge/D3_finetune.py`, `bge/D4_T2match.py`, `bge/D6_threshold_eval.py`, `bge/D7_iterate_pipeline.py`, `bge/export_tier1_label_studio.py`, `bge/format_label_studio_requirements.py`, `bge/slice_label_studio_json.py`, `preprocessing/parse_desc.py`, `rag/qc_utils.py`, `skill_extraction/iterate_flat_skill_dictionary.py`, `skill_extraction/iteration_rules.py`, `skill_extraction/skill_dictionary_workflow.py`, `tests/parse_job_desc.py`, `tests/qwen_doc_text.py`, `tests/test_api_key.py`, `tests/test_job_title_matching_fixes.py`, `tests/test_llm_router.py`, `tests/test_match_validation_escalation.py`, `tests/test_occupation_iteration_utils.py`, `tests/test_skill_extraction_config.py`, `tests/test_skill_label_cleaning.py`, `utils/utils.py`。
- 公开接口缺少 docstring：184 处。
- 公开接口缺少类型提示：178 处。
- `bert/llm_annotate.py` 因语法错误未纳入 AST 统计，修复后需要复扫。

### 2.2 docstring 缺失高发文件

| 文件 | 缺失数量 |
| --- | ---: |
| `job_title_parsing/occupation_dictionary_pipeline.py` | 21 |
| `tests/build_gpt_knowledge_files.py` | 16 |
| `preprocessing/parse_desc.py` | 13 |
| `bert/bert_ner.py` | 10 |
| `utils/llm_labeling_utils.py` | 10 |
| `utils/llm_router.py` | 8 |
| `bert/train_bert.py` | 7 |
| `bge/format_label_studio_requirements.py` | 6 |
| `penghui/train_rag_weighted.py` | 6 |
| `skill_extraction/iteration_rules.py` | 5 |
| `skill_extraction/regression_eval.py` | 5 |
| `skill_extraction/skill_dictionary_workflow.py` | 5 |

### 2.3 类型提示缺失高发文件

| 文件 | 缺失数量 |
| --- | ---: |
| `analysis/occupation_salary_analysis.py` | 13 |
| `analysis/education_distribution_analysis.py` | 10 |
| `bert/bert_ner.py` | 8 |
| `preprocessing/integrate_occupation.py` | 8 |
| `analysis/industry_trend_analysis.py` | 7 |
| `tests/migrate_duckdb_to_pg.py` | 7 |
| `analysis/generate_standardized_tables.py` | 6 |
| `bert/ner_predict.py` | 6 |
| `bert/train_bert.py` | 6 |
| `nlp_analysis/text_preprocessing.py` | 6 |
| `penghui/train_rag_weighted.py` | 6 |
| `bert/prepare_data.py` | 5 |

### 2.4 建议采用的 Google 风格模板

```python
def parse_salary(salary_str: str) -> tuple[float | None, float | None]:
    """Parse a raw salary string into lower and upper monthly salary bounds.

    Args:
        salary_str: Raw salary text from a job posting, such as "10-15K".

    Returns:
        A tuple of `(lower_bound, upper_bound)`. Returns `(None, None)` when
        the value cannot be parsed.

    Raises:
        ValueError: If strict parsing is enabled and the salary format is invalid.
    """
```

自动生成的逐项模板见附件 `docstring_todo_templates.md`。

## 3. 代码风格与一致性

### 3.1 PEP 8 / black 风格问题

- 超过 88 列的行共 988 处。
- 检测到 `preprocessing/duckdb_cleaner.py` 的 docstring 中存在 Windows 反斜杠转义警告：`invalid escape sequence '\P'`。建议将模块 docstring 改成 raw string 或把反斜杠写成 `\\`。
- `analysis/occupation_salary_analysis.py` 存在重复方法定义，属于可运行但高风险的维护问题。

| 文件 | 超长行数量 |
| --- | ---: |
| `skill_extraction/llm_label_regression_dataset.py` | 44 |
| `job_title_parsing/cli.py` | 41 |
| `penghui/reproduce_round2_validity.py` | 40 |
| `preprocessing/parse_desc.py` | 40 |
| `job_title_parsing/occupation_dictionary_pipeline.py` | 33 |
| `job_title_parsing/matching_pipeline.py` | 30 |
| `llm/archive/vllm_launcher.py` | 26 |
| `skill_extraction/history/qwen_skill_workflow.py` | 25 |
| `bge/D2_filter.py` | 24 |
| `bge/D4_T2match.py` | 23 |
| `job_title_parsing/scoring.py` | 22 |
| `skill_extraction/context_classifier.py` | 22 |

### 3.2 推荐配置：`pyproject.toml`

```toml
[tool.black]
line-length = 88
target-version = ["py310", "py311"]
exclude = '''
/(
    \.git
  | \.venv
  | __pycache__
  | archive
  | history
  | output
  | models
  | dist
  | build
)/
'''

[tool.isort]
profile = "black"
line_length = 88
known_first_party = ["src", "analysis", "bert", "bge", "job_title_parsing", "rag", "skill_extraction", "utils"]
skip = ["archive", "history", "__pycache__", ".venv", "output"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-q"

[tool.mypy]
python_version = "3.10"
ignore_missing_imports = true
warn_unused_ignores = true
warn_return_any = false
exclude = "(archive|history|__pycache__|output|models)"
```

### 3.3 推荐配置：`.flake8`

```ini
[flake8]
max-line-length = 88
extend-ignore = E203,W503
exclude =
    .git,
    .venv,
    __pycache__,
    archive,
    history,
    output,
    models,
    dist,
    build
per-file-ignores =
    __init__.py:F401
    */archive/*:E501,F401,F403
    */history/*:E501,F401,F403
```

### 3.4 本地检查命令

```powershell
python -m compileall -q src
black --check src tests
flake8 src tests
pytest -q
```

确认格式化后执行：

```powershell
black src tests
isort src tests
```

## 4. 协作规范设计

已单独生成 `CONTRIBUTING.md` 草案，覆盖分支策略、提交信息、代码审查清单、本地环境设置、禁止直接 push 到主分支、配置与密钥管理、归档规则。

核心建议：

- `main` 只接受 PR 合并，禁止直接 push。
- 采用 `feature/`、`fix/`、`refactor/`、`experiment/` 分支。
- 提交信息采用 Conventional Commits。
- PR 必须通过 `compileall + black --check + flake8 + pytest`。
- 实验代码必须在 README 或归档记录中说明责任人、目标、替代入口和是否仍被主流程依赖。

## 5. 文件归档判断

### 5.1 立即删除，不建议归档

`.pyc` 与 `__pycache__` 是解释器缓存，不属于项目资产。本次发现 204 个 `.pyc`，其中部分缓存对应的源码已不存在，说明版本库状态不干净。

| 缓存文件 | 缺失源码 |
| --- | --- |
| `analysis/__pycache__/generate_excel_report.cpython-311.pyc` | `analysis/generate_excel_report.py` |
| `analysis/__pycache__/salary_analysis.cpython-311.pyc` | `analysis/salary_analysis.py` |
| `analysis/__pycache__/salary_analysis.cpython-313.pyc` | `analysis/salary_analysis.py` |
| `analysis/__pycache__/skill_combination.cpython-311.pyc` | `analysis/skill_combination.py` |
| `analysis/__pycache__/skill_combination.cpython-313.pyc` | `analysis/skill_combination.py` |
| `llm/__pycache__/batch_annotator.cpython-311.pyc` | `llm/batch_annotator.py` |
| `llm/__pycache__/ner_schema.cpython-311.pyc` | `llm/ner_schema.py` |
| `llm/__pycache__/prompt_builder.cpython-311.pyc` | `llm/prompt_builder.py` |
| `llm/__pycache__/qwen3_extractor.cpython-311.pyc` | `llm/qwen3_extractor.py` |
| `llm/__pycache__/vllm_launcher.cpython-311.pyc` | `llm/vllm_launcher.py` |
| `llm/__pycache__/vllm_launcher.cpython-313.pyc` | `llm/vllm_launcher.py` |
| `skill_extraction/__pycache__/clean_skill_dictionary.cpython-311.pyc` | `skill_extraction/clean_skill_dictionary.py` |
| `skill_extraction/__pycache__/clean_skill_dictionary.cpython-313.pyc` | `skill_extraction/clean_skill_dictionary.py` |
| `skill_extraction/__pycache__/coverage.cpython-310.pyc` | `skill_extraction/coverage.py` |
| `skill_extraction/__pycache__/coverage.cpython-311.pyc` | `skill_extraction/coverage.py` |
| `skill_extraction/__pycache__/coverage.cpython-313.pyc` | `skill_extraction/coverage.py` |
| `skill_extraction/__pycache__/data_source.cpython-310.pyc` | `skill_extraction/data_source.py` |
| `skill_extraction/__pycache__/data_source.cpython-311.pyc` | `skill_extraction/data_source.py` |
| `skill_extraction/__pycache__/data_source.cpython-313.pyc` | `skill_extraction/data_source.py` |
| `skill_extraction/__pycache__/dictionary_store.cpython-310.pyc` | `skill_extraction/dictionary_store.py` |
| `skill_extraction/__pycache__/dictionary_store.cpython-311.pyc` | `skill_extraction/dictionary_store.py` |
| `skill_extraction/__pycache__/dictionary_store.cpython-313.pyc` | `skill_extraction/dictionary_store.py` |
| `skill_extraction/__pycache__/import_llm_results.cpython-313.pyc` | `skill_extraction/import_llm_results.py` |
| `skill_extraction/__pycache__/init_llm_output_layout.cpython-311.pyc` | `skill_extraction/init_llm_output_layout.py` |
| `skill_extraction/__pycache__/init_llm_output_layout.cpython-313.pyc` | `skill_extraction/init_llm_output_layout.py` |
| `skill_extraction/__pycache__/match_hard_skills_to_duckdb.cpython-310.pyc` | `skill_extraction/match_hard_skills_to_duckdb.py` |
| `skill_extraction/__pycache__/match_hard_skills_to_duckdb.cpython-313.pyc` | `skill_extraction/match_hard_skills_to_duckdb.py` |
| `skill_extraction/__pycache__/optimize_hard_skill_dictionary.cpython-313.pyc` | `skill_extraction/optimize_hard_skill_dictionary.py` |
| `skill_extraction/__pycache__/qwen_skill_workflow.cpython-311.pyc` | `skill_extraction/qwen_skill_workflow.py` |
| `skill_extraction/__pycache__/qwen_skill_workflow.cpython-313.pyc` | `skill_extraction/qwen_skill_workflow.py` |
| `tests/__pycache__/merge_similar_skills.cpython-313.pyc` | `tests/merge_similar_skills.py` |
| `tests/__pycache__/start_session.cpython-311.pyc` | `tests/start_session.py` |
| `tests/__pycache__/test1.cpython-311.pyc` | `tests/test1.py` |
| `tests/__pycache__/test1.cpython-313.pyc` | `tests/test1.py` |
| `tests/__pycache__/vllm_launcher.cpython-313.pyc` | `tests/vllm_launcher.py` |

### 5.2 建议归档或隔离的文件/目录

| 文件或目录 | 理由 | 建议动作 |
| --- | --- | --- |
| `skill_extraction/history/` | 目录名明确表示历史版本，应并入统一 archive/history 归档区，避免活跃代码误依赖。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `bert/#U5de5#U4f5c#U5468#U62a5_bert.md` | 疑似导出/周报文件且文件名编码异常；建议移至 docs/archive 或改名为可读中文/英文名。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `bge/#U5de5#U4f5c#U5468#U62a5_bge.md` | 疑似导出/周报文件且文件名编码异常；建议移至 docs/archive 或改名为可读中文/英文名。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `job_title_parsing/#U5de5#U4f5c#U5468#U62a5_job_title_parsing.md` | 疑似导出/周报文件且文件名编码异常；建议移至 docs/archive 或改名为可读中文/英文名。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `penghui/deep_analysis_round2.py` | 文件名包含版本/轮次/临时标记，疑似实验迭代脚本，应迁入 archive/experiments 并保留实验说明。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `penghui/reproduce_round2_validity.py` | 文件名包含版本/轮次/临时标记，疑似实验迭代脚本，应迁入 archive/experiments 并保留实验说明。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `penghui/train_rag_round2.py` | 文件名包含版本/轮次/临时标记，疑似实验迭代脚本，应迁入 archive/experiments 并保留实验说明。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `penghui/train_rag_round2_v3.py` | 文件名包含版本/轮次/临时标记，疑似实验迭代脚本，应迁入 archive/experiments 并保留实验说明。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `penghui/train_rag_round2_v4.py` | 文件名包含版本/轮次/临时标记，疑似实验迭代脚本，应迁入 archive/experiments 并保留实验说明。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `skill_extraction/pipeline_v1.py` | 文件名包含版本/轮次/临时标记，疑似实验迭代脚本，应迁入 archive/experiments 并保留实验说明。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `skill_extraction/pipeline_v2.py` | 文件名包含版本/轮次/临时标记，疑似实验迭代脚本，应迁入 archive/experiments 并保留实验说明。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `tests/benchmark_vllm_speed.py` | tests 下非 test_ 命名脚本更像基准/迁移/临时验证工具，不会被 pytest 标准发现；建议移到 scripts/ 或 archive/experiments。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `tests/build_gpt_knowledge_files.py` | tests 下非 test_ 命名脚本更像基准/迁移/临时验证工具，不会被 pytest 标准发现；建议移到 scripts/ 或 archive/experiments。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `tests/migrate_duckdb_to_pg.py` | tests 下非 test_ 命名脚本更像基准/迁移/临时验证工具，不会被 pytest 标准发现；建议移到 scripts/ 或 archive/experiments。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `tests/parse_desc.py` | tests 下非 test_ 命名脚本更像基准/迁移/临时验证工具，不会被 pytest 标准发现；建议移到 scripts/ 或 archive/experiments。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `tests/parse_job_desc.py` | tests 下非 test_ 命名脚本更像基准/迁移/临时验证工具，不会被 pytest 标准发现；建议移到 scripts/ 或 archive/experiments。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `tests/qwen_doc_text.py` | tests 下非 test_ 命名脚本更像基准/迁移/临时验证工具，不会被 pytest 标准发现；建议移到 scripts/ 或 archive/experiments。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `skill_extraction/history/__init__.py` | 已位于 archive/history，但仍是可导入 Python 文件；建议隔离为非包目录或排除 CI 检查，防止活跃代码误依赖。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `skill_extraction/history/build_merged_hard_skill_dictionary.py` | 已位于 archive/history，但仍是可导入 Python 文件；建议隔离为非包目录或排除 CI 检查，防止活跃代码误依赖。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `skill_extraction/history/clean_skill_dictionary.py` | 已位于 archive/history，但仍是可导入 Python 文件；建议隔离为非包目录或排除 CI 检查，防止活跃代码误依赖。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `skill_extraction/history/coverage.py` | 已位于 archive/history，但仍是可导入 Python 文件；建议隔离为非包目录或排除 CI 检查，防止活跃代码误依赖。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `skill_extraction/history/data_source.py` | 已位于 archive/history，但仍是可导入 Python 文件；建议隔离为非包目录或排除 CI 检查，防止活跃代码误依赖。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `skill_extraction/history/dictionary_store.py` | 已位于 archive/history，但仍是可导入 Python 文件；建议隔离为非包目录或排除 CI 检查，防止活跃代码误依赖。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `skill_extraction/history/import_llm_results.py` | 已位于 archive/history，但仍是可导入 Python 文件；建议隔离为非包目录或排除 CI 检查，防止活跃代码误依赖。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `skill_extraction/history/init_llm_output_layout.py` | 已位于 archive/history，但仍是可导入 Python 文件；建议隔离为非包目录或排除 CI 检查，防止活跃代码误依赖。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `skill_extraction/history/match_hard_skills_to_duckdb.py` | 已位于 archive/history，但仍是可导入 Python 文件；建议隔离为非包目录或排除 CI 检查，防止活跃代码误依赖。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `skill_extraction/history/occupation_skill_pipeline.py` | 已位于 archive/history，但仍是可导入 Python 文件；建议隔离为非包目录或排除 CI 检查，防止活跃代码误依赖。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `skill_extraction/history/optimize_hard_skill_dictionary.py` | 已位于 archive/history，但仍是可导入 Python 文件；建议隔离为非包目录或排除 CI 检查，防止活跃代码误依赖。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `skill_extraction/history/qwen_skill_workflow.py` | 已位于 archive/history，但仍是可导入 Python 文件；建议隔离为非包目录或排除 CI 检查，防止活跃代码误依赖。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `llm/archive/__init__.py` | 已位于 archive/history，但仍是可导入 Python 文件；建议隔离为非包目录或排除 CI 检查，防止活跃代码误依赖。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `llm/archive/batch_annotator.py` | 已位于 archive/history，但仍是可导入 Python 文件；建议隔离为非包目录或排除 CI 检查，防止活跃代码误依赖。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `llm/archive/bio_converter.py` | 已位于 archive/history，但仍是可导入 Python 文件；建议隔离为非包目录或排除 CI 检查，防止活跃代码误依赖。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `llm/archive/ner_schema.py` | 已位于 archive/history，但仍是可导入 Python 文件；建议隔离为非包目录或排除 CI 检查，防止活跃代码误依赖。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `llm/archive/prompt_builder.py` | 已位于 archive/history，但仍是可导入 Python 文件；建议隔离为非包目录或排除 CI 检查，防止活跃代码误依赖。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `llm/archive/qwen3_extractor.py` | 已位于 archive/history，但仍是可导入 Python 文件；建议隔离为非包目录或排除 CI 检查，防止活跃代码误依赖。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `llm/archive/vllm_launcher.py` | 已位于 archive/history，但仍是可导入 Python 文件；建议隔离为非包目录或排除 CI 检查，防止活跃代码误依赖。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `job_title_parsing/archive/evaluate_matching.py` | 已位于 archive/history，但仍是可导入 Python 文件；建议隔离为非包目录或排除 CI 检查，防止活跃代码误依赖。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `job_title_parsing/archive/evaluate_parser.py` | 已位于 archive/history，但仍是可导入 Python 文件；建议隔离为非包目录或排除 CI 检查，防止活跃代码误依赖。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `job_title_parsing/archive/occupation_dict_manager.py` | 已位于 archive/history，但仍是可导入 Python 文件；建议隔离为非包目录或排除 CI 检查，防止活跃代码误依赖。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |
| `job_title_parsing/archive/occupation_parser.py` | 已位于 archive/history，但仍是可导入 Python 文件；建议隔离为非包目录或排除 CI 检查，防止活跃代码误依赖。 | 迁入 archive/ 并在 README 记录归档原因和替代入口。 |

### 5.3 归档命令预案

已生成 `archive_plan.ps1`，默认带 `-WhatIf`，不会实际移动。建议维护者先预览，再逐项确认。

```powershell
.rchive_plan.ps1
```

## 6. 目录与文件命名规范评估

### 6.1 Python 文件命名问题

| 当前文件 | 推荐名称 | 问题 |
| --- | --- | --- |
| `bge/D1_quchong.py` | `d1_quchong.py` 或阶段化命名如 `step_01_deduplicate.py` | Python 文件名不是 snake_case |
| `bge/D2_filter.py` | `d2_filter.py` 或阶段化命名如 `step_01_deduplicate.py` | Python 文件名不是 snake_case |
| `bge/D3_finetune.py` | `d3_finetune.py` 或阶段化命名如 `step_01_deduplicate.py` | Python 文件名不是 snake_case |
| `bge/D4_T2match.py` | `d4_t2match.py` 或阶段化命名如 `step_01_deduplicate.py` | Python 文件名不是 snake_case |
| `bge/D5_qwen3_auto_label.py` | `d5_qwen3_auto_label.py` 或阶段化命名如 `step_01_deduplicate.py` | Python 文件名不是 snake_case |
| `bge/D6_threshold_eval.py` | `d6_threshold_eval.py` 或阶段化命名如 `step_01_deduplicate.py` | Python 文件名不是 snake_case |
| `bge/D7_iterate_pipeline.py` | `d7_iterate_pipeline.py` 或阶段化命名如 `step_01_deduplicate.py` | Python 文件名不是 snake_case |

更具体的推荐映射：

```powershell
git mv src/bge/D1_quchong.py src/bge/step_01_deduplicate.py
git mv src/bge/D2_filter.py src/bge/step_02_filter.py
git mv src/bge/D3_finetune.py src/bge/step_03_finetune.py
git mv src/bge/D4_T2match.py src/bge/step_04_tier2_match.py
git mv src/bge/D5_qwen3_auto_label.py src/bge/step_05_qwen3_auto_label.py
git mv src/bge/D6_threshold_eval.py src/bge/step_06_threshold_eval.py
git mv src/bge/D7_iterate_pipeline.py src/bge/step_07_iterate_pipeline.py
```

### 6.2 编码异常文件名

| 当前文件 | 建议 |
| --- | --- |
| `bert/#U5de5#U4f5c#U5468#U62a5_bert.md` | 建议改为可读英文名并移入 `docs/archive/` |
| `bge/#U5de5#U4f5c#U5468#U62a5_bge.md` | 建议改为可读英文名并移入 `docs/archive/` |
| `job_title_parsing/#U5de5#U4f5c#U5468#U62a5_job_title_parsing.md` | 建议改为可读英文名并移入 `docs/archive/` |

### 6.3 目录命名建议

- `utils/` 可以保留，但建议避免继续堆积“万能工具函数”；新增工具应按领域拆到 `io_utils.py`、`llm_router.py`、`path_utils.py` 等明确模块。
- `penghui/` 是人名目录，不适合作为长期主干目录。建议迁入 `archive/experiments/penghui/` 或重命名为表达业务含义的目录。
- `history/` 与 `archive/` 含义重叠，建议统一为顶层 `archive/`，活跃代码不得依赖归档目录。

## 7. 路径管理与导入优化

### 7.1 硬编码路径样例

| 位置 | 命中内容 |
| --- | --- |
| `bge/D2_filter.py:71` | `RAG_EMBEDDING_MODEL = r"D:\model\bge-base-zh-v1.5"` |
| `bge/D2_filter.py:78` | `QWEN_MODEL_PATH = r"D:\model\Qwen3-8B"` |
| `bge/D3_finetune.py:24` | `LOCAL_MODEL_PATH = r"D:\model\bge-base-zh-v1.5"` |
| `bge/D3_finetune.py:26` | `OUTPUT_MODEL_PATH = r"D:\model\bge-base-zh-finetuned"` |
| `bge/D4_T2match.py:26` | `FINETUNED_MODEL_PATH = r"D:\model\bge-base-zh-finetuned"` |
| `bge/D5_qwen3_auto_label.py:58` | `MODEL_PATH = r"D:\model\Qwen3-8B"` |
| `bge/D5_qwen3_auto_label.py:102` | `RAG_EMBEDDING_MODEL = r"D:\model\bge-base-zh-finetuned"` |
| `bge/D5_qwen3_auto_label.py:517` | `f.write("raw_output:\n")` |
| `bge/export_tier1_label_studio.py:31` | `RAG_EMBEDDING_MODEL = r"D:\model\bge-base-zh-finetuned"` |
| `job_title_parsing/archive/occupation_dict_manager.py:20` | `DEFAULT_DB_PATH = r"D:\PythonProjects\Employ26\output\recruit.duckdb"` |
| `llm/archive/qwen3_extractor.py:6` | `模型路径：D:\\model\\Qwen3-8B` |
| `llm/archive/qwen3_extractor.py:20` | `MODEL_PATH = r"D:\model\Qwen3-8B"` |
| `penghui/disagreement_deep_analysis.py:23` | `MODEL_PATH = r"D:\model\bge-large-zh-v1.5"` |
| `penghui/eval_models_multimetric.py:34` | `"baseline (bge-large)": r"D:\model\bge-large-zh-v1.5",` |
| `penghui/multidim_validation.py:32` | `MODEL_PATH = r"D:\model\bge-large-zh-v1.5"` |
| `penghui/train_rag_round2.py:42` | `BASE_MODEL_PATH = r"D:\model\bge-large-zh-v1.5"` |
| `penghui/train_rag_round2_v3.py:34` | `BASE_MODEL_PATH = r"D:\model\bge-large-zh-v1.5"` |
| `penghui/train_rag_round2_v4.py:29` | `BASE_MODEL_PATH = r"D:\model\bge-large-zh-v1.5"` |
| `penghui/train_rag_weighted.py:32` | `BASE_MODEL_PATH = r"D:\model\bge-large-zh-v1.5"` |
| `preprocessing/duckdb_cleaner.py:6` | `1. 连接 DuckDB 数据库（D:\PythonProjects\Employ26\output\recruit.duckdb）` |
| `preprocessing/duckdb_cleaner.py:207` | `default=r"D:\PythonProjects\Employ26\output\recruit.duckdb",` |
| `preprocessing/prepare_skill_extraction_requirement_matches.py:8` | `3. 使用本地 BGE 微调模型 `D:\\model\\bge-base-zh-finetuned` 做职业细类语义匹配` |
| `preprocessing/sample_data.py:19` | `DEFAULT_DB_PATH = r"D:\PythonProjects\Employ26\output\recruit.duckdb"` |
| `rag/batch_rag_match.py:97` | `embedding_model_path: str = r"D:\model\bge-large-zh-v1.5"` |
| `rag/cli.py:68` | `query_p.add_argument("--embedding-model", default=r"D:\model\bge-large-zh-v1.5")` |

建议将路径集中到配置对象：

```python
from dataclasses import dataclass
from pathlib import Path
import os

@dataclass(frozen=True)
class ProjectPaths:
    duckdb_path: Path = Path(os.getenv("EMPLOYDATA_DUCKDB_PATH", "output/recruit.duckdb"))
    bge_model_path: Path = Path(os.getenv("EMPLOYDATA_BGE_MODEL_PATH", "models/bge-base-zh-finetuned"))
    qwen_model_path: Path = Path(os.getenv("EMPLOYDATA_QWEN_MODEL_PATH", "models/Qwen3-8B"))
```

在 CLI 中允许覆盖：

```python
parser.add_argument("--config", type=Path, default=Path("config/database.yaml"))
parser.add_argument("--embedding-model", type=Path, default=paths.bge_model_path)
```

### 7.2 `sys.path` 修改位置

| 位置 | 代码 |
| --- | --- |
| `bert/run_bert_training.py:22` | `sys.path.insert(0, str(Path(__file__).parent.parent.parent))` |
| `bge/D2_filter.py:31` | `sys.path.insert(0, _PROJECT_ROOT)` |
| `bge/D5_qwen3_auto_label.py:39` | `sys.path.insert(0, _PROJECT_ROOT)` |
| `bge/D7_iterate_pipeline.py:22` | `sys.path.insert(0, _PROJECT_ROOT)` |
| `bge/export_tier1_label_studio.py:14` | `sys.path.insert(0, _PROJECT_ROOT)` |
| `job_title_parsing/archive/evaluate_parser.py:16` | `sys.path.insert(0, str(Path(__file__).parent.parent.parent))` |
| `llm/archive/batch_annotator.py:22` | `sys.path.insert(0, str(ROOT / "src" / "llm"))` |
| `llm/archive/bio_converter.py:200` | `sys.path.insert(0, str(Path(__file__).parent))` |
| `llm/start_session.py:19` | `sys.path.insert(0, str(PROJECT_ROOT))` |
| `llm/vllm_server.py:12` | `sys.path.insert(0, str(PROJECT_ROOT))` |
| `rag/batch_rag_match.py:68` | `sys.path.insert(0, str(PROJECT_ROOT))` |
| `tests/parse_job_desc.py:32` | `sys.path.insert(0, _PROJECT_ROOT)` |

替代方式：

```powershell
# 从仓库根目录运行
python -m src.rag.cli query --help
python -m src.skill_extraction.pipeline_v2 prepare
```

包内代码使用相对导入：

```python
# 不推荐
from src.rag.config import RAGConfig

# 推荐，在 src/rag 内部
from .config import RAGConfig
```

### 7.3 星号导入

`skill_extraction/llm_labeling_utils.py:7` 使用 `from ..utils.llm_labeling_utils import *`。建议改为显式导入：

```python
from ..utils.llm_labeling_utils import run_openai_prompt_pairs, safe_text

__all__ = ["run_openai_prompt_pairs", "safe_text"]
```

### 7.4 循环引用风险

检测到潜在循环：

```text
skill_extraction.context_classifier
 -> skill_extraction.match_flat_skills_to_duckdb
 -> skill_extraction.regression_eval
 -> skill_extraction.match_flat_skills_to_duckdb
```

建议将 `SkillContextCandidate`、`FlatHardSkillMatcher` 使用的共享数据结构抽离到 `skill_extraction/types.py` 或 `skill_extraction/protocols.py`，运行时导入放在函数内部或 `typing.TYPE_CHECKING` 下。

### 7.5 `__init__.py` 暴露接口

- `job_title_parsing/__init__.py` 与 `skill_extraction/__init__.py` 使用延迟加载，方向合理。
- `rag/__init__.py` 直接导入 `RAGConfig` 和 `OccupationRAG`，可接受，但要确保导入时没有重型模型加载副作用。
- `analysis/__init__.py`、`utils/__init__.py`、`visualization/__init__.py` 只有简短说明，未显式 `__all__`。如果这些包需要对外提供稳定接口，应补充 `__all__`；如果只是内部目录，可以保持空导出。
- 不建议归档目录包含 `__init__.py` 并暴露接口；否则归档代码仍可能被误导入。

## 8. 建议的三阶段治理路线

### 阶段 1：恢复基础健康状态

```powershell
# 修复 bert/llm_annotate.py 后
python -m compileall -q src
Get-ChildItem -Path src -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -Path src -Recurse -File -Filter "*.pyc" | Remove-Item -Force
Copy-Item .gitignore.recommended .gitignore
```

### 阶段 2：建立团队门禁

```powershell
Copy-Item pyproject.toml .\pyproject.toml
Copy-Item .flake8 .\.flake8
pip install black flake8 isort pytest
black --check src tests
flake8 src tests
pytest -q
```

### 阶段 3：结构化重构

1. 先改配置层，消除硬编码路径。
2. 再补主链路 docstring 与类型提示：`job_title_parsing/`, `rag/`, `skill_extraction/`, `analysis/`。
3. 最后处理实验归档、文件重命名和超长模块拆分。

## 9. 附件

- `src_audit_findings.csv`：全量问题清单，可按严重程度、文件、类别过滤。
- `docstring_todo_templates.md`：公开接口 docstring/type-hint 补充模板。
- `CONTRIBUTING.md`：团队协作规范草案。
- `pyproject.toml`：black/isort/pytest/mypy 建议配置。
- `.flake8`：flake8 建议配置。
- `.gitignore.recommended`：推荐忽略规则。
- `archive_plan.ps1`：归档/清理预案，默认预览不执行。
