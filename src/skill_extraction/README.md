# 技能词典流程

当前 `src/skill_extraction` 目录只保留和“职业细类技能词典”直接相关的脚本，不再保留旧的 Word2Vec / BERT / 实验型技能抽取脚本。

## 目录职责

- `config.py`
  读取 `config/database.yaml`，统一 DuckDB、BGE 模型、输出目录、词典目录配置。
- `bge_matcher.py`
  使用 `D:\model\bge-base-zh-finetuned` 做职业细类匹配，返回 Top5 候选，并按 Top1 到 Top5 依次回退选择首个可用细类。
- `data_source.py`
  从 DuckDB 读取招聘样本，匹配职业细类，生成训练清单和验证池。
- `dictionary_store.py`
  统一管理 `dicts/occupation_skill_dictionary.json`。
- `import_llm_results.py`
  把 LLM 输出的 `JSON / Markdown / 文本` 结果自动解析并合并回职业技能词典。
- `clean_skill_dictionary.py`
  清洗职业技能词典，移除软技能、学历门槛、福利待遇、工作时间等非硬技能噪音，并输出审计报告。
- `init_llm_output_layout.py`
  初始化 `output/skill_extraction/llm_outputs` 目录规范模板。
- `coverage.py`
  评估职业细类技能词典对验证集“任职要求”条目的覆盖率。
- `occupation_skill_pipeline.py`
  主 CLI。负责抽样、切割任职要求、生成 LLM prompt、覆盖率验证与迭代补词。

## 流程说明

1. 每个职业细类默认抽取 100 条训练样本。
2. 训练样本通过 `src.preprocessing.parse_desc.parse_desc_df` 切出“任职要求”，减少发给 LLM 的 token。
3. LLM 根据每个细类的训练 prompt 输出技能词典，并写回 `dicts/occupation_skill_dictionary.json`。
4. 每轮从同一细类的验证池中再随机抽 10 条作为验证集。
5. 如果某细类覆盖率低于 95%，自动生成补词 prompt，补词后继续下一轮验证。

## 使用方式

### 1. 生成训练样本与训练 prompt

```bash
python -m src.skill_extraction.occupation_skill_pipeline prepare ^
  --train-size 100 ^
  --validation-batch-size 10 ^
  --parse-workers 1
```

输出：

- `output/skill_extraction/occupation_skill_training_manifest.csv`
- `output/skill_extraction/occupation_skill_training_requirements.csv`
- `output/skill_extraction/occupation_skill_validation_pool.csv`
- `output/skill_extraction/occupation_skill_category_summary.csv`
- `output/skill_extraction/prompts/train/*.md`
- `dicts/occupation_skill_dictionary.json`

### 2. 执行一轮覆盖率验证

```bash
python -m src.skill_extraction.occupation_skill_pipeline iterate ^
  --validation-batch-size 10 ^
  --coverage-threshold 0.95 ^
  --parse-workers 1
```

输出：

- `output/skill_extraction/reports/round_XX/validation_samples.csv`
- `output/skill_extraction/reports/round_XX/coverage_summary.csv`
- `output/skill_extraction/reports/round_XX/coverage_items.csv`
- `output/skill_extraction/reports/round_XX/uncovered_items.csv`
- `output/skill_extraction/prompts/supplement/round_XX/*.md`

### 3. 导入 LLM 返回的 JSON

```bash
python -m src.skill_extraction.import_llm_results ^
  --input output/skill_extraction/llm_outputs
```

### 4. 初始化 LLM 输出目录模板

```bash
python -m src.skill_extraction.init_llm_output_layout
```

默认会生成：

- `output/skill_extraction/llm_outputs/README.md`
- `output/skill_extraction/llm_outputs/train/round_00/raw/`
- `output/skill_extraction/llm_outputs/train/round_00/json/`
- `output/skill_extraction/llm_outputs/train/round_00/imported/`
- `output/skill_extraction/llm_outputs/supplement/round_01/raw/`
- `output/skill_extraction/llm_outputs/supplement/round_01/json/`
- `output/skill_extraction/llm_outputs/supplement/round_01/imported/`
- `output/skill_extraction/llm_outputs/examples/*.json`

支持：

- 单个 `.json`
- 单个 `.md`
- 单个 `.txt`
- 一个目录下的上述文件，默认递归扫描

如果只想先看导入结果，不写回词典：

```bash
python -m src.skill_extraction.import_llm_results ^
  --input output/skill_extraction/llm_outputs ^
  --dry-run
```

### 5. 查看当前状态

```bash
python -m src.skill_extraction.occupation_skill_pipeline status
```

### 6. 清洗技能词典中的软技能与招聘噪音

```bash
python -m src.skill_extraction.clean_skill_dictionary
```

默认输出：

- `dicts/occupation_skill_dictionary.cleaned.json`
- `output/skill_extraction/reports/dictionary_cleaning/<timestamp>/cleaning_summary.json`
- `output/skill_extraction/reports/dictionary_cleaning/<timestamp>/cleaning_details.csv`

如果只想先看清洗结果，不写新词典：

```bash
python -m src.skill_extraction.clean_skill_dictionary --dry-run
```

如果确认结果后要直接覆盖原词典：

```bash
python -m src.skill_extraction.clean_skill_dictionary --in-place
```

覆盖模式会自动先备份原始词典，再写回清洗后的结果。

## 词典格式

词典统一保存在：

- `dicts/occupation_skill_dictionary.json`

示例：

```json
{
  "metadata": {
    "schema_version": 1,
    "description": "按职业细类维护的技能词典"
  },
  "categories": {
    "信息传输、软件和信息技术服务人员 > 软件和信息技术服务人员 > 软件开发人员 > 后端开发人员": {
      "detail_name": "后端开发人员",
      "hierarchy": {
        "大类": "信息传输、软件和信息技术服务人员",
        "中类": "软件和信息技术服务人员",
        "小类": "软件开发人员",
        "细类": "后端开发人员"
      },
      "skills": [
        {
          "name": "Python",
          "aliases": ["python"],
          "skill_type": "编程语言",
          "notes": ""
        }
      ]
    }
  }
}
```

## 约束

- 职业技能词典必须保存在 `dicts/`。
- 训练 prompt 只使用“任职要求”或其回退文本，避免把整段岗位描述直接送入 LLM。
- 职业细类匹配统一使用 `D:\model\bge-base-zh-finetuned`。
- 覆盖率默认目标为 95%。
- 某职业细类未达标时，继续迭代验证与补词。


请你读取@src\skill_extraction
1，请生成一个脚本，读取技能词典@dicts\flat_skill_dictionary.json
对数据库recruit.main.skill_extraction_requirement_matches的“任职要求_items_text”进行正则匹配，若字段为空，则选用“岗位职责_items_text”或者“岗位描述_清洗”。
注意，词典中的aliases也要参与匹配
2，生成的“skill_name”保存在recruit.main.hard_skill_match_results_dev
3，请添加验证逻辑，使用本地Qwen3进行验证生成的skill_name。若缺少相应的skill或者匹配不正确的skill或者因为aliases不正确匹配了错误的skill，请及时更改词典
4，请勿重复生成函数，请检查该项目是否已有相同功能的函数
5，规范注释



请你读取recruit.main.hard_skill_match_results_dev的前十行数据
1，评判skill_name的提取准确率和覆盖率，并且总结为什么会不足或者错误
2，根据你的总结修改匹配脚本@src\skill_extraction\match_flat_skills_to_duckdb.py
和技能词典生成脚本@src\skill_extraction\occupation_skill_pipeline.py
3，告诉我有没有更好的方法，提取出硬技能，兼顾准确率和效率
4，规范注释
