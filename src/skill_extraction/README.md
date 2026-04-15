# Skill Extraction

`src/skill_extraction` 当前分成两套入口：

- `v1`
  历史版，按职业细类分层维护技能词典，相关模块已收拢到 `history/`
- `v2`
  当前主流程，直接构造平面化职业硬技能词典，并支持自动评测、自动标注和二阶段过滤

## 目录说明

- [pipeline_v1.py](/d:/PythonProjects/Employ26/src/skill_extraction/pipeline_v1.py:1)
  `v1` 入口
- [pipeline_v2.py](/d:/PythonProjects/Employ26/src/skill_extraction/pipeline_v2.py:1)
  `v2` 平面词典构造入口
- [occupation_skill_pipeline.py](/d:/PythonProjects/Employ26/src/skill_extraction/occupation_skill_pipeline.py:1)
  `v2` 主实现
- [match_flat_skills_to_duckdb.py](/d:/PythonProjects/Employ26/src/skill_extraction/match_flat_skills_to_duckdb.py:1)
  平面词典匹配、LLM 抽检、上下文判别接入
- [regression_eval.py](/d:/PythonProjects/Employ26/src/skill_extraction/regression_eval.py:1)
  回归评测
- [llm_label_regression_dataset.py](/d:/PythonProjects/Employ26/src/skill_extraction/llm_label_regression_dataset.py:1)
  自动生成回归集
- [llm_label_context_dataset.py](/d:/PythonProjects/Employ26/src/skill_extraction/llm_label_context_dataset.py:1)
  自动生成上下文判别训练集
- [context_classifier.py](/d:/PythonProjects/Employ26/src/skill_extraction/context_classifier.py:1)
  多分类上下文判别器
- [DESIGN_v2.md](/d:/PythonProjects/Employ26/src/skill_extraction/DESIGN_v2.md:1)
  详细设计文档

## v1 运行方式

```bash
python -m src.skill_extraction.pipeline_v1 prepare
python -m src.skill_extraction.pipeline_v1 iterate
python -m src.skill_extraction.pipeline_v1 status
```

`v1` 适合需要保留“职业细类 -> 技能词典”分层结构的场景。

## v2 运行流程

### 1. 构造平面词典

```bash
python -m src.skill_extraction.pipeline_v2
```

### 2. 自动生成回归集

```bash
python -m src.skill_extraction.llm_label_regression_dataset ^
  --sample-size 400 ^
  --num-votes 3
```

### 3. 自动生成上下文训练集

```bash
python -m src.skill_extraction.llm_label_context_dataset ^
  --regression-dataset output/skill_extraction/regression/flat_skill_regression_dataset.jsonl ^
  --dictionary dicts/flat_skill_dictionary.json ^
  --num-votes 3
```

### 4. 训练上下文判别器

```bash
python -m src.skill_extraction.context_classifier train ^
  --dataset output/skill_extraction/context_classifier/context_dataset_llm.jsonl ^
  --output-dir output/skill_extraction/context_classifier/model
```

### 5. 运行回归评测

```bash
python -m src.skill_extraction.regression_eval ^
  --dataset output/skill_extraction/regression/flat_skill_regression_dataset.jsonl ^
  --dictionary dicts/flat_skill_dictionary.json ^
  --fail-under-precision 0.90 ^
  --fail-under-f1 0.80
```

### 6. 启用二阶段过滤做匹配

```bash
python -m src.skill_extraction.match_flat_skills_to_duckdb match ^
  --dictionary dicts/flat_skill_dictionary.json ^
  --context-classifier-model output/skill_extraction/context_classifier/model ^
  --context-threshold 0.80
```

## 当前 v2 的原理

### 词典构造

1. 读取 DuckDB 中已完成职业匹配的岗位文本
2. 按职业中类采样
3. 用本地 LLM 抽取硬技能并构造平面词典

### 词典匹配

1. 第一阶段做高速词典召回
2. 第二阶段用多分类上下文判别器过滤误报
3. 最终把保留下来的技能写回 DuckDB

### 自动训练数据

1. 用 LLM 从 JD 自动生成 `gold_skills`
2. 用 LLM 对词典召回候选打多分类标签
3. 用这些自动标注数据训练上下文判别器

## 标签定义

- `valid_hard_skill`
  候选在当前文本中是有效硬技能
- `too_generic`
  候选过于泛化，不适合作为技能词典项
- `wrong_alias_mapping`
  alias 命中了文本，但语义不属于当前技能
- `not_skill`
  命中内容不是硬技能

## 配置来源

统一读取：

- [config/database.yaml](/d:/PythonProjects/Employ26/config/database.yaml:1)

当前关键字段：

```yaml
LLM_model_path: D:/model/Qwen3-8B
BERT_path: D:\model\chinese-roberta-wwm-ext
```

## 说明

- `v1` 的相关功能模块已经整体下沉到 `history/`
- `v2` 的词典构造主逻辑保持不变，这次新增的是自动标注、回归评测和多分类上下文过滤链路
- 更完整的设计说明见 [DESIGN_v2.md](/d:/PythonProjects/Employ26/src/skill_extraction/DESIGN_v2.md:1)
