# src/llm —— 基于 Qwen3-8B 的招聘数据 NER 标注模块

## 模块结构

```
src/llm/
├── ner_schema.py        # NER 实体类型定义 + BIO 标签集
├── prompt_builder.py    # Prompt 构建（zero_shot / few_shot）
├── qwen3_extractor.py   # Qwen3-8B 本地推理封装
├── batch_annotator.py   # 批量标注主程序（支持断点续跑）
├── bio_converter.py     # BIO 格式转换 + 质量验证 + 数据集划分
├── test_pipeline.py     # 基础逻辑测试（无需 GPU）
└── __init__.py
```

---

## 提取的实体类型

> 以下字段已在原始数据集中存在，本模块**不重复提取**：
> 发布时间、岗位名称、工作城市、薪资水平、经验要求、学历要求、岗位描述、公司名称、公司规模、公司行业

| 实体类型 | BIO 前缀 | 说明 | 示例 | 研究意义 |
|---|---|---|---|---|
| SKILL | `SKILL` | 专业能力/业务技能 | 数据分析、财务核算 | 技能-薪资溢价分析 |
| TOOL | `TOOL` | 软件/框架/编程语言 | Python、Tableau、SAP | 行业技术栈演变 |
| CERT | `CERT` | 证书/资质/执照 | CPA、PMP、驾照 | 职业准入门槛研究 |
| BENEFIT | `BNF` | 福利待遇 | 五险一金、带薪年假 | 就业质量评估 |
| DUTY | `DUTY` | 核心工作职责 | 负责产品需求分析 | 岗位职能画像 |
| HEADCOUNT | `HC` | 招聘人数 | 1人、若干 | 劳动力需求规模 |
| JOB_TYPE | `JT` | 工作性质 | 全职、实习 | 非标准就业趋势 |

BIO 标签集（共 15 个）：`O`, `B/I-SKILL`, `B/I-TOOL`, `B/I-CERT`, `B/I-BNF`, `B/I-DUTY`, `B/I-HC`, `B/I-JT`

---

## 为什么选择 LLM 标注？方法对比

| 方法 | 准确率 | 成本 | 适用场景 |
|---|---|---|---|
| 规则/正则 | 低（60-70%） | 极低 | 结构化字段（薪资、学历） |
| LLM 零样本（zero_shot） | 中（70-80%） | 低 | 快速冷启动 |
| **LLM 少样本（few_shot，本方案）** | **中高（80-88%）** | **低** | **大规模自动标注** |
| LLM few_shot + 人工抽检修正 | 高（88-93%） | 中 | 推荐最终方案 |
| 人工标注 + 微调 BERT/RoBERTa | 最高（92-96%） | 高 | 生产 NER 模型 |

**推荐路径**：
```
Qwen3 few_shot 批量标注
    → 人工抽检 5%（Label Studio 导入 ls_jd_tasks.json 的 predictions 字段）
    → 修正后转 BIO 格式
    → 微调 chinese-roberta-wwm-ext NER 模型
    → 全量数据推理
```

---

## 研究意义

1. **劳动经济学**：建立技能-薪资回报模型（哪些工具/证书带来最高溢价）
2. **教育政策**：高校课程与市场需求的匹配度分析（哪些专业技能供不应求）
3. **人力资源**：行业间技能迁移路径（跨行业跳槽可行性评估）
4. **区域经济**：广东省不同城市对技术栈/工具的偏好差异
5. **就业预测**：新兴技能出现频率的时序趋势（AI/大模型相关岗位增长）
6. **福利研究**：非薪资福利指数构建，评估广东省就业质量

---

## 快速开始

### 1. 环境要求

```bash
pip install transformers torch accelerate tqdm
```

模型路径：`D:\model\Qwen3-8B`（已在 `qwen3_extractor.py` 中配置）

### 2. 基础逻辑测试（无需 GPU）

```bash
cd d:/pythonProject/leisure/Employ26
python src/llm/test_pipeline.py
```

### 3. 小批量试跑（建议先跑 10 条验证效果）

```bash
python src/llm/batch_annotator.py \
    --input data/ls_jd_tasks.json \
    --output output/llm_annotations \
    --limit 10 \
    --mode few_shot
```

### 4. 全量标注（支持断点续跑）

```bash
# 全量，开启思维链（更准确，速度较慢）
python src/llm/batch_annotator.py \
    --input data/ls_jd_tasks.json \
    --output output/llm_annotations \
    --mode few_shot \
    --thinking

# 中断后续跑（自动跳过已处理条目）
python src/llm/batch_annotator.py \
    --input data/ls_jd_tasks.json \
    --output output/llm_annotations \
    --mode few_shot
```

### 5. 转换为 NER 训练格式

```bash
python src/llm/bio_converter.py \
    --input output/llm_annotations/ner_bio_samples.jsonl \
    --output output/ner_dataset
```

输出：
```
output/ner_dataset/
├── conll/
│   ├── train.conll   # CoNLL 格式（token\tlabel）
│   ├── dev.conll
│   └── test.conll
└── hf_json/
    ├── train.jsonl   # HuggingFace datasets 格式
    ├── dev.jsonl
    └── test.jsonl
```

---

## 输出文件说明

| 文件 | 格式 | 说明 |
|---|---|---|
| `output/llm_annotations/extracted_fields.jsonl` | JSONL | 每条记录包含 row_id + 7类抽取字段 |
| `output/llm_annotations/ner_bio_samples.jsonl` | JSONL | BIO 格式，含 tokens/labels/row_id |
| `output/ner_dataset/conll/*.conll` | CoNLL | 可直接用于 spaCy、seqeval 训练 |
| `output/ner_dataset/hf_json/*.jsonl` | JSONL | 可直接用于 HuggingFace Trainer |

