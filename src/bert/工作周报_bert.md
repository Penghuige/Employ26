# `bert` 模块工作周报

## 1. 周报说明

本文档面向 `src/bert` 目录，按照时间顺序整理该模块的研发变更、阶段性交付结果、技术方案演进与当前状态，并补充该模块实际使用到的 Prompt 信息。

需要特别说明的是：

- 当前 `src/bert` 目录在仓库中处于 **未跟踪（untracked）** 状态，无法直接通过 Git 提交历史逐次还原 commit。
- 因此，本周报的时间线依据以下信息重建：
  - 文件系统 `LastWriteTime`
  - 脚本职责与调用关系
  - 输入输出路径与模型产物
  - 训练流程、标注流程和 CLI 设计
- 本报告更适合作为“阶段研发周报 / 里程碑周报”，重点反映该模块从分类训练、规则抽取、LLM 标注到 BERT-NER 训练与推理工具建设的过程。

---

## 2. 模块总体目标

`src/bert` 模块的核心目标，是围绕招聘 JD 构建两类能力：

- **职业类别分类**
  - 将岗位标题 + JD 摘要映射到职业类别
- **命名实体识别（NER）**
  - 从招聘 JD 中提取职位、技能、证书、语言、学历、经验、专业、领域等实体

从当前代码形态看，这个目录并不是单一训练脚本，而是覆盖了以下几种能力路径：

- 基于 Label Studio 标注数据的 BERT 分类训练
- 基于规则词典的职位/技能抽取
- 基于本地 Ollama 大模型的自动 NER 标注
- 基于 BIO 标注的 BERT-NER 训练与推理
- 面向文本文件 / JSON 文件 / 交互式命令行的批量预测

也就是说，`bert` 模块既承担了“模型训练”的职责，也承担了“标注数据生产”和“下游推理工具”的职责。

---

## 3. 本期研发总览

从时间上看，`bert` 模块经历了三个明显阶段：

### 阶段一：建立分类训练与基础抽取工具（2026-03-17）

最早出现的脚本包括：

- `run_bert_training.py`
- `train_bert.py`
- `ner_extract.py`
- `llm_annotate.py`

这说明项目在初始阶段同时推进了两条线：

- 一条是职业类别分类训练链路
- 一条是实体抽取与自动标注链路

### 阶段二：补齐 BERT-NER 训练与推理闭环（2026-03-24）

新增：

- `bert_ner.py`

这标志着 NER 能力从规则抽取和 LLM 标注，进一步升级为：

- 自动构造 BIO 标注数据
- BERT Token Classification 训练
- 实体级评估
- 推理器封装

### 阶段三：拆分独立的数据准备与预测入口（2026-04-11）

新增：

- `prepare_data.py`
- `ner_predict.py`

这说明系统开始从“功能脚本”向“分层可复用工具”演进：

- 分类任务的数据准备独立化
- NER 预测脚本独立化
- 训练、推理、标注三条流程的边界更清晰

---

## 4. 按时间顺序的详细变更周报

## 2026-03-17：完成职业分类训练链路和基础 NER 工具原型

### 变更文件

- `run_bert_training.py`
- `train_bert.py`
- `ner_extract.py`
- `llm_annotate.py`

### 主要工作内容

#### 1. 建立一键式 BERT 职业类别分类训练流程

`run_bert_training.py` 作为总入口，串起了：

- 数据准备
- BERT 微调训练
- 结果展示

脚本会：

- 调用 `prepare_data.main()`
- 调用 `train_bert.main()`
- 从 DuckDB 中读取训练指标与测试预测样例

这说明一开始就考虑了“不要手动分步执行多个脚本”，而是希望形成一个完整的训练入口。

#### 2. 完成 BERT 职业类别分类训练主逻辑

`train_bert.py` 构建了完整的文本分类训练链路，主要能力包括：

- 从 DuckDB 中读取：
  - `train_set`
  - `val_set`
  - `test_set`
  - `label_map`
- 使用 `BertForSequenceClassification` 做职业类别分类
- 使用 `BertTokenizerFast` 统一文本编码
- 支持：
  - AdamW 优化器
  - 线性 warmup 学习率调度
  - Early Stopping
  - val 集 macro-F1 作为最佳模型选择依据
- 将训练指标写回 `train_metrics`
- 将测试预测写回 `predictions`
- 落盘：
  - `output/models/bert_occ_category/`
  - `train_config.json`

这条链路比较完整，已经具备基础实验管理能力，而不是只输出终端日志。

#### 3. 构建规则式 NER 提取器

`ner_extract.py` 提供了一个轻量、快速、可直接上线试用的规则抽取方案，主要包括：

- 从 `ls_jd_tasks.json` 中构建职位词典和技能词典
- 对技能做分隔符切分、黑名单过滤和频次筛选
- 使用最长匹配提取职位
- 使用不重叠匹配提取技能
- 使用正则补充英文技术栈，例如：
  - Python
  - Java
  - MySQL
  - Docker
  - Kubernetes
  - BERT
  - GPT
- 输出批量抽取结果到 `output/ner_results.csv`

这一方案虽然不是深度模型，但优点很明显：

- 快
- 易解释
- 无训练成本
- 适合作为基线方案或弱监督辅助工具

#### 4. 引入基于本地 Ollama 的自动 NER 标注器

`llm_annotate.py` 是 `bert` 目录中非常关键的一个脚本，它负责用本地大模型对招聘 JD 做自动实体标注。主要能力包括：

- 对接本地 Ollama 服务
- 支持模型选择：
  - `qwen2.5:7b`
  - `qwen2.5:14b`
  - `deepseek-r1:7b`
  - `qwen3:8b`
- 使用 `SYSTEM_PROMPT + USER_TEMPLATE` 构建 NER 标注 prompt
- 支持断点续传，已完成 `row_id` 自动跳过
- 支持三类输出：
  - JSONL
  - CoNLL
  - 人工审核 CSV
- 对模型输出做 JSON 容错提取
- 对实体做合法性校验：
  - type 必须合法
  - text 必须真实出现在原文中
  - 去重

### 技术价值

- 完成了职业分类和实体识别两条主线的第一版工具链建设
- 不仅能训练模型，也能生产标注数据
- 把本地 LLM 纳入标注流程，为后续 BERT-NER 训练提供自动化数据来源

---

## 2026-03-24：完成 BERT-NER 训练与推理闭环

### 变更文件

- `bert_ner.py`

### 主要工作内容

`bert_ner.py` 是整个 `bert` 目录里最完整的 NER 训练与推理脚本，它把前期的标注/规则能力整合成了标准的 BERT-NER 闭环。

主要能力包括：

#### 1. 自动生成字符级 BIO 标注

脚本中的 `auto_bio_label()` 和 `build_bio_dataset()` 负责：

- 根据现有实体信息自动生成字符级 BIO 序列
- 输出样本结构：
  - `text`
  - `chars`
  - `labels`
- 自动过滤没有实体的样本
- 支持缓存到：
  - `output/ner_data/bio_samples.json`

这一步的价值在于：把 LLM 或人工标注结果转成 BERT 可训练格式，完成了监督学习数据桥接。

#### 2. 处理 sub-word 到字符级标签对齐

`NERDataset` 负责：

- 调用 `BertTokenizerFast`
- 利用 `word_ids()` 将 token 映射回字符位置
- 将字符级标签对齐为 token 级标签
- 对特殊 token 与无效位置填充 `-100`

这是中文 BERT-NER 训练中最关键的工程细节之一，决定了 BIO 标注能否正确进入损失函数。

#### 3. 训练 Token Classification 模型

`train_ner()` 实现了标准的 NER 训练流程：

- 加载 BIO 缓存或重新构建样本
- 将样本划分为 train/val/test
- 使用 `BertForTokenClassification`
- 使用 `seqeval` 做实体级评估
- 使用 val F1 保存最佳模型
- 将模型保存到：
  - `output/models/bert_ner`

这意味着 NER 模块已经从“规则抽取”正式升级为“监督式训练模型”。

#### 4. 提供统一的 NER 推理器

`NERPredictor` 封装了：

- 模型加载
- 文本清洗
- tokenize
- BIO 解码
- 实体 span 还原

同时支持：

- 单条预测
- 批量预测
- 演示模式
- `--predict`
- `--train`
- `--check-data`

### 技术价值

- 打通了“自动标注 -> BIO 数据 -> BERT-NER 训练 -> 实体预测”的完整闭环
- 技术路线较标准，后续便于继续引入更高质量标注数据
- 使 NER 部分不再依赖单一规则或单一大模型

---

## 2026-04-11：拆分数据准备与独立预测入口，提升工程可复用性

### 变更文件

- `prepare_data.py`
- `ner_predict.py`

### 主要工作内容

#### 1. 数据准备脚本独立化

`prepare_data.py` 把分类任务的数据准备从训练脚本中显式拆出，主要包括：

- 读取 `data/ls_jd_tasks.json`
- 提取训练字段：
  - `row_id`
  - `job_title`
  - `clean_title`
  - `jd_snippet`
  - `occ_category`
  - `occ_core`
  - `ai_level`
  - `ai_edu`
  - `ai_exp`
  - `hard_skills`
  - `ner_status`
- 组装训练文本：
  - `clean_title + [SEP] + jd_snippet`
- 过滤空标签和极低频类别
- 使用 `LabelEncoder` 生成标签映射
- 划分 `train/val/test`
- 写入 DuckDB：
  - `jd_raw`
  - `train_set`
  - `val_set`
  - `test_set`
  - `label_map`
  - `train_metrics`
  - `predictions`

这一步让分类训练链路边界更清晰，也更适合后续复用。

#### 2. NER 预测工具独立化

`ner_predict.py` 则将 NER 推理功能单独拆出，主要包括：

- 加载训练好的 `output/models/bert_ner`
- 提供统一的 `NERPredictor`
- 支持：
  - 单文本预测
  - 文本文件逐行预测
  - JSON 文件批量预测
  - 交互式命令行
- 输出结果到 CSV

这说明项目已经开始区分：

- 训练脚本
- 数据准备脚本
- 纯推理脚本

这是工程成熟度提升的重要信号。

### 阶段产出评价

截至 2026-04-11，`bert` 模块已经形成较完整的“分类 + NER + 自动标注 + 推理工具”组合，既可用于实验，也具备一定的离线生产能力。

---

## 5. 当前模块架构总结

从当前目录结构看，`bert` 模块可以抽象为以下五层：

### 1. 数据准备层

- `prepare_data.py`

职责是从标注 JSON 中整理分类任务训练数据，写入 DuckDB，并维护训练数据集和标签映射。

### 2. 分类训练层

- `run_bert_training.py`
- `train_bert.py`

职责是完成职业类别分类模型的训练、评估、保存和结果落库。

### 3. 自动标注层

- `llm_annotate.py`

职责是通过本地 Ollama 大模型生成 NER 标注结果，并导出为 JSONL / CoNLL / CSV。

### 4. 规则与监督式 NER 层

- `ner_extract.py`
- `bert_ner.py`

职责是分别提供：

- 规则基线抽取能力
- BERT-NER 训练和实体识别能力

### 5. 预测交付层

- `ner_predict.py`

职责是面向文本文件、JSON 文件和交互式使用场景，提供可直接运行的实体抽取工具。

---

## 6. 本期关键技术亮点

### 1. 同时覆盖分类与 NER 两条能力线

很多项目只做一种任务，而当前 `bert` 模块同时建设了：

- 职业类别分类
- 招聘文本 NER

这意味着它既能做岗位整体归类，也能做局部实体提取，具备较好的下游可扩展性。

### 2. 自动标注和监督训练衔接较自然

`llm_annotate.py` 负责把 LLM 输出转成结构化实体；
`bert_ner.py` 负责把结构化实体转成 BIO 标注并训练 BERT。

这条链路说明项目已经具备“LLM 生产弱监督数据，再用传统模型吸收知识”的能力。

### 3. DuckDB 作为训练中间层使用得比较合理

在分类任务中，DuckDB 被用于沉淀：

- 原始解析数据
- 数据集划分结果
- 标签映射
- 训练指标
- 测试预测

这比纯 CSV 散落式管理更适合后续追踪实验结果。

### 4. NER 工具形态比较完整

当前 NER 相关能力不只是一个训练脚本，而是同时具备：

- 自动标注
- BIO 转换
- 模型训练
- 文件批量预测
- JSON 批量预测
- 交互式预测

这让模块更像一个“可交付工具箱”，而不只是实验代码。

---

## 7. 当前风险与待改进点

### 1. 目录未纳入 Git 跟踪

当前 `src/bert` 是未跟踪目录，会导致：

- 无法精确还原每次改动
- 无法追踪 prompt 演进
- 无法绑定训练结果与代码版本

建议尽快纳入版本控制。

### 2. 分类与 NER 两条链路耦合在同一目录中

当前目录内同时存在：

- 分类训练
- LLM 标注
- 规则抽取
- BERT-NER

虽然功能丰富，但也带来一定复杂度：

- 模块边界不够清晰
- 维护时容易混淆输入输出依赖

建议未来按任务类型进一步拆分子目录。

### 3. LLM 标注依赖本地 Ollama 服务

`llm_annotate.py` 对本地环境依赖较强，包括：

- Ollama 服务启动状态
- 本地模型是否已拉取
- 推理超时与服务稳定性

这使它更适合个人离线标注环境，不太适合直接迁移到无状态环境。

### 4. 自动标注质量仍需人工复核机制补强

虽然 `llm_annotate.py` 已支持导出 review CSV 和 CoNLL，但从当前目录结构看，还缺少更系统化的：

- 标注质量抽样规范
- 人工修正回流流程
- 版本化 gold 数据集

这一点未来仍有提升空间。

### 5. 分类和 NER 的评估口径尚未统一

当前：

- 分类任务有 DuckDB 训练指标和测试预测
- NER 任务有 `seqeval` 和导出样本

但两者尚未形成统一的实验记录和版本报告方式。若后续长期维护，建议建立统一实验台账。

---

## 8. 脚本 Prompt 附录

## 8.1 Prompt 使用情况结论

与 `job_title_parsing` 不同，`bert` 目录 **确实存在实际的大模型 Prompt 调用链路**，主要集中在：

- `src/bert/llm_annotate.py`

其调用方式为：

- 向本地 Ollama `/api/chat` 发送消息
- 使用：
  - `SYSTEM_PROMPT`
  - `USER_TEMPLATE`

也就是说：

- `bert` 既是 Prompt 的定义方，也是调用方

---

## 8.2 System Prompt

`llm_annotate.py` 中定义的系统提示词核心内容如下：

```text
你是专业的招聘信息NER标注专家。从招聘JD中识别以下实体类型：

TITLE  : 职位核心词
SKILL  : 硬技能/工具/框架
CERT   : 证书/资质
LANG   : 语言能力
EDU    : 学历要求
EXP    : 工作经验要求
MAJOR  : 专业要求
DOMAIN : 行业/领域知识

【必须排除，不标注】
软素质、泛动作、福利、条件、特质

【输出要求】
- 只输出纯JSON，不要任何解释
- text必须是原文中的完整字符串
- entities按原文出现顺序排列
```

### 设计特点

- 明确给出 8 类实体定义
- 明确列出排除项，避免模型把软素质和福利误标成实体
- 强调必须输出纯 JSON
- 强调实体 text 必须来自原文，降低幻觉风险

---

## 8.3 User Prompt 模板

`llm_annotate.py` 中的用户模板如下：

```text
职位名称：{job_title}
JD内容：{jd_text}

输出格式：
{"entities": [{"text": "原文字符串", "type": "实体类型"}]}
```

### 设计特点

- 输入结构非常简洁
- 重点让模型围绕“职位名称 + JD内容”做标注
- 输出结构足够简单，便于后处理和容错解析

---

## 8.4 Prompt 调用方式

`llm_annotate.py` 中调用 Ollama 的请求体结构如下：

```json
{
  "model": "qwen2.5:7b",
  "messages": [
    {"role": "system", "content": "SYSTEM_PROMPT"},
    {"role": "user", "content": "USER_TEMPLATE.format(...)"} 
  ],
  "stream": false,
  "options": {
    "temperature": 0.1,
    "top_p": 0.9,
    "num_predict": 1024
  }
}
```

### 设计特点

- `temperature=0.1` 说明该任务偏向稳定抽取，而不是创造性生成
- 使用 `messages` 结构，和标准对话式大模型接口一致
- `num_predict=1024` 为复杂 JD 留出较充足输出空间

---

## 8.5 Prompt 后处理与安全约束

除了 prompt 本身，脚本还做了若干重要的后处理约束：

- `_parse(...)`
  - 支持直接 JSON 解析
  - 支持从 markdown 代码块提取 JSON
  - 支持从原始文本中提取最外层 `{...}`
- `_validate(...)`
  - 校验 type 是否在 `VALID_TYPES`
  - 校验实体 text 是否真实出现在原文
  - 去重
- `entities_to_bio(...)`
  - 将实体结果转换为字符级 BIO，用于训练数据生产

这说明当前 Prompt 设计并不是孤立的，而是与完整的数据清洗、解析、验证和训练链路配套的。

---

## 9. 结论

综合来看，`src/bert` 模块在本期内完成了从“职业类别分类训练脚手架”到“LLM 自动标注 + BERT-NER 训练 + 独立预测工具”的关键扩展，已经形成一个覆盖分类与实体识别两条主线的训练与推理工具集。

如果按研发成熟度划分：

- 2026-03-17：处于分类训练与自动标注原型建设期
- 2026-03-24：进入 BERT-NER 闭环建设期
- 2026-04-11：进入数据准备与预测入口独立化阶段

从当前代码形态判断，后续最值得继续推进的方向包括：

- 将目录与模型结果纳入 Git 版本管理
- 建立统一的分类 / NER 实验台账
- 增强自动标注后的人工复核闭环
- 把规则抽取、LLM 标注、BERT-NER 的结果做更系统的对比评估
- 继续沉淀更高质量的 gold NER 数据集

如果你需要，我还可以继续补一版：

- `src/bert` 的管理汇报精简版
- 将 `bert + bge + job_title_parsing` 合并成统一总周报
- 按“数据、模型、评估、风险”四个维度重组为更偏汇报风格的版本

