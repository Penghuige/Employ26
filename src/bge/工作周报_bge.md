# `bge` 模块工作周报

## 1. 周报说明

本文档面向 `src/bge` 目录，按照时间顺序整理该模块的研发变更、阶段性交付结果、技术方案演进与当前状态，并补充该模块实际使用到的 Prompt 信息。

需要特别说明的是：

- 当前 `src/bge` 目录在仓库中处于 **未跟踪（untracked）** 状态，无法直接通过 Git 提交历史逐次还原 commit。
- 因此，本周报的时间线依据以下信息重建：
  - 文件系统 `LastWriteTime`
  - 脚本职责与调用关系
  - 中间产物与输出文件时间
  - 文档模板与自动化流水线结构
- 本报告更适合作为“阶段研发周报 / 里程碑周报”，重点反映该模块从数据清洗、规则过滤、BGE 微调、Qwen 质检到人工复核与自动迭代闭环的建设过程。

---

## 2. 模块总体目标

`src/bge` 模块的核心目标，是围绕“中国职业大典”构建一个 **招聘岗位到职业编码的分层匹配系统**，并通过 BGE 向量检索、Qwen3 质检与人工标注回流，逐步提升匹配精度和覆盖率。

从当前代码形态看，该模块不是单一脚本，而是一条比较完整的“数据生产线”，大体分为以下阶段：

- Step 1：招聘样本清洗与严格去重
- Step 2：Tier1 规则匹配 + RAG 质检
- Step 3：基于高置信样本做 BGE 微调
- Step 4：对 Pending 样本做 Tier2 向量检索与阈值分流
- Step 5：用 Qwen3 + RAG 对边界样本自动标注
- Step 6：基于自动标注结果评估阈值
- Step 7：形成 D4 -> D5 -> D3 -> D6 的自动迭代闭环

此外，还补齐了：

- Label Studio 导出
- JSON 分片
- 标注文本格式化
- 人工标注模板
- 阈值评估模板

---

## 3. 本期研发总览

从时间上看，`bge` 模块经历了四个明显阶段：

### 阶段一：基础数据准备与初版向量训练（2026-03-27）

最早出现的脚本是：

- `D1_quchong.py`
- `D3_finetune.py`

说明项目初期的重点是：

- 先把招聘原始数据做严格去重
- 再尽快构建一版可用的领域向量模型

这是一种典型的“先把训练样本质量拉起来，再提升检索器效果”的建设路径。

### 阶段二：引入 Qwen 自动质检与困难样本反馈（2026-04-01）

新增：

- `D5_qwen3_auto_label.py`

这标志着系统开始从单纯的向量匹配，升级为：

- BGE 检索负责召回
- Qwen3 负责对边界样本进行语义质检和纠错标注
- 自动标注结果进一步反哺微调训练

### 阶段三：补齐人工复核与标注平台链路（2026-04-02 ~ 2026-04-03）

新增：

- `slice_label_studio_json.py`
- `format_label_studio_requirements.py`
- `export_tier1_label_studio.py`
- `label_studio_softskill_template.xml`
- `docs/manual_label_template.txt`
- `docs/threshold_evaluation_template.txt`

这一阶段说明项目开始系统性考虑：

- 如何做人工抽检
- 如何将候选结果导出到标注平台
- 如何做大文件分片
- 如何规范人工标注字段
- 如何沉淀阈值实验记录

### 阶段四：完成 Tier2 检索、阈值评估与自动迭代闭环（2026-04-11）

集中新增：

- `D2_filter.py`
- `D4_T2match.py`
- `D6_threshold_eval.py`
- `D7_iterate_pipeline.py`

这意味着整个系统在 4 月 11 日形成了真正意义上的生产化闭环：

- Tier1 规则 + RAG 质检
- Tier2 向量检索 + 阈值决策
- Qwen 自动标注
- BGE 困难样本增量微调
- 阈值实验评估
- 自动轮次迭代

---

## 4. 按时间顺序的详细变更周报

## 2026-03-27：完成数据去重与基础微调训练框架

### 变更文件

- `D1_quchong.py`
- `D3_finetune.py`

### 主要工作内容

#### 1. 构建严格去重流程

`D1_quchong.py` 完成了招聘数据进入 BGE 链路前的基础样本治理，核心能力包括：

- 从 `config/database.yaml` 读取 DuckDB 路径与 `jobs_table`
- 合并多个招聘来源表
- 对岗位描述进行 HTML 剥离和全空白清洗
- 构建 `dedup_desc_fingerprint`
- 按以下维度执行联合去重：
  - 岗位名称
  - 工作城市
  - 岗位描述指纹
  - 公司名称
  - 公司行业
  - `YearMonth`
- 保留同月内最早发布记录
- 输出严格去重后的训练/检索基准数据

这一步的意义在于：向量模型对重复样本非常敏感，如果不做前置清洗，很容易出现热门岗位过度主导训练分布的问题。

#### 2. 建立领域微调训练脚手架

`D3_finetune.py` 则完成了 BGE 领域适配的初版训练框架，主要包括：

- 读取 `Tier1_Matched_Data.csv` 作为高置信监督样本
- 读取《中国职业大典》标准文件构造正样本文本
- 清洗岗位描述，剥离薪资、福利、HTML、无关符号
- 构造 `InputExample` 训练样本
- 执行类别均衡化降采样，缓解头部职业过拟合
- 通过 pseudo anchor 机制补齐长尾职业
- 加载本地 `bge-base-zh-v1.5`
- 输出微调后的 `bge-base-zh-finetuned`

### 技术价值

- 打通了“原始数据 -> 训练样本 -> 向量微调模型”的最小闭环
- 为后续所有 RAG 检索和 Tier2 向量匹配提供了领域化模型基础
- 体现出较强的数据工程意识，而不是直接拿通用 embedding 模型生硬应用

---

## 2026-04-01：引入 Qwen3 自动标注，建立语义质检反馈机制

### 变更文件

- `D5_qwen3_auto_label.py`

### 主要工作内容

`D5_qwen3_auto_label.py` 是整个 `bge` 模块中的关键升级点，它把系统从“单纯向量检索”推进到了“检索 + 生成式质检”的协同模式。

核心能力包括：

- 读取不同阶段输出：
  - `tier1`
  - `tier2`
  - `tier3`
- 统一清洗岗位描述，防止 prompt 过长
- 使用微调后的 BGE 检索模型做 TopK 候选召回
- 将岗位信息、系统预测结果和候选职业上下文拼接成 prompt
- 批量调用本地 Qwen3-8B
- 对模型输出 JSON 做鲁棒解析
- 生成自动标注文件：
  - `qwen3_8b_rag_labels_*.csv`
  - `qwen3_8b_rag_labels_latest.csv`
  - `qwen3_8b_rag_failed_*.txt`

### 技术价值

- 将 LLM 的角色限定为“质检员 / 自动审阅员”，而不是让其直接替代整个匹配系统
- 通过 RAG 候选约束输出，显著降低自由生成带来的幻觉风险
- 为后续困难样本回流训练提供了自动标注入口

### 阶段判断

从这一天开始，`bge` 模块不再只是检索系统，而是开始具备半自动闭环学习能力。

---

## 2026-04-02：增加大规模标注数据处理能力

### 变更文件

- `slice_label_studio_json.py`

### 主要工作内容

该脚本解决的是大规模 Label Studio JSON 导出文件的工程问题，主要包括：

- 流式读取顶层 JSON 数组
- 避免一次性加载超大文件导致内存压力
- 支持只切出前 N 条任务
- 输出精简版 JSON，便于预览、抽样和人工测试

### 技术价值

- 说明项目规模已超出“小文件人工处理”阶段
- 开始考虑大体量标注数据在本地机器上的可操作性

---

## 2026-04-03：补齐 Label Studio 人工复核链路

### 变更文件

- `format_label_studio_requirements.py`
- `export_tier1_label_studio.py`
- `label_studio_softskill_template.xml`
- `docs/manual_label_template.txt`
- `docs/threshold_evaluation_template.txt`

### 主要工作内容

#### 1. 输出可供人工复核的 Label Studio 任务

`export_tier1_label_studio.py` 负责把系统输出转成可交给人工标注平台的任务格式，能力包括：

- 读取 `Tier2_Matched_Data.csv` 和 `Tier3_Pending_Data.csv`
- 基于 BGE + task chunk 再补候选
- 过滤不适合参与复核的职业编码
- 组装 Top 候选和候补候选
- 输出：
  - Label Studio JSON
  - preview CSV
  - 分片 JSON
  - manifest

这说明系统已经不是“只看最终结果”，而是把人工复核设计成正式流程。

#### 2. 优化标注文本可读性

`format_label_studio_requirements.py` 针对岗位要求文本做专门格式化，包括：

- 统一换行
- 识别 `|`、`;` 等内联分隔符
- 在编号、标题字段、项目符号前插入换行
- 清理多余空白行

这一步对标注效率非常关键，因为招聘 JD 常常是脏格式、长文本、混合语言，不做整理会明显降低人工审核效率。

#### 3. 沉淀人工标注与阈值评估规范

两个文档模板的作用非常明确：

- `manual_label_template.txt`
  - 定义 `sample_id / stage / predicted_code / gold_code / error_type / reviewer` 等字段
  - 规范错误类型和填写规则
  - 为后续监督训练沉淀 gold 数据
- `threshold_evaluation_template.txt`
  - 规范阈值实验记录字段
  - 约束如何比较 precision、matched_rate、决策建议

### 技术价值

- 完成了“模型输出 -> 人工审核 -> 结构化反馈”这一关键链路
- 把实验和标注从个人经验操作，提升为可复用、可交接的流程资产

---

## 2026-04-11：完成分层匹配主流程、阈值评估和自动迭代闭环

### 变更文件

- `D2_filter.py`
- `D4_T2match.py`
- `D6_threshold_eval.py`
- `D7_iterate_pipeline.py`

### 主要工作内容

#### 1. 建立 Tier1 规则过滤 + RAG 质检主流程

`D2_filter.py` 是整个系统的一级分流入口。它采用的是“规则层快速筛选 + RAG 质检补强”的混合模式，主要包括：

- 精确匹配
- 子串匹配
- 模糊匹配
- 敏感词过滤
- 岗位描述切分解析
- 多种检索模式：
  - `legacy`
  - `parsed_adaptive_task`
  - `compare`
- 调用 `build_qc_prompt` 生成质检 prompt
- 对高置信规则命中结果做 Qwen 质检
- 输出：
  - `Tier1_Matched_Data.csv`
  - `Tier2_Pending_Data.csv`

这一步的设计思路很稳：先用低成本规则吃掉明显样本，再把边界案例交给更贵的模型判断。

#### 2. 建立 Tier2 向量检索与阈值分流主流程

`D4_T2match.py` 负责处理 `Tier2_Pending_Data.csv` 中更困难的样本，核心能力包括：

- 加载微调后的 BGE 模型
- 读取职业大典标准库
- 严格复用微调阶段的清洗规则，保证特征空间一致
- 为每条样本生成候选职业相似度分数
- 将全量中间分数写入 `Tier2_Intermediate_Cache.csv`
- 根据阈值将样本分为：
  - `Tier2_Matched_Data.csv`
  - `Tier3_Pending_Data.csv`
- 输出阈值巡检文件：
  - `Tier2_Threshold_Inspection.xlsx`
- 额外生成子类覆盖分析样本和 Label Studio JSON：
  - `Tier2_Subclass_Coverage_10each.csv`
  - `Tier2_Subclass_Coverage_10each.label_studio.json`

这说明团队在 Tier2 处理上不仅关注“分对没分对”，还开始关注：

- 阈值灵敏度
- 误差边界
- 子类覆盖均衡性

#### 3. 建立阈值实验评估模块

`D6_threshold_eval.py` 的职责是把阈值试验标准化。当前关键配置包括：

- `THRESHOLD_START = 0.65`
- `THRESHOLD_END = 0.90`
- `THRESHOLD_STEP = 0.01`
- `MODEL_VERSION = "bge-base-zh-finetuned"`

脚本会生成标准化阈值评估表，帮助回答几个核心问题：

- 不同阈值下，系统覆盖率如何变化
- 哪个阈值能达到目标精度
- 主要错误类型集中在哪些模式
- 当前阈值应当接受、拒绝还是继续观察

#### 4. 建立自动轮次迭代闭环

`D7_iterate_pipeline.py` 是整个 `bge` 模块最能体现工程成熟度的脚本。它按轮次自动执行：

- D4：重算 Tier2 分数
- D5：自动标注边界样本
- D3：基于困难样本增量微调
- D6：评估阈值表现

并沉淀：

- `iteration_log.csv`
- 最新阈值评估报告
- 每轮最佳 precision

这意味着系统已经从“人工串脚本”升级为“可重复执行的自举迭代流程”。

### 阶段产出评价

截至 2026-04-11，`bge` 模块已经形成一个较完整的职业匹配与持续优化系统，具备：

- 分层策略
- 领域向量模型
- RAG 检索
- LLM 质检
- 人工复核
- 阈值实验
- 自动迭代

这已经不是单点算法验证，而是具备明显生产流程特征的工程化子系统。

---

## 5. 当前模块架构总结

从当前目录结构看，`bge` 模块可以抽象为以下六层：

### 1. 数据治理层

- `D1_quchong.py`

职责是统一招聘样本、清洗缺失值、按多维指纹做严格去重，为下游提供可信样本基线。

### 2. 一级规则分流层

- `D2_filter.py`

职责是用规则、模糊匹配与轻量 RAG 质检快速分流明显样本，控制高成本推理规模。

### 3. 向量检索与领域模型层

- `D3_finetune.py`
- `D4_T2match.py`

职责是训练领域化 BGE 模型，并在困难样本上做向量召回与阈值分流。

### 4. LLM 自动质检层

- `D5_qwen3_auto_label.py`

职责是将 BGE 候选结果交给 Qwen3 进行约束式 JSON 质检与纠偏。

### 5. 评估与迭代优化层

- `D6_threshold_eval.py`
- `D7_iterate_pipeline.py`

职责是对阈值和模型效果做版本化实验管理，并实现自动轮次迭代。

### 6. 人工复核与标注交付层

- `export_tier1_label_studio.py`
- `format_label_studio_requirements.py`
- `slice_label_studio_json.py`
- `docs/manual_label_template.txt`
- `docs/threshold_evaluation_template.txt`
- `label_studio_softskill_template.xml`

职责是把模型输出转为人工可审、可回流、可沉淀的数据资产。

---

## 6. 本期关键技术亮点

### 1. 采用分层策略控制成本与质量

系统不是所有样本都直接走大模型，而是分成：

- Tier1：规则 + RAG 质检
- Tier2：BGE 检索 + 阈值
- Tier3：无法高置信分配的 Pending

这种设计兼顾了：

- 运行成本
- 推理速度
- 精度控制
- 人工复核可管理性

### 2. 向量模型、RAG 与 LLM 分工清晰

当前方案中：

- BGE 负责相似度召回
- RAG 负责约束候选上下文
- Qwen 负责质检和纠错

不是让单一模型“一把梭”，而是把不同能力放在最适合的位置上，工程上更稳。

### 3. 难样本回流机制比较成熟

`D5 -> D3 -> D6 -> D7` 形成了完整回路：

- 先自动标注错误样本
- 再将困难样本加入训练
- 然后重新评估阈值
- 最终沉淀到下一轮模型版本

这使系统具有持续提升的可能，而不只是一次性训练。

### 4. 人工复核链路建设较完整

很多项目到自动标注阶段就结束了，但当前 `bge` 已经补齐了：

- Label Studio 任务导出
- 大文件分片
- 文本格式化
- 标注模板
- 阈值实验模板

这说明项目不仅考虑模型本身，也考虑长期运营和团队协作效率。

---

## 7. 当前风险与待改进点

### 1. 目录未纳入 Git 跟踪

当前 `src/bge` 为未跟踪目录，会带来几个明显问题：

- 无法还原真实迭代顺序
- 无法精确统计每轮修改范围
- 不利于对模型版本、阈值版本和 prompt 版本做严格审计

建议尽快纳入版本控制，并对关键产物建立版本命名规范。

### 2. 脚本依赖较强的本地绝对路径

例如：

- `D:\model\bge-base-zh-v1.5`
- `D:\model\bge-base-zh-finetuned`
- `D:\model\Qwen3-8B`

这会导致：

- 迁移成本较高
- 多人协作环境不稳定
- 自动化部署难度上升

建议后续统一改为配置文件或环境变量驱动。

### 3. Prompt 与质检逻辑定义跨目录

`bge` 实际调用的 prompt 构建函数位于 `src/rag/qc_utils.py`，这在工程上可行，但也意味着：

- 目录边界不够清晰
- 后续维护时容易漏改
- 周报或审计时需要跨模块追踪

建议未来将 `bge` 所依赖的 prompt 模板版本显式记录。

### 4. 自动标注仍受候选召回质量约束

当前 Qwen 的输出被限定必须从候选列表中选取，这能减少幻觉，但也带来一个边界：

- 若候选集本身缺少正确职业，则 Qwen 只能在错误候选中做相对最优选择
- 这会使 `dictionary_gap` 和召回缺口成为系统上限约束

### 5. 阈值与模型版本需要更严格联动

从当前结构看，阈值实验已经规范化，但如果未来模型频繁迭代，仍需要更强约束：

- 阈值必须绑定模型版本
- 阈值必须绑定数据版本
- 阈值变更应有固定决策记录

---

## 8. 脚本 Prompt 附录

## 8.1 Prompt 使用情况结论

与 `job_title_parsing` 不同，`bge` 模块 **确实存在实际的大模型 Prompt 调用链路**。

调用关系为：

- `src/bge/D2_filter.py`
  - 在 Tier1 质检阶段调用 `build_qc_prompt(...)`
- `src/bge/D5_qwen3_auto_label.py`
  - 在自动标注阶段调用 `build_qc_prompt(...)`
- Prompt 模板定义位置：
  - `src/rag/qc_utils.py`

也就是说：

- `bge` 是 Prompt 的 **业务调用方**
- `src/rag/qc_utils.py` 是 Prompt 的 **模板定义方**

---

## 8.2 Prompt 模板一：Tier1 / Tier2 质检 Prompt

适用场景：

- 当前样本已有系统预测结果
- 需要 Qwen 判断预测是否正确

实际模板核心结构如下：

```text
/no_think
你是职业分类质检员。请基于岗位信息、当前预测和知识库候选，判断预测是否正确。
只能输出一行合法 JSON，禁止输出解释、markdown、<think> 内容。
JSON 键：is_correct,gold_title,gold_code,error_type,error_note
规则：
1) is_correct=1 预测正确，=0 预测错误。
2) is_correct=1 时：gold_title 与 gold_code 必须从知识库候选中选取与预测最吻合的一项，error_type/error_note 置空。
3) is_correct=0 时：gold_title/gold_code 必须从知识库候选中选取最正确的一项，且该项名称必须与 predicted_title 不同；
   严禁凭空编造不在候选列表中的职业名称或代码。
   error_type 从以下选一：title_ambiguous,desc_noise,assistant_intern_confusion,
   coarse_to_fine_mismatch,cross_domain_confusion,dictionary_gap,
   low_confidence_borderline,other
4) 判断依据：仅凭岗位描述与候选职业定义/任务的语义匹配程度，不受预测分数高低影响。
5) 若知识库候选中没有比当前预测更合适的职业，则判 is_correct=1。

岗位名称: {job_title}
岗位描述: {job_desc[:400]}
当前预测职业: {predicted_title}
当前预测代码: {predicted_code}
当前预测分数: {predicted_score}

知识库候选（gold_title/gold_code 只能从此列表中选取）:
{rag_context}
```

### 设计特点

- 使用 `/no_think` 和 `enable_thinking=False` 双重约束，抑制 Qwen3 输出思维链
- 强制输出单行 JSON，便于脚本批量解析
- 明确规定 `gold_title/gold_code` 必须从候选中选取，降低幻觉风险
- 强制 `is_correct=0` 时的 `gold_title != predicted_title`，减少自相矛盾输出

---

## 8.3 Prompt 模板二：Tier3 无预测推荐 Prompt

适用场景：

- 当前样本没有系统预测结果
- 需要 Qwen 直接从候选中推荐最合适职业

实际模板核心结构如下：

```text
/no_think
你是职业分类标注员。该岗位尚无系统预测结果，请根据岗位信息和知识库候选，
直接为该岗位推荐最合适的职业分类。
只能输出一行合法 JSON，禁止输出解释、markdown、<think> 内容。
JSON 键：is_correct,gold_title,gold_code,error_type,error_note
规则：
1) 无系统预测，is_correct 固定填 0。
2) gold_title/gold_code 必须从知识库候选中选取最合适的一项。
3) error_type 填 dictionary_gap（系统未覆盖此职业）。
4) error_note 简要说明推荐理由。

岗位名称: {job_title}
岗位描述: {job_desc[:400]}

知识库候选（请从中选取 gold_title/gold_code）:
{rag_context}
```

### 设计特点

- 把 Tier3 定义成“未覆盖职业的推荐任务”，而不是简单拒绝
- 强制 `dictionary_gap`，使这类样本天然成为后续字典补全和模型改进的素材
- 保留 `error_note` 作为简要解释，方便复核

---

## 8.4 Prompt 上下文格式

候选职业上下文由 `build_rag_context(...)` 生成，格式大致如下：

```text
[候选1] 代码:{code} 名称:{title} 细分工种:{sub_titles} 相似度:{score}
定义:{desc}
任务:{tasks}
```

其作用是：

- 给 Qwen 一个受控、结构化、长度可控的候选职业列表
- 让模型在“限定候选集合”内做判断，而不是开放生成职业名称

---

## 8.5 Prompt 相关实现细节

除了 Prompt 文本本身，当前链路还做了几项很关键的工程处理：

- `D5_qwen3_auto_label.py`
  - 对岗位描述进行截断，默认控制在 400 字符以内
  - 通过 `RAG_TOP_K` 控制 prompt 长度与显存开销
- `src/rag/qc_utils.py`
  - 使用 `tokenizer.apply_chat_template(..., enable_thinking=False)`
  - 对模型输出做 `extract_json(...)` 鲁棒解析
  - 若 JSON 解析失败，回退为 `qwen_output_parse_failed`
  - 对标签做 `normalize_label(...)` 后处理，修复 `gold==predicted` 等矛盾

这说明当前 Prompt 设计并不是“只写一段指令”，而是形成了较完整的推理约束、解析、纠错链条。

---

## 9. 结论

综合来看，`src/bge` 模块在本期内完成了从“招聘数据去重 + 初版向量训练”到“分层匹配 + Qwen 质检 + 人工复核 + 自动迭代”的关键跃迁，已经具备明显的生产化和持续优化特征。

如果按研发成熟度划分：

- 2026-03-27：处于数据准备与基础微调期
- 2026-04-01：进入 LLM 质检增强期
- 2026-04-02 ~ 2026-04-03：进入人工复核与标注平台建设期
- 2026-04-11：进入完整闭环迭代期

从当前代码形态判断，后续最值得继续推进的方向包括：

- 将目录和关键产物纳入 Git 版本管理
- 统一模型路径与配置管理
- 把 Prompt、模型版本、阈值版本做更严格的版本绑定
- 增强召回侧对 `dictionary_gap` 职业的补全能力
- 形成更稳定的人工标注回流与 gold 集积累机制

如果你需要，我还可以继续补一版：

- 更偏管理汇报口径的精简版周报
- 带表格和里程碑状态的汇报版周报
- 把 `src/bge` 和 `src/job_title_parsing` 合并成一份总周报

