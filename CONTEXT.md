# Employ26

Employ26 是一个围绕广东省招聘数据进行岗位匹配、技能抽取与职业检索分析的项目。它把多平台招聘文本、标注结果和派生特征收敛成可复用的数据资产。

## Language

**Experimental Workspace**:
用于复现实验、比较方案和沉淀阶段性结论的代码区域。它服务于探索和验证，不直接等同于正式长期维护的生产入口。
_Avoid_: Production pipeline, formal module, official entrypoint

**Retrieval Effectiveness**:
在检索实验语境中，优先优化候选选择准确率、排序质量、层级命中率和 MRR 等离线检索质量指标，而不是优先优化训练耗时、推理时延或模型体积。
_Avoid_: Training efficiency as the primary goal, deployment cost as the default objective

**Baseline-First Model Selection**:
在底座模型存在多个候选时，先固化当前表现最好且可复现的基线训练与评估流程，再让新底座作为挑战者在同口径下比较，而不是在基线尚未冻结时同时漂移多个变量。
_Avoid_: Changing base model and training recipe at the same time, comparing models under different evaluation setups

**Unified Retrieval Evaluation**:
检索基线与挑战者模型的优劣，必须以统一数据集、统一指标定义和统一评估脚本得出的结果判定；训练脚本各自附带的局部评估只可用于单次实验内诊断，不作为跨方案定胜负的正式标准。
_Avoid_: Comparing models by per-script test sets, mixing local experiment metrics with canonical benchmark results

**Baseline Recipe**:
检索基线不是单个已训练模型目录，而是一套可重复执行并可重新产出同类结果的固定实验配方。它至少同时冻结训练数据构造规则、train/test split 规则、训练超参数和统一评估流程。
_Avoid_: Treating one saved model artifact as the whole baseline, changing data rules while claiming the baseline is unchanged

**Frozen Baseline Data Contract**:
当实验基线被声明为可复现配方时，其输入数据源也必须冻结到具体 PostgreSQL 表名与字段口径，而不是只说“来自标注数据”或“来自职业词典”。后续挑战者模型必须在同一数据契约上比较。
_Avoid_: Vague data provenance, switching between tables/views/field semantics without renaming the recipe

**Single-Variable Challenger Run**:
基线的第一轮挑战应只改变一个主变量，并把其他配方元素全部冻结不动。对于底座模型挑战，这意味着只替换 embedding base model，本轮不同时调整样本构造、split 规则、训练超参数或正式评估口径。
_Avoid_: Multi-variable “challenge” runs, crediting a new base model for gains caused by recipe changes

**Challenger Win Criteria**:
底座模型挑战只有在统一评估中同时守住主检索指标并未明显牺牲层级准确率时，才可视为取代基线。对 Penghui 检索实验而言，第一轮主胜负指标是 `candidate_acc` 与 `mrr`，而 `subclass_acc` 与 `midclass_acc` 作为约束指标不应明显退步。
_Avoid_: Declaring victory from a single flattering metric, trading away core retrieval quality for local gains

**Parameterized Baseline Runner**:
当检索基线被定义为可复现配方时，应优先通过参数化同一训练入口来切换底座模型和产物命名，而不是复制出新的专用脚本。这样可以把挑战者运行约束在同一配方框架中，减少实现漂移。
_Avoid_: Forking a new training script per base model, encoding recipe changes as script proliferation

**Recipe-Backbone Artifact Naming**:
检索实验产物应同时显式表达“配方名”和“底座名”，使同一配方下的不同底座挑战者可以直接对照。目录名和结果文件名不应再只暴露历史版本号或只暴露某一侧信息。
_Avoid_: Ambiguous artifact names, model directories that hide whether the recipe or the backbone changed

**Forward-Only Artifact Renaming**:
新的产物命名规范从下一次新跑开始生效，不主动重命名已经验证过的历史产物目录。历史目录可继续作为已确认实例存在，直到新的参数化入口稳定后再决定是否做兼容迁移。
_Avoid_: Renaming validated historical artifacts while introducing a new recipe contract, mixing compatibility migration with experiment definition

**Parameterized Canonical Evaluation**:
正式统一评估入口也应接受显式模型列表或模型映射参数，而不是把待评估模型写死在源码中。这样挑战者模型可以在不修改评估代码的前提下进入同口径比较。
_Avoid_: Editing benchmark source code for each challenger run, hidden model lists baked into the evaluator

**Minimal Challenger CLI**:
在基线固化和第一轮挑战阶段，训练入口与统一评估入口只暴露支撑单变量挑战所需的最小参数集合，而不是提前扩张成通用实验平台。当前最小集合是训练入口支持 `--base-model-path`、`--output-model-name`、`--run-label`，统一评估入口支持重复传入 `--model NAME=PATH`。
_Avoid_: Premature general-purpose experiment CLIs, adding knobs unrelated to the first challenger comparison

**Public Feature Layer**:
供多个下游流程复用的正式结构化数据层。它的职责是沉淀稳定、可重复使用的派生特征，而不是充当某个单独脚本的私有中间结果。
_Avoid_: Private temp table, one-off cache, script-local intermediate

**Job Description Parsing Result**:
对单条招聘岗位描述做结构化切分后形成的可复用文本特征。它属于 Public Feature Layer，面向后续匹配、抽取和检索流程共享使用。
_Avoid_: Cleaned-data copy, disposable parsing output

**Recruitment Record**:
来自某个招聘平台的一条完整职位发布记录。它是项目中的基础数据单元，其中包含岗位名称、岗位描述等原始招聘信息；同岗跨平台默认视为不同 Recruitment Record。
_Avoid_: Cleaned-data row, parsing result

**Job Description Text**:
Recruitment Record 中承载职责、要求、福利等招聘叙述的自由文本。描述解析关注的是这段文本本身，而不是整张招聘表的复制或清洗。
_Avoid_: Whole cleaned table, copied job table

**Source Locator**:
用于唯一定位某条来源招聘记录的标准标识符，格式为 `"{source_table}:{source_row_number}"`。新流程中的来源回溯、跨步骤传递和结果引用都应以它为准。
_Avoid_: sample_row_id, source_record_id, ad-hoc source key

**Recruitment Record ID**:
用于跨时间稳定标识一条招聘记录的正式业务主键。它优先承接平台原生岗位 ID；若源平台没有提供，则在统一规范层首次生成并冻结。
_Avoid_: Source Locator, row_number, temp hash

**Recruitment Normalized Layer**:
面向跨平台复用的招聘统一规范层。它负责把不同平台的招聘记录收敛为一致的公共字段，并承载稳定的 Recruitment Record ID。
_Avoid_: Raw mirror, feature layer, experiment output

**Recruitment Normalized Table**:
Recruitment Normalized Layer 的实体表实现。在当前阶段，它以 `public.recruitment_jobs_normalized` 的形式承载统一字段、来源定位信息和冻结后的 Recruitment Record ID。
_Avoid_: View-only projection, disposable export

**Frozen Recruitment Record ID**:
在 Recruitment Normalized Table 中首次确定后不再重算的 Recruitment Record ID。后续同步可以更新记录内容，但不应改变它所代表的业务身份。
_Avoid_: Recomputed ID, per-run temporary key

**Dedupe Fingerprint**:
在缺少平台原生岗位 ID 时，用于识别“这次来的记录是否与历史记录属于同一招聘记录”的内部认同字段。它服务于 Frozen Recruitment Record ID 的稳定分配，但不应作为下游公共引用契约。
_Avoid_: Public identifier, Source Locator, temporary checksum

**Normalized Recruitment Input**:
供公共特征层优先消费的统一招聘输入。长期来看，岗位描述解析等下游复用流程应以 Recruitment Normalized Table 为主上游，而不是直接面向各平台源表。
_Avoid_: Direct raw-table input as the long-term default

**Current Authoritative Parsing Result**:
面向下游消费的当前唯一权威岗位描述解析结果。它按 `recruitment_record_id` 唯一存储，重跑时覆盖为最新结果，而不是并存保留多个 parser 版本。
_Avoid_: Versioned parsing history as the default public contract

**Recruitment Record Reference**:
新公共表之间引用招聘记录时使用的标准关联字段名。该字段统一命名为 `recruitment_record_id`，用于表达正式业务身份，而不是流程内定位符；现有活跃公共表在重构时也应收敛到该字段。
_Avoid_: sample_row_id, source_record_id, generic job_id

**Requirement Match Result**:
面向技能抽取等下游流程复用的岗位要求匹配结果。它作为公共结果表时，应以 `recruitment_record_id` 作为标准引用字段，而不是继续传播 `sample_row_id`。
_Avoid_: sample_row_id as the public reference contract

**Requirement Match Preparation Script**:
用于生成 Requirement Match Result 的实现脚本名，例如 `src.data_pipeline.requirement_match_prep`。它是技术实现入口，不是领域中的正式数据概念。
_Avoid_: Treating a script name as the canonical data model term

**Opaque Recruitment Record ID**:
无业务语义、仅用于稳定标识 Recruitment Record 的正式 ID。它不编码平台、标题、日期等业务信息，可读信息应由其他字段承载。
_Avoid_: Readable composite key, semantic identifier

**Recruitment Record ID Assignment**:
`recruitment_record_id` 只在记录首次进入 Recruitment Normalized Table 且未命中历史记录时生成一次。后续同步可以更新记录内容，但不应重新分配该 ID。
_Avoid_: Reassignment on content update, regenerated ID on re-run

**Source Trace Fields**:
用于回溯招聘记录原始来源位置的辅助字段，例如 `source_table` 和 `source_row_number`。它们服务于溯源和排查，但不参与 Recruitment Record 的业务身份定义，也不替代 `recruitment_record_id`。
_Avoid_: Business primary key, standard cross-table reference

**Deprecated Reference Fields**:
旧链路中用于引用招聘记录的历史字段名，例如 `sample_row_id`、`source_record_id`、`row_id` 和旧式 `job_id`。在新公共链路中它们应视为废弃字段，不再作为标准引用键。
_Avoid_: Standard identifier, canonical reference

**Historical Annotation RRID Backfill**:
为历史 `annotations.label_studio_tasks_v2` 任务补齐 `recruitment_record_id` 的一次性治理动作。它只回填正式业务身份字段，不改写任务文本、候选或标注语义，并要求同时保留可复演的审计证据。
_Avoid_: Silent remap, semantic overwrite

**Historical Annotation Snapshot Row ID**:
历史 `label_studio_tasks_v2.row_id` 在第二轮导出链路中表示 Label Studio 导出快照里的行号，而不是招聘源表主键。它只能用于回放历史任务对应的快照行，不能直接当作招聘记录身份。
_Avoid_: public.jd_raw.row_id, canonical source primary key
