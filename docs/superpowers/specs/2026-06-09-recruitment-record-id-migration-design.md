# Recruitment Record ID Migration Design

**Goal**

将 Employ26 当前活跃公共链路从 `sample_row_id`、`source_record_id`、`row_id`、旧式 `job_id` 等历史身份字段，收敛为统一的 `recruitment_record_id` 主轴，并让后续公共表都基于该字段关联招聘记录。

## Scope

本次设计只覆盖第一批活跃链路：

- 统一规范层 `public.recruitment_jobs_normalized`
- 岗位描述解析结果表 `public.job_description_parsed`
- 岗位要求匹配结果表 `public.skill_extraction_requirement_matches`
- 直接消费上述结果的活跃读取逻辑与配置

本次设计不覆盖：

- `archive/` 中的历史脚本
- BERT 历史训练链路的全面重构
- 一次性清理全仓库所有旧字段引用

## Context

仓库已经明确了以下领域决策：

- `recruitment_record_id` 是新的唯一标准招聘记录身份
- `public.recruitment_jobs_normalized` 是短期统一规范层的实体表
- `public.job_description_parsed` 是当前唯一权威解析结果表
- 新公共表之间统一使用 `recruitment_record_id` 关联
- `sample_row_id`、`source_record_id`、`row_id`、旧式 `job_id` 都是废弃身份字段
- `source_table` 与 `source_row_number` 仅保留为溯源字段

## Recommended Approach

推荐采用“先建立统一规范层，再改下游公共表契约”的路径，而不是直接在现有派生表中拼补新字段。

原因：

- `recruitment_record_id` 的首次分配与冻结必须在统一规范层完成
- `job_description_parsed` 和 `skill_extraction_requirement_matches` 都应依赖规范层输入，而不是继续从三套平台表各自生成身份
- 这样可以一次性拆开“业务身份”和“来源定位符”

## Alternatives Considered

### Option A: 直接在现有派生表中回填 `recruitment_record_id`

优点：

- 改动看起来较小
- 不需要先创建统一规范层

缺点：

- 派生层反过来定义源记录身份，边界错误
- 无法稳定承载 “首次生成并冻结” 的规则
- 之后仍需要再引入统一规范层，形成二次迁移

结论：不采用。

### Option B: 先做统一规范层，再驱动下游公共表迁移

优点：

- 领域边界清晰
- 能支撑冻结后的 `recruitment_record_id`
- 让后续公共链路有单一上游入口

缺点：

- 首批改动面更大
- 需要同步调整配置、写库逻辑和消费方读取逻辑

结论：采用。

### Option C: 保留旧字段兼容过渡

优点：

- 短期更稳
- 旧脚本更少报错

缺点：

- 与已经确认的“直接硬切到 `recruitment_record_id`”冲突
- 会继续传播双轨身份模型

结论：不采用。

## Design

### 1. 统一规范层

新增实体表 `public.recruitment_jobs_normalized`，作为跨平台统一招聘记录入口。

建议最小字段集：

- `recruitment_record_id`
- `source_platform`
- `source_table`
- `source_row_number`
- `source_native_job_id`
- `dedupe_fingerprint`
- `job_title`
- `job_description_raw`
- `work_city`
- `company_name`
- `publish_date`
- `created_at`
- `updated_at`

身份规则：

- 若源平台存在原生岗位 ID，则优先以“平台 + 原生岗位 ID”命中历史记录
- 若不存在，则使用内部 `dedupe_fingerprint` 命中历史记录
- 仅当未命中历史记录时，首次生成新的无业务语义 `recruitment_record_id`
- 记录内容允许更新，但 `recruitment_record_id` 不重发

同步规则：

- 采用增量 upsert
- 不采用 `CREATE OR REPLACE TABLE`
- `dedupe_fingerprint` 仅作为内部认同字段，不作为下游公共引用契约

### 2. 岗位描述解析结果表

`public.job_description_parsed` 迁移为以 `recruitment_record_id` 为核心身份的当前权威解析表。

字段契约调整：

- 新增 `recruitment_record_id`
- 删除 `source_record_id`
- 保留 `source_platform`
- 保留 `source_table`
- 保留 `source_row_number`
- 保留 `parser_version`，但仅作结果元数据，不参与唯一身份

表语义：

- 一条 `Recruitment Record` 在解析表中只保留一条当前结果
- 重跑解析时覆盖该记录的最新结果
- 不保留多个解析版本并存作为公共契约

唯一性规则：

- 核心唯一键为 `recruitment_record_id`

### 3. 岗位要求匹配结果表

`public.skill_extraction_requirement_matches` 迁移为以 `recruitment_record_id` 为标准引用字段的公共结果表。

字段契约调整：

- 新增并强制使用 `recruitment_record_id`
- 删除 `sample_row_id` 作为标准字段
- 保留 `source_table`、`source_row_number` 作为溯源字段
- 保留 `occupation_code`、`occupation_title`、TopK 候选列等业务结果字段

上游输入调整：

- 长期主输入切到 `public.recruitment_jobs_normalized`
- 描述解析优先读取 `public.job_description_parsed`

### 4. 活跃读取逻辑

第一批读取逻辑要同步切换：

- `src.db.job_description_parsed`
- `src.data_pipeline.requirement_match_prep`
- `src.utils.llm_labeling_utils`
- `src.skill_extraction.occupation_skill_pipeline`
- 与 `skill_extraction_requirement_matches` 直接耦合的活跃读取逻辑

这些逻辑中：

- 读取招聘记录身份时统一改为 `recruitment_record_id`
- 仅在确实需要回溯原始来源时才读取 `source_table/source_row_number`
- 不再生成或透传 `sample_row_id`、`source_record_id`

## Data Flow

目标数据流如下：

1. 三平台源表进入 `public.recruitment_jobs_normalized`
2. 统一规范层为每条记录分配或命中冻结的 `recruitment_record_id`
3. 描述解析从统一规范层读取 `Job Description Text`
4. 解析结果写入 `public.job_description_parsed`
5. 岗位要求匹配流程使用 `recruitment_record_id` 贯穿读取与写入
6. 下游技能抽取和标注辅助逻辑只通过 `recruitment_record_id` 关联公共结果表

## Error Handling

需要显式处理的异常场景：

- 源记录缺失必要字段，无法生成 `dedupe_fingerprint`
- 同一同步批次中出现重复 `source_native_job_id`
- 同一同步批次中出现重复 `dedupe_fingerprint`
- 下游读取逻辑仍然请求已删除的旧字段
- 规范层缺失 `recruitment_record_id` 时禁止继续写入派生公共表

处理原则：

- 统一规范层写入失败应中止当批同步
- 下游公共写入在缺少 `recruitment_record_id` 时应显式报错
- 不做静默回退到 `sample_row_id` 或 `row_id`

## Testing Strategy

第一批测试至少覆盖：

- 规范层首次生成 `recruitment_record_id`
- 同步重跑时命中历史记录并保留原 ID
- `job_description_parsed` 按 `recruitment_record_id` 覆盖写入
- `source_record_id` 不再出现在解析结果写入行中
- `skill_extraction_requirement_matches` 结果行包含 `recruitment_record_id`
- 读取工具在新表结构下仍能正常构造样本与文本

验证方式：

- 新增或更新局部单元测试
- 运行 `python -m compileall -q src`
- 运行与本次改动直接相关的 `pytest` 用例

## Implementation Phasing

建议分三步执行：

### Phase 1

建立统一规范层与公共身份模型：

- 配置中新增 `public.recruitment_jobs_normalized`
- 新增建表、upsert 与 ID 分配逻辑

### Phase 2

迁移解析与要求匹配公共表：

- `public.job_description_parsed`
- `public.skill_extraction_requirement_matches`
- 对应 Python 写库与读取逻辑

### Phase 3

收口活跃消费方与文档：

- `llm_labeling_utils`
- `occupation_skill_pipeline`
- 数据库文档与表说明

## Success Criteria

完成后应满足：

- 活跃公共表统一使用 `recruitment_record_id`
- `sample_row_id`、`source_record_id`、`row_id` 不再作为第一批活跃公共链路的标准身份字段
- `job_description_parsed` 与 `skill_extraction_requirement_matches` 可在无旧字段前提下运行
- 下游活跃读取逻辑能仅依赖 `recruitment_record_id` 工作

## Open Boundaries

本设计明确保留以下后续问题，不在本次第一批迁移内解决：

- 标注 schema 全量迁移到 `recruitment_record_id`
- BERT 历史数据集与训练链路改造
- 跨平台职位聚合为更高层实体
- `recruitment.jobs` 独立 schema 的中期演进
