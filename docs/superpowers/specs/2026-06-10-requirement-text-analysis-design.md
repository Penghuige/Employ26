# Requirement Text Analysis Design

**Goal**

建立第一阶段 `Requirement Text Statistical Analysis` 的正式数据底座与词汇资源治理方式，完成以下三件事：

1. 扩展 `public.recruitment_jobs_normalized`，把招聘源表已有的高频结构化字段纳入统一规范层。
2. 引入 PostgreSQL `analysis_lexicon` schema，作为第一阶段 requirement text 统计所依赖词汇资源的唯一正式 source of truth。
3. 为第一阶段 requirement text 统计建立稳定输入、稳定输出和稳定版本口径，并判断职业词典相关 4 张表的保留策略。

## Scope

本次设计覆盖：

- `public.recruitment_jobs_normalized` 字段扩展
- 三家平台 `sample` 数据补充进入 `recruitment_jobs_normalized`
- `analysis_lexicon` 新 schema 及其资源表
- 第一阶段 requirement text 统计的脚本骨架、输出目录和结果表结构
- `src/analysis/README.md` 与相关数据库说明更新
- `occ_dict` / `occ_dict_detailed` / `occ_dict_pro` / `occ_dict_class` 的比较与保留建议

本次设计不覆盖：

- 主题模型、聚类、LLM 标注、自动分类器
- 新的正式分析结果回写表
- 全项目通用词汇资源平台
- `archive/` 历史链路

## Confirmed Domain Decisions

基于本轮确认，第一阶段 requirement text 统计已固定以下约束：

- 主输入为 `public.recruitment_jobs_normalized` + `public.job_description_parsed`
- 每个 `recruitment_record_id` 只读取当前唯一权威解析结果
- 主语料只分析 `requirements_text` 非空记录
- `duties_text` fallback 仅用于诊断，不并入主口径
- unigram / bigram / trigram 都按“命中记录数”计数，而不是单文重复次数
- bigram / trigram 仅在 requirement item 内部生成
- 主结果按发布记录口径计数，并附去重文本对照
- 分词资源、停用词、短语规则统一进入 `analysis_lexicon`

## Current-State Findings

### 1. `recruitment_jobs_normalized` 覆盖不足

实查 PostgreSQL 后发现：

- `public.recruitment_jobs_normalized` 当前仅 `18611` 行
- 来源全部来自三家 `sample` 表，但覆盖远未补齐：
  - `"51job".sample`: `45512` 行，其中已进入规范层 `7594` 行
  - `"Liepin".sample`: `20047` 行，其中已进入规范层 `3280` 行
  - `"Zhilian".sample`: `38379` 行，其中已进入规范层 `7737` 行

结论：

- 统一规范层目前更像“标注链路回填副产物”，还不是 sample 层的完整统一入口
- 本次必须补一条“样本表 -> 统一规范层”的正式补库入口

### 2. 四张职业词典表并非完全重复

数据库实查结果：

- `public.occ_dict`: `1698` 行，列为 `code/title/desc/tasks`
- `public.occ_dict_detailed`: `1698` 行，是 `occ_dict` 加层级字段增强版
- `public.occ_dict_pro`: `1698` 行，是 `occ_dict_detailed` 再加检索预处理字段增强版
- `public.occ_dict_class`: `2324` 行，只含层级分类字段，不含 `desc/tasks/retrieval_*`

重叠判断：

- `occ_dict` 与 `occ_dict_detailed` 是一对一同码增强关系
- `occ_dict_detailed` 与 `occ_dict_pro` 在业务上也是增强关系，当前 `1691/1698` 码对齐，剩余差异主要来自尾随空格与括号字符规范化
- `occ_dict_class` 不是简单重复表，它是完整层级分类骨架，包含大量中间层级节点；`occ_dict_detailed` 只覆盖其中 `758` 个具体职业代码

结论：

- 不建议“只保留一张表”
- 更合理的保留策略是：
  - `occ_dict_pro` 作为检索/下游主用职业词典
  - `occ_dict_class` 作为完整职业层级骨架
  - `occ_dict_detailed` 作为中间增强层，可视后续实际引用决定是否逐步退役
  - `occ_dict` 可视为最薄基表，若全仓库无直接依赖，可作为后续候选退役对象

## Recommended Approach

推荐采用“三段式落地”：

### A. 先补统一规范层，再做第一阶段统计

先把 `salary_raw`、`education_requirement_raw`、`experience_requirement_raw`、`company_size_raw`、`company_industry_raw` 等字段补进 `public.recruitment_jobs_normalized`，并把三家 `sample` 表补齐到统一层。

原因：

- 第一阶段 requirement text 统计需要这些结构化维度做交叉分析
- 如果不先扩规范层，后面每个分析脚本都得回源联三张 sample 表

### B. 再建立 `analysis_lexicon`

新增 `analysis_lexicon` schema，至少包含：

- `lexicon_release`
- `user_dictionary`
- `stopwords`
- `phrase_rules`

并采用：

- `bigserial id` 主键
- `text + check` 域值约束
- 三张资源表显式挂 `lexicon_release`
- 运行时只读取 `is_current = true` 的正式版本

### C. 最后补第一阶段统计脚手架

不做模型化，只做：

- 总体高频表
- 分层高频表
- 覆盖率 / fallback 诊断表
- 去重文本 vs 发布记录口径对照表
- `lexicon summary`
- `noise_terms_filtered.csv`

## Alternatives Considered

### Option A: 继续从 `output/integrated` / CSV 起步

优点：

- 改动少
- 脚本上手快

缺点：

- 与当前已确认的 PG 公共层口径冲突
- 会让第一阶段又建立一套临时 source of truth

结论：不采用。

### Option B: 直接在 `dicts/` 里管理词汇资源

优点：

- 版本化直观
- 更接近现有仓库习惯

缺点：

- 与本轮已确认的 PostgreSQL 新 schema 决策冲突
- 无法满足“集中维护、共享更新”的目标

结论：不采用。

### Option C: 一次性把词汇资源做成全项目通用平台

优点：

- 长期看可能减少重复建设

缺点：

- 设计边界会迅速膨胀
- 会拖慢第一阶段 requirement text 统计落地

结论：不采用，先做 stage-scoped schema。

## Data Model

### 1. `public.recruitment_jobs_normalized`

在现有字段基础上新增原始结构化维度：

- `salary_raw`
- `education_requirement_raw`
- `experience_requirement_raw`
- `company_size_raw`
- `company_industry_raw`

命名原则：

- 统一用英文公共列名
- 原始值显式保留 `*_raw`
- 后续若做标准化，再增量补 `salary_min` / `salary_max` / `education_level` 等列

### 2. `analysis_lexicon.lexicon_release`

最小字段：

- `id`
- `version`
- `is_current`
- `released_at`
- `released_by`
- `notes`
- `created_at`
- `updated_at`

### 3. `analysis_lexicon.user_dictionary`

最小字段：

- `id`
- `release_id`
- `term`
- `normalized_term`
- `preferred_term`
- `term_type`
- `category`
- `variants_json`
- `enabled`
- `source`
- `notes`
- `created_at`
- `updated_at`

约束：

- `normalized_term + term_type` 唯一
- 同一 `normalized_term` 可以跨多个 `term_type`

### 4. `analysis_lexicon.stopwords`

最小字段：

- `id`
- `release_id`
- `term`
- `normalized_term`
- `scope`
- `stop_strength`
- `enabled`
- `source`
- `notes`
- `created_at`
- `updated_at`

约束：

- `scope + normalized_term` 唯一
- `enabled = false` 的停用词保留，不物理删除

### 5. `analysis_lexicon.phrase_rules`

最小字段：

- `id`
- `release_id`
- `rule_type`
- `source_term`
- `normalized_source_term`
- `replacement_term`
- `priority`
- `enabled`
- `source`
- `notes`
- `created_at`
- `updated_at`

约束：

- `normalized_source_term + rule_type` 唯一
- 规则采用确定性显式替换，不做模糊 pattern 改写

## Output Design

第一阶段输出目录：

- `output/reports/req_analysis_{mm-dd}/`

固定输出：

- `run_manifest.json`
- 总体高频表 CSV
- 分层高频表 CSV
- 覆盖率/回退率诊断 CSV
- 去重文本 vs 发布记录对照 CSV
- `noise_terms_filtered.csv`
- `lexicon_summary.csv`
- 主报告 TXT

固定章节顺序：

1. 运行摘要
2. 样本覆盖率与回退诊断
3. 总体高频 unigram
4. 总体高频 bigram/trigram
5. 按 `term_type/category` 的结构分析
6. 按职业/城市/行业/公司规模分层
7. 月度趋势
8. 去重文本 vs 发布记录口径对照

## Error Handling

需要显式处理：

- `lexicon_release` 无当前正式版本
- 当前 release 下某类资源为空
- `requirements_text` 缺失
- `publish_date` 无法解析
- 词典规则冲突
- 规范层缺失新增 raw 字段导致下游联查失败

处理原则：

- release 缺失时直接失败
- 某类资源为空时继续跑，但在输出中记录
- 弱 item 切分文本只做 unigram，不做 phrase 统计

## Testing Strategy

至少覆盖：

- `recruitment_jobs_normalized` 新字段建表 / upsert
- 三家 `sample` 写入统一层
- `analysis_lexicon` schema 与 4 张表建表
- 当前 release 读取逻辑
- 空资源类别时不中断分析
- 第一阶段输出目录与文件命名
- 高频表最小列集

## Success Criteria

完成后应满足：

- 三家 `sample` 能完整补入 `public.recruitment_jobs_normalized`
- 统一层带有 raw 结构化维度，可直接支撑 requirement text 分层分析
- `analysis_lexicon` 成为第一阶段唯一正式词汇资源 source of truth
- 第一阶段可在 PostgreSQL 公共层之上直接产出稳定 CSV/TXT 结果
- 对 4 张职业词典表给出明确保留策略，而不是简单粗暴只留一张
