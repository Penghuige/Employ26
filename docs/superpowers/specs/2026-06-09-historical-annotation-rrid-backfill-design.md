# 历史标注任务 `recruitment_record_id` 回填设计

## 目标

仅为历史 `annotations.label_studio_tasks_v2` 任务回填正式业务身份字段 `recruitment_record_id`，不改动任务文本、候选、标注结果等原始语义字段。

## 已确认事实

- 历史任务 JSON 与 `annotations.label_studio_tasks_v2` 当前都不自带 `recruitment_record_id`
- 历史 `row_id` 不是招聘源表主键，而是 `export_tier1_label_studio.py` 导出时生成的快照行号
- 导出脚本将 `Tier2_Matched_Data.csv` 的前 30 行与 `Tier3_Pending_Data.csv` 全量拼接后重置索引，再写入 Label Studio `data.row_id`
- 当前 PostgreSQL 中 `public.recruitment_jobs_normalized` 尚未落表，因此回填必须同时补齐规范层中的被引用记录

## 一级证据链

一级证据链采用“导出快照行号回放”：

1. 读取 `annotations.label_studio_tasks_v2.row_id`
2. 依据导出脚本语义回放到历史快照行：
   - `row_id < 30` 对应 `output/data5/Tier2_Matched_Data.csv` 的第 `row_id` 行
   - `row_id >= 30` 对应 `output/data5/Tier3_Pending_Data.csv` 的第 `row_id - 30` 行
3. 从快照行提取招聘源字段，与三家 `sample` 表逐条匹配

这条链路可复演、可审计，不依赖错误的 `annotations.label_studio_tasks_v2.row_id -> public.jd_raw.row_id` 历史假关联。

## 自动回填规则

### Rule 1: `exact_full_row_unique`

使用招聘源的 10 个基础字段构造严格归一化键：

- `发布时间`
- `岗位名称`
- `工作城市`
- `薪资水平`
- `经验要求`
- `学历要求`
- `岗位描述`
- `公司名称`
- `公司规模`
- `公司行业`

若快照行在三家 `sample` 表中命中且仅命中 1 条，则：

- 认定来源记录唯一
- 生成或命中该来源记录的 `recruitment_record_id`
- 自动回填

### Rule 2: `normalized_full_row_unique`

若严格键未命中，则使用更强归一化后的文本键再次匹配：

- 去空白差异
- 统一常见中英文标点
- 统一大小写

若仍能唯一命中 1 条，则自动回填，但置信等级低于 Rule 1。

### Rule 2.5: `exact_duplicate_rows_same_source_table`

若严格键命中多个候选，但这些候选：

- 全部来自同一 `source_table`
- 10 个基础字段完全一致

则将其视为同一招聘记录在同一采样表中的重复行，自动选择最小 `source_row_number` 作为规范层来源行，并收敛到同一个 `recruitment_record_id`。

### Rule 3: `strong_text_similarity_unique`

若前两条规则未命中，则只在候选收窄后启用强文本相似自动绑定：

- 先按 `岗位名称`、`公司名称` 的强归一化值筛候选
- 再用 `岗位描述` 的相似度做排序
- 仅当第一名相似度足够高，且明显高于第二名时，才允许自动绑定

否则不自动回填。

## 不自动回填的情形

以下情形进入人工复核或保持空值：

- 找不到候选来源记录
- 命中多个等价候选，无法稳定区分来源
- 相似度虽高但与第二名差距不足
- 快照 `row_id` 超出历史导出范围

## 审计产物

本次回填除更新任务表外，还必须落一张审计表，至少记录：

- `task_id`
- `historical_row_id`
- `mapping_status`
- `mapping_rule`
- `confidence_tier`
- `source_table`
- `source_row_number`
- `recruitment_record_id`
- `candidate_count`
- `best_similarity_score`
- `second_similarity_score`
- `evidence_summary`
- `backfill_version`
- `backfilled_at`

## 写入策略

- 在 `annotations.label_studio_tasks_v2` 上新增 `recruitment_record_id` 列
- 仅更新本次规则判定为可自动回填的任务
- 未命中与待复核任务保留空值
- 审计表每次允许按 `task_id` 覆盖，确保同一版本可重跑

## 结果预期

- 历史任务获得正式 `recruitment_record_id`
- 回填来源与规则可复演、可解释
- 后续 `src/penghui` 等读取任务表时，可直接消费正式业务身份字段
