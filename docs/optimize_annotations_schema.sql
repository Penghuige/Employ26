-- Employ26 annotations schema optimization.
-- Purpose:
--   1. Add primary/foreign keys and common indexes for Label Studio tables.
--   2. Add jsonb mirror columns for legacy text JSON fields.
--   3. Add standardized choice_code and timestamp mirror fields.
--   4. Add a convenience view for task + annotation querying.
--
-- This script is intended to be idempotent.

begin;

-- Core constraints for second-round Label Studio tables.
do $$
begin
    if not exists (
        select 1
        from pg_constraint
        where conname = 'label_studio_tasks_v2_pkey'
          and conrelid = 'annotations.label_studio_tasks_v2'::regclass
    ) then
        alter table annotations.label_studio_tasks_v2
        add constraint label_studio_tasks_v2_pkey primary key (id);
    end if;

    if not exists (
        select 1
        from pg_constraint
        where conname = 'label_studio_annotations_v2_pkey'
          and conrelid = 'annotations.label_studio_annotations_v2'::regclass
    ) then
        alter table annotations.label_studio_annotations_v2
        add constraint label_studio_annotations_v2_pkey primary key (task_id, annotation_id);
    end if;

    if not exists (
        select 1
        from pg_constraint
        where conname = 'label_studio_annotations_v2_task_id_fkey'
          and conrelid = 'annotations.label_studio_annotations_v2'::regclass
    ) then
        alter table annotations.label_studio_annotations_v2
        add constraint label_studio_annotations_v2_task_id_fkey
        foreign key (task_id)
        references annotations.label_studio_tasks_v2 (id);
    end if;
end $$;

-- JSONB mirror columns for current task table.
alter table annotations.label_studio_tasks_v2
add column if not exists annotations_completed_jsonb jsonb,
add column if not exists data_raw_jsonb jsonb,
add column if not exists created_at_ts timestamptz,
add column if not exists updated_at_ts timestamptz;

update annotations.label_studio_tasks_v2
set
    annotations_completed_jsonb = case
        when annotations_completed is null or annotations_completed = '' then '[]'::jsonb
        else annotations_completed::jsonb
    end,
    data_raw_jsonb = case
        when data_raw is null or data_raw = '' then '{}'::jsonb
        else data_raw::jsonb
    end,
    created_at_ts = nullif(created_at, '')::timestamptz,
    updated_at_ts = nullif(updated_at, '')::timestamptz
where
    annotations_completed_jsonb is null
    or data_raw_jsonb is null
    or created_at_ts is null
    or updated_at_ts is null;

-- Standardized annotation fields.
alter table annotations.label_studio_annotations_v2
add column if not exists choice_code text,
add column if not exists created_at_ts timestamptz;

update annotations.label_studio_annotations_v2
set
    choice_code = case
        when best_candidate = '候选A' then 'A'
        when best_candidate = '候选B' then 'B'
        when best_candidate = '候选C' then 'C'
        when best_candidate = '候选D' then 'D'
        when best_candidate = '候选E' then 'E'
        when best_candidate = '以上选项都不属于' then 'NONE'
        else null
    end,
    created_at_ts = nullif(created_at, '')::timestamptz
where choice_code is null or created_at_ts is null;

do $$
begin
    if not exists (
        select 1
        from pg_constraint
        where conname = 'label_studio_annotations_v2_choice_code_check'
          and conrelid = 'annotations.label_studio_annotations_v2'::regclass
    ) then
        alter table annotations.label_studio_annotations_v2
        add constraint label_studio_annotations_v2_choice_code_check
        check (choice_code in ('A', 'B', 'C', 'D', 'E', 'NONE'));
    end if;
end $$;

-- JSONB mirror columns for legacy task-level annotation table.
alter table annotations.label_studio_annotations
add column if not exists annotations_jsonb jsonb,
add column if not exists data_jsonb jsonb,
add column if not exists meta_jsonb jsonb,
add column if not exists drafts_jsonb jsonb,
add column if not exists predictions_jsonb jsonb,
add column if not exists comment_authors_jsonb jsonb,
add column if not exists created_at_ts timestamptz,
add column if not exists updated_at_ts timestamptz;

update annotations.label_studio_annotations
set
    annotations_jsonb = case when annotations is null or annotations = '' then '[]'::jsonb else annotations::jsonb end,
    data_jsonb = case when data is null or data = '' then '{}'::jsonb else data::jsonb end,
    meta_jsonb = case when meta is null or meta = '' then '{}'::jsonb else meta::jsonb end,
    drafts_jsonb = case when drafts is null or drafts = '' then '[]'::jsonb else drafts::jsonb end,
    predictions_jsonb = case when predictions is null or predictions = '' then '[]'::jsonb else predictions::jsonb end,
    comment_authors_jsonb = case when comment_authors is null or comment_authors = '' then '[]'::jsonb else comment_authors::jsonb end,
    created_at_ts = nullif(created_at, '')::timestamptz,
    updated_at_ts = nullif(updated_at, '')::timestamptz
where
    annotations_jsonb is null
    or data_jsonb is null
    or meta_jsonb is null
    or drafts_jsonb is null
    or predictions_jsonb is null
    or comment_authors_jsonb is null
    or created_at_ts is null
    or updated_at_ts is null;

-- Common lookup indexes.
create index if not exists idx_label_studio_tasks_v2_row_id
on annotations.label_studio_tasks_v2 (row_id);

create index if not exists idx_label_studio_tasks_v2_sample_source
on annotations.label_studio_tasks_v2 (sample_source);

create index if not exists idx_label_studio_annotations_v2_task_id
on annotations.label_studio_annotations_v2 (task_id);

create index if not exists idx_label_studio_annotations_v2_annotator_id
on annotations.label_studio_annotations_v2 (annotator_id);

create index if not exists idx_label_studio_annotations_v2_choice_code
on annotations.label_studio_annotations_v2 (choice_code);

create index if not exists idx_match_training_features_task_id
on public.match_training_features (task_id);

-- JSONB indexes. These are useful for ad-hoc JSON path and containment queries.
create index if not exists idx_label_studio_tasks_v2_data_raw_jsonb_gin
on annotations.label_studio_tasks_v2 using gin (data_raw_jsonb);

create index if not exists idx_label_studio_tasks_v2_annotations_completed_jsonb_gin
on annotations.label_studio_tasks_v2 using gin (annotations_completed_jsonb);

create index if not exists idx_label_studio_annotations_data_jsonb_gin
on annotations.label_studio_annotations using gin (data_jsonb);

create index if not exists idx_label_studio_annotations_annotations_jsonb_gin
on annotations.label_studio_annotations using gin (annotations_jsonb);

-- Convenience view: task-level fields plus one row per annotation.
create or replace view annotations.v_label_studio_task_annotations_v2 as
select
    t.id as task_id,
    t.row_id,
    t.sample_source,
    t.job_title,
    t.job_requirements,
    t.is_validation,
    t.cand_a_code,
    t.cand_a_title,
    t.cand_a_source,
    t.cand_b_code,
    t.cand_b_title,
    t.cand_b_source,
    t.cand_c_code,
    t.cand_c_title,
    t.cand_c_source,
    t.cand_d_code,
    t.cand_d_title,
    t.cand_d_source,
    t.cand_e_code,
    t.cand_e_title,
    t.cand_e_source,
    t.data_raw_jsonb,
    a.annotation_id,
    a.annotator_id,
    a.lead_time_sec,
    a.best_candidate,
    a.choice_code,
    a.soft_skill,
    a.reason,
    a.created_at_ts as annotation_created_at
from annotations.label_studio_tasks_v2 t
left join annotations.label_studio_annotations_v2 a
  on a.task_id = t.id;

-- Documentation comments for database tools.
comment on table annotations.label_studio_tasks_v2 is
'第二轮 Label Studio 标注任务主表；一行对应一个任务。';

comment on table annotations.label_studio_annotations_v2 is
'第二轮 Label Studio 扁平标注明细表；一行对应一个人工 annotation。';

comment on table annotations.label_studio_annotations is
'旧版 Label Studio 标注任务归档表；保留原始 JSON 风格字段。';

comment on table annotations.deepseek_relabel_raw is
'DeepSeek 对 Label Studio 任务的重标结果；task_id 为主键。';

comment on view annotations.v_label_studio_task_annotations_v2 is
'任务表和标注明细表的便捷查询视图，不改变底层一对多结构。';

comment on column annotations.label_studio_tasks_v2.data_raw_jsonb is
'data_raw 的 jsonb 镜像字段，用于数据库侧 JSON 查询和索引。';

comment on column annotations.label_studio_tasks_v2.annotations_completed_jsonb is
'annotations_completed 的 jsonb 镜像字段，用于数据库侧 JSON 查询和索引。';

comment on column annotations.label_studio_annotations_v2.choice_code is
'标准化候选选择结果，取值 A/B/C/D/E/NONE。';

commit;
