# Recruitment Record ID Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立 `recruitment_record_id` 统一身份主轴，并完成第一批活跃公共链路迁移。

**Architecture:** 先新增统一规范层 `public.recruitment_jobs_normalized` 与身份分配逻辑，再把 `public.job_description_parsed` 和 `public.skill_extraction_requirement_matches` 的写入/读取契约改成 `recruitment_record_id`。最后收口直接依赖这些公共结果的活跃读取逻辑与文档。

**Tech Stack:** Python 3.10+, pandas, SQLAlchemy, PostgreSQL, pytest

---

### Task 1: Add Recruitment Normalized Table Support

**Files:**
- Create: `src/db/recruitment_jobs_normalized.py`
- Modify: `config/database.yaml`
- Test: `src/tests/test_recruitment_jobs_normalized.py`

- [ ] Add `public.recruitment_jobs_normalized` to table config.
- [ ] Implement PostgreSQL table creation and upsert helpers with `recruitment_record_id`, `source_native_job_id`, and `dedupe_fingerprint`.
- [ ] Add tests covering first insert and frozen-ID upsert behavior.

### Task 2: Migrate Job Description Parsed Table Identity

**Files:**
- Modify: `src/db/job_description_parsed.py`
- Modify: `src/data_pipeline/description_schema.py`
- Modify: `src/data_pipeline/description_parsing.py`
- Test: `src/tests/test_description_parsing.py`

- [ ] Replace `source_record_id`-based row construction with `recruitment_record_id`.
- [ ] Change table schema and upsert key to use `recruitment_record_id`.
- [ ] Keep `source_platform/source_table/source_row_number` only as trace fields.
- [ ] Update tests to assert the new write contract.

### Task 3: Migrate Requirement Match Result Contract

**Files:**
- Modify: `src/data_pipeline/requirement_match_prep.py`
- Modify: `src/skill_extraction/config.py`
- Modify: `src/utils/llm_labeling_utils.py`
- Test: `src/tests/test_requirement_match_prep.py`

- [ ] Stop propagating `sample_row_id` in requirement match public outputs.
- [ ] Require and emit `recruitment_record_id` in requirement match rows.
- [ ] Update loading utilities to read `recruitment_record_id` and trace fields.
- [ ] Add focused tests for the new result schema.

### Task 4: Update Active Consumers and Docs

**Files:**
- Modify: `src/skill_extraction/occupation_skill_pipeline.py`
- Modify: `Employ26-database.md`
- Modify: `src/penghui/README.md` only if identity wording appears

- [ ] Update active consumer code paths to assume `recruitment_record_id`.
- [ ] Remove first-batch references that still document `sample_row_id` or `source_record_id` as the standard identifier.
- [ ] Refresh database documentation for `job_description_parsed` and `skill_extraction_requirement_matches`.

### Task 5: Verify the Migration Surface

**Files:**
- No new files required

- [ ] Run `python -m compileall -q src`.
- [ ] Run targeted `pytest` for the touched tests.
- [ ] Summarize remaining unmigrated identity usage outside the first-batch scope.
