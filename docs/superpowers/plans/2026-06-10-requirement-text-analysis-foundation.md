# Requirement Text Analysis Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 扩展统一规范层、落地 `analysis_lexicon` schema、补齐三家 `sample` 到 `recruitment_jobs_normalized`，并建立第一阶段 requirement text 统计脚手架。

**Architecture:** 先扩展 `src.db.recruitment_jobs_normalized` 的表结构和写入行模型，再补一条正式的 sample backfill 入口。随后新增 `analysis_lexicon` 数据库访问层与 requirement text 统计脚本，直接消费 `public.recruitment_jobs_normalized` 与 `public.job_description_parsed`。职业词典 4 表的比较结果只落文档与配置建议，不在本轮做物理删表。

**Tech Stack:** Python 3.10+, pandas, SQLAlchemy, PostgreSQL, pytest

---

### Task 1: Expand `recruitment_jobs_normalized`

**Files:**
- Modify: `src/db/recruitment_jobs_normalized.py`
- Test: `src/tests/test_recruitment_jobs_normalized.py`

- [ ] Add raw structured fields to `RecruitmentNormalizedRow` and table DDL.
- [ ] Extend row builders to map `薪资水平` / `学历要求` / `经验要求` / `公司规模` / `公司行业`.
- [ ] Keep `*_raw` naming in the public table contract.
- [ ] Add tests covering the expanded row payload and stable fingerprint behavior.

### Task 2: Add sample backfill entrypoint

**Files:**
- Create: `src/data_pipeline/backfill_recruitment_jobs_normalized.py`
- Modify: `config/database.yaml`
- Test: `src/tests/test_recruitment_jobs_normalized.py`

- [ ] Implement a CLI that reads `"51job".sample`, `"Liepin".sample`, `"Zhilian".sample`.
- [ ] Convert rows with `build_normalized_rows_from_dataframe()`.
- [ ] Upsert into `public.recruitment_jobs_normalized`.
- [ ] Reuse existing DB helpers instead of duplicating insert logic.

### Task 3: Add `analysis_lexicon` schema support

**Files:**
- Create: `src/db/analysis_lexicon.py`
- Create: `src/tests/test_analysis_lexicon.py`

- [ ] Add DDL helpers for `analysis_lexicon.lexicon_release`.
- [ ] Add DDL helpers for `analysis_lexicon.user_dictionary`.
- [ ] Add DDL helpers for `analysis_lexicon.stopwords`.
- [ ] Add DDL helpers for `analysis_lexicon.phrase_rules`.
- [ ] Enforce the confirmed uniqueness and check-constraint strategy.

### Task 4: Build lexicon runtime readers

**Files:**
- Modify: `src/db/analysis_lexicon.py`
- Create: `src/tests/test_analysis_lexicon.py`

- [ ] Implement readers that load only `is_current = true`.
- [ ] Return empty categories gracefully while exposing diagnostics.
- [ ] Add summary helpers for term type/category/scope/enabled counts.
- [ ] Add snapshot export helper as optional utility, without auto-run side effects.

### Task 5: Create requirement text analysis scaffold

**Files:**
- Create: `src/analysis/requirement_text_analysis.py`
- Modify: `src/analysis/README.md`
- Create: `src/tests/test_requirement_text_analysis.py`

- [ ] Load `public.recruitment_jobs_normalized` + `public.job_description_parsed`.
- [ ] Restrict main sample to non-empty `requirements_text`.
- [ ] Build unigram counts on record presence.
- [ ] Build bigram/trigram only inside requirement items.
- [ ] Wire in lexicon resources from `analysis_lexicon`.
- [ ] Output CSV/TXT artifacts into `output/reports/req_analysis_{mm-dd}/`.

### Task 6: Emit diagnostics and fixed output contracts

**Files:**
- Modify: `src/analysis/requirement_text_analysis.py`
- Create: `src/tests/test_requirement_text_analysis.py`

- [ ] Add `run_manifest.json`.
- [ ] Add coverage/fallback diagnostics.
- [ ] Add deduplicated-text vs record-weighted comparison.
- [ ] Add `noise_terms_filtered.csv`.
- [ ] Fix the minimal CSV column sets and TXT section order.

### Task 7: Document occupation dictionary retention recommendation

**Files:**
- Modify: `src/analysis/README.md`
- Modify: `Employ26-database.md`

- [ ] Record that `occ_dict` / `occ_dict_detailed` / `occ_dict_pro` are layered variants.
- [ ] Record that `occ_dict_class` is a broader classification backbone, not a drop-in duplicate.
- [ ] Recommend operational primary tables without dropping physical tables in this task.

### Task 8: Verify and summarize

**Files:**
- No new files required

- [ ] Run `python -m compileall -q src`.
- [ ] Run targeted pytest files for normalized jobs, analysis lexicon, and requirement text analysis.
- [ ] Re-run focused PostgreSQL queries for sample coverage and dictionary table comparison.
- [ ] Summarize implemented changes, remaining risks, and follow-up cleanup candidates.
