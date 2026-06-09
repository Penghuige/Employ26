# Historical Annotation RRID Backfill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `annotations.label_studio_tasks_v2` 历史任务回填可审计的 `recruitment_record_id`。

**Architecture:** 先根据历史导出快照回放每个任务对应的样本行，再与三家 `sample` 表做唯一来源匹配，随后增量补齐 `public.recruitment_jobs_normalized` 并回填任务表。所有自动判定同时写入审计表，未命中任务保留空值。

**Tech Stack:** Python, pandas, SQLAlchemy, PostgreSQL, pytest

---

### Task 1: 实现回填脚本与纯函数

**Files:**
- Create: `src/utils/backfill_label_studio_recruitment_record_ids.py`
- Test: `src/tests/test_backfill_label_studio_recruitment_record_ids.py`

- [ ] 编写快照回放、键生成、候选裁决纯函数，并补单元测试
- [ ] 实现 PostgreSQL 读取样本表、创建审计表、补齐任务表列、写回 `recruitment_record_id`
- [ ] 实现脚本 CLI，支持 `--dry-run`

### Task 2: 同步文档

**Files:**
- Modify: `Employ26-database.md`
- Modify: `CONTEXT.md`

- [ ] 记录历史标注回填规则与新增审计表
- [ ] 补充必要术语说明

### Task 3: 运行验证

**Files:**
- Modify: `src/penghui/README.md`

- [ ] 运行目标测试
- [ ] 执行回填脚本并汇总自动命中、待复核、未命中数量
- [ ] 视结果更新实验说明中的任务身份读取口径
