# Recruitment Normalized Backfill Concurrency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add chunked concurrent SQL backfill, resumable chunk logging, and pooled database access for `recruitment_jobs_normalized` so 10M+ row runs can be tuned and retried safely.

**Architecture:** Keep heavy read and write work inside PostgreSQL. Python only plans chunks, schedules them to worker threads, records run metrics, and exposes CLI controls. The database layer owns locator snapshots, chunk-status persistence, and pooled engine configuration.

**Tech Stack:** Python 3.10+, SQLAlchemy, PostgreSQL, pytest

---

### Task 1: Add pooled PostgreSQL engine options

**Files:**
- Modify: `src/db/postgres.py`
- Test: `src/tests/test_backfill_recruitment_jobs_normalized.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_pg_engine_options_include_pooling_defaults():
    options = build_pg_engine_options(
        pool_size=4,
        max_overflow=2,
        pool_recycle=1800,
        pool_pre_ping=True,
        application_name="backfill-test",
    )
    assert options["pool_size"] == 4
    assert options["max_overflow"] == 2
    assert options["pool_recycle"] == 1800
    assert options["pool_pre_ping"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/tests/test_backfill_recruitment_jobs_normalized.py -k pooling -v`
Expected: FAIL because `build_pg_engine_options` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def build_pg_engine_options(...):
    return {
        "future": True,
        "pool_size": pool_size,
        "max_overflow": max_overflow,
        "pool_recycle": pool_recycle,
        "pool_pre_ping": pool_pre_ping,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/tests/test_backfill_recruitment_jobs_normalized.py -k pooling -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/db/postgres.py src/tests/test_backfill_recruitment_jobs_normalized.py
git commit -m "feat: add pooled postgres engine options"
```

### Task 2: Add chunk planning and benchmark summarization helpers

**Files:**
- Modify: `src/data_pipeline/backfill_recruitment_jobs_normalized.py`
- Create: `src/tests/test_backfill_recruitment_jobs_normalized.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_plan_chunks_splits_ranges_deterministically():
    chunks = plan_chunks(total_rows=5, chunk_size=2, limit_rows=None)
    assert [(chunk.range_start, chunk.range_end) for chunk in chunks] == [(1, 2), (3, 4), (5, 5)]

def test_build_run_summary_computes_rows_per_second():
    summary = build_run_summary(
        chunk_results=[
            ChunkResult("run-1", '"51job".sample', 1, 1, 2, 2, 2, 2.0, 1, "succeeded", ""),
            ChunkResult("run-1", '"51job".sample', 2, 3, 4, 2, 2, 1.0, 1, "succeeded", ""),
        ]
    )
    assert summary["written_rows"] == 4
    assert summary["rows_per_second"] == 4 / 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest src/tests/test_backfill_recruitment_jobs_normalized.py -k "plan_chunks or build_run_summary" -v`
Expected: FAIL because the helper types and functions do not exist.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(frozen=True)
class ChunkPlan:
    ...

def plan_chunks(...):
    ...

def build_run_summary(...):
    ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest src/tests/test_backfill_recruitment_jobs_normalized.py -k "plan_chunks or build_run_summary" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_pipeline/backfill_recruitment_jobs_normalized.py src/tests/test_backfill_recruitment_jobs_normalized.py
git commit -m "feat: add chunk planning and benchmark summaries"
```

### Task 3: Add SQL locator, chunk execution, and resumable run-state persistence

**Files:**
- Modify: `src/data_pipeline/backfill_recruitment_jobs_normalized.py`
- Modify: `src/db/recruitment_jobs_normalized.py`
- Test: `src/tests/test_backfill_recruitment_jobs_normalized.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_build_chunk_insert_sql_uses_conflict_do_nothing_for_missing_only():
    sql = build_chunk_insert_sql(
        locator_table="public.recruitment_jobs_normalized_locator",
        normalized_table="public.recruitment_jobs_normalized",
        source_table='"51job".sample',
        only_missing=True,
    )
    assert "ON CONFLICT (source_table, source_row_number) DO NOTHING" in sql

def test_build_chunk_insert_sql_uses_conditional_update_for_full_refresh():
    sql = build_chunk_insert_sql(
        locator_table="public.recruitment_jobs_normalized_locator",
        normalized_table="public.recruitment_jobs_normalized",
        source_table='"51job".sample',
        only_missing=False,
    )
    assert "IS DISTINCT FROM" in sql
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest src/tests/test_backfill_recruitment_jobs_normalized.py -k build_chunk_insert_sql -v`
Expected: FAIL because the SQL builder does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
def ensure_backfill_runtime_tables(...):
    ...

def build_chunk_insert_sql(...):
    ...

def execute_backfill_chunk(...):
    ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest src/tests/test_backfill_recruitment_jobs_normalized.py -k build_chunk_insert_sql -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_pipeline/backfill_recruitment_jobs_normalized.py src/db/recruitment_jobs_normalized.py src/tests/test_backfill_recruitment_jobs_normalized.py
git commit -m "feat: add resumable sql chunk backfill runtime"
```

### Task 4: Wire CLI concurrency, benchmark output, and smoke validation

**Files:**
- Modify: `src/data_pipeline/backfill_recruitment_jobs_normalized.py`
- Test: `src/tests/test_backfill_recruitment_jobs_normalized.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_build_parser_exposes_concurrency_flags():
    parser = build_parser()
    args = parser.parse_args(["--workers", "4", "--chunk-size", "200000", "--benchmark"])
    assert args.workers == 4
    assert args.chunk_size == 200000
    assert args.benchmark is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest src/tests/test_backfill_recruitment_jobs_normalized.py -k concurrency_flags -v`
Expected: FAIL because the CLI flags do not exist.

- [ ] **Step 3: Write minimal implementation**

```python
parser.add_argument("--workers", type=int, default=3)
parser.add_argument("--chunk-size", type=int, default=200000)
parser.add_argument("--benchmark", action="store_true")
```

- [ ] **Step 4: Run targeted tests and compile**

Run: `pytest src/tests/test_backfill_recruitment_jobs_normalized.py -v`
Expected: PASS

Run: `python -m compileall -q src`
Expected: PASS with no syntax errors

- [ ] **Step 5: Commit**

```bash
git add src/data_pipeline/backfill_recruitment_jobs_normalized.py src/tests/test_backfill_recruitment_jobs_normalized.py
git commit -m "feat: add concurrent backfill cli and benchmark logging"
```
