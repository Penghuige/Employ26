# Analysis CLI Markdown Reports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add one preferred `src.analysis.cli` entrypoint and make active `src/analysis` human-readable reports Markdown-first.

**Architecture:** Keep existing analysis classes and functions as the implementation units. Add a thin orchestration CLI that calls those units, update report writers to emit `.md`, and update direct consumers/tests/docs to use Markdown names.

**Tech Stack:** Python `argparse`, `pathlib`, `logging`, pandas, pytest, existing `src/analysis` modules.

---

## File Structure

- Create: `src/analysis/cli.py` as the unified orchestration CLI.
- Modify: `src/analysis/requirement_text_analysis.py` to write `report.md` and expose shared CLI args cleanly.
- Modify: `src/analysis/occupation_salary_analysis.py`, `src/analysis/education_distribution_analysis.py`, `src/analysis/industry_trend_analysis.py` to write Markdown report filenames and headings.
- Modify: `src/analysis/generate_excel_summary.py` to prefer Markdown report summaries with legacy TXT fallback.
- Modify: `src/analysis/README.md` to document the unified CLI and Markdown outputs.
- Modify: `CONTEXT.md` to keep the glossary aligned on Markdown report deliverables.
- Modify: `src/tests/test_requirement_text_analysis.py` and create `src/tests/test_analysis_cli.py`.

### Task 1: Requirement Report Markdown

**Files:**
- Modify: `src/analysis/requirement_text_analysis.py`
- Modify: `src/tests/test_requirement_text_analysis.py`

- [x] **Step 1: Update the existing test expectation**

Change the output filename tuple in `src/tests/test_requirement_text_analysis.py` so it includes `"report.md"` instead of `"report.txt"`, and read `tmp_path / "report.md"` for content assertions.

- [x] **Step 2: Run the focused test and verify failure**

Run: `pytest src/tests/test_requirement_text_analysis.py -q`

Expected before implementation: failure because `report.md` is not created.

- [x] **Step 3: Write Markdown report output**

In `src/analysis/requirement_text_analysis.py`, change the final report write from:

```python
(output_dir / "report.txt").write_text(...)
```

to:

```python
(output_dir / "report.md").write_text(...)
```

Keep `_build_report_text` content stable unless a small heading marker improves Markdown readability without changing assertions.

- [x] **Step 4: Run the focused test and verify pass**

Run: `pytest src/tests/test_requirement_text_analysis.py -q`

Expected: pass.

### Task 2: Unified Analysis CLI

**Files:**
- Create: `src/analysis/cli.py`
- Create: `src/tests/test_analysis_cli.py`

- [x] **Step 1: Add parser tests**

Create tests that import `src.analysis.cli.build_parser` and assert these parse successfully:

```python
parser.parse_args(["structured"])
parser.parse_args(["structured", "--with-integration", "--sample", "--with-excel"])
parser.parse_args(["requirements", "--top-n", "5", "--min-group-size", "2"])
```

- [x] **Step 2: Run parser tests and verify failure**

Run: `pytest src/tests/test_analysis_cli.py -q`

Expected before implementation: failure because `src.analysis.cli` does not exist.

- [x] **Step 3: Implement `src.analysis.cli`**

Add:

- `build_parser() -> argparse.ArgumentParser`
- `run_structured(args: argparse.Namespace) -> None`
- `run_requirements(args: argparse.Namespace) -> None`
- `main() -> None`

The structured command should call existing classes:

```python
OccupationSalaryAnalyzer().run()
EducationDistributionAnalyzer().run()
IndustryTrendAnalyzer().run()
StandardizedTableGenerator().generate_all()
ExcelReportGenerator().create_summary_report()
```

Only call `DataIntegrator(use_full_data=not args.sample).integrate_all()` when `--with-integration` is present. Only call Excel when `--with-excel` is present. Skip standardized tables when `--skip-standardized` is present.

The requirements command should call:

```python
analyze_requirement_texts(
    output_dir=build_current_output_dir(),
    params=AnalysisParams(
        top_n=args.top_n,
        min_group_size=args.min_group_size,
        min_monthly_group_size=args.min_monthly_group_size,
        extractor_version=args.extractor_version.strip() or DEFAULT_EXTRACTOR_VERSION,
    ),
)
```

- [x] **Step 4: Run parser tests and CLI help smoke checks**

Run:

```bash
pytest src/tests/test_analysis_cli.py -q
python -m src.analysis.cli --help
python -m src.analysis.cli structured --help
python -m src.analysis.cli requirements --help
```

Expected: tests pass and help commands exit with code 0.

### Task 3: Structured Report Markdown

**Files:**
- Modify: `src/analysis/occupation_salary_analysis.py`
- Modify: `src/analysis/education_distribution_analysis.py`
- Modify: `src/analysis/industry_trend_analysis.py`

- [x] **Step 1: Rename structured report outputs**

Change report paths to:

```python
self.output_dir / "职业类别薪资分析报告.md"
self.output_dir / "学历需求分布分析报告.md"
self.output_dir / "行业景气度分析报告.md"
```

- [x] **Step 2: Convert plain banners to Markdown headings**

Use top-level headings like:

```python
f.write("# 广东省招聘数据 - 职业类别薪资分析报告\n\n")
f.write("## 一、职业类别薪资统计\n\n")
```

Preserve existing statistics and loop logic.

- [x] **Step 3: Update log output strings**

Replace logger references to `.txt` report files with `.md` names.

- [x] **Step 4: Run syntax check**

Run: `python -m compileall src/analysis`

Expected: all modified analysis files compile.

### Task 4: Excel Summary Markdown Ingestion

**Files:**
- Modify: `src/analysis/generate_excel_summary.py`

- [x] **Step 1: Update docstring and report list**

Change docstring references from `*.txt` reports to Markdown reports. Update primary report list to:

```python
[
    ("职业类别薪资分析报告.md", "职业薪资分析"),
    ("学历需求分布分析报告.md", "学历需求分布"),
    ("行业景气度分析报告.md", "行业景气度"),
]
```

- [x] **Step 2: Add legacy fallback resolution**

Add a helper inside `ExcelReportGenerator`:

```python
def _resolve_report_file(self, markdown_filename: str) -> Path | None:
    markdown_path = self.reports_dir / markdown_filename
    if markdown_path.exists():
        return markdown_path
    legacy_path = markdown_path.with_suffix(".txt")
    if legacy_path.exists():
        return legacy_path
    return None
```

Use it in `_add_text_summary`.

- [x] **Step 3: Run syntax check**

Run: `python -m compileall src/analysis/generate_excel_summary.py`

Expected: compile succeeds.

### Task 5: README and Context Documentation

**Files:**
- Modify: `src/analysis/README.md`
- Modify: `CONTEXT.md`

- [x] **Step 1: Update README recommended entrypoints**

Add preferred commands:

```bash
python -m src.analysis.cli structured --with-integration --with-excel
python -m src.analysis.cli requirements
```

Move individual script commands under a debug/partial rerun section.

- [x] **Step 2: Update output lists**

Replace `report.txt` with `report.md`. Add structured Markdown report filenames.

- [x] **Step 3: Confirm context terminology**

Ensure `CONTEXT.md` contains `CSV-Plus-Markdown Deliverables` and `Fixed Markdown Report Order`, with no active `CSV-Plus-TXT Deliverables` or `Fixed TXT Report Order` terms.

- [x] **Step 4: Search for active stale report references**

Run: `rg -n "report\\.txt|报告\\.txt|CSV-Plus-TXT|Fixed TXT" src/analysis src/tests CONTEXT.md`

Expected: no active `src/analysis` or test references to report-class TXT outputs.

### Task 6: Final Verification

**Files:**
- Verify all modified files.

- [x] **Step 1: Run focused tests**

Run:

```bash
pytest src/tests/test_requirement_text_analysis.py src/tests/test_analysis_cli.py -q
```

Expected: pass.

- [x] **Step 2: Run CLI help smoke checks**

Run:

```bash
python -m src.analysis.cli --help
python -m src.analysis.cli structured --help
python -m src.analysis.cli requirements --help
```

Expected: all exit with code 0.

- [x] **Step 3: Review diff for unrelated changes**

Run:

```bash
git diff -- src/analysis src/tests/test_requirement_text_analysis.py src/tests/test_analysis_cli.py CONTEXT.md docs/superpowers/plans/2026-06-11-analysis-cli-markdown-reports.md
```

Expected: diff only includes this implementation and previously staged Markdown context correction.
