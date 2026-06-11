# Analysis CLI and Markdown Reports Design

**Goal**

Unify the active `src/analysis` workflows behind one preferred CLI entrypoint and make human-readable report deliverables Markdown-first. The change should reduce missed manual steps while preserving the existing statistical logic and CSV / Excel outputs.

## Scope

This design covers the active analysis workflows under `src/analysis` and their direct references:

- Structured statistics workflow based on `output/integrated` and `output/reports`
- Requirement text constraint analysis based on PostgreSQL inputs
- Human-readable report file naming and formatting
- Excel summary ingestion of report summaries
- `src/analysis/README.md` usage guidance

This design does not cover:

- `archive/` historical scripts
- `src/penghui` report files
- Non-report `.txt` assets such as dictionaries, manifests, annotation tasks, or word lists
- Rewriting the statistical methods or changing existing CSV schemas

## Current-State Findings

The structured statistics workflow is currently documented as a manual run order:

1. `src.data_pipeline.occupation_integration`
2. `src.analysis.occupation_salary_analysis`
3. `src.analysis.education_distribution_analysis`
4. `src.analysis.industry_trend_analysis`
5. `src.analysis.generate_standardized_tables`
6. `src.analysis.generate_excel_summary`

Each script can still run independently, but the workflow has no single canonical command. That makes repeat runs easy to partially execute, especially when optional standardized tables or Excel output are needed.

The requirement text workflow already has a CLI-style module entrypoint, but it writes `report.txt`. Tests and README content currently reference that TXT filename.

Several structured analysis scripts write human-readable reports as `.txt`, while `generate_excel_summary.py` reads those `.txt` names explicitly. This conflicts with the new project convention that report-class documents should be Markdown.

## Recommended Approach

Add a lightweight `src.analysis.cli` module that orchestrates existing scripts without rewriting their internals.

Recommended commands:

```bash
python -m src.analysis.cli structured
python -m src.analysis.cli structured --with-integration --with-excel
python -m src.analysis.cli requirements
```

The single CLI should become the preferred README entrypoint. Existing per-script module commands remain supported as lower-level debugging or partial rerun entrypoints.

## Alternatives Considered

### Option A: Only rename TXT reports to Markdown

This is low risk, but it leaves the structured workflow as a manual checklist. It solves the document-format issue but not the repeatability issue.

### Option B: Add a thin orchestration CLI and rename reports to Markdown

This is the recommended path. It improves repeatability while keeping analysis logic stable. The CLI can call existing classes and functions, so the implementation risk stays bounded.

### Option C: Extract shared libraries and refactor the full analysis workflow

This would create cleaner long-term boundaries, especially for duplicated salary parsing, but it is too broad for this change. It risks changing statistical behavior while solving an entrypoint and report-format problem.

## CLI Design

### `structured` subcommand

The `structured` command should run the active structured statistics workflow.

Default behavior:

- Run `occupation_salary_analysis`
- Run `education_distribution_analysis`
- Run `industry_trend_analysis`
- Run `generate_standardized_tables`
- Do not run integration unless requested
- Do not generate Excel unless requested

Options:

- `--with-integration`: run `src.data_pipeline.occupation_integration` first
- `--sample`: pass sample mode to integration when `--with-integration` is used
- `--with-excel`: run `generate_excel_summary` after standardized tables
- `--skip-standardized`: skip `generate_standardized_tables`

The command should fail fast if a required step raises an exception. It should log the step name before running each step so partial failures are easy to locate.

### `requirements` subcommand

The `requirements` command should call `requirement_text_analysis.analyze_requirement_texts` with the same parameters currently exposed by `requirement_text_analysis`.

Options:

- `--top-n`
- `--min-group-size`
- `--min-monthly-group-size`
- `--extractor-version`

The existing `python -m src.analysis.requirement_text_analysis` command should remain compatible and may delegate to the same parser or implementation.

## Markdown Report Design

Human-readable reports created by active `src/analysis` workflows should use `.md` filenames.

Structured workflow report outputs:

- `output/reports/职业类别薪资分析报告.md`
- `output/reports/学历需求分布分析报告.md`
- `output/reports/行业景气度分析报告.md`

Requirement text workflow report output:

- `output/reports/req_analysis_{mm-dd}/report.md`

Formatting rules:

- Use Markdown headings instead of separator-only TXT banners
- Keep the same section order and statistical content
- Keep CSV filenames unchanged
- Keep HTML visualization filenames unchanged
- Do not delete historical `.txt` files that already exist on disk

`generate_excel_summary.py` should read the Markdown report filenames. If needed for a softer migration, it may fall back to legacy `.txt` names, but Markdown should be the primary source.

## README Updates

`src/analysis/README.md` should present the unified CLI as the recommended entrypoint.

The README should still document individual scripts as debug or partial rerun commands, but they should no longer be the first-choice workflow.

The output list should reference `report.md` for requirement text analysis and Markdown filenames for structured reports.

## Error Handling

The orchestration CLI should:

- Log every step before it starts
- Stop on the first failed step
- Surface the original exception rather than hiding it behind a generic success/failure message
- Avoid deleting or cleaning output directories automatically

Markdown report generation should:

- Create parent output directories before writing
- Use UTF-8 encoding
- Leave CSV generation behavior unchanged

## Testing Strategy

Update or add tests for:

- Requirement text analysis writes `report.md`
- Requirement text output file list no longer expects `report.txt`
- Report content still includes the core Chinese section titles or summary strings
- CLI parser accepts `structured` and `requirements` subcommands

Manual smoke checks:

```bash
python -m src.analysis.cli --help
python -m src.analysis.cli structured --help
python -m src.analysis.cli requirements --help
pytest src/tests/test_requirement_text_analysis.py
```

If local data and PostgreSQL are available, run one full workflow smoke test after implementation.

## Success Criteria

The change is successful when:

- A user can run the structured statistics workflow from one preferred CLI command
- Active human-readable analysis reports are Markdown files
- Existing CSV, HTML, and Excel outputs remain available
- The Excel summary reads Markdown report summaries
- README guidance no longer requires manually chaining six commands for the normal path
- Requirement text tests pass with `report.md`
