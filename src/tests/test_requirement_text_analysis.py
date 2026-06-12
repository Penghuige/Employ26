from pathlib import Path

import pandas as pd

from src.analysis import requirement_text_analysis as rta


def test_build_requirement_analysis_query_falls_back_to_legacy_source_locator():
    query = rta.build_requirement_analysis_query(
        parsed_columns={
            "__source_table",
            "__source_row_number",
            "requirements_text",
            "parser_version",
            "parsed_at",
        }
    )

    assert "p.recruitment_record_id" not in query
    assert 'p.parsed_source_table = n.source_table' in query
    assert 'p.parsed_source_row_number = n.source_row_number' in query


def test_analyze_requirement_texts_writes_phase2_outputs(monkeypatch, tmp_path: Path):
    resources = {
        "release": {"version": "v_test_phase2"},
        "user_dictionary": pd.DataFrame(columns=["term_type", "category", "enabled"]),
        "stopwords": pd.DataFrame(columns=["scope", "stop_strength", "enabled"]),
        "phrase_rules": pd.DataFrame(columns=["rule_type", "source", "enabled"]),
        "requirement_rules": pd.DataFrame(
            [
                {
                    "id": 1,
                    "rule_type": "extract",
                    "dimension_name": "certificate",
                    "pattern_text": "驾驶证",
                    "replacement_text": "",
                    "normalized_value": "驾驶证",
                    "operator": "contains",
                    "priority": 10,
                    "enabled": True,
                    "source": "manual",
                    "notes": "",
                },
                {
                    "id": 2,
                    "rule_type": "extract",
                    "dimension_name": "language",
                    "pattern_text": "英语四级",
                    "replacement_text": "",
                    "normalized_value": "英语四级",
                    "operator": "contains",
                    "priority": 10,
                    "enabled": True,
                    "source": "manual",
                    "notes": "",
                },
                {
                    "id": 3,
                    "rule_type": "extract",
                    "dimension_name": "travel",
                    "pattern_text": "出差",
                    "replacement_text": "",
                    "normalized_value": "接受出差",
                    "operator": "allow",
                    "priority": 10,
                    "enabled": True,
                    "source": "manual",
                    "notes": "",
                },
                {
                    "id": 4,
                    "rule_type": "extract",
                    "dimension_name": "shift",
                    "pattern_text": "夜班",
                    "replacement_text": "",
                    "normalized_value": "接受夜班",
                    "operator": "allow",
                    "priority": 10,
                    "enabled": True,
                    "source": "manual",
                    "notes": "",
                },
                {
                    "id": 5,
                    "rule_type": "extract",
                    "dimension_name": "work_condition",
                    "pattern_text": "加班",
                    "replacement_text": "",
                    "normalized_value": "接受加班",
                    "operator": "allow",
                    "priority": 10,
                    "enabled": True,
                    "source": "manual",
                    "notes": "",
                },
                {
                    "id": 6,
                    "rule_type": "extract",
                    "dimension_name": "physical_condition",
                    "pattern_text": "身体健康",
                    "replacement_text": "",
                    "normalized_value": "身体健康",
                    "operator": "allow",
                    "priority": 10,
                    "enabled": True,
                    "source": "manual",
                    "notes": "",
                },
                {
                    "id": 7,
                    "rule_type": "template_noise",
                    "dimension_name": "",
                    "pattern_text": "责任心强",
                    "replacement_text": "",
                    "normalized_value": "责任心强",
                    "operator": "exclude",
                    "priority": 10,
                    "enabled": True,
                    "source": "manual",
                    "notes": "template phrase",
                },
            ]
        ),
    }
    source_df = pd.DataFrame(
        [
            {
                "recruitment_record_id": "r1",
                "source_platform": "51job",
                "source_table": '"51job".sample',
                "source_row_number": 1,
                "source_native_job_id": "a1",
                "job_title": "数据分析师",
                "work_city": "广州",
                "company_name": "A公司",
                "publish_date": "2026-06-01",
                "salary_raw": "10-20K",
                "education_requirement_raw": "本科",
                "experience_requirement_raw": "3年",
                "company_size_raw": "100-499人",
                "company_industry_raw": "互联网,AI",
                "requirements_text": "本科以上 | 3-5年经验 | 20-35岁 | 男女不限 | 驾驶证 | 英语四级 | 接受出差夜班加班 | 身体健康 | 责任心强",
                "duties_text": "",
                "sections_brief": "",
                "parser_version": "description_parsing_v11",
                "parsed_at": "2026-06-09 00:00:00",
            },
            {
                "recruitment_record_id": "r2",
                "source_platform": "Liepin",
                "source_table": '"Liepin".sample',
                "source_row_number": 2,
                "source_native_job_id": "b2",
                "job_title": "招商主管",
                "work_city": "深圳",
                "company_name": "B公司",
                "publish_date": "2026-06-02",
                "salary_raw": "20-30K",
                "education_requirement_raw": "大专",
                "experience_requirement_raw": "5年",
                "company_size_raw": "100-499人",
                "company_industry_raw": "地产",
                "requirements_text": "大专以上; 5年以上经验; 35岁以下; 仅限女性; 接受出差",
                "duties_text": "",
                "sections_brief": "",
                "parser_version": "description_parsing_v11",
                "parsed_at": "2026-06-09 00:00:00",
            },
            {
                "recruitment_record_id": "r3",
                "source_platform": "Zhilian",
                "source_table": '"Zhilian".sample',
                "source_row_number": 3,
                "source_native_job_id": "c3",
                "job_title": "销售顾问",
                "work_city": "东莞",
                "company_name": "C公司",
                "publish_date": "2026-06-03",
                "salary_raw": "8-12K",
                "education_requirement_raw": "大专",
                "experience_requirement_raw": "1年",
                "company_size_raw": "20-99人",
                "company_industry_raw": "消费品",
                "requirements_text": "责任心强，团队合作精神，服从安排",
                "duties_text": "",
                "sections_brief": "",
                "parser_version": "description_parsing_v11",
                "parsed_at": "2026-06-09 00:00:00",
            },
            {
                "recruitment_record_id": "r4",
                "source_platform": "Zhilian",
                "source_table": '"Zhilian".sample',
                "source_row_number": 4,
                "source_native_job_id": "d4",
                "job_title": "销售顾问",
                "work_city": "东莞",
                "company_name": "D公司",
                "publish_date": "2026-06-04",
                "salary_raw": "8-12K",
                "education_requirement_raw": "大专",
                "experience_requirement_raw": "1年",
                "company_size_raw": "20-99人",
                "company_industry_raw": "消费品",
                "requirements_text": "",
                "duties_text": "负责客户沟通",
                "sections_brief": "",
                "parser_version": "description_parsing_v11",
                "parsed_at": "2026-06-09 00:00:00",
            },
        ]
    )

    captured_rows = {}

    def _fake_replace(rows, *, extractor_version, table_name=rta.replace_requirement_constraint_facts.__defaults__[0] if rta.replace_requirement_constraint_facts.__defaults__ else None):
        captured_rows["rows"] = rows
        captured_rows["extractor_version"] = extractor_version
        return len(rows)

    def _fake_load(*, extractor_version="", table_name="public.requirement_constraint_facts"):
        rows = captured_rows.get("rows", [])
        return pd.DataFrame([{"fact_id": index + 1, **row.__dict__} for index, row in enumerate(rows)])

    monkeypatch.setattr(rta, "load_current_lexicon_resources", lambda: resources)
    monkeypatch.setattr(rta, "load_requirement_analysis_dataframe", lambda: source_df.copy())
    monkeypatch.setattr(rta, "replace_requirement_constraint_facts", _fake_replace)
    monkeypatch.setattr(rta, "load_requirement_constraint_facts_dataframe", _fake_load)

    result = rta.analyze_requirement_texts(
        output_dir=tmp_path,
        params=rta.AnalysisParams(top_n=20, min_group_size=1, min_monthly_group_size=1),
    )

    assert result["fact_rows"] > 0
    assert captured_rows["extractor_version"] == rta.DEFAULT_EXTRACTOR_VERSION

    for filename in (
        "run_manifest.json",
        "coverage_diagnostics.csv",
        "lexicon_summary.csv",
        "constraint_dimension_frequency.csv",
        "constraint_value_distribution.csv",
        "constraint_by_city_industry.csv",
        "template_noise_report.csv",
        "requirement_stringency_index.csv",
        "report.md",
    ):
        assert (tmp_path / filename).exists()

    diagnostics_df = pd.read_csv(tmp_path / "coverage_diagnostics.csv")
    assert int(diagnostics_df.iloc[0]["requirements_nonempty_records"]) == 3
    assert int(diagnostics_df.iloc[0]["duties_fallback_records"]) == 1
    assert int(diagnostics_df.iloc[0]["records_with_constraints"]) == 2

    dimension_df = pd.read_csv(tmp_path / "constraint_dimension_frequency.csv")
    assert "experience" in set(dimension_df["dimension_name"].tolist())
    assert "education" in set(dimension_df["dimension_name"].tolist())

    noise_df = pd.read_csv(tmp_path / "template_noise_report.csv")
    assert "责任心强" in set(noise_df["noise_text"].tolist())

    stringency_df = pd.read_csv(tmp_path / "requirement_stringency_index.csv")
    r1 = stringency_df[stringency_df["recruitment_record_id"] == "r1"].iloc[0]
    r3 = stringency_df[stringency_df["recruitment_record_id"] == "r3"].iloc[0]
    assert int(r1["stringency_score"]) > int(r3["stringency_score"])

    report_text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "hard skill / soft skill" in report_text
    assert "六、模板噪声报告" in report_text
