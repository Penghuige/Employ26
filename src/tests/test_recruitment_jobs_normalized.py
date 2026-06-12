import pandas as pd

from src.db.recruitment_jobs_normalized import (
    build_dedupe_fingerprint,
    build_normalized_rows_from_dataframe,
)


def test_build_dedupe_fingerprint_is_stable_for_same_values():
    first = build_dedupe_fingerprint(
        source_platform="51job",
        company_name="OpenAI",
        job_title="数据分析师",
        job_description_raw="熟悉 SQL 和 Python。",
        publish_date="2026-06-01",
        work_city="广州",
    )
    second = build_dedupe_fingerprint(
        source_platform="51job",
        company_name="OpenAI",
        job_title="数据分析师",
        job_description_raw="熟悉 SQL 和 Python。",
        publish_date="2026-06-01",
        work_city="广州",
    )

    assert first == second
    assert len(first) == 40


def test_build_normalized_rows_from_dataframe_maps_raw_structured_fields():
    dataframe = pd.DataFrame(
        [
            {
                "__source_row_number": 42,
                "岗位名称": "数据分析师",
                "岗位描述": "熟悉 SQL 和 Python。",
                "工作城市": "广州",
                "公司名称": "OpenAI",
                "发布时间": "2026-06-01",
                "薪资水平": "15-25K",
                "学历要求": "本科",
                "经验要求": "3年",
                "公司规模": "500-999人",
                "公司行业": "人工智能",
            }
        ]
    )

    rows = build_normalized_rows_from_dataframe(
        dataframe,
        source_table='"51job".sample',
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.source_row_number == 42
    assert row.salary_raw == "15-25K"
    assert row.education_requirement_raw == "本科"
    assert row.experience_requirement_raw == "3年"
    assert row.company_size_raw == "500-999人"
    assert row.company_industry_raw == "人工智能"
