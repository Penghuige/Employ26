import pandas as pd

from src.analysis.structured_pg_source import normalize_structured_source_dataframe


def test_normalize_structured_source_dataframe_adds_analysis_columns():
    source_df = pd.DataFrame(
        [
            {
                "recruitment_record_id": "r1",
                "source_platform": "51job",
                "source_table": '"51job".sample',
                "source_row_number": 1,
                "job_title": "算法工程师",
                "work_city": "广东省深圳市",
                "company_name": "A公司",
                "publish_date": "2026-06-01",
                "salary_raw": "20-30K",
                "education_requirement_raw": "本科",
                "experience_requirement_raw": "3-5年",
                "company_size_raw": "100-499人",
                "company_industry_raw": "互联网/人工智能",
                "occupation_code": "2-02-10-09",
                "occupation_title": "人工智能工程技术人员",
                "occupation_major_category": "专业技术人员",
                "occupation_middle_category": "工程技术人员",
                "occupation_minor_category": "信息和通信工程技术人员",
                "occupation_detail_category": "人工智能工程技术人员",
                "occupation_confidence": 0.91,
                "occupation_is_matched": True,
            }
        ]
    )

    result_df = normalize_structured_source_dataframe(source_df)
    row = result_df.iloc[0]

    assert row["publish_month"] == "2026-06"
    assert row["city_normalized"] == "深圳"
    assert row["industry_normalized"] == "互联网"
    assert row["occupation_core"] == "人工智能工程技术人员"
    assert row["occupation_category"] == "工程技术人员"
    assert row["薪资水平"] == "20-30K"
    assert row["学历要求"] == "本科"
    assert row["city_clean"] == "深圳"
    assert row["industry_clean"] == "互联网"
