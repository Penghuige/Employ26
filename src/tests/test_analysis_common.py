from datetime import datetime

import pandas as pd

from src.analysis.analysis_common import (
    build_requirement_output_dir,
    enrich_common_dimension_columns,
)


def test_build_requirement_output_dir_uses_batch_folder(tmp_path):
    output_dir = build_requirement_output_dir(
        datetime(2026, 6, 11),
        base_output_dir=tmp_path,
    )

    assert output_dir == tmp_path / "req_analysis_06-11"


def test_enrich_common_dimension_columns_adds_shared_fields():
    source_df = pd.DataFrame(
        [
            {
                "publish_date": "2026-06-01",
                "work_city": "广东省深圳市",
                "company_industry_raw": "互联网/人工智能",
                "company_size_raw": "100-499人",
            }
        ]
    )

    result_df = enrich_common_dimension_columns(source_df)
    row = result_df.iloc[0]

    assert row["publish_month"] == "2026-06"
    assert row["city_normalized"] == "深圳"
    assert row["industry_normalized"] == "互联网"
    assert row["company_size_normalized"] == "100-499人"
