import json

import pandas as pd

from src.data_pipeline.description_parsing import (
    PARSER_VERSION,
    build_parsed_pg_rows,
    parse_desc_df,
)


def test_parse_desc_df_extracts_explicit_requirements_and_duties():
    df = pd.DataFrame(
        [
            {
                "岗位名称": "Java开发工程师",
                "岗位描述": "<p>岗位职责：1. 负责后端开发；2. 维护系统</p><p>任职要求：熟悉 Java；掌握 MySQL</p>",
            }
        ]
    )

    parsed = parse_desc_df(df)

    assert "负责后端开发" in parsed.loc[0, "岗位职责_items_text"]
    assert "熟悉 Java" in parsed.loc[0, "任职要求_items_text"]
    assert parsed.loc[0, "RAG匹配来源"] == "任职要求"


def test_parse_desc_df_infers_section_when_no_heading():
    df = pd.DataFrame(
        [
            {
                "岗位名称": "数据分析师",
                "岗位描述": "本科以上学历，熟练使用 SQL 和 Python，有数据分析经验优先。",
            }
        ]
    )

    parsed = parse_desc_df(df)

    assert "SQL" in parsed.loc[0, "任职要求_items_text"]
    assert parsed.loc[0, "RAG匹配文本"]


def test_build_parsed_pg_rows_uses_english_columns_and_json_payload():
    df = pd.DataFrame(
        [
            {
                "__source_row_number": 7,
                "sample_row_id": '"51job".sample:7',
                "岗位名称": "测试工程师",
                "岗位描述": "任职要求：熟悉自动化测试。",
            }
        ]
    )
    parsed = parse_desc_df(df)

    rows = build_parsed_pg_rows(parsed, source_table='"51job".sample')

    assert rows[0]["source_platform"] == "51job"
    assert rows[0]["source_row_number"] == 7
    assert rows[0]["source_record_id"] == '"51job".sample:7'
    assert rows[0]["job_title"] == "测试工程师"
    assert rows[0]["requirements_text"]
    assert rows[0]["parser_version"] == PARSER_VERSION
    assert json.loads(str(rows[0]["description_sections"]))["sections"]
