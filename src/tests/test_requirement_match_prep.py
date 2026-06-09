import pandas as pd

from src.data_pipeline.requirement_match_prep import _build_requirement_query_columns


def test_build_requirement_query_columns_prefers_requirements_text():
    dataframe = pd.DataFrame(
        [
            {
                "recruitment_record_id": "rrid-1",
                "任职要求_items_text": "熟悉 Python | 熟悉 SQL",
                "RAG匹配文本": "岗位描述兜底文本",
            }
        ]
    )

    result = _build_requirement_query_columns(dataframe)

    assert result.loc[0, "职业匹配文本"] == "熟悉 Python | 熟悉 SQL"
    assert result.loc[0, "职业匹配来源"] == "任职要求_items_text"
