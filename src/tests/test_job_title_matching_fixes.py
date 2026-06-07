import pandas as pd

from src.job_title_parsing.match_utils import load_config
from src.job_title_parsing.matching_pipeline import MatchPipeline
from src.job_title_parsing.title_cleaner import JobTitleCleaner


def test_title_cleaner_preserves_common_english_roles():
    cleaner = JobTitleCleaner(load_config())

    assert cleaner.clean("HRBP") == "HRBP"
    assert cleaner.clean("CEO") == "CEO"
    assert cleaner.clean("SRE") == "SRE"
    assert cleaner.clean("GNZW03") == ""


def test_match_pipeline_returns_no_candidates_when_retrieval_has_no_hits():
    catalog_df = pd.DataFrame(
        [
            {
                "code": "1",
                "title": "商品营业员",
                "title_clean": "商品营业员",
                "retrieval_title_text": "商品营业员 导购",
                "retrieval_task_text": "销售商品 接待顾客",
                "retrieval_desc_text": "在商店销售商品",
                "task_list": ["销售商品", "接待顾客"],
                "aliases": ["导购"],
                "hierarchy_text": "商业 服务业人员",
                "大类": "4",
            }
        ]
    )
    pipeline = MatchPipeline(catalog_df=catalog_df)

    result = pipeline.match_one("火星基地总管", "", debug=True)

    assert result["candidates"] == []
    assert result["top1_code"] == ""
    assert result["confidence_level"] == "low"
    assert "no_candidates" in result["risk_flags"]
    assert result["debug_info"]["no_candidate_reason"]
