from src.job_title_parsing.occupation_dictionary_pipeline import normalize_job_title, retrieve_top_candidates


SAMPLE_DICT = {
    "canonical_occupations": [
        {"name": "Java开发工程师", "aliases": ["Java工程师", "高级Java工程师"]},
        {"name": "数据分析师", "aliases": ["数据分析", "DA"]},
        {"name": "招聘专员", "aliases": ["HR招聘", "招聘"]},
    ]
}


class TestOccupationIterationUtils:
    def test_normalize_job_title_removes_common_noise(self):
        assert normalize_job_title("高级Java开发工程师（深圳）- 20k") == "高级Java开发工程师"

    def test_retrieve_top_candidates_prefers_exact_and_alias_match(self):
        candidates = retrieve_top_candidates("Java工程师", SAMPLE_DICT, top_k=3)
        assert candidates
        assert candidates[0]["canonical_name"] == "Java开发工程师"
