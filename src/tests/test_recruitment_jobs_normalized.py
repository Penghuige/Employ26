from src.db.recruitment_jobs_normalized import build_dedupe_fingerprint


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
