from src.utils.backfill_label_studio_recruitment_record_ids import (
    SourceCandidate,
    build_match_key,
    choose_best_candidate,
    resolve_snapshot_row,
    similarity_score,
)


def test_resolve_snapshot_row_maps_first_30_rows_to_tier2():
    import pandas as pd

    tier2 = pd.DataFrame([{"岗位名称": "A"}, {"岗位名称": "B"}])
    tier3 = pd.DataFrame([{"岗位名称": "C"}])

    resolved = resolve_snapshot_row(1, 1, tier2, tier3)

    assert resolved.snapshot_source == "tier2_validation"
    assert resolved.snapshot_row_number == 1
    assert resolved.payload["岗位名称"] == "B"


def test_resolve_snapshot_row_maps_later_rows_to_tier3():
    import pandas as pd

    tier2 = pd.DataFrame([{"岗位名称": "A"}] * 30)
    tier3 = pd.DataFrame([{"岗位名称": "C"}, {"岗位名称": "D"}])

    resolved = resolve_snapshot_row(1, 31, tier2, tier3)

    assert resolved.snapshot_source == "tier3_main"
    assert resolved.snapshot_row_number == 1
    assert resolved.payload["岗位名称"] == "D"


def test_choose_best_candidate_prefers_strict_unique_match():
    payload = {
        "发布时间": "2022-01-01",
        "岗位名称": "数据分析师",
        "工作城市": "广州",
        "薪资水平": "10k",
        "经验要求": "3年",
        "学历要求": "本科",
        "岗位描述": "熟悉SQL",
        "公司名称": "甲公司",
        "公司规模": "100人",
        "公司行业": "互联网",
    }
    candidate = SourceCandidate(
        source_table='"51job".sample',
        source_platform="51job",
        source_row_number=7,
        payload=payload,
        strict_key=build_match_key(payload, normalized=False),
        normalized_key=build_match_key(payload, normalized=True),
    )

    status, rule, candidates, _, _ = choose_best_candidate(payload, [candidate])

    assert status == "AUTO_CONFIRMED"
    assert rule == "exact_full_row_unique"
    assert candidates[0].source_row_number == 7


def test_choose_best_candidate_rejects_ambiguous_similarity_match():
    payload = {
        "发布时间": "2022-01-01",
        "岗位名称": "数据分析师",
        "工作城市": "广州",
        "薪资水平": "10k",
        "经验要求": "3年",
        "学历要求": "本科",
        "岗位描述": "熟悉SQL和Python",
        "公司名称": "甲公司",
        "公司规模": "100人",
        "公司行业": "互联网",
    }
    payload2 = dict(payload)
    payload2["岗位描述"] = "熟悉SQL和Python，能做报表"
    payload3 = dict(payload)
    payload3["岗位描述"] = "熟悉SQL和Python，负责数据建模"

    candidate2 = SourceCandidate(
        source_table='"51job".sample',
        source_platform="51job",
        source_row_number=8,
        payload=payload2,
        strict_key=("x",),
        normalized_key=("x",),
    )
    candidate3 = SourceCandidate(
        source_table='"Liepin".sample',
        source_platform="Liepin",
        source_row_number=9,
        payload=payload3,
        strict_key=("y",),
        normalized_key=("y",),
    )

    status, rule, _, best_score, second_score = choose_best_candidate(payload, [candidate2, candidate3])

    assert status == "REVIEW_REQUIRED"
    assert rule == "strong_text_similarity_ambiguous"
    assert best_score is not None
    assert second_score is not None


def test_choose_best_candidate_collapses_exact_duplicates_in_same_source_table():
    payload = {
        "发布时间": "2022-01-01",
        "岗位名称": "数据分析师",
        "工作城市": "广州",
        "薪资水平": "10k",
        "经验要求": "3年",
        "学历要求": "本科",
        "岗位描述": "熟悉SQL和Python",
        "公司名称": "甲公司",
        "公司规模": "100人",
        "公司行业": "互联网",
    }
    candidate1 = SourceCandidate(
        source_table='"51job".sample',
        source_platform="51job",
        source_row_number=8,
        payload=payload,
        strict_key=build_match_key(payload, normalized=False),
        normalized_key=build_match_key(payload, normalized=True),
    )
    candidate2 = SourceCandidate(
        source_table='"51job".sample',
        source_platform="51job",
        source_row_number=11,
        payload=dict(payload),
        strict_key=build_match_key(payload, normalized=False),
        normalized_key=build_match_key(payload, normalized=True),
    )

    status, rule, candidates, _, _ = choose_best_candidate(payload, [candidate2, candidate1])

    assert status == "AUTO_CONFIRMED"
    assert rule == "exact_duplicate_rows_same_source_table"
    assert candidates[0].source_row_number == 8


def test_similarity_score_rewards_closer_text():
    left = {"岗位名称": "电工", "公司名称": "甲公司", "岗位描述": "负责设备维修"}
    right = {"岗位名称": "电工", "公司名称": "甲公司", "岗位描述": "负责设备维修"}
    other = {"岗位名称": "电工", "公司名称": "甲公司", "岗位描述": "负责仓库分拣"}

    assert similarity_score(left, right) > similarity_score(left, other)
