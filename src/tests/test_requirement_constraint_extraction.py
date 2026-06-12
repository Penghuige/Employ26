from src.analysis.requirement_constraint_extraction import (
    convert_constraints_to_fact_rows,
    extract_requirement_constraints,
    normalize_item_text,
    parse_requirement_rules,
    split_requirement_items,
)

import pandas as pd


def test_split_requirement_items_handles_pipe_and_semicolon():
    items, reliable = split_requirement_items("熟悉 Office | 本科以上；接受出差")

    assert reliable is True
    assert items == ["熟悉 Office", "本科以上", "接受出差"]


def test_split_requirement_items_handles_mixed_punctuation():
    items, reliable = split_requirement_items("1. 本科以上\n2. 3-5年经验\n3. 接受夜班")

    assert reliable is True
    assert items == ["本科以上", "3-5年经验", "接受夜班"]


def test_normalize_item_text_normalizes_year_case_and_gender():
    normalized = normalize_item_text("三年 OFFICE经验，性别不限")

    assert "3年" in normalized
    assert "Office" in normalized
    assert "男女不限" in normalized


def test_extract_requirement_constraints_covers_required_dimensions():
    requirement_rules_df = pd.DataFrame(
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
    )

    result = extract_requirement_constraints(
        "本科以上 | 3-5年经验 | 20-35岁 | 男女不限 | 需持有驾驶证 | 英语四级 | 接受出差夜班加班 | 身体健康 | 责任心强",
        rules_by_type=parse_requirement_rules(requirement_rules_df),
    )

    dimensions = {row.dimension_name for row in result.constraints}
    assert {"experience", "education", "age", "gender", "certificate", "language", "travel", "shift", "work_condition", "physical_condition"} <= dimensions
    assert any(hit.noise_text == "责任心强" for hit in result.template_noise_hits)


def test_convert_constraints_to_fact_rows_preserves_stable_keys():
    result = extract_requirement_constraints("本科以上 | 3年经验")
    rows = convert_constraints_to_fact_rows(
        recruitment_record_id="rr1",
        source_table='"51job".sample',
        source_row_number=1,
        constraints=result.constraints,
    )

    assert rows
    assert all(row.recruitment_record_id == "rr1" for row in rows)
    assert all(row.source_row_number == 1 for row in rows)
