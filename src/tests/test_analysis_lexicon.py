import pandas as pd

from src.db.analysis_lexicon import (
    build_lexicon_summary_frames,
    build_bootstrap_payload,
    build_bootstrap_requirement_rules,
    classify_bootstrap_term,
    normalize_lexicon_term,
)


def test_normalize_lexicon_term_normalizes_case_and_whitespace():
    assert normalize_lexicon_term("  Python\u3000Tool  ") == "python tool"


def test_build_lexicon_summary_frames_groups_resources():
    resources = {
        "user_dictionary": pd.DataFrame(
            [
                {"term_type": "hard_skill_hint", "category": "skill", "enabled": True},
                {"term_type": "hard_skill_hint", "category": "skill", "enabled": True},
                {"term_type": "noise_term", "category": "noise", "enabled": False},
            ]
        ),
        "stopwords": pd.DataFrame(
            [
                {"scope": "unigram", "stop_strength": "hard_stop", "enabled": True},
                {"scope": "unigram", "stop_strength": "hard_stop", "enabled": True},
            ]
        ),
        "phrase_rules": pd.DataFrame(
            [
                {"rule_type": "merge", "source": "manual", "enabled": True},
                {"rule_type": "exclude", "source": "manual", "enabled": False},
            ]
        ),
        "requirement_rules": pd.DataFrame(
            [
                {"rule_type": "extract", "dimension_name": "certificate", "enabled": True},
                {"rule_type": "template_noise", "dimension_name": "", "enabled": True},
            ]
        ),
    }

    summary = build_lexicon_summary_frames(resources)

    user_summary = summary["user_dictionary"]
    assert int(user_summary.loc[0, "row_count"]) == 2
    stopword_summary = summary["stopwords"]
    assert int(stopword_summary.loc[0, "row_count"]) == 2
    phrase_summary = summary["phrase_rules"]
    assert set(phrase_summary["rule_type"].tolist()) == {"exclude", "merge"}
    requirement_summary = summary["requirement_rules"]
    assert set(requirement_summary["rule_type"].tolist()) == {"extract", "template_noise"}


def test_build_lexicon_summary_frames_handles_empty_resources():
    resources = {
        "user_dictionary": pd.DataFrame(columns=["term_type", "category", "enabled"]),
        "stopwords": pd.DataFrame(columns=["scope", "stop_strength", "enabled"]),
        "phrase_rules": pd.DataFrame(columns=["rule_type", "source", "enabled"]),
        "requirement_rules": pd.DataFrame(columns=["rule_type", "dimension_name", "enabled"]),
    }

    summary = build_lexicon_summary_frames(resources)

    assert summary["user_dictionary"].empty
    assert list(summary["user_dictionary"].columns) == ["term_type", "category", "enabled", "row_count"]
    assert summary["stopwords"].empty
    assert summary["phrase_rules"].empty
    assert summary["requirement_rules"].empty


def test_classify_bootstrap_term_uses_expected_heuristics():
    assert classify_bootstrap_term("Python") == ("tool_hint", "tool")
    assert classify_bootstrap_term("本科") == ("noise_term", "noise")
    assert classify_bootstrap_term("沟通能力") == ("soft_skill_hint", "soft_trait")
    assert classify_bootstrap_term("电工证") == ("certificate_hint", "certificate")
    assert classify_bootstrap_term("算法工程师") == ("noise_term", "noise")
    assert classify_bootstrap_term("1-3年") == ("noise_term", "noise")
    assert classify_bootstrap_term("统招本科") == ("noise_term", "noise")
    assert classify_bootstrap_term("工作经验") == ("noise_term", "noise")


def test_build_bootstrap_payload_includes_core_sections(tmp_path):
    userdict_path = tmp_path / "userdict.txt"
    userdict_path.write_text("Python 20000 eng\n算法工程师 20000 nz\n沟通能力 20000 nz\n", encoding="utf-8")

    stopwords_short_path = tmp_path / "stop_short.txt"
    stopwords_short_path.write_text("and\nor\n", encoding="utf-8")

    stopwords_optional_path = tmp_path / "stop_optional.txt"
    stopwords_optional_path.write_text("五险一金\n福利\n", encoding="utf-8")

    generic_terms_path = tmp_path / "generic.txt"
    generic_terms_path.write_text("经理\n工程师\n", encoding="utf-8")

    payload = build_bootstrap_payload(
        userdict_path=userdict_path,
        stopwords_short_path=stopwords_short_path,
        stopwords_optional_path=stopwords_optional_path,
        generic_terms_path=generic_terms_path,
    )

    assert payload["user_dictionary"]
    assert payload["stopwords"]
    assert payload["phrase_rules"]
    assert payload["requirement_rules"]
    assert any(row["term"] == "Python" and row["term_type"] == "tool_hint" for row in payload["user_dictionary"])
    assert any(row["term"] == "经理" and row["term_type"] == "noise_term" for row in payload["user_dictionary"])
    assert any(row["term"] == "沟通能力" and row["term_type"] == "soft_skill_hint" for row in payload["user_dictionary"])
    assert any(row["term"] == "福利" and row["scope"] == "requirement_analysis" for row in payload["stopwords"])


def test_build_bootstrap_requirement_rules_covers_phase2_dimensions():
    rules = build_bootstrap_requirement_rules()

    assert any(row["rule_type"] == "extract" and row["dimension_name"] == "certificate" for row in rules)
    assert any(row["rule_type"] == "extract" and row["dimension_name"] == "language" for row in rules)
    assert any(row["rule_type"] == "template_noise" and row["pattern_text"] == "责任心强" for row in rules)
