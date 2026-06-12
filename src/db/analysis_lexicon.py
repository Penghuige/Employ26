"""第一阶段 requirement text 分析词汇资源 schema。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.db.postgres import create_pg_engine, ensure_schema


DEFAULT_LEXICON_SCHEMA = "analysis_lexicon"
DEFAULT_RELEASE_TABLE = f"{DEFAULT_LEXICON_SCHEMA}.lexicon_release"
DEFAULT_USER_DICTIONARY_TABLE = f"{DEFAULT_LEXICON_SCHEMA}.user_dictionary"
DEFAULT_STOPWORDS_TABLE = f"{DEFAULT_LEXICON_SCHEMA}.stopwords"
DEFAULT_PHRASE_RULES_TABLE = f"{DEFAULT_LEXICON_SCHEMA}.phrase_rules"
DEFAULT_REQUIREMENT_RULES_TABLE = f"{DEFAULT_LEXICON_SCHEMA}.requirement_rules"

TERM_TYPES = (
    "hard_skill_hint",
    "soft_skill_hint",
    "certificate_hint",
    "tool_hint",
    "noise_term",
)
CATEGORIES = (
    "skill",
    "soft_trait",
    "certificate",
    "tool",
    "noise",
)
RULE_TYPES = (
    "merge",
    "exclude",
)
RULE_SOURCES = (
    "manual",
    "derived",
    "imported",
)
REQUIREMENT_RULE_TYPES = (
    "item_split",
    "normalize",
    "extract",
    "template_noise",
)
REQUIREMENT_DIMENSIONS = (
    "experience",
    "education",
    "age",
    "gender",
    "certificate",
    "language",
    "travel",
    "shift",
    "physical_condition",
    "availability",
    "work_condition",
)
REQUIREMENT_OPERATORS = (
    "eq",
    "ge",
    "le",
    "between",
    "contains",
    "allow",
    "exclude",
)
STOPWORD_SCOPES = (
    "global",
    "requirement_analysis",
    "unigram",
    "bigram",
    "trigram",
)
STOPWORD_STRENGTHS = (
    "hard_stop",
    "soft_stop",
)
DEFAULT_BOOTSTRAP_VERSION = "v1_bootstrap_requirement_analysis"
DEFAULT_RELEASED_BY = "codex"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_USERDICT_PATH = PROJECT_ROOT / "dicts" / "userdict_zh_recruitment.txt"
DEFAULT_STOPWORDS_SHORT_PATH = PROJECT_ROOT / "dicts" / "stopwords_recruitment_short.txt"
DEFAULT_STOPWORDS_OPTIONAL_PATH = PROJECT_ROOT / "dicts" / "stopwords_zh_recruitment_optional.txt"
DEFAULT_GENERIC_TERMS_PATH = PROJECT_ROOT / "dicts" / "job_generic_terms.txt"
BOOTSTRAP_REQUIREMENT_STOPWORDS = (
    "熟悉",
    "具备",
    "优先",
    "相关",
    "专业",
    "工作经验",
    "经验",
    "以上学历",
    "学历",
    "熟练",
    "良好",
    "较强",
    "具有",
    "能够",
)
EXPLICIT_NOISE_TERMS = {
    "工作经验",
    "统招本科",
}
EXPLICIT_SOFT_SKILL_TERMS = (
    "沟通",
    "沟通能力",
    "沟通协调",
    "协调",
    "协调能力",
    "团队",
    "团队合作",
    "团队协作",
    "学习",
    "学习能力",
    "抗压",
    "抗压能力",
    "责任心",
    "服务意识",
    "执行力",
    "表达能力",
    "合作精神",
    "积极主动",
    "吃苦耐劳",
    "解决问题",
    "逻辑思维",
    "亲和力",
)
EXPLICIT_TOOL_TERMS = (
    "Office",
    "Excel",
    "Word",
    "PPT",
    "SAP",
)

JOB_TITLE_SUFFIXES = (
    "工程师",
    "经理",
    "专员",
    "主管",
    "总监",
    "助理",
    "顾问",
    "实习生",
    "文员",
    "客服",
    "销售",
    "运营",
    "分析师",
    "设计师",
)
DEGREE_OR_STAGE_TERMS = {
    "博士",
    "硕士",
    "本科",
    "大专",
    "中专",
    "高中",
    "应届生",
    "应届毕业生",
}
WELFARE_NOISE_TERMS = {
    "五险一金",
    "六险一金",
}
SOFT_SKILL_HINT_KEYWORDS = (
    "沟通",
    "协调",
    "抗压",
    "责任心",
    "执行力",
    "学习能力",
    "团队",
    "表达",
    "逻辑",
    "亲和力",
    "主动",
    "细致",
    "认真",
)
CERTIFICATE_HINT_KEYWORDS = (
    "证",
    "资格",
    "认证",
    "执业",
    "注册",
    "上岗",
    "建造师",
    "会计师",
    "教师资格",
    "驾驶证",
)


def _empty_summary_frame(columns: list[str]) -> pd.DataFrame:
    """返回带固定列名的空摘要表。"""
    return pd.DataFrame(columns=columns)


def split_table_name(table_name: str) -> tuple[str, str]:
    """拆分 schema.table。"""
    normalized = str(table_name).strip()
    if "." not in normalized:
        return "public", normalized.strip('"')
    schema, table = normalized.split(".", 1)
    return schema.strip().strip('"'), table.strip().strip('"')


def quote_table_name(table_name: str) -> str:
    """返回双引号安全表名。"""
    schema_name, raw_table_name = split_table_name(table_name)
    return f'"{schema_name}"."{raw_table_name}"'


def normalize_lexicon_term(value: object) -> str:
    """生成稳定归一化词面。"""
    if value is None:
        return ""
    text_value = str(value).strip()
    if not text_value:
        return ""
    text_value = text_value.replace("\u3000", " ").replace("\xa0", " ")
    text_value = re.sub(r"\s+", " ", text_value)
    text_value = text_value.casefold()
    return text_value


def _sql_in(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{value}'" for value in values)


def _read_term_lines(path: str | Path) -> list[str]:
    """读取简单词表，每行一个词，忽略注释和空行。"""
    term_path = Path(path)
    return [
        line.strip()
        for line in term_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _read_jieba_userdict(path: str | Path) -> list[dict[str, str]]:
    """读取 jieba userdict 格式词典。"""
    rows: list[dict[str, str]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text_value = line.strip()
        if not text_value or text_value.startswith("#"):
            continue
        parts = text_value.split()
        term = parts[0].strip()
        freq = parts[1].strip() if len(parts) >= 2 else ""
        tag = parts[2].strip() if len(parts) >= 3 else ""
        if term:
            rows.append({"term": term, "freq": freq, "tag": tag})
    return rows


def _looks_ascii_technical(term: str) -> bool:
    """判断是否更像英文技术词/工具词。"""
    return bool(re.search(r"[A-Za-z]", term))


def _is_year_experience_term(normalized_term: str) -> bool:
    """判断是否为年限/经验年数类词项。"""
    return bool(
        re.fullmatch(r"\d+\s*-\s*\d+年", normalized_term)
        or re.fullmatch(r"\d+年以上", normalized_term)
        or re.fullmatch(r"\d+年", normalized_term)
    )


def _is_education_term(normalized_term: str) -> bool:
    """判断是否为学历/学段相关词项。"""
    education_keywords = ("本科", "大专", "硕士", "博士", "中专", "高中", "学历", "应届")
    return any(keyword in normalized_term for keyword in education_keywords)


def _push_bootstrap_user_term(
    row_map: dict[str, dict[str, object]],
    *,
    term: str,
    term_type: str,
    category: str,
    source: str,
    notes: str = "",
) -> None:
    """按 normalized_term 写入或覆盖 bootstrap user_dictionary 词项。"""
    normalized_term = normalize_lexicon_term(term)
    if not normalized_term:
        return
    row_map[normalized_term] = {
        "term": term,
        "normalized_term": normalized_term,
        "preferred_term": term,
        "term_type": term_type,
        "category": category,
        "variants_json": [],
        "enabled": True,
        "source": source,
        "notes": notes,
    }


def classify_bootstrap_term(term: str) -> tuple[str, str]:
    """把旧词典词项映射到 v1 release 的 term_type/category。"""
    normalized_term = normalize_lexicon_term(term)
    if not normalized_term:
        return "noise_term", "noise"

    if normalized_term in {normalize_lexicon_term(item) for item in EXPLICIT_NOISE_TERMS}:
        return "noise_term", "noise"
    if _is_year_experience_term(normalized_term):
        return "noise_term", "noise"
    if _is_education_term(normalized_term):
        return "noise_term", "noise"
    if normalized_term in {
        normalize_lexicon_term(item) for item in DEGREE_OR_STAGE_TERMS.union(WELFARE_NOISE_TERMS)
    }:
        return "noise_term", "noise"
    if any(normalized_term.endswith(normalize_lexicon_term(suffix)) for suffix in JOB_TITLE_SUFFIXES):
        return "noise_term", "noise"
    if any(keyword in term for keyword in CERTIFICATE_HINT_KEYWORDS):
        return "certificate_hint", "certificate"
    if any(keyword in term for keyword in SOFT_SKILL_HINT_KEYWORDS):
        return "soft_skill_hint", "soft_trait"
    if _looks_ascii_technical(term):
        return "tool_hint", "tool"
    return "hard_skill_hint", "skill"


def build_bootstrap_requirement_rules() -> list[dict[str, object]]:
    """构建 Phase 2 requirement 规则表的首版种子。"""
    rows: list[dict[str, object]] = [
        {
            "rule_type": "item_split",
            "dimension_name": "",
            "pattern_text": "\n",
            "replacement_text": " | ",
            "normalized_value": "",
            "operator": "contains",
            "priority": 10,
            "enabled": True,
            "source": "imported",
            "notes": "line break item splitter",
        },
        {
            "rule_type": "item_split",
            "dimension_name": "",
            "pattern_text": "；",
            "replacement_text": " | ",
            "normalized_value": "",
            "operator": "contains",
            "priority": 10,
            "enabled": True,
            "source": "imported",
            "notes": "semicolon item splitter",
        },
        {
            "rule_type": "normalize",
            "dimension_name": "",
            "pattern_text": "性别不限",
            "replacement_text": "男女不限",
            "normalized_value": "男女不限",
            "operator": "eq",
            "priority": 20,
            "enabled": True,
            "source": "manual",
            "notes": "normalize gender free text",
        },
        {
            "rule_type": "normalize",
            "dimension_name": "",
            "pattern_text": "不限性别",
            "replacement_text": "男女不限",
            "normalized_value": "男女不限",
            "operator": "eq",
            "priority": 20,
            "enabled": True,
            "source": "manual",
            "notes": "normalize gender free text",
        },
        {
            "rule_type": "normalize",
            "dimension_name": "",
            "pattern_text": "普通话流利",
            "replacement_text": "普通话标准",
            "normalized_value": "普通话标准",
            "operator": "eq",
            "priority": 20,
            "enabled": True,
            "source": "manual",
            "notes": "normalize language phrase",
        },
        {
            "rule_type": "template_noise",
            "dimension_name": "",
            "pattern_text": "能力",
            "replacement_text": "",
            "normalized_value": "能力",
            "operator": "exclude",
            "priority": 10,
            "enabled": True,
            "source": "manual",
            "notes": "generic token noise",
        },
        {
            "rule_type": "template_noise",
            "dimension_name": "",
            "pattern_text": "强",
            "replacement_text": "",
            "normalized_value": "强",
            "operator": "exclude",
            "priority": 10,
            "enabled": True,
            "source": "manual",
            "notes": "generic token noise",
        },
        {
            "rule_type": "template_noise",
            "dimension_name": "",
            "pattern_text": "良好",
            "replacement_text": "",
            "normalized_value": "良好",
            "operator": "exclude",
            "priority": 10,
            "enabled": True,
            "source": "manual",
            "notes": "generic token noise",
        },
        {
            "rule_type": "template_noise",
            "dimension_name": "",
            "pattern_text": "责任心强",
            "replacement_text": "",
            "normalized_value": "责任心强",
            "operator": "exclude",
            "priority": 20,
            "enabled": True,
            "source": "manual",
            "notes": "template phrase",
        },
        {
            "rule_type": "template_noise",
            "dimension_name": "",
            "pattern_text": "团队合作精神",
            "replacement_text": "",
            "normalized_value": "团队合作精神",
            "operator": "exclude",
            "priority": 20,
            "enabled": True,
            "source": "manual",
            "notes": "template phrase",
        },
        {
            "rule_type": "template_noise",
            "dimension_name": "",
            "pattern_text": "服从安排",
            "replacement_text": "",
            "normalized_value": "服从安排",
            "operator": "exclude",
            "priority": 30,
            "enabled": True,
            "source": "manual",
            "notes": "platform boilerplate",
        },
        {
            "rule_type": "template_noise",
            "dimension_name": "",
            "pattern_text": "抗压能力强",
            "replacement_text": "",
            "normalized_value": "抗压能力强",
            "operator": "exclude",
            "priority": 30,
            "enabled": True,
            "source": "manual",
            "notes": "platform boilerplate",
        },
        {
            "rule_type": "template_noise",
            "dimension_name": "",
            "pattern_text": "能承受较大压力",
            "replacement_text": "",
            "normalized_value": "能承受较大压力",
            "operator": "exclude",
            "priority": 30,
            "enabled": True,
            "source": "manual",
            "notes": "platform boilerplate",
        },
    ]
    for dimension_name, pattern_text, normalized_value, operator, notes in (
        ("certificate", "驾驶证", "驾驶证", "contains", "extract certificate"),
        ("certificate", "会计证", "会计证", "contains", "extract certificate"),
        ("certificate", "建造师", "建造师", "contains", "extract certificate"),
        ("language", "英语四级", "英语四级", "contains", "extract language"),
        ("language", "普通话标准", "普通话标准", "contains", "extract language"),
        ("travel", "出差", "接受出差", "allow", "extract travel condition"),
        ("shift", "夜班", "接受夜班", "allow", "extract shift condition"),
        ("work_condition", "加班", "接受加班", "allow", "extract work condition"),
        ("physical_condition", "身体健康", "身体健康", "allow", "extract physical condition"),
        ("physical_condition", "无不良记录", "不良记录", "exclude", "extract physical condition"),
        ("physical_condition", "无犯罪记录", "犯罪记录", "exclude", "extract physical condition"),
    ):
        rows.append(
            {
                "rule_type": "extract",
                "dimension_name": dimension_name,
                "pattern_text": pattern_text,
                "replacement_text": "",
                "normalized_value": normalized_value,
                "operator": operator,
                "priority": 40,
                "enabled": True,
                "source": "manual",
                "notes": notes,
            }
        )
    return rows


def build_bootstrap_payload(
    userdict_path: str | Path = DEFAULT_USERDICT_PATH,
    stopwords_short_path: str | Path = DEFAULT_STOPWORDS_SHORT_PATH,
    stopwords_optional_path: str | Path = DEFAULT_STOPWORDS_OPTIONAL_PATH,
    generic_terms_path: str | Path = DEFAULT_GENERIC_TERMS_PATH,
) -> dict[str, list[dict[str, object]]]:
    """基于仓库已有 dicts 生成首版 release 资源。"""
    user_dictionary_map: dict[str, dict[str, object]] = {}

    for item in _read_jieba_userdict(userdict_path):
        term = str(item["term"]).strip()
        term_type, category = classify_bootstrap_term(term)
        _push_bootstrap_user_term(
            user_dictionary_map,
            term=term,
            term_type=term_type,
            category=category,
            source="bootstrap:userdict_zh_recruitment",
            notes=f"origin_tag={item['tag']}" if item["tag"] else "",
        )

    for term in _read_term_lines(generic_terms_path):
        _push_bootstrap_user_term(
            user_dictionary_map,
            term=term,
            term_type="noise_term",
            category="noise",
            source="bootstrap:job_generic_terms",
            notes="generic job title noise term",
        )

    for term in EXPLICIT_SOFT_SKILL_TERMS:
        _push_bootstrap_user_term(
            user_dictionary_map,
            term=term,
            term_type="soft_skill_hint",
            category="soft_trait",
            source="bootstrap:soft_skill_terms",
            notes="manual soft skill hint",
        )

    for term in EXPLICIT_TOOL_TERMS:
        _push_bootstrap_user_term(
            user_dictionary_map,
            term=term,
            term_type="tool_hint",
            category="tool",
            source="bootstrap:office_tool_terms",
            notes="manual office/tool hint",
        )

    user_dictionary_rows = list(user_dictionary_map.values())

    stopword_rows: list[dict[str, object]] = []
    seen_stopwords: set[tuple[str, str]] = set()
    for term in _read_term_lines(stopwords_short_path):
        normalized_term = normalize_lexicon_term(term)
        dedupe_key = ("global", normalized_term)
        if not normalized_term or dedupe_key in seen_stopwords:
            continue
        seen_stopwords.add(dedupe_key)
        stopword_rows.append(
            {
                "term": term,
                "normalized_term": normalized_term,
                "scope": "global",
                "stop_strength": "hard_stop",
                "enabled": True,
                "source": "bootstrap:stopwords_recruitment_short",
                "notes": "",
            }
        )

    for term in _read_term_lines(stopwords_optional_path):
        normalized_term = normalize_lexicon_term(term)
        dedupe_key = ("requirement_analysis", normalized_term)
        if not normalized_term or dedupe_key in seen_stopwords:
            continue
        seen_stopwords.add(dedupe_key)
        stopword_rows.append(
            {
                "term": term,
                "normalized_term": normalized_term,
                "scope": "requirement_analysis",
                "stop_strength": "soft_stop",
                "enabled": True,
                "source": "bootstrap:stopwords_zh_recruitment_optional",
                "notes": "",
            }
        )

    for term in BOOTSTRAP_REQUIREMENT_STOPWORDS:
        normalized_term = normalize_lexicon_term(term)
        dedupe_key = ("requirement_analysis", normalized_term)
        if not normalized_term or dedupe_key in seen_stopwords:
            continue
        seen_stopwords.add(dedupe_key)
        stopword_rows.append(
            {
                "term": term,
                "normalized_term": normalized_term,
                "scope": "requirement_analysis",
                "stop_strength": "hard_stop",
                "enabled": True,
                "source": "bootstrap:requirement_boilerplate",
                "notes": "requirement boilerplate term",
            }
        )

    phrase_rule_rows: list[dict[str, object]] = [
        {
            "rule_type": "merge",
            "source_term": "office",
            "normalized_source_term": normalize_lexicon_term("office"),
            "replacement_term": "Office",
            "priority": 10,
            "enabled": True,
            "source": "imported",
            "notes": "bootstrap normalized casing",
        },
        {
            "rule_type": "merge",
            "source_term": "excel",
            "normalized_source_term": normalize_lexicon_term("excel"),
            "replacement_term": "Excel",
            "priority": 10,
            "enabled": True,
            "source": "imported",
            "notes": "bootstrap normalized casing",
        },
        {
            "rule_type": "merge",
            "source_term": "word",
            "normalized_source_term": normalize_lexicon_term("word"),
            "replacement_term": "Word",
            "priority": 10,
            "enabled": True,
            "source": "imported",
            "notes": "bootstrap normalized casing",
        },
        {
            "rule_type": "merge",
            "source_term": "ppt",
            "normalized_source_term": normalize_lexicon_term("ppt"),
            "replacement_term": "PPT",
            "priority": 10,
            "enabled": True,
            "source": "imported",
            "notes": "bootstrap normalized casing",
        },
    ]

    return {
        "user_dictionary": user_dictionary_rows,
        "stopwords": stopword_rows,
        "phrase_rules": phrase_rule_rows,
        "requirement_rules": build_bootstrap_requirement_rules(),
    }


def upsert_release_metadata(
    connection,
    *,
    version: str,
    released_by: str,
    notes: str,
    is_current: bool = True,
) -> int:
    """创建或更新 release 元数据，并维护唯一 current。"""
    qualified_table = quote_table_name(DEFAULT_RELEASE_TABLE)
    release_id = connection.execute(
        text(
            f"""
            SELECT id
            FROM {qualified_table}
            WHERE version = :version
            """
        ),
        {"version": version},
    ).scalar_one_or_none()

    if is_current:
        connection.execute(
            text(
                f"""
                UPDATE {qualified_table}
                SET is_current = false,
                    updated_at = now()
                WHERE is_current = true
                  AND version <> :version
                """
            ),
            {"version": version},
        )

    if release_id is None:
        release_id = connection.execute(
            text(
                f"""
                INSERT INTO {qualified_table} (
                    version,
                    is_current,
                    released_at,
                    released_by,
                    notes
                )
                VALUES (
                    :version,
                    :is_current,
                    now(),
                    :released_by,
                    :notes
                )
                RETURNING id
                """
            ),
            {
                "version": version,
                "is_current": is_current,
                "released_by": released_by,
                "notes": notes,
            },
        ).scalar_one()
    else:
        connection.execute(
            text(
                f"""
                UPDATE {qualified_table}
                SET is_current = :is_current,
                    released_at = now(),
                    released_by = :released_by,
                    notes = :notes,
                    updated_at = now()
                WHERE id = :release_id
                """
            ),
            {
                "release_id": int(release_id),
                "is_current": is_current,
                "released_by": released_by,
                "notes": notes,
            },
        )
    return int(release_id)


def replace_release_resources(
    connection,
    *,
    release_id: int,
    payload: dict[str, list[dict[str, object]]],
) -> None:
    """全量替换指定 release 下的资源。"""
    connection.execute(
        text(f"DELETE FROM {quote_table_name(DEFAULT_USER_DICTIONARY_TABLE)} WHERE release_id = :release_id"),
        {"release_id": int(release_id)},
    )
    connection.execute(
        text(f"DELETE FROM {quote_table_name(DEFAULT_STOPWORDS_TABLE)} WHERE release_id = :release_id"),
        {"release_id": int(release_id)},
    )
    connection.execute(
        text(f"DELETE FROM {quote_table_name(DEFAULT_PHRASE_RULES_TABLE)} WHERE release_id = :release_id"),
        {"release_id": int(release_id)},
    )
    connection.execute(
        text(f"DELETE FROM {quote_table_name(DEFAULT_REQUIREMENT_RULES_TABLE)} WHERE release_id = :release_id"),
        {"release_id": int(release_id)},
    )

    if payload["user_dictionary"]:
        connection.execute(
            text(
                f"""
                INSERT INTO {quote_table_name(DEFAULT_USER_DICTIONARY_TABLE)} (
                    release_id,
                    term,
                    normalized_term,
                    preferred_term,
                    term_type,
                    category,
                    variants_json,
                    enabled,
                    source,
                    notes
                )
                VALUES (
                    :release_id,
                    :term,
                    :normalized_term,
                    :preferred_term,
                    :term_type,
                    :category,
                    CAST(:variants_json AS jsonb),
                    :enabled,
                    :source,
                    :notes
                )
                """
            ),
            [
                {
                    **row,
                    "release_id": int(release_id),
                    "variants_json": json.dumps(row["variants_json"], ensure_ascii=False),
                }
                for row in payload["user_dictionary"]
            ],
        )

    if payload["stopwords"]:
        connection.execute(
            text(
                f"""
                INSERT INTO {quote_table_name(DEFAULT_STOPWORDS_TABLE)} (
                    release_id,
                    term,
                    normalized_term,
                    scope,
                    stop_strength,
                    enabled,
                    source,
                    notes
                )
                VALUES (
                    :release_id,
                    :term,
                    :normalized_term,
                    :scope,
                    :stop_strength,
                    :enabled,
                    :source,
                    :notes
                )
                """
            ),
            [{**row, "release_id": int(release_id)} for row in payload["stopwords"]],
        )

    if payload["phrase_rules"]:
        connection.execute(
            text(
                f"""
                INSERT INTO {quote_table_name(DEFAULT_PHRASE_RULES_TABLE)} (
                    release_id,
                    rule_type,
                    source_term,
                    normalized_source_term,
                    replacement_term,
                    priority,
                    enabled,
                    source,
                    notes
                )
                VALUES (
                    :release_id,
                    :rule_type,
                    :source_term,
                    :normalized_source_term,
                    :replacement_term,
                    :priority,
                    :enabled,
                    :source,
                    :notes
                )
                """
            ),
            [{**row, "release_id": int(release_id)} for row in payload["phrase_rules"]],
        )

    if payload.get("requirement_rules"):
        connection.execute(
            text(
                f"""
                INSERT INTO {quote_table_name(DEFAULT_REQUIREMENT_RULES_TABLE)} (
                    release_id,
                    rule_type,
                    dimension_name,
                    pattern_text,
                    replacement_text,
                    normalized_value,
                    operator,
                    priority,
                    enabled,
                    source,
                    notes
                )
                VALUES (
                    :release_id,
                    :rule_type,
                    :dimension_name,
                    :pattern_text,
                    :replacement_text,
                    :normalized_value,
                    :operator,
                    :priority,
                    :enabled,
                    :source,
                    :notes
                )
                """
            ),
            [{**row, "release_id": int(release_id)} for row in payload["requirement_rules"]],
        )


def bootstrap_initial_release(
    *,
    version: str = DEFAULT_BOOTSTRAP_VERSION,
    released_by: str = DEFAULT_RELEASED_BY,
    notes: str = "Bootstrap from repository dicts for requirement text analysis v1.",
) -> dict[str, object]:
    """初始化首版 analysis_lexicon 正式 release。"""
    engine = create_pg_engine()
    payload = build_bootstrap_payload()
    try:
        with engine.begin() as connection:
            ensure_analysis_lexicon_schema(connection)
            release_id = upsert_release_metadata(
                connection,
                version=version,
                released_by=released_by,
                notes=notes,
                is_current=True,
            )
            replace_release_resources(
                connection,
                release_id=release_id,
                payload=payload,
            )
        return {
            "version": version,
            "release_id": release_id,
            "user_dictionary_rows": len(payload["user_dictionary"]),
            "stopword_rows": len(payload["stopwords"]),
            "phrase_rule_rows": len(payload["phrase_rules"]),
        }
    finally:
        engine.dispose()


def ensure_lexicon_release_table(connection, table_name: str = DEFAULT_RELEASE_TABLE) -> None:
    """确保 lexicon release 表存在。"""
    schema_name, _ = split_table_name(table_name)
    ensure_schema(connection, schema_name)
    qualified_table = quote_table_name(table_name)
    object_prefix = "analysis_lexicon_release"
    connection.execute(
        text(
            f"""
            CREATE TABLE IF NOT EXISTS {qualified_table} (
                id bigserial PRIMARY KEY,
                version text NOT NULL UNIQUE,
                is_current boolean NOT NULL DEFAULT false,
                released_at timestamptz NOT NULL DEFAULT now(),
                released_by text,
                notes text,
                created_at timestamptz NOT NULL DEFAULT now(),
                updated_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
    )
    connection.execute(
        text(
            f"""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_{object_prefix}_current
            ON {qualified_table} (is_current)
            WHERE is_current = true
            """
        )
    )


def ensure_user_dictionary_table(connection, table_name: str = DEFAULT_USER_DICTIONARY_TABLE) -> None:
    """确保用户词典表存在。"""
    ensure_lexicon_release_table(connection)
    qualified_table = quote_table_name(table_name)
    release_table = quote_table_name(DEFAULT_RELEASE_TABLE)
    connection.execute(
        text(
            f"""
            CREATE TABLE IF NOT EXISTS {qualified_table} (
                id bigserial PRIMARY KEY,
                release_id bigint NOT NULL REFERENCES {release_table}(id),
                term text NOT NULL,
                normalized_term text NOT NULL,
                preferred_term text NOT NULL,
                term_type text NOT NULL CHECK (term_type IN ({_sql_in(TERM_TYPES)})),
                category text NOT NULL CHECK (category IN ({_sql_in(CATEGORIES)})),
                variants_json jsonb NOT NULL DEFAULT '[]'::jsonb,
                enabled boolean NOT NULL DEFAULT true,
                source text,
                notes text,
                created_at timestamptz NOT NULL DEFAULT now(),
                updated_at timestamptz NOT NULL DEFAULT now(),
                UNIQUE (release_id, normalized_term, term_type)
            )
            """
        )
    )


def ensure_stopwords_table(connection, table_name: str = DEFAULT_STOPWORDS_TABLE) -> None:
    """确保停用词表存在。"""
    ensure_lexicon_release_table(connection)
    qualified_table = quote_table_name(table_name)
    release_table = quote_table_name(DEFAULT_RELEASE_TABLE)
    connection.execute(
        text(
            f"""
            CREATE TABLE IF NOT EXISTS {qualified_table} (
                id bigserial PRIMARY KEY,
                release_id bigint NOT NULL REFERENCES {release_table}(id),
                term text NOT NULL,
                normalized_term text NOT NULL,
                scope text NOT NULL CHECK (scope IN ({_sql_in(STOPWORD_SCOPES)})),
                stop_strength text NOT NULL CHECK (stop_strength IN ({_sql_in(STOPWORD_STRENGTHS)})),
                enabled boolean NOT NULL DEFAULT true,
                source text,
                notes text,
                created_at timestamptz NOT NULL DEFAULT now(),
                updated_at timestamptz NOT NULL DEFAULT now(),
                UNIQUE (release_id, scope, normalized_term)
            )
            """
        )
    )


def ensure_phrase_rules_table(connection, table_name: str = DEFAULT_PHRASE_RULES_TABLE) -> None:
    """确保短语规则表存在。"""
    ensure_lexicon_release_table(connection)
    qualified_table = quote_table_name(table_name)
    release_table = quote_table_name(DEFAULT_RELEASE_TABLE)
    connection.execute(
        text(
            f"""
            CREATE TABLE IF NOT EXISTS {qualified_table} (
                id bigserial PRIMARY KEY,
                release_id bigint NOT NULL REFERENCES {release_table}(id),
                rule_type text NOT NULL CHECK (rule_type IN ({_sql_in(RULE_TYPES)})),
                source_term text NOT NULL,
                normalized_source_term text NOT NULL,
                replacement_term text NOT NULL,
                priority integer NOT NULL DEFAULT 100,
                enabled boolean NOT NULL DEFAULT true,
                source text CHECK (source IS NULL OR source IN ({_sql_in(RULE_SOURCES)})),
                notes text,
                created_at timestamptz NOT NULL DEFAULT now(),
                updated_at timestamptz NOT NULL DEFAULT now(),
                UNIQUE (release_id, normalized_source_term, rule_type)
            )
            """
        )
    )


def ensure_requirement_rules_table(connection, table_name: str = DEFAULT_REQUIREMENT_RULES_TABLE) -> None:
    """确保 Phase 2 requirement 规则表存在。"""
    ensure_lexicon_release_table(connection)
    qualified_table = quote_table_name(table_name)
    release_table = quote_table_name(DEFAULT_RELEASE_TABLE)
    connection.execute(
        text(
            f"""
            CREATE TABLE IF NOT EXISTS {qualified_table} (
                id bigserial PRIMARY KEY,
                release_id bigint NOT NULL REFERENCES {release_table}(id),
                rule_type text NOT NULL CHECK (rule_type IN ({_sql_in(REQUIREMENT_RULE_TYPES)})),
                dimension_name text,
                pattern_text text NOT NULL,
                replacement_text text NOT NULL DEFAULT '',
                normalized_value text NOT NULL DEFAULT '',
                operator text CHECK (operator IS NULL OR operator IN ({_sql_in(REQUIREMENT_OPERATORS)})),
                priority integer NOT NULL DEFAULT 100,
                enabled boolean NOT NULL DEFAULT true,
                source text CHECK (source IS NULL OR source IN ({_sql_in(RULE_SOURCES)})),
                notes text,
                created_at timestamptz NOT NULL DEFAULT now(),
                updated_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
    )
    connection.execute(
        text(
            f"""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_requirement_rules_unique
            ON {qualified_table} (
                release_id,
                rule_type,
                COALESCE(dimension_name, ''),
                pattern_text
            )
            """
        )
    )


def ensure_analysis_lexicon_schema(connection) -> None:
    """确保 analysis_lexicon 所有表存在。"""
    ensure_lexicon_release_table(connection)
    ensure_user_dictionary_table(connection)
    ensure_stopwords_table(connection)
    ensure_phrase_rules_table(connection)
    ensure_requirement_rules_table(connection)


def ensure_analysis_lexicon_schema_in_database() -> None:
    """在数据库中创建 analysis_lexicon 结构。"""
    engine = create_pg_engine()
    try:
        with engine.begin() as connection:
            ensure_analysis_lexicon_schema(connection)
    finally:
        engine.dispose()


def ensure_requirement_rules_seeded_for_current_release() -> bool:
    """若当前正式 release 缺少 requirement_rules，则补入默认规则。"""
    engine = create_pg_engine()
    payload = build_bootstrap_requirement_rules()
    try:
        with engine.begin() as connection:
            ensure_analysis_lexicon_schema(connection)
            release = load_current_release(connection)
            release_id = int(release["id"])
            qualified_table = quote_table_name(DEFAULT_REQUIREMENT_RULES_TABLE)
            existing_count = int(
                connection.execute(
                    text(
                        f"""
                        SELECT count(*)
                        FROM {qualified_table}
                        WHERE release_id = :release_id
                        """
                    ),
                    {"release_id": release_id},
                ).scalar_one()
            )
            if existing_count > 0:
                return False
            connection.execute(
                text(
                    f"""
                    INSERT INTO {qualified_table} (
                        release_id,
                        rule_type,
                        dimension_name,
                        pattern_text,
                        replacement_text,
                        normalized_value,
                        operator,
                        priority,
                        enabled,
                        source,
                        notes
                    )
                    VALUES (
                        :release_id,
                        :rule_type,
                        :dimension_name,
                        :pattern_text,
                        :replacement_text,
                        :normalized_value,
                        :operator,
                        :priority,
                        :enabled,
                        :source,
                        :notes
                    )
                    """
                ),
                [{**row, "release_id": release_id} for row in payload],
            )
            return True
    finally:
        engine.dispose()


def load_current_release(connection) -> dict[str, object]:
    """加载当前唯一正式 release。"""
    release_table = quote_table_name(DEFAULT_RELEASE_TABLE)
    rows = connection.execute(
        text(
            f"""
            SELECT id, version, is_current, released_at, released_by, notes
            FROM {release_table}
            WHERE is_current = true
            """
        )
    ).mappings().fetchall()
    if not rows:
        raise RuntimeError("analysis_lexicon 当前没有 is_current = true 的正式 release。")
    if len(rows) > 1:
        raise RuntimeError("analysis_lexicon 存在多个 is_current = true 的正式 release。")
    return dict(rows[0])


def _load_release_bound_dataframe(connection, table_name: str, release_id: int) -> pd.DataFrame:
    qualified_table = quote_table_name(table_name)
    return pd.read_sql_query(
        text(
            f"""
            SELECT *
            FROM {qualified_table}
            WHERE release_id = :release_id
            """
        ),
        connection,
        params={"release_id": int(release_id)},
    )


def load_current_lexicon_resources() -> dict[str, object]:
    """读取当前正式 release 下的三类资源。"""
    ensure_requirement_rules_seeded_for_current_release()
    engine = create_pg_engine()
    try:
        with engine.connect() as connection:
            release = load_current_release(connection)
            release_id = int(release["id"])
            user_dictionary_df = _load_release_bound_dataframe(
                connection,
                DEFAULT_USER_DICTIONARY_TABLE,
                release_id,
            )
            stopwords_df = _load_release_bound_dataframe(
                connection,
                DEFAULT_STOPWORDS_TABLE,
                release_id,
            )
            phrase_rules_df = _load_release_bound_dataframe(
                connection,
                DEFAULT_PHRASE_RULES_TABLE,
                release_id,
            )
            requirement_rules_df = _load_release_bound_dataframe(
                connection,
                DEFAULT_REQUIREMENT_RULES_TABLE,
                release_id,
            )
            return {
                "release": release,
                "user_dictionary": user_dictionary_df,
                "stopwords": stopwords_df,
                "phrase_rules": phrase_rules_df,
                "requirement_rules": requirement_rules_df,
            }
    finally:
        engine.dispose()


def build_lexicon_summary_frames(resources: dict[str, object]) -> dict[str, pd.DataFrame]:
    """构建词汇资源 summary DataFrame。"""
    summary_frames: dict[str, pd.DataFrame] = {}

    user_dictionary_df = resources["user_dictionary"].copy()
    stopwords_df = resources["stopwords"].copy()
    phrase_rules_df = resources["phrase_rules"].copy()
    requirement_rules_df = resources.get("requirement_rules", pd.DataFrame()).copy()

    if user_dictionary_df.empty:
        summary_frames["user_dictionary"] = _empty_summary_frame(
            ["term_type", "category", "enabled", "row_count"]
        )
    else:
        summary_frames["user_dictionary"] = (
            user_dictionary_df.groupby(["term_type", "category", "enabled"], dropna=False)
            .size()
            .reset_index(name="row_count")
            .sort_values(["term_type", "category", "enabled"])
        )

    if stopwords_df.empty:
        summary_frames["stopwords"] = _empty_summary_frame(
            ["scope", "stop_strength", "enabled", "row_count"]
        )
    else:
        summary_frames["stopwords"] = (
            stopwords_df.groupby(["scope", "stop_strength", "enabled"], dropna=False)
            .size()
            .reset_index(name="row_count")
            .sort_values(["scope", "stop_strength", "enabled"])
        )

    if phrase_rules_df.empty:
        summary_frames["phrase_rules"] = _empty_summary_frame(
            ["rule_type", "source", "enabled", "row_count"]
        )
    else:
        summary_frames["phrase_rules"] = (
            phrase_rules_df.groupby(["rule_type", "source", "enabled"], dropna=False)
            .size()
            .reset_index(name="row_count")
            .sort_values(["rule_type", "source", "enabled"])
        )

    if requirement_rules_df.empty:
        summary_frames["requirement_rules"] = _empty_summary_frame(
            ["rule_type", "dimension_name", "enabled", "row_count"]
        )
    else:
        summary_frames["requirement_rules"] = (
            requirement_rules_df.groupby(["rule_type", "dimension_name", "enabled"], dropna=False)
            .size()
            .reset_index(name="row_count")
            .sort_values(["rule_type", "dimension_name", "enabled"])
        )
    return summary_frames


def export_current_lexicon_snapshot(output_dir: str | Path) -> Path:
    """按需导出当前正式 lexicon 快照。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    resources = load_current_lexicon_resources()
    release = resources["release"]
    snapshot_payload = {
        "release": {
            key: str(value) for key, value in release.items()
        },
        "user_dictionary": resources["user_dictionary"].to_dict(orient="records"),
        "stopwords": resources["stopwords"].to_dict(orient="records"),
        "phrase_rules": resources["phrase_rules"].to_dict(orient="records"),
        "requirement_rules": resources["requirement_rules"].to_dict(orient="records"),
    }
    target = output_path / f"lexicon_snapshot_{release['version']}.json"
    target.write_text(
        json.dumps(snapshot_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return target


def build_parser() -> argparse.ArgumentParser:
    """构建 analysis_lexicon 管理 CLI。"""
    parser = argparse.ArgumentParser(description="analysis_lexicon schema 管理工具")
    parser.add_argument(
        "--ensure-schema",
        action="store_true",
        help="在 PostgreSQL 中创建 analysis_lexicon schema 及其表结构",
    )
    parser.add_argument(
        "--export-current-snapshot",
        default="",
        help="可选：导出当前正式 release 快照到指定目录",
    )
    parser.add_argument(
        "--bootstrap-v1",
        action="store_true",
        help="基于仓库 dicts 初始化首版 requirement analysis 正式 release",
    )
    parser.add_argument(
        "--version",
        default=DEFAULT_BOOTSTRAP_VERSION,
        help="bootstrap release 版本号，默认创建 v1 初始版本",
    )
    parser.add_argument(
        "--released-by",
        default=DEFAULT_RELEASED_BY,
        help="bootstrap release 发布人",
    )
    parser.add_argument(
        "--notes",
        default="Bootstrap from repository dicts for requirement text analysis v1.",
        help="bootstrap release 说明",
    )
    return parser


def main() -> None:
    """CLI 入口。"""
    args = build_parser().parse_args()
    if args.ensure_schema:
        ensure_analysis_lexicon_schema_in_database()
    if args.bootstrap_v1:
        summary = bootstrap_initial_release(
            version=str(args.version).strip(),
            released_by=str(args.released_by).strip(),
            notes=str(args.notes).strip(),
        )
        print(json.dumps(summary, ensure_ascii=False))
    if args.export_current_snapshot:
        export_current_lexicon_snapshot(args.export_current_snapshot)


if __name__ == "__main__":
    main()
