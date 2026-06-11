"""Phase 2 requirement 条目切分、约束抽取与噪声识别。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re
from typing import Any

from src.db.analysis_lexicon import normalize_lexicon_term
from src.db.requirement_constraint_facts import RequirementConstraintFactRow


DEFAULT_EXTRACTOR_VERSION = "requirement_constraint_extractor_v2"
DEGREE_RANKS = {
    "高中": 1,
    "中专": 2,
    "大专": 3,
    "本科": 4,
    "硕士": 5,
    "博士": 6,
}
CHINESE_DIGITS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


@dataclass(frozen=True)
class RequirementRule:
    """从 PostgreSQL requirement_rules 读取出的规则。"""

    id: int | None
    rule_type: str
    dimension_name: str
    pattern_text: str
    replacement_text: str
    normalized_value: str
    operator: str
    priority: int
    enabled: bool
    source: str
    notes: str


@dataclass(frozen=True)
class ExtractedConstraint:
    """单条 item 中抽取出的约束。"""

    item_index: int
    item_text_raw: str
    item_text_normalized: str
    dimension_name: str
    constraint_type: str
    raw_value: str
    normalized_value: str
    operator: str
    value_min: float | None
    value_max: float | None
    unit: str
    evidence_text: str
    rule_id: int | None


@dataclass(frozen=True)
class TemplateNoiseHit:
    """单条 item 上识别出的模板噪声。"""

    item_index: int
    item_text_raw: str
    item_text_normalized: str
    noise_type: str
    noise_text: str
    rule_id: int | None
    rule_source: str


@dataclass(frozen=True)
class RequirementItem:
    """切分后的 requirement item。"""

    item_index: int
    raw_text: str
    normalized_text: str


@dataclass(frozen=True)
class ExtractionResult:
    """单条招聘 requirement text 的抽取结果。"""

    items: list[RequirementItem]
    constraints: list[ExtractedConstraint]
    template_noise_hits: list[TemplateNoiseHit]
    reliable_itemization: bool


def parse_requirement_rules(requirement_rules_df) -> dict[str, list[RequirementRule]]:
    """把 requirement_rules DataFrame 转成按 rule_type 分组的规则。"""
    grouped: dict[str, list[RequirementRule]] = defaultdict(list)
    if requirement_rules_df is None or requirement_rules_df.empty:
        return grouped

    sort_columns = [column for column in ["priority", "id"] if column in requirement_rules_df.columns]
    work_df = requirement_rules_df.sort_values(sort_columns, ascending=True) if sort_columns else requirement_rules_df
    for _, row in work_df.iterrows():
        if not bool(row.get("enabled", True)):
            continue
        rule = RequirementRule(
            id=int(row["id"]) if row.get("id") is not None and str(row.get("id")) != "nan" else None,
            rule_type=str(row.get("rule_type", "")).strip(),
            dimension_name=str(row.get("dimension_name", "")).strip(),
            pattern_text=str(row.get("pattern_text", "")).strip(),
            replacement_text=str(row.get("replacement_text", "") or "").strip(),
            normalized_value=str(row.get("normalized_value", "") or "").strip(),
            operator=str(row.get("operator", "") or "").strip(),
            priority=int(row.get("priority", 100) or 100),
            enabled=bool(row.get("enabled", True)),
            source=str(row.get("source", "") or "").strip(),
            notes=str(row.get("notes", "") or "").strip(),
        )
        if rule.rule_type and rule.pattern_text:
            grouped[rule.rule_type].append(rule)
    return grouped


def chinese_numeral_to_int(text_value: str) -> int | None:
    """将常见中文数字转为整数，覆盖年龄/经验场景。"""
    raw = str(text_value or "").strip()
    if not raw:
        return None
    if raw.isdigit():
        return int(raw)
    if raw == "十":
        return 10
    if "十" in raw:
        left, right = raw.split("十", 1)
        tens = CHINESE_DIGITS.get(left, 1 if left == "" else None)
        if tens is None:
            return None
        ones = CHINESE_DIGITS.get(right, 0 if right == "" else None)
        if ones is None:
            return None
        return tens * 10 + ones
    if raw in CHINESE_DIGITS:
        return CHINESE_DIGITS[raw]
    return None


def normalize_numeric_expressions(text_value: str) -> str:
    """把中文数字的年/岁/级表达转成阿拉伯数字。"""
    work_text = str(text_value or "")

    def _replace_range(match: re.Match[str]) -> str:
        left = chinese_numeral_to_int(match.group(1))
        right = chinese_numeral_to_int(match.group(2))
        unit = match.group(3)
        if left is None or right is None:
            return match.group(0)
        return f"{left}-{right}{unit}"

    def _replace_single(match: re.Match[str]) -> str:
        number = chinese_numeral_to_int(match.group(1))
        unit = match.group(2)
        if number is None:
            return match.group(0)
        return f"{number}{unit}"

    chinese_digits_pattern = r"[零一二两三四五六七八九十]+"
    work_text = re.sub(
        rf"({chinese_digits_pattern})\s*[-~至到]\s*({chinese_digits_pattern})(年|岁|级)",
        _replace_range,
        work_text,
    )
    work_text = re.sub(
        rf"({chinese_digits_pattern})(年|岁|级)",
        _replace_single,
        work_text,
    )
    return work_text


def apply_item_split_rules(text_value: str, rules: list[RequirementRule] | None = None) -> str:
    """应用 item 切分预处理规则。"""
    work_text = str(text_value or "")
    if rules:
        for rule in rules:
            work_text = work_text.replace(rule.pattern_text, rule.replacement_text or " | ")
    work_text = work_text.replace("\r\n", "\n").replace("\r", "\n")
    work_text = re.sub(r"[•●■▪◆★]+", " | ", work_text)
    work_text = re.sub(r"[;；]+", " | ", work_text)
    work_text = re.sub(r"\n+", " | ", work_text)
    work_text = re.sub(r"(?:^|\s)[（(]?\d+[）).、]", " | ", work_text)
    work_text = re.sub(r"\s*\|\s*", " | ", work_text)
    return work_text


def split_requirement_items(
    requirements_text: str,
    split_rules: list[RequirementRule] | None = None,
) -> tuple[list[str], bool]:
    """切分 requirement items，并返回是否可靠切分。"""
    prepared = apply_item_split_rules(requirements_text, split_rules)
    text_value = str(prepared or "").strip()
    if not text_value:
        return [], False
    if " | " in text_value:
        items = [item.strip(" ,，。；;|") for item in text_value.split(" | ") if item.strip(" ,，。；;|")]
        return items, len(items) >= 2
    items = [item.strip() for item in re.split(r"[\n；;]+", text_value) if item.strip()]
    if len(items) >= 2:
        return items, True
    return [text_value], False


def normalize_item_text(item_text: str, normalize_rules: list[RequirementRule] | None = None) -> str:
    """对单条 item 做 Phase 2 规范化。"""
    work_text = normalize_numeric_expressions(str(item_text or "").strip())
    work_text = work_text.replace("性别不限", "男女不限")
    work_text = work_text.replace("不限性别", "男女不限")
    work_text = work_text.replace("普通话流利", "普通话标准")
    work_text = re.sub(r"office", "Office", work_text, flags=re.IGNORECASE)
    work_text = re.sub(r"excel", "Excel", work_text, flags=re.IGNORECASE)
    work_text = re.sub(r"word", "Word", work_text, flags=re.IGNORECASE)
    work_text = re.sub(r"ppt", "PPT", work_text, flags=re.IGNORECASE)
    work_text = re.sub(r"英语\s*4级", "英语四级", work_text, flags=re.IGNORECASE)
    work_text = re.sub(r"cet[-\s]*4", "英语四级", work_text, flags=re.IGNORECASE)
    if normalize_rules:
        for rule in normalize_rules:
            replacement = rule.replacement_text or rule.normalized_value or rule.pattern_text
            work_text = work_text.replace(rule.pattern_text, replacement)
    work_text = re.sub(r"\s+", " ", work_text)
    return work_text.strip(" ,，。；;")


def _infer_noise_type(rule: RequirementRule) -> str:
    notes = rule.notes.lower()
    if "generic" in notes:
        return "generic_token"
    if "platform" in notes:
        return "platform_phrase"
    return "template_phrase"


def detect_template_noise(
    item_index: int,
    item_text_raw: str,
    item_text_normalized: str,
    noise_rules: list[RequirementRule] | None = None,
) -> list[TemplateNoiseHit]:
    """识别 item 内的模板噪声。"""
    hits: list[TemplateNoiseHit] = []
    normalized_item = normalize_lexicon_term(item_text_normalized)
    if not normalized_item:
        return hits
    for rule in noise_rules or []:
        normalized_pattern = normalize_lexicon_term(rule.pattern_text)
        if normalized_pattern and normalized_pattern in normalized_item:
            hits.append(
                TemplateNoiseHit(
                    item_index=item_index,
                    item_text_raw=item_text_raw,
                    item_text_normalized=item_text_normalized,
                    noise_type=_infer_noise_type(rule),
                    noise_text=rule.pattern_text,
                    rule_id=rule.id,
                    rule_source=rule.source,
                )
            )
    return hits


def _dedupe_constraints(rows: list[ExtractedConstraint]) -> list[ExtractedConstraint]:
    seen: set[tuple[Any, ...]] = set()
    deduped: list[ExtractedConstraint] = []
    for row in rows:
        key = (
            row.item_index,
            row.dimension_name,
            row.normalized_value,
            row.operator,
            row.constraint_type,
        )
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    return deduped


def _make_constraint(
    *,
    item_index: int,
    item_text_raw: str,
    item_text_normalized: str,
    dimension_name: str,
    constraint_type: str,
    raw_value: str,
    normalized_value: str,
    operator: str,
    value_min: float | None = None,
    value_max: float | None = None,
    unit: str = "",
    evidence_text: str = "",
    rule_id: int | None = None,
) -> ExtractedConstraint:
    return ExtractedConstraint(
        item_index=item_index,
        item_text_raw=item_text_raw,
        item_text_normalized=item_text_normalized,
        dimension_name=dimension_name,
        constraint_type=constraint_type,
        raw_value=raw_value,
        normalized_value=normalized_value,
        operator=operator,
        value_min=value_min,
        value_max=value_max,
        unit=unit,
        evidence_text=evidence_text or raw_value or normalized_value,
        rule_id=rule_id,
    )


def extract_experience_constraints(
    item_index: int,
    item_text_raw: str,
    item_text_normalized: str,
) -> list[ExtractedConstraint]:
    """抽取经验年限约束。"""
    rows: list[ExtractedConstraint] = []
    text_value = item_text_normalized
    for match in re.finditer(r"(?P<min>\d+)\s*[-~至到]\s*(?P<max>\d+)\s*年", text_value):
        left = float(match.group("min"))
        right = float(match.group("max"))
        rows.append(
            _make_constraint(
                item_index=item_index,
                item_text_raw=item_text_raw,
                item_text_normalized=item_text_normalized,
                dimension_name="experience",
                constraint_type="range",
                raw_value=match.group(0),
                normalized_value=f"{int(left)}-{int(right)}年",
                operator="between",
                value_min=left,
                value_max=right,
                unit="年",
                evidence_text=match.group(0),
            )
        )
    if rows:
        return rows

    for pattern in (
        r"(?P<min>\d+)\s*年以上",
        r"至少\s*(?P<min>\d+)\s*年",
        r"(?P<min>\d+)\s*年(?:以上)?(?:工作)?经验",
    ):
        match = re.search(pattern, text_value)
        if match:
            left = float(match.group("min"))
            rows.append(
                _make_constraint(
                    item_index=item_index,
                    item_text_raw=item_text_raw,
                    item_text_normalized=item_text_normalized,
                    dimension_name="experience",
                    constraint_type="hard_gate",
                    raw_value=match.group(0),
                    normalized_value=f"{int(left)}年",
                    operator="ge",
                    value_min=left,
                    unit="年",
                    evidence_text=match.group(0),
                )
            )
            return rows

    match = re.search(r"(?<![-~至到])(?P<min>\d+)\s*年(?!\s*[-~至到])", text_value)
    if match and ("经验" in text_value or "年以上" not in text_value):
        left = float(match.group("min"))
        rows.append(
            _make_constraint(
                item_index=item_index,
                item_text_raw=item_text_raw,
                item_text_normalized=item_text_normalized,
                dimension_name="experience",
                constraint_type="hard_gate",
                raw_value=match.group(0),
                normalized_value=f"{int(left)}年",
                operator="ge",
                value_min=left,
                unit="年",
                evidence_text=match.group(0),
            )
        )
    return rows


def extract_education_constraints(
    item_index: int,
    item_text_raw: str,
    item_text_normalized: str,
) -> list[ExtractedConstraint]:
    """抽取学历约束。"""
    rows: list[ExtractedConstraint] = []
    for degree, rank in sorted(DEGREE_RANKS.items(), key=lambda item: item[1], reverse=True):
        if degree in item_text_normalized:
            rows.append(
                _make_constraint(
                    item_index=item_index,
                    item_text_raw=item_text_raw,
                    item_text_normalized=item_text_normalized,
                    dimension_name="education",
                    constraint_type="hard_gate",
                    raw_value=degree,
                    normalized_value=degree,
                    operator="ge",
                    value_min=float(rank),
                    unit="degree_rank",
                    evidence_text=degree,
                )
            )
            break
    return rows


def extract_age_constraints(
    item_index: int,
    item_text_raw: str,
    item_text_normalized: str,
) -> list[ExtractedConstraint]:
    """抽取年龄约束。"""
    rows: list[ExtractedConstraint] = []
    match = re.search(r"(?P<min>\d+)\s*[-~至到]\s*(?P<max>\d+)\s*岁", item_text_normalized)
    if match:
        left = float(match.group("min"))
        right = float(match.group("max"))
        rows.append(
            _make_constraint(
                item_index=item_index,
                item_text_raw=item_text_raw,
                item_text_normalized=item_text_normalized,
                dimension_name="age",
                constraint_type="range",
                raw_value=match.group(0),
                normalized_value=f"{int(left)}-{int(right)}岁",
                operator="between",
                value_min=left,
                value_max=right,
                unit="岁",
                evidence_text=match.group(0),
            )
        )
        return rows
    match = re.search(r"(?P<max>\d+)\s*岁(?:以下|以内)", item_text_normalized)
    if match:
        right = float(match.group("max"))
        rows.append(
            _make_constraint(
                item_index=item_index,
                item_text_raw=item_text_raw,
                item_text_normalized=item_text_normalized,
                dimension_name="age",
                constraint_type="hard_gate",
                raw_value=match.group(0),
                normalized_value=f"{int(right)}岁以下",
                operator="le",
                value_max=right,
                unit="岁",
                evidence_text=match.group(0),
            )
        )
        return rows
    match = re.search(r"(?P<min>\d+)\s*岁以上", item_text_normalized)
    if match:
        left = float(match.group("min"))
        rows.append(
            _make_constraint(
                item_index=item_index,
                item_text_raw=item_text_raw,
                item_text_normalized=item_text_normalized,
                dimension_name="age",
                constraint_type="hard_gate",
                raw_value=match.group(0),
                normalized_value=f"{int(left)}岁以上",
                operator="ge",
                value_min=left,
                unit="岁",
                evidence_text=match.group(0),
            )
        )
    return rows


def extract_gender_constraints(
    item_index: int,
    item_text_raw: str,
    item_text_normalized: str,
) -> list[ExtractedConstraint]:
    """抽取性别限制。"""
    text_value = item_text_normalized
    if "男女不限" in text_value:
        return [
            _make_constraint(
                item_index=item_index,
                item_text_raw=item_text_raw,
                item_text_normalized=item_text_normalized,
                dimension_name="gender",
                constraint_type="binary",
                raw_value="男女不限",
                normalized_value="不限",
                operator="allow",
                evidence_text="男女不限",
            )
        ]
    for pattern, normalized_value in (
        (r"(仅限|限)\s*(女性|女)", "女性"),
        (r"(仅限|限)\s*(男性|男)", "男性"),
    ):
        match = re.search(pattern, text_value)
        if match:
            return [
                _make_constraint(
                    item_index=item_index,
                    item_text_raw=item_text_raw,
                    item_text_normalized=item_text_normalized,
                    dimension_name="gender",
                    constraint_type="hard_gate",
                    raw_value=match.group(0),
                    normalized_value=normalized_value,
                    operator="eq",
                    evidence_text=match.group(0),
                )
            ]
    return []


def extract_rule_based_constraints(
    item_index: int,
    item_text_raw: str,
    item_text_normalized: str,
    extract_rules: list[RequirementRule] | None = None,
) -> list[ExtractedConstraint]:
    """抽取基于 requirement_rules 的确定性约束。"""
    rows: list[ExtractedConstraint] = []
    normalized_item = normalize_lexicon_term(item_text_normalized)
    if not normalized_item:
        return rows
    for rule in extract_rules or []:
        normalized_pattern = normalize_lexicon_term(rule.pattern_text)
        if not normalized_pattern or normalized_pattern not in normalized_item:
            continue
        normalized_value = rule.normalized_value or rule.pattern_text
        if rule.dimension_name in {"certificate", "language"}:
            constraint_type = "hard_gate"
        else:
            constraint_type = "binary"
        rows.append(
            _make_constraint(
                item_index=item_index,
                item_text_raw=item_text_raw,
                item_text_normalized=item_text_normalized,
                dimension_name=rule.dimension_name,
                constraint_type=constraint_type,
                raw_value=rule.pattern_text,
                normalized_value=normalized_value,
                operator=rule.operator or "contains",
                evidence_text=rule.pattern_text,
                rule_id=rule.id,
            )
        )
    return rows


def extract_constraints_from_item(
    item_index: int,
    item_text_raw: str,
    item_text_normalized: str,
    extract_rules: list[RequirementRule] | None = None,
) -> list[ExtractedConstraint]:
    """从单条 item 中抽取全部正式约束。"""
    rows: list[ExtractedConstraint] = []
    rows.extend(extract_experience_constraints(item_index, item_text_raw, item_text_normalized))
    rows.extend(extract_education_constraints(item_index, item_text_raw, item_text_normalized))
    rows.extend(extract_age_constraints(item_index, item_text_raw, item_text_normalized))
    rows.extend(extract_gender_constraints(item_index, item_text_raw, item_text_normalized))
    rows.extend(extract_rule_based_constraints(item_index, item_text_raw, item_text_normalized, extract_rules))
    return _dedupe_constraints(rows)


def extract_requirement_constraints(
    requirements_text: str,
    rules_by_type: dict[str, list[RequirementRule]] | None = None,
) -> ExtractionResult:
    """对整段 requirement text 执行切分、规范化、噪声识别和约束抽取。"""
    rules_by_type = rules_by_type or {}
    raw_items, reliable_itemization = split_requirement_items(
        requirements_text,
        split_rules=rules_by_type.get("item_split", []),
    )
    items: list[RequirementItem] = []
    constraints: list[ExtractedConstraint] = []
    template_noise_hits: list[TemplateNoiseHit] = []
    for index, item_text_raw in enumerate(raw_items, start=1):
        item_text_normalized = normalize_item_text(
            item_text_raw,
            normalize_rules=rules_by_type.get("normalize", []),
        )
        item = RequirementItem(
            item_index=index,
            raw_text=item_text_raw,
            normalized_text=item_text_normalized,
        )
        items.append(item)
        template_noise_hits.extend(
            detect_template_noise(
                item_index=index,
                item_text_raw=item_text_raw,
                item_text_normalized=item_text_normalized,
                noise_rules=rules_by_type.get("template_noise", []),
            )
        )
        constraints.extend(
            extract_constraints_from_item(
                item_index=index,
                item_text_raw=item_text_raw,
                item_text_normalized=item_text_normalized,
                extract_rules=rules_by_type.get("extract", []),
            )
        )
    return ExtractionResult(
        items=items,
        constraints=_dedupe_constraints(constraints),
        template_noise_hits=template_noise_hits,
        reliable_itemization=reliable_itemization,
    )


def convert_constraints_to_fact_rows(
    *,
    recruitment_record_id: str,
    source_table: str,
    source_row_number: int,
    constraints: list[ExtractedConstraint],
    extractor_version: str = DEFAULT_EXTRACTOR_VERSION,
) -> list[RequirementConstraintFactRow]:
    """把抽取结果转成数据库写入行。"""
    rows: list[RequirementConstraintFactRow] = []
    for row in constraints:
        rows.append(
            RequirementConstraintFactRow(
                recruitment_record_id=recruitment_record_id,
                source_table=source_table,
                source_row_number=int(source_row_number),
                item_index=int(row.item_index),
                item_text_raw=row.item_text_raw,
                item_text_normalized=row.item_text_normalized,
                dimension_name=row.dimension_name,
                constraint_type=row.constraint_type,
                raw_value=row.raw_value,
                normalized_value=row.normalized_value,
                operator=row.operator,
                value_min=row.value_min,
                value_max=row.value_max,
                unit=row.unit,
                evidence_text=row.evidence_text,
                rule_id=row.rule_id,
                extractor_version=extractor_version,
            )
        )
    return rows
