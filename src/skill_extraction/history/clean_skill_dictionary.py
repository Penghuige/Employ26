"""
清洗职业细类技能词典，移除软技能、学历门槛、福利待遇等非硬技能噪音。

设计目标：
1. 默认不覆盖原始词典，先生成一份清洗后的副本，降低误删风险。
2. 除了输出清洗后的 JSON，还同时输出详细审计报告，方便复核每一条为何被删除。
3. 规则尽量保守，优先删除“明确不应进入硬技能词典”的条目。

典型要删除的内容：
- 软技能：团队协作、学习能力、抗压能力、责任心、服务意识等
- 招聘门槛：学历要求、年龄要求、经验要求、证书优先
- 福利环境：五险一金、包吃包住、双休、工作环境、薪资待遇
- 已明确标注为“排除项”的条目

用法示例：
python -m src.skill_extraction.history.clean_skill_dictionary
python -m src.skill_extraction.history.clean_skill_dictionary --dry-run
python -m src.skill_extraction.history.clean_skill_dictionary --in-place
python -m src.skill_extraction.history.clean_skill_dictionary --input dicts/occupation_skill_dictionary.json --output dicts/occupation_skill_dictionary.cleaned.json
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
from typing import Dict, Iterable, List, Sequence

from ..config import load_skill_extraction_config


# 这些 skill_type 一旦出现，基本可以确定不是“硬技能词典”应保留的内容。
# 这里使用“精确匹配”为主，避免误删像“办公软件技能”这类正常技能分类。
SKILL_TYPE_BLACKLIST = {
    "软素质",
    "软技能",
    "福利待遇",
    "福利管理",
    "学历",
    "学历条件",
    "学历要求",
    "教育要求",
    "教育背景",
    "年龄要求",
    "工作时间",
    "工作环境",
    "经验要求",
    "资质要求",
    "资格条件",
    "资格认证",
    "政治背景",
    "职业背景要求",
    "健康资质",
    "人员筛选",
}


# 这些名称本身就是典型软技能或招聘条件，直接删除。
# 这里尽量只放“歧义极低”的词，降低误删真实专业技能的风险。
EXACT_NAME_BLACKLIST = {
    "团队协作",
    "团队合作",
    "团队协作能力",
    "团队协作精神",
    "团队合作精神",
    "团队建设",
    "团队管理经验",
    "学习能力",
    "培训学习能力",
    "抗压能力",
    "责任心",
    "执行力",
    "亲和力",
    "服务意识",
    "职业道德",
    "学历要求",
    "专科及以上学历",
    "大专学历",
    "本科及以上学历",
    "福利待遇",
    "薪资待遇",
    "工作时间",
    "工作环境",
    "年龄要求",
}


# 名称里出现这些词时，通常属于软技能或招聘条件。
# 但“沟通/协调”具有一定业务语义，因此不会单独因为这两个字就删；
# 对沟通类条目要走额外的“是否为泛化沟通能力”判定。
SUBSTRING_NAME_BLACKLIST = (
    "五险一金",
    "包吃",
    "包住",
    "双休",
    "福利",
    "宿舍",
    "工龄奖",
    "学历",
    "本科",
    "大专",
    "硕士",
    "博士",
    "年龄",
    "经验要求",
    "证书优先",
    "资质证书优先",
)


# 这些关键词若出现在 notes 中，基本说明该条目被标为排除项或招聘条件。
NOTE_EXCLUSION_MARKERS = (
    "排除项",
    "不纳入",
    "不计入",
    "福利内容",
    "学历门槛",
    "年龄限制",
)


NOTE_NON_SKILL_PATTERNS = (
    re.compile(r"(五险一金|带薪假期|带薪年休假|双休|周末双休|朝九晚六|包吃|包住|住宿条件|宿舍)"),
    re.compile(r"(薪资|工资|月收入|补贴|福利方案|福利待遇|福利发放)"),
    re.compile(r"(证书优先|资格证书优先|持有.*证书优先)"),
)


# 泛化软技能关键词。这里的词本身不是“业务对象”，更像“个人特质”。
GENERIC_SOFT_SKILL_KEYWORDS = (
    "团队",
    "协作",
    "合作",
    "学习能力",
    "抗压",
    "责任心",
    "执行力",
    "亲和力",
    "服务意识",
    "职业道德",
    "逻辑思维",
    "表达能力",
    "沟通能力",
    "协调能力",
    "组织协调",
    "领导力",
)


# 下列词出现在“沟通/协调”类名称中时，往往表示它仍然是岗位相关能力，
# 不应被简单归为泛化软技能。例如：
# - 英语沟通能力：本质上是语言能力
# - 商务沟通与谈判能力：本质上是商务技能
# - 医患沟通：本质上是医疗岗位专业沟通能力
DOMAIN_CONTEXT_WHITELIST = (
    "英语",
    "日语",
    "韩语",
    "俄语",
    "法语",
    "德语",
    "西班牙语",
    "阿拉伯语",
    "粤语",
    "朝鲜语",
    "外语",
    "商务",
    "谈判",
    "客户",
    "医患",
    "售后",
    "供应商",
    "政府",
    "飞行",
    "临床",
    "翻译",
    "多语言",
    "双语",
    "普通话",
    "国语",
    "前端",
    "后端",
    "前后端",
    "现场",
    "施工",
    "渠道",
    "电话",
    "直播",
    "面试",
)


LANGUAGE_SKILL_MARKERS = (
    "英语",
    "日语",
    "韩语",
    "俄语",
    "法语",
    "德语",
    "西班牙语",
    "阿拉伯语",
    "粤语",
    "朝鲜语",
    "外语",
    "多语言",
    "双语",
    "普通话",
    "国语",
)


# 这些词表明条目已经带有明确的技术对象、业务对象或任务对象。
# 当条目里同时出现 “团队/管理/协调/沟通/逻辑思维” 等软性成分时，
# 如果还能命中这些对象，优先认为它是“业务/技术能力”，不应直接删掉。
TECHNICAL_TASK_MARKERS = (
    "Java",
    "Python",
    "SQL",
    "ERP",
    "MES",
    "CRM",
    "OA",
    "APQP",
    "PPAP",
    "SPC",
    "MSA",
    "FMEA",
    "MPS",
    "MRP",
    "Office",
    "Excel",
    "Word",
    "PPT",
    "CAD",
    "BIM",
    "PLC",
    "系统",
    "平台",
    "开发",
    "运维",
    "部署",
    "架构",
    "代码",
    "编程",
    "算法",
    "模型",
    "数据库",
    "数据",
    "流程",
    "项目",
    "订单",
    "交期",
    "单据",
    "门店",
    "网点",
    "供应链",
    "生产",
    "制造",
    "工艺",
    "质量",
    "工具",
    "图纸",
    "文档",
    "公文",
    "文字",
    "写作",
    "文案",
    "方案",
    "策划",
    "报告",
    "财务",
    "会计",
    "税务",
    "审计",
    "销售",
    "采购",
    "招投标",
    "产品",
    "技术",
    "测试",
    "安全",
    "检验",
)


# 这些表达通常属于招聘措辞，而不是技能别名。
# 例如 “熟练使用Excel表格” 不应该作为 alias 留在最终词典里。
DESCRIPTIVE_ALIAS_MARKERS = (
    "熟练",
    "精通",
    "掌握",
    "具备",
    "能够",
    "良好",
    "较强",
    "优先",
    "经验者优先",
    "一年以上",
    "三年以上",
)


@dataclass
class AuditRow:
    """记录一次清洗动作，方便后续写出审计 CSV。"""

    action: str
    reason: str
    detail_path: str
    detail_name: str
    skill_name: str
    field: str
    original_text: str
    cleaned_text: str
    skill_type: str


def _safe_text(value: object) -> str:
    """安全转为字符串并去首尾空白。"""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _unique_keep_order(items: Iterable[object]) -> List[str]:
    """按原顺序去重。"""
    seen = set()
    result: List[str] = []
    for item in items:
        text = _safe_text(item)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _contains_any(text: str, patterns: Sequence[str]) -> bool:
    """判断文本是否包含任一关键词。"""
    return any(pattern in text for pattern in patterns)


def _needs_note_exclusion(note_text: str) -> bool:
    """notes 中出现明确排除语气时，整条技能直接删除。"""
    return _contains_any(note_text, NOTE_EXCLUSION_MARKERS)


def _contains_ascii_skill_token(text: str) -> bool:
    """判断文本里是否带有明显的英文缩写或技术代号。"""
    return bool(re.search(r"[A-Za-z][A-Za-z0-9_./#+-]{1,}", text))


def _has_hard_skill_context(text: str) -> bool:
    """判断文本是否带有明确的技术、业务或任务语境。"""
    return (
        _contains_any(text, DOMAIN_CONTEXT_WHITELIST)
        or _contains_any(text, TECHNICAL_TASK_MARKERS)
        or _contains_ascii_skill_token(text)
    )


def _has_normalizable_hard_component(text: str) -> bool:
    """判断文本是否适合被抽成更标准的硬技能主名称。

    这里比 `_has_hard_skill_context` 更严格：
    - 技术对象、任务对象、英文技术词可以
    - 纯“客户 / 政府 / 供应商”这类业务对象名词不够具体，不单独作为主技能名抽出
    """
    return (
        _contains_any(text, TECHNICAL_TASK_MARKERS)
        or _contains_any(text, LANGUAGE_SKILL_MARKERS)
        or _contains_ascii_skill_token(text)
    )


def _is_valid_normalized_skill_name(text: str) -> bool:
    """判断提取后的主技能名是否足够像“可落词典”的技能项。"""
    value = _safe_text(text)
    if not value:
        return False
    if len(value) < 3 and not _contains_ascii_skill_token(value) and not _contains_any(value, LANGUAGE_SKILL_MARKERS):
        return False
    if "团队" in value:
        return False
    if re.search(r"(意识|态度|积极性|作风|机构|忠诚度)$", value):
        return False
    return True


def _is_generic_communication_or_soft_skill(name_text: str) -> bool:
    """判断某条名称是否属于泛化软技能。

    规则说明：
    - 如果命中明显软技能词，直接删除。
    - 如果出现“沟通/协调/表达/逻辑思维”等泛化能力词，默认倾向删除。
    - 但如果同时出现语言、系统、流程、项目、数据等明确业务上下文，则保留。
    """
    if not _contains_any(name_text, GENERIC_SOFT_SKILL_KEYWORDS):
        return False
    if _has_hard_skill_context(name_text):
        return False
    return True


def _is_non_skill_requirement(name_text: str, skill_type_text: str, note_text: str) -> bool:
    """判断条目是否属于学历/福利/工时/年龄/经验等招聘条件。

    这里刻意不再把 notes 整体并入主判断。
    原因是很多真实技能的 notes 会写成：
    - 薪酬福利制度设计
    - 向客户宣导保险产品及福利方案
    这类说明里虽然带“福利”，但条目本身仍然是业务技能，不应因此被误删。
    """
    del note_text
    joined = " | ".join([name_text, skill_type_text])
    return _contains_any(joined, SUBSTRING_NAME_BLACKLIST)


def _strip_descriptive_wrappers(text: str) -> str:
    """去掉招聘措辞中的修饰词，尽量保留技能核心。

    这一步专门用来修复两类高频误删：
    - `APQP/PPAP/SPC/MSA/FMEA掌握` 这类“真实技能 + 掌握/熟练”
    - `良好的英语听说读写能力` 这类“真实技能 + 良好的/较强的”
    """
    value = _safe_text(text)
    if not value:
        return ""

    original = value
    patterns = [
        (r"^(具备|能够|可以|善于|熟练掌握|熟练使用|熟练运用|熟练操作|熟练|精通|掌握|了解|熟悉|良好的|较强的|较强|良好)\s*", ""),
        (r"\s*(掌握能力|应用能力|使用能力|操作能力)$", ""),
        (r"\s*(掌握|精通|熟练使用|熟练运用|熟练操作|熟练|熟悉|良好|较强)$", ""),
    ]
    for pattern, replacement in patterns:
        value = re.sub(pattern, replacement, value)

    value = re.sub(r"^[与和及、/]+", "", value)
    value = re.sub(r"[与和及、/]+$", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) <= 1:
        return original
    return value or original


def _strip_priority_suffix_if_hard_skill(text: str) -> str:
    """仅在文本本身明显像真实技能时，去掉“优先/优先考虑”等尾巴。

    例子：
    - ERP系统优先 -> ERP系统
    - 英语优先 -> 英语

    但像：
    - 电工证优先
    - 法律职业资格优先
    这类更像资格条件，不会在这里强行保留。
    """
    value = _safe_text(text)
    if not value:
        return ""

    stripped = re.sub(r"(优先考虑|优先录用|优先)$", "", value).strip()
    if stripped == value:
        return value
    if not _has_normalizable_hard_component(stripped):
        return value
    if re.search(r"(证|证书|资格|学历|专业背景|背景)$", stripped):
        return value
    return stripped


def _extract_hard_skill_component(text: str) -> str:
    """从“硬技能 + 软性成分”的混合短语中提取更像硬技能的部分。

    例子：
    - `Java开发平台与团队管理` -> `Java开发平台`
    - `订单交期跟踪与协调能力` -> `订单交期跟踪`
    """
    value = _safe_text(text)
    if not value or not any(token in value for token in ("与", "和", "及")):
        return value

    parts = [part.strip() for part in re.split(r"[与和及]", value) if part.strip()]
    if len(parts) < 2:
        return value

    kept_parts = [part for part in parts if _has_hard_skill_context(part) and not _is_generic_communication_or_soft_skill(part)]
    if not kept_parts:
        return value
    return " / ".join(_unique_keep_order(kept_parts))


def _normalize_alias_candidate(alias_text: str) -> tuple[str, str]:
    """对 alias 做保守归一化。"""
    normalized = _strip_descriptive_wrappers(alias_text)
    if normalized != _safe_text(alias_text):
        return normalized, "strip_descriptive_wrapper"
    normalized = _strip_priority_suffix_if_hard_skill(alias_text)
    if normalized != _safe_text(alias_text):
        return normalized, "strip_priority_suffix"
    return alias_text, ""


def _clean_split_part(part_text: str) -> str:
    """清理拆分后的候选片段。

    目标是把长句中的真实技能对象提取出来，同时去掉附着在后面的泛化包装词。
    """
    value = _strip_descriptive_wrappers(part_text)
    value = re.sub(r"(等)?(绘图及办公软件|绘图软件|办公软件|设计软件|软件工具|工具应用|应用工具|工具)$", "", value)
    value = re.sub(r"^[、，,;/；\\s]+", "", value)
    value = re.sub(r"[、，,;/；\\s]+$", "", value)
    return value.strip()


def _split_alias_candidates(alias_text: str) -> tuple[List[str], str]:
    """把长 alias 中并列出现的多个技能词拆出来。

    典型场景：
    - APQP/PPAP/SPC/MSA/FMEA掌握
    - Excel/Word/PowerPoint精通
    - 熟练操作CAD、pro-E或solidworks设计软件熟练、会使用办公软件

    返回多个候选 alias；如果无法安全拆分，则返回空列表。
    """
    value = _safe_text(alias_text)
    if not value:
        return [], ""

    normalized = _strip_descriptive_wrappers(value)
    if len(normalized) > 80:
        return [], ""

    has_series_separator = any(token in normalized for token in ("、", "，", ",", "；", ";", "/", "或", "以及"))
    ascii_tokens = re.findall(r"[A-Za-z][A-Za-z0-9.+#-]{1,}", normalized)

    # 英文缩写/工具并列时，优先直接拆 token，避免把长句拆出垃圾片段。
    if has_series_separator and len(ascii_tokens) >= 2:
        candidates = _unique_keep_order(ascii_tokens)
        return (candidates, "split_series_alias") if len(candidates) >= 2 else ([], "")

    split_pattern = r"[、，,；;]|/|或|以及"
    raw_parts = [part.strip() for part in re.split(split_pattern, normalized) if part.strip()]
    candidates: List[str] = []
    for part in raw_parts:
        cleaned = _clean_split_part(part)
        if not cleaned or cleaned == normalized:
            continue
        if _has_hard_skill_context(cleaned):
            candidates.append(cleaned)

    candidates = _unique_keep_order(candidates)
    return (candidates, "split_series_alias") if len(candidates) >= 2 else ([], "")


def _normalize_skill_candidate(skill_name: str) -> tuple[str, str]:
    """对 skill 主名称做保守归一化。"""
    stripped = _strip_descriptive_wrappers(skill_name)
    if stripped != _safe_text(skill_name):
        return stripped, "strip_descriptive_wrapper"
    stripped = _strip_priority_suffix_if_hard_skill(skill_name)
    if stripped != _safe_text(skill_name):
        return stripped, "strip_priority_suffix"

    extracted = _extract_hard_skill_component(skill_name)
    if extracted != _safe_text(skill_name):
        return extracted, "extract_hard_component"
    return skill_name, ""


def _should_remove_skill(skill: Dict) -> tuple[bool, str]:
    """判断整条技能是否应从词典中删除。"""
    name_text = _safe_text(skill.get("name", ""))
    skill_type_text = _safe_text(skill.get("skill_type", ""))
    note_text = _safe_text(skill.get("notes", ""))

    if not name_text:
        return True, "empty_name"
    if _needs_note_exclusion(note_text):
        return True, "note_marked_exclusion"
    if skill_type_text in SKILL_TYPE_BLACKLIST:
        return True, "skill_type_blacklist"
    if name_text in EXACT_NAME_BLACKLIST:
        return True, "exact_name_blacklist"
    if _is_non_skill_requirement(name_text, skill_type_text, note_text):
        return True, "non_skill_requirement"
    if _is_generic_communication_or_soft_skill(name_text):
        return True, "generic_soft_skill"
    return False, ""


def _should_drop_alias(alias_text: str, skill_name: str) -> tuple[bool, str]:
    """判断某个 alias 是否应删除。

    alias 的清洗比 skill 主名称更严格一些，因为 alias 更容易混入：
    - 招聘措辞
    - 条件描述
    - 解释性长短语
    """
    if not alias_text:
        return True, "empty_alias"
    if alias_text.casefold() == skill_name.casefold():
        return True, "duplicate_of_name"
    if _contains_any(alias_text, NOTE_EXCLUSION_MARKERS):
        return True, "alias_marked_exclusion"
    if _contains_any(alias_text, DESCRIPTIVE_ALIAS_MARKERS):
        return True, "descriptive_alias"
    if _contains_any(alias_text, SUBSTRING_NAME_BLACKLIST):
        return True, "alias_non_skill_requirement"
    if alias_text in EXACT_NAME_BLACKLIST:
        return True, "alias_exact_blacklist"
    if _is_generic_communication_or_soft_skill(alias_text):
        return True, "alias_generic_soft_skill"
    return False, ""


def _clean_note(note_text: str) -> tuple[str, str]:
    """清理 notes 字段。

    这里不尝试做复杂重写，只做“删脏不改义”：
    - 明确的排除项或招聘条件说明，直接清空
    - 其余说明保持原样，避免误删技能范围描述
    """
    if not note_text:
        return "", ""
    if _contains_any(note_text, NOTE_EXCLUSION_MARKERS):
        return "", "note_marked_exclusion"
    return note_text, ""


def _load_dictionary(path: Path) -> Dict:
    """读取词典 JSON。"""
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _save_json(path: Path, payload: Dict) -> None:
    """保存 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def _build_default_output_path(input_path: Path) -> Path:
    """默认输出为同目录下的 `.cleaned.json` 副本。"""
    return input_path.with_name(f"{input_path.stem}.cleaned.json")


def clean_dictionary(dictionary: Dict) -> tuple[Dict, Dict, List[AuditRow]]:
    """执行清洗并返回：

    1. 清洗后的词典
    2. 汇总统计
    3. 审计明细
    """
    cleaned = json.loads(json.dumps(dictionary, ensure_ascii=False))
    audit_rows: List[AuditRow] = []
    counters = Counter()

    categories = cleaned.setdefault("categories", {})
    for detail_path, category in categories.items():
        detail_name = _safe_text(category.get("detail_name", "")) or detail_path
        original_skills = category.get("skills", []) or []
        cleaned_skills: List[Dict] = []

        for skill in original_skills:
            name_text = _safe_text(skill.get("name", ""))
            skill_type_text = _safe_text(skill.get("skill_type", ""))
            note_text = _safe_text(skill.get("notes", ""))
            aliases = [_safe_text(alias) for alias in (skill.get("aliases", []) or [])]

            counters["skills_before"] += 1

            normalized_name, normalize_reason = _normalize_skill_candidate(name_text)
            if normalized_name != name_text:
                original_remove, _ = _should_remove_skill(skill)
                trial_skill = dict(skill)
                trial_skill["name"] = normalized_name
                normalized_remove, _ = _should_remove_skill(trial_skill)
                should_apply_normalization = original_remove and not normalized_remove
                if (
                    not should_apply_normalization
                    and normalize_reason == "extract_hard_component"
                    and _contains_any(name_text, GENERIC_SOFT_SKILL_KEYWORDS)
                    and _has_normalizable_hard_component(normalized_name)
                    and _is_valid_normalized_skill_name(normalized_name)
                    and not normalized_remove
                ):
                    should_apply_normalization = True

                if should_apply_normalization:
                    counters["skills_normalized"] += 1
                    counters[f"skill_normalized__{normalize_reason}"] += 1
                    audit_rows.append(
                        AuditRow(
                            action="normalize_skill",
                            reason=normalize_reason,
                            detail_path=detail_path,
                            detail_name=detail_name,
                            skill_name=name_text,
                            field="skill",
                            original_text=name_text,
                            cleaned_text=normalized_name,
                            skill_type=skill_type_text,
                        )
                    )
                    skill["name"] = normalized_name
                    if _safe_text(name_text) and _safe_text(name_text).casefold() != _safe_text(normalized_name).casefold():
                        skill.setdefault("aliases", [])
                        skill["aliases"] = _unique_keep_order([*skill.get("aliases", []), name_text])
                    name_text = normalized_name

            remove_skill, remove_reason = _should_remove_skill(skill)
            if remove_skill:
                counters["skills_removed"] += 1
                counters[f"removed__{remove_reason}"] += 1
                audit_rows.append(
                    AuditRow(
                        action="remove_skill",
                        reason=remove_reason,
                        detail_path=detail_path,
                        detail_name=detail_name,
                        skill_name=name_text,
                        field="skill",
                        original_text=name_text,
                        cleaned_text="",
                        skill_type=skill_type_text,
                    )
                )
                continue

            cleaned_aliases: List[str] = []
            for alias_text in aliases:
                split_aliases, split_reason = _split_alias_candidates(alias_text)
                if split_aliases:
                    original_drop, _ = _should_drop_alias(alias_text, skill_name=name_text)
                    surviving_split_aliases = [
                        candidate
                        for candidate in split_aliases
                        if not _should_drop_alias(candidate, skill_name=name_text)[0]
                    ]
                    if original_drop and surviving_split_aliases:
                        counters["aliases_split"] += len(surviving_split_aliases)
                        counters[f"alias_split__{split_reason}"] += len(surviving_split_aliases)
                        for candidate in surviving_split_aliases:
                            audit_rows.append(
                                AuditRow(
                                    action="split_alias",
                                    reason=split_reason,
                                    detail_path=detail_path,
                                    detail_name=detail_name,
                                    skill_name=name_text,
                                    field="aliases",
                                    original_text=alias_text,
                                    cleaned_text=candidate,
                                    skill_type=skill_type_text,
                                )
                            )
                            cleaned_aliases.append(candidate)
                        continue

                alias_candidate, alias_normalize_reason = _normalize_alias_candidate(alias_text)
                if alias_candidate != alias_text:
                    original_drop, _ = _should_drop_alias(alias_text, skill_name=name_text)
                    normalized_drop, _ = _should_drop_alias(alias_candidate, skill_name=name_text)
                    if original_drop and not normalized_drop:
                        counters["aliases_normalized"] += 1
                        counters[f"alias_normalized__{alias_normalize_reason}"] += 1
                        audit_rows.append(
                            AuditRow(
                                action="normalize_alias",
                                reason=alias_normalize_reason,
                                detail_path=detail_path,
                                detail_name=detail_name,
                                skill_name=name_text,
                                field="aliases",
                                original_text=alias_text,
                                cleaned_text=alias_candidate,
                                skill_type=skill_type_text,
                            )
                        )
                        cleaned_aliases.append(alias_candidate)
                        continue

                drop_alias, alias_reason = _should_drop_alias(alias_text, skill_name=name_text)
                if drop_alias:
                    counters["aliases_removed"] += 1
                    counters[f"alias_removed__{alias_reason}"] += 1
                    audit_rows.append(
                        AuditRow(
                            action="remove_alias",
                            reason=alias_reason,
                            detail_path=detail_path,
                            detail_name=detail_name,
                            skill_name=name_text,
                            field="aliases",
                            original_text=alias_text,
                            cleaned_text="",
                            skill_type=skill_type_text,
                        )
                    )
                    continue
                cleaned_aliases.append(alias_text)

            deduped_aliases = _unique_keep_order(cleaned_aliases)
            if len(deduped_aliases) != len(cleaned_aliases):
                counters["aliases_deduplicated"] += len(cleaned_aliases) - len(deduped_aliases)

            cleaned_note_text, note_reason = _clean_note(note_text)
            if note_reason:
                counters["notes_cleared"] += 1
                counters[f"note_cleared__{note_reason}"] += 1
                audit_rows.append(
                    AuditRow(
                        action="clear_note",
                        reason=note_reason,
                        detail_path=detail_path,
                        detail_name=detail_name,
                        skill_name=name_text,
                        field="notes",
                        original_text=note_text,
                        cleaned_text=cleaned_note_text,
                        skill_type=skill_type_text,
                    )
                )

            skill["aliases"] = deduped_aliases
            skill["notes"] = cleaned_note_text
            cleaned_skills.append(skill)

        category["skills"] = cleaned_skills
        counters["skills_after"] += len(cleaned_skills)
        if cleaned_skills:
            counters["categories_with_skills_after"] += 1

    metadata = cleaned.setdefault("metadata", {})
    metadata["cleaned_at"] = datetime.now().isoformat(timespec="seconds")
    metadata["cleaning_summary"] = {
        "skills_before": counters["skills_before"],
        "skills_after": counters["skills_after"],
        "skills_removed": counters["skills_removed"],
        "skills_normalized": counters["skills_normalized"],
        "aliases_removed": counters["aliases_removed"],
        "aliases_split": counters["aliases_split"],
        "aliases_normalized": counters["aliases_normalized"],
        "aliases_deduplicated": counters["aliases_deduplicated"],
        "notes_cleared": counters["notes_cleared"],
    }

    summary = {
        "categories_total": len(categories),
        "categories_with_skills_after": counters["categories_with_skills_after"],
        "skills_before": counters["skills_before"],
        "skills_after": counters["skills_after"],
        "skills_removed": counters["skills_removed"],
        "skills_normalized": counters["skills_normalized"],
        "aliases_removed": counters["aliases_removed"],
        "aliases_split": counters["aliases_split"],
        "aliases_normalized": counters["aliases_normalized"],
        "aliases_deduplicated": counters["aliases_deduplicated"],
        "notes_cleared": counters["notes_cleared"],
        "reasons": dict(sorted(counters.items())),
    }
    return cleaned, summary, audit_rows


def _write_audit_report(report_dir: Path, summary: Dict, audit_rows: Sequence[AuditRow]) -> None:
    """写出 JSON 汇总和 CSV 明细报告。"""
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_path = report_dir / "cleaning_summary.json"
    detail_path = report_dir / "cleaning_details.csv"

    with open(summary_path, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)

    with open(detail_path, "w", encoding="utf-8-sig", newline="") as file_obj:
        import csv

        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "action",
                "reason",
                "detail_path",
                "detail_name",
                "skill_name",
                "field",
                "original_text",
                "cleaned_text",
                "skill_type",
            ],
        )
        writer.writeheader()
        for row in audit_rows:
            writer.writerow(asdict(row))


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数。"""
    parser = argparse.ArgumentParser(description="清洗职业技能词典中的软技能与招聘噪音")
    parser.add_argument("--input", help="输入词典路径，默认使用 dicts/occupation_skill_dictionary.json")
    parser.add_argument("--output", help="输出词典路径；未指定时默认输出为 *.cleaned.json")
    parser.add_argument("--report-dir", help="审计报告输出目录；未指定时写到 output/skill_extraction/reports/dictionary_cleaning/<timestamp>")
    parser.add_argument("--in-place", action="store_true", help="直接覆盖输入词典；会自动备份原文件")
    parser.add_argument("--dry-run", action="store_true", help="只生成审计报告，不写清洗后的词典")
    return parser


def main() -> None:
    """CLI 入口。"""
    parser = build_parser()
    args = parser.parse_args()

    config = load_skill_extraction_config()
    input_path = Path(args.input) if args.input else config.dictionary_path
    if not input_path.exists():
        raise FileNotFoundError(f"词典不存在: {input_path}")

    if args.in_place and args.output:
        raise ValueError("--in-place 与 --output 不能同时使用")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = (
        Path(args.report_dir)
        if args.report_dir
        else config.report_dir / "dictionary_cleaning" / timestamp
    )

    if args.in_place:
        output_path = input_path
    elif args.output:
        output_path = Path(args.output)
    else:
        output_path = _build_default_output_path(input_path)

    dictionary = _load_dictionary(input_path)
    cleaned_dictionary, summary, audit_rows = clean_dictionary(dictionary)
    summary["input_path"] = str(input_path)
    summary["output_path"] = str(output_path)
    summary["dry_run"] = bool(args.dry_run)
    summary["generated_at"] = datetime.now().isoformat(timespec="seconds")

    _write_audit_report(report_dir=report_dir, summary=summary, audit_rows=audit_rows)

    if not args.dry_run:
        if args.in_place:
            backup_path = input_path.with_name(f"{input_path.stem}.backup_{timestamp}.json")
            shutil.copy2(input_path, backup_path)
            summary["backup_path"] = str(backup_path)
            _write_audit_report(report_dir=report_dir, summary=summary, audit_rows=audit_rows)
        _save_json(output_path, cleaned_dictionary)

    print(f"输入词典: {input_path}")
    print(f"输出词典: {output_path}")
    print(f"审计目录: {report_dir}")
    print(f"技能条目: {summary['skills_before']} -> {summary['skills_after']}")
    print(f"删除技能: {summary['skills_removed']}")
    print(f"删除 alias: {summary['aliases_removed']}")
    print(f"清空 notes: {summary['notes_cleared']}")
    if args.dry_run:
        print("当前为 dry-run，未写出清洗后的词典文件。")


if __name__ == "__main__":
    main()
