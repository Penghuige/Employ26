"""
构建“跨职业合并版”硬技能词典，供全局匹配脚本使用。

设计目标：
1. 输入是按职业细类分层的硬技能词典，例如 `occupation_skill_dictionary_v2.4.json`；
2. 输出是一份“单池化”的全局词典，减少跨职业重复技能带来的重复匹配；
3. 在合并过程中，继续复用现有清洗规则，保守移除非硬技能；
4. 额外清理一批“只适合职业内上下文，不适合全局匹配”的低区分度泛化标签。

为什么需要单独做这一步：
1. 原词典按职业细类分层，同名技能会在多个细类里重复出现；
2. 当匹配脚本放开职业细类限制后，这些重复词条会增加匹配开销，也更容易造成泛化匹配；
3. 对于“全局匹配”场景，更适合使用一份已经去重、合并 alias、收紧泛化词的单池词典。

这份脚本并不会重写你精修过的职业细类词典，而是生成一个新的“全局合并版”副本。

用法示例：
python -m src.skill_extraction.build_merged_hard_skill_dictionary ^
  --input dicts/occupation_skill_dictionary_v2.4.json ^
  --output dicts/occupation_skill_dictionary_v2.5_merged.json
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Dict, Iterable, List, Tuple

from .clean_skill_dictionary import clean_dictionary


MERGED_POOL_PATH = "通用技能池 > 合并硬技能词典"

# 这类词在“职业细类词典”里未必完全错误，但一旦放到全局池里，区分度太低，
# 会明显增加泛化误匹配，因此在合并版词典里直接去掉。
LOW_SIGNAL_EXACT_BLACKLIST = {
    "问题解决",
    "应急处理",
    "异常处理",
    "数据记录",
    "数据录入",
    "工具使用",
    "流程执行",
}

# 这类后缀如果同时满足“高频跨职业复用”与“名称过泛”，在全局匹配里价值较低。
LOW_SIGNAL_SUFFIX_PATTERN = re.compile(r"(操作|维护|控制|规范|检测|记录|使用|应用|流程|处理|执行)$")


SKILL_TYPE_PRIORITY = [
    "软件/编程",
    "业务系统/数据库",
    "设备/仪器",
    "工艺/方法",
    "办公软件",
    "专业知识",
    "语言能力",
    "证书/资质",
]


@dataclass
class MergeAuditRow:
    """记录合并或过滤动作，便于输出审计报告。"""

    action: str
    reason: str
    skill_name: str
    cleaned_text: str
    source_category_count: int
    source_detail_paths: str


def _safe_text(value: object) -> str:
    """安全转字符串并去首尾空白。"""
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


def _contains_ascii_token(text: str) -> bool:
    """判断文本里是否带有英文/缩写 token。"""
    return bool(re.search(r"[A-Za-z][A-Za-z0-9.+#/\-]*", text))


def _choose_skill_type(skill_types: Counter) -> str:
    """
    为合并后的 skill 选一个代表性的 skill_type。

    规则：
    1. 优先取出现次数最高的类型；
    2. 若次数相同，则按预设优先级选更具体的类型；
    3. 最后再按名称排序兜底，保证输出稳定。
    """
    if not skill_types:
        return ""

    def sort_key(item: Tuple[str, int]) -> Tuple[int, int, str]:
        skill_type, count = item
        try:
            priority = SKILL_TYPE_PRIORITY.index(skill_type)
        except ValueError:
            priority = len(SKILL_TYPE_PRIORITY)
        return (-count, priority, skill_type)

    return sorted(skill_types.items(), key=sort_key)[0][0]


def _should_drop_low_signal_global_skill(skill_name: str, source_category_count: int) -> tuple[bool, str]:
    """
    判断某个条目是否应从“全局合并版”词典中移除。

    这里删除的并不一定是“错误条目”，更多是“在职业内可接受、但在全局池中过泛”的条目。
    这样做的目标是降低全局匹配时的低质量命中。
    """
    name = _safe_text(skill_name)
    if not name:
        return True, "empty_name"

    if name in LOW_SIGNAL_EXACT_BLACKLIST:
        return True, "low_signal_exact_name"

    # 高频、短中文、泛化动作词，通常不适合作为全局硬技能名。
    if (
        source_category_count >= 20
        and len(name) <= 8
        and not _contains_ascii_token(name)
        and LOW_SIGNAL_SUFFIX_PATTERN.search(name)
    ):
        return True, "high_frequency_low_signal_label"

    return False, ""


def _load_dictionary(path: Path) -> Dict:
    """读取 JSON 词典。"""
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _save_json(path: Path, payload: Dict) -> None:
    """保存 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def build_merged_dictionary(dictionary: Dict) -> tuple[Dict, Dict, List[MergeAuditRow]]:
    """
    构建合并后的全局硬技能词典。

    处理顺序：
    1. 先复用现有清洗脚本，确保输入词典已经是干净的硬技能版本；
    2. 按 skill 主名称跨职业合并；
    3. 合并 aliases、skill_type、来源职业细类；
    4. 对全局池额外删除低区分度标签。
    """
    cleaned_dictionary, clean_summary, _ = clean_dictionary(dictionary)

    buckets: Dict[str, Dict] = {}
    for detail_path, category in cleaned_dictionary.get("categories", {}).items():
        for skill in category.get("skills", []) or []:
            name = _safe_text(skill.get("name", ""))
            if not name:
                continue

            key = name.casefold()
            bucket = buckets.setdefault(
                key,
                {
                    "name": name,
                    "aliases": [],
                    "skill_type_counter": Counter(),
                    "source_detail_paths": set(),
                    "notes": [],
                },
            )

            bucket["aliases"].extend([_safe_text(alias) for alias in (skill.get("aliases", []) or [])])
            skill_type = _safe_text(skill.get("skill_type", ""))
            if skill_type:
                bucket["skill_type_counter"][skill_type] += 1
            bucket["source_detail_paths"].add(detail_path)

            note = _safe_text(skill.get("notes", ""))
            if note:
                bucket["notes"].append(note)

    merged_skills: List[Dict] = []
    audit_rows: List[MergeAuditRow] = []
    counters = Counter()

    for bucket in buckets.values():
        name = bucket["name"]
        aliases = _unique_keep_order(bucket["aliases"])
        aliases = [alias for alias in aliases if alias.casefold() != name.casefold()]
        source_detail_paths = sorted(bucket["source_detail_paths"])
        source_category_count = len(source_detail_paths)
        skill_type = _choose_skill_type(bucket["skill_type_counter"])

        drop_skill, drop_reason = _should_drop_low_signal_global_skill(
            skill_name=name,
            source_category_count=source_category_count,
        )
        if drop_skill:
            counters["skills_removed"] += 1
            counters[f"removed__{drop_reason}"] += 1
            audit_rows.append(
                MergeAuditRow(
                    action="remove_skill",
                    reason=drop_reason,
                    skill_name=name,
                    cleaned_text="",
                    source_category_count=source_category_count,
                    source_detail_paths=" | ".join(source_detail_paths),
                )
            )
            continue

        merged_skills.append(
            {
                "name": name,
                "aliases": aliases,
                "skill_type": skill_type,
                "notes": "",
                "source_category_count": source_category_count,
                "source_detail_paths": source_detail_paths,
            }
        )
        counters["skills_after_merge"] += 1

    merged_skills.sort(key=lambda item: (item["name"].casefold(), item["skill_type"].casefold()))

    merged_dictionary = {
        "metadata": {
            **cleaned_dictionary.get("metadata", {}),
            "schema_version": "merged_hard_skill_dictionary_v1",
            "merged_at": datetime.now().isoformat(timespec="seconds"),
            "description": "跨职业合并后的全局硬技能词典，适用于全局硬技能匹配。",
            "merge_summary": {
                "categories_before": len(cleaned_dictionary.get("categories", {})),
                "skills_before": sum(
                    len(category.get("skills", []) or [])
                    for category in cleaned_dictionary.get("categories", {}).values()
                ),
                "unique_skill_names_before_filter": len(buckets),
                "skills_after_merge": counters["skills_after_merge"],
                "skills_removed": counters["skills_removed"],
                "removed_low_signal_exact_name": counters["removed__low_signal_exact_name"],
                "removed_high_frequency_low_signal_label": counters["removed__high_frequency_low_signal_label"],
                "clean_summary": clean_summary,
            },
        },
        "categories": {
            MERGED_POOL_PATH: {
                "detail_name": "合并硬技能词典",
                "hierarchy": {
                    "大类": "通用技能池",
                    "中类": "合并硬技能词典",
                    "小类": "合并硬技能词典",
                    "细类": "合并硬技能词典",
                },
                "available_count": len(merged_skills),
                "train_count": 0,
                "validation_pool_count": 0,
                "skills": merged_skills,
            }
        },
    }

    summary = {
        "categories_before": len(cleaned_dictionary.get("categories", {})),
        "skills_before": sum(
            len(category.get("skills", []) or [])
            for category in cleaned_dictionary.get("categories", {}).values()
        ),
        "unique_skill_names_before_filter": len(buckets),
        "skills_after_merge": counters["skills_after_merge"],
        "skills_removed": counters["skills_removed"],
        "removed_low_signal_exact_name": counters["removed__low_signal_exact_name"],
        "removed_high_frequency_low_signal_label": counters["removed__high_frequency_low_signal_label"],
    }
    return merged_dictionary, summary, audit_rows


def write_audit(report_dir: Path, summary: Dict, audit_rows: List[MergeAuditRow]) -> None:
    """输出 JSON 汇总和 CSV 审计。"""
    report_dir.mkdir(parents=True, exist_ok=True)
    _save_json(report_dir / "merge_summary.json", summary)

    import csv

    with open(report_dir / "merge_details.csv", "w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "action",
                "reason",
                "skill_name",
                "cleaned_text",
                "source_category_count",
                "source_detail_paths",
            ],
        )
        writer.writeheader()
        for row in audit_rows:
            writer.writerow(row.__dict__)


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数。"""
    parser = argparse.ArgumentParser(description="把职业细类硬技能词典合并成全局硬技能词典")
    parser.add_argument("--input", required=True, help="输入词典 JSON")
    parser.add_argument("--output", required=True, help="输出词典 JSON")
    parser.add_argument(
        "--report-dir",
        default="output/skill_extraction/reports/merged_hard_skill_dictionary",
        help="审计报告目录",
    )
    return parser


def main() -> None:
    """CLI 入口。"""
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_dir = Path(args.report_dir)

    dictionary = _load_dictionary(input_path)
    merged_dictionary, summary, audit_rows = build_merged_dictionary(dictionary)
    summary["input_path"] = str(input_path)
    summary["output_path"] = str(output_path)
    summary["generated_at"] = datetime.now().isoformat(timespec="seconds")

    _save_json(output_path, merged_dictionary)
    write_audit(report_dir=report_dir, summary=summary, audit_rows=audit_rows)

    print(f"输入词典: {input_path}")
    print(f"输出词典: {output_path}")
    print(f"审计目录: {report_dir}")
    print(f"原始技能条目: {summary['skills_before']}")
    print(f"唯一技能名(过滤前): {summary['unique_skill_names_before_filter']}")
    print(f"合并后技能条目: {summary['skills_after_merge']}")
    print(f"移除低区分度条目: {summary['skills_removed']}")


if __name__ == "__main__":
    main()
