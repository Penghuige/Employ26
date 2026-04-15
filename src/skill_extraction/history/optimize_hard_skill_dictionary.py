"""
对手工维护的硬技能词典做结构校验与轻量优化。

设计目标：
1. 不推翻人工词典内容，只修复“结构性问题”。
2. 重点处理：
   - 空职业细类节点
   - 同一技能条目内重复 alias
   - alias 与主名称重复
   - 首尾空白与空字符串
3. 输出下一版本词典，并附带审计报告。

适用场景：
- 你已经人工精修了一版词典，希望在不改变核心判断的前提下做自动体检。
- 你希望把“结构是否健康”这件事固化成脚本，而不是每次手工检查。

用法示例：
python -m src.skill_extraction.history.optimize_hard_skill_dictionary ^
  --input dicts/occupation_skill_dictionary_v2.3.json ^
  --output dicts/occupation_skill_dictionary_v2.4.json
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _safe_text(value: object) -> str:
    """安全转字符串并去首尾空白。"""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _normalize_key(text: object) -> str:
    """生成去重键。"""
    return _safe_text(text).casefold()


def _unique_keep_order(items: Iterable[object]) -> List[str]:
    """按原顺序去重。"""
    seen = set()
    result: List[str] = []
    for item in items:
        text = _safe_text(item)
        if not text:
            continue
        key = _normalize_key(text)
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def load_dictionary(path: Path) -> Dict:
    """读取词典 JSON。"""
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def save_dictionary(path: Path, dictionary: Dict) -> None:
    """保存词典 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(dictionary, file_obj, ensure_ascii=False, indent=2)


def optimize_dictionary(dictionary: Dict) -> Tuple[Dict, Dict, List[Dict]]:
    """优化词典结构，并返回优化后的词典、汇总信息与审计明细。"""
    optimized = json.loads(json.dumps(dictionary, ensure_ascii=False))
    categories = optimized.setdefault("categories", {})

    audit_rows: List[Dict] = []
    counters = Counter()

    kept_categories: Dict[str, Dict] = {}
    for detail_path, category in categories.items():
        detail_name = _safe_text(category.get("detail_name", "")) or detail_path
        skills = category.get("skills", []) or []
        counters["categories_before"] += 1

        if not skills:
            counters["removed_empty_categories"] += 1
            audit_rows.append(
                {
                    "action": "remove_empty_category",
                    "detail_path": detail_path,
                    "detail_name": detail_name,
                    "skill_name": "",
                    "field": "category",
                    "original_text": "",
                    "cleaned_text": "",
                    "reason": "empty_skill_list",
                }
            )
            continue

        normalized_skills: List[Dict] = []
        for skill in skills:
            counters["skills_before"] += 1

            name = _safe_text(skill.get("name", ""))
            skill_type = _safe_text(skill.get("skill_type", ""))
            notes = _safe_text(skill.get("notes", ""))
            aliases = skill.get("aliases", []) or []

            if not name:
                counters["removed_empty_name_skills"] += 1
                audit_rows.append(
                    {
                        "action": "remove_skill",
                        "detail_path": detail_path,
                        "detail_name": detail_name,
                        "skill_name": "",
                        "field": "skill",
                        "original_text": "",
                        "cleaned_text": "",
                        "reason": "empty_name",
                    }
                )
                continue

            cleaned_aliases = []
            for alias in aliases:
                alias_text = _safe_text(alias)
                if not alias_text:
                    counters["removed_empty_aliases"] += 1
                    continue
                if _normalize_key(alias_text) == _normalize_key(name):
                    counters["removed_alias_equal_name"] += 1
                    audit_rows.append(
                        {
                            "action": "remove_alias",
                            "detail_path": detail_path,
                            "detail_name": detail_name,
                            "skill_name": name,
                            "field": "aliases",
                            "original_text": alias_text,
                            "cleaned_text": "",
                            "reason": "alias_equals_name",
                        }
                    )
                    continue
                cleaned_aliases.append(alias_text)

            deduped_aliases = _unique_keep_order(cleaned_aliases)
            if len(deduped_aliases) < len(cleaned_aliases):
                removed_count = len(cleaned_aliases) - len(deduped_aliases)
                counters["deduplicated_aliases"] += removed_count
                audit_rows.append(
                    {
                        "action": "deduplicate_aliases",
                        "detail_path": detail_path,
                        "detail_name": detail_name,
                        "skill_name": name,
                        "field": "aliases",
                        "original_text": " | ".join(cleaned_aliases),
                        "cleaned_text": " | ".join(deduped_aliases),
                        "reason": "duplicate_alias_within_skill",
                    }
                )

            normalized_skills.append(
                {
                    "name": name,
                    "aliases": deduped_aliases,
                    "skill_type": skill_type,
                    "notes": notes,
                }
            )
            counters["skills_after"] += 1

        if not normalized_skills:
            counters["removed_empty_categories_after_skill_cleanup"] += 1
            audit_rows.append(
                {
                    "action": "remove_empty_category",
                    "detail_path": detail_path,
                    "detail_name": detail_name,
                    "skill_name": "",
                    "field": "category",
                    "original_text": "",
                    "cleaned_text": "",
                    "reason": "empty_after_skill_cleanup",
                }
            )
            continue

        category["skills"] = sorted(
            normalized_skills,
            key=lambda item: (_safe_text(item.get("name", "")).casefold(), _safe_text(item.get("skill_type", "")).casefold()),
        )
        kept_categories[detail_path] = category
        counters["categories_after"] += 1

    optimized["categories"] = dict(sorted(kept_categories.items(), key=lambda item: item[0]))
    optimized.setdefault("metadata", {})
    optimized["metadata"]["optimized_at"] = datetime.now().isoformat(timespec="seconds")
    optimized["metadata"]["optimization_summary"] = {
        "categories_before": counters["categories_before"],
        "categories_after": counters["categories_after"],
        "removed_empty_categories": counters["removed_empty_categories"] + counters["removed_empty_categories_after_skill_cleanup"],
        "skills_before": counters["skills_before"],
        "skills_after": counters["skills_after"],
        "removed_empty_name_skills": counters["removed_empty_name_skills"],
        "removed_empty_aliases": counters["removed_empty_aliases"],
        "removed_alias_equal_name": counters["removed_alias_equal_name"],
        "deduplicated_aliases": counters["deduplicated_aliases"],
    }

    summary = {
        "categories_before": counters["categories_before"],
        "categories_after": counters["categories_after"],
        "removed_empty_categories": counters["removed_empty_categories"] + counters["removed_empty_categories_after_skill_cleanup"],
        "skills_before": counters["skills_before"],
        "skills_after": counters["skills_after"],
        "removed_empty_name_skills": counters["removed_empty_name_skills"],
        "removed_empty_aliases": counters["removed_empty_aliases"],
        "removed_alias_equal_name": counters["removed_alias_equal_name"],
        "deduplicated_aliases": counters["deduplicated_aliases"],
    }
    return optimized, summary, audit_rows


def write_audit(report_dir: Path, summary: Dict, audit_rows: List[Dict]) -> None:
    """写出 JSON 汇总和 CSV 审计。"""
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "optimization_summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)

    import csv

    with open(report_dir / "optimization_details.csv", "w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "action",
                "detail_path",
                "detail_name",
                "skill_name",
                "field",
                "original_text",
                "cleaned_text",
                "reason",
            ],
        )
        writer.writeheader()
        for row in audit_rows:
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""
    parser = argparse.ArgumentParser(description="优化人工维护的硬技能词典结构")
    parser.add_argument("--input", required=True, help="输入词典 JSON")
    parser.add_argument("--output", required=True, help="输出词典 JSON")
    parser.add_argument("--report-dir", default="output/skill_extraction/reports/hard_skill_dictionary_optimization", help="审计报告目录")
    return parser


def main() -> None:
    """CLI 入口。"""
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_dir = Path(args.report_dir)

    dictionary = load_dictionary(input_path)
    optimized, summary, audit_rows = optimize_dictionary(dictionary)
    summary["input_path"] = str(input_path)
    summary["output_path"] = str(output_path)
    summary["generated_at"] = datetime.now().isoformat(timespec="seconds")

    save_dictionary(output_path, optimized)
    write_audit(report_dir=report_dir, summary=summary, audit_rows=audit_rows)

    print(f"输入词典: {input_path}")
    print(f"输出词典: {output_path}")
    print(f"审计目录: {report_dir}")
    print(f"职业细类: {summary['categories_before']} -> {summary['categories_after']}")
    print(f"技能条目: {summary['skills_before']} -> {summary['skills_after']}")
    print(f"删除空细类: {summary['removed_empty_categories']}")
    print(f"去重 alias: {summary['deduplicated_aliases']}")


if __name__ == "__main__":
    main()
