"""
职业细类技能词典覆盖率评估模块。

覆盖率定义：
1. 先把任职要求切分成要求条目
2. 过滤明显非技能条目
3. 统计“被词典命中的技能条目 / 技能条目总数”
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import pandas as pd

from .dictionary_store import OccupationSkillDictionaryStore


NON_SKILL_ITEM_PATTERNS = [
    re.compile(r"(本科|大专|硕士|博士|学历|学位)"),
    re.compile(r"(年龄|性别|形象|身高|户籍)"),
    re.compile(r"(\d+\s*年.*经验|工作经验|经验要求)"),
    re.compile(r"(能接受|可接受).*(出差|加班|轮班|夜班)"),
    re.compile(r"(责任心|抗压|沟通能力|团队协作|执行力|学习能力|稳定性)"),
]


def normalize_match_text(text: str) -> str:
    """归一化匹配文本，便于中英文技能词命中。"""
    lowered = str(text).lower()
    lowered = re.sub(r"[\s\u3000]+", "", lowered)
    lowered = re.sub(r"[，,。；;：:（）()\[\]【】{}<>《》“”\"'‘’/\\_.-]", "", lowered)
    return lowered


def safe_lower_text(text: str) -> str:
    """保留边界信息的英文匹配文本。"""
    return re.sub(r"\s+", " ", str(text).lower()).strip()


class RequirementCoverageEvaluator:
    """技能词典覆盖率评估器。"""

    def __init__(self, dictionary_store: OccupationSkillDictionaryStore):
        self.dictionary_store = dictionary_store

    @staticmethod
    def split_requirement_items(text: str) -> List[str]:
        """把任职要求文本拆成条目。"""
        content = str(text or "").strip()
        if not content:
            return []
        items = [item.strip() for item in content.split(" | ")]
        return [item for item in items if item]

    @staticmethod
    def is_skill_like_item(item: str) -> bool:
        """判断要求条目是否属于技能覆盖率评估范围。"""
        text = str(item).strip()
        if not text:
            return False
        if len(text) <= 2:
            return False
        for pattern in NON_SKILL_ITEM_PATTERNS:
            if pattern.search(text):
                return False
        return True

    def match_item_skills(self, item: str, skill_terms: List[str]) -> List[str]:
        """返回某个要求条目中命中的技能词。"""
        matched_terms: List[str] = []
        normalized_item = normalize_match_text(item)
        raw_item = safe_lower_text(item)

        unique_terms = sorted({term.strip() for term in skill_terms if str(term).strip()}, key=len, reverse=True)
        for term in unique_terms:
            raw_term = str(term).strip()
            lower_term = raw_term.lower()
            normalized_term = normalize_match_text(raw_term)
            if not normalized_term:
                continue

            if re.fullmatch(r"[a-z][a-z0-9]*", lower_term):
                if re.search(rf"(?<![a-z0-9]){re.escape(lower_term)}(?![a-z0-9])", raw_item):
                    matched_terms.append(raw_term)
                    continue

            if normalized_term in normalized_item:
                matched_terms.append(raw_term)

        deduplicated: List[str] = []
        seen = set()
        for term in matched_terms:
            normalized = normalize_match_text(term)
            if normalized in seen:
                continue
            seen.add(normalized)
            deduplicated.append(term)
        return deduplicated

    def evaluate_batch(self, validation_df: pd.DataFrame, dictionary: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """评估一批验证样本，返回汇总、明细和未覆盖条目。"""
        summary_rows: List[Dict] = []
        item_rows: List[Dict] = []
        uncovered_rows: List[Dict] = []

        if validation_df.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        for detail_path, group in validation_df.groupby("detail_path", sort=False):
            skill_terms = self.dictionary_store.get_skill_terms(dictionary, detail_path)
            total_skill_items = 0
            covered_skill_items = 0
            uncovered_item_count = 0

            for row in group.to_dict(orient="records"):
                requirement_text = row.get("任职要求_items_text") or row.get("需求文本") or ""
                for item in self.split_requirement_items(requirement_text):
                    skill_like = self.is_skill_like_item(item)
                    matched_skills = self.match_item_skills(item, skill_terms) if skill_like else []
                    is_covered = bool(matched_skills)

                    if skill_like:
                        total_skill_items += 1
                        covered_skill_items += int(is_covered)
                        if not is_covered:
                            uncovered_item_count += 1
                            uncovered_rows.append(
                                {
                                    "detail_path": detail_path,
                                    "detail_name": row.get("detail_name", ""),
                                    "sample_row_id": row.get("sample_row_id", ""),
                                    "岗位名称": row.get("岗位名称", ""),
                                    "未覆盖要求条目": item,
                                }
                            )

                    item_rows.append(
                        {
                            "detail_path": detail_path,
                            "detail_name": row.get("detail_name", ""),
                            "sample_row_id": row.get("sample_row_id", ""),
                            "岗位名称": row.get("岗位名称", ""),
                            "要求条目": item,
                            "is_skill_like_item": skill_like,
                            "is_covered": is_covered,
                            "matched_skills": " | ".join(matched_skills),
                        }
                    )

            coverage = 1.0 if total_skill_items == 0 else covered_skill_items / total_skill_items
            summary_rows.append(
                {
                    "detail_path": detail_path,
                    "detail_name": group["detail_name"].iloc[0],
                    "validation_sample_count": int(group["sample_row_id"].nunique()),
                    "skill_item_count": total_skill_items,
                    "covered_skill_item_count": covered_skill_items,
                    "uncovered_skill_item_count": uncovered_item_count,
                    "coverage": round(coverage, 6),
                    "dictionary_skill_count": len(skill_terms),
                }
            )

        return (
            pd.DataFrame(summary_rows).sort_values(["coverage", "detail_name"], ascending=[True, True]),
            pd.DataFrame(item_rows),
            pd.DataFrame(uncovered_rows),
        )
