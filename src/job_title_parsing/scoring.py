"""打分与融合模块。"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .match_utils import min_max_normalize, read_text_lines, unique_keep_order


class ScoreFusion:
    """融合 title/tasks/desc/hierarchy 等多路信号。"""

    def __init__(self, config: Dict[str, Any]):
        scoring = config.get("scoring", {})
        self.title_weight = float(scoring.get("title_weight", 0.45))
        self.task_weight = float(scoring.get("task_weight", 0.30))
        self.desc_weight = float(scoring.get("desc_weight", 0.15))
        self.hierarchy_weight = float(scoring.get("hierarchy_weight", 0.10))
        self.alias_bonus = float(scoring.get("alias_bonus", 0.10))
        self.title_direct_bonus = float(scoring.get("title_direct_bonus", 0.18))
        self.task_overlap_bonus = float(scoring.get("task_overlap_bonus", 0.08))
        self.generic_title_penalty = float(scoring.get("generic_title_penalty", 0.08))
        self.conflict_penalty = float(scoring.get("conflict_penalty", 0.12))
        self.low_confidence_margin = float(scoring.get("low_confidence_margin", 0.08))
        self.low_confidence_score_threshold = float(scoring.get("low_confidence_score_threshold", 0.35))
        self.generic_terms = set(read_text_lines("dicts/job_generic_terms.txt"))

    def normalize_scores(self, score_map: Dict[int, float]) -> Dict[int, float]:
        """分数归一化。"""
        return min_max_normalize(score_map)

    def compute_task_overlap(self, core_task_sentences: List[str], task_list: List[str]) -> float:
        """简单 token overlap/fuzzy bonus。"""
        if not core_task_sentences or not task_list:
            return 0.0
        max_score = 0.0
        for sent in core_task_sentences:
            sent_chars = set(sent)
            for task in task_list:
                task_chars = set(task)
                if not sent_chars or not task_chars:
                    continue
                score = len(sent_chars & task_chars) / max(len(sent_chars | task_chars), 1)
                max_score = max(max_score, score)
        return max_score

    def alias_exact_bonus(self, clean_title: str, title: str, aliases: Iterable[str]) -> float:
        """clean_title 精确命中 title / aliases 给 bonus。"""
        target = str(clean_title).strip()
        if not target:
            return 0.0
        all_names = unique_keep_order([title, *list(aliases)])
        return self.alias_bonus if target in all_names else 0.0

    def compute_generic_penalty(self, clean_title: str, function_terms: List[str], object_terms: List[str]) -> float:
        """对只体现泛职能、缺少业务对象的标题施加降权。"""
        if not clean_title:
            return 0.0
        generic_hits = [term for term in self.generic_terms if term and term in clean_title]
        if not generic_hits:
            return 0.0
        if object_terms:
            return round(self.generic_title_penalty * 0.5, 6)
        if function_terms and len(clean_title) <= 8:
            return round(self.generic_title_penalty, 6)
        return round(min(self.generic_title_penalty, 0.04 * len(generic_hits)), 6)

    def compute_conflict_penalty(self, conflict_terms: List[str], candidate_text: str) -> float:
        """根据高风险冲突词对明显不匹配候选施加惩罚。"""
        text = str(candidate_text)
        if not conflict_terms or not text:
            return 0.0

        has_real_estate_candidate = any(word in text for word in ["房地产", "房地", "楼盘", "置业"])
        has_ecom_candidate = any(word in text for word in ["电商", "互联网营销", "跨境", "平台", "网店"])

        real_estate_conflicts = {
            "亚马逊", "Amazon", "amazon", "FBA", "fba", "CPC", "cpc", "A+页面", "站内广告", "广告投放", "listing", "选品", "店铺"
        }
        ecom_conflicts = {"楼盘", "房源", "案场", "置业"}

        penalty = 0.0
        if has_real_estate_candidate and any(term in real_estate_conflicts for term in conflict_terms):
            penalty += self.conflict_penalty
        if has_ecom_candidate and any(term in ecom_conflicts for term in conflict_terms):
            penalty += self.conflict_penalty
        return round(min(penalty, self.conflict_penalty * 1.5), 6)

    def title_direct_match_bonus(self, title_match_level: str) -> float:
        """title 命中层级奖励：exact > normalized_exact > fuzzy。"""
        level = str(title_match_level or "")
        if level == "exact":
            return self.title_direct_bonus
        if level == "normalized_exact":
            return round(self.title_direct_bonus * 0.85, 6)
        if level == "fuzzy":
            return round(self.title_direct_bonus * 0.35, 6)
        return 0.0

    def final_score(
        self,
        title_score: float,
        task_score: float,
        desc_score: float,
        hierarchy_score: float,
        alias_bonus: float,
        title_direct_bonus: float,
        task_overlap_bonus: float,
        generic_penalty: float = 0.0,
        conflict_penalty: float = 0.0,
    ) -> float:
        """按照配置公式计算最终融合分数。"""
        return (
            self.title_weight * title_score
            + self.task_weight * task_score
            + self.desc_weight * desc_score
            + self.hierarchy_weight * hierarchy_score
            + alias_bonus
            + title_direct_bonus
            + self.task_overlap_bonus * task_overlap_bonus
            - generic_penalty
            - conflict_penalty
        )

    def build_confidence_flags(self, top_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """根据候选分数与惩罚信息构造低置信标记。"""
        if not top_candidates:
            return {
                "confidence_level": "low",
                "risk_flags": ["no_candidates"],
                "top1_top2_margin": 0.0,
                "is_review_needed": True,
            }

        top1 = top_candidates[0]
        top2 = top_candidates[1] if len(top_candidates) > 1 else None
        margin = round(top1.get("final_score", 0.0) - (top2.get("final_score", 0.0) if top2 else 0.0), 6)
        risk_flags: List[str] = []

        if margin < self.low_confidence_margin:
            risk_flags.append("small_top1_top2_margin")
        if float(top1.get("final_score", 0.0)) < self.low_confidence_score_threshold:
            risk_flags.append("low_top1_score")
        if float(top1.get("generic_penalty", 0.0)) > 0:
            risk_flags.append("generic_title_penalty")
        if float(top1.get("conflict_penalty", 0.0)) > 0:
            risk_flags.append("domain_conflict_penalty")
        if float(top1.get("task_bm25_score", 0.0)) == 0.0:
            risk_flags.append("task_signal_missing")
        if float(top1.get("title_bm25_score", 0.0)) == 0.0 and float(top1.get("title_fuzzy_score", 0.0)) == 0.0:
            risk_flags.append("title_signal_missing")

        if any(flag in risk_flags for flag in ["domain_conflict_penalty", "small_top1_top2_margin", "title_signal_missing"]):
            confidence_level = "low"
        elif risk_flags:
            confidence_level = "medium"
        else:
            confidence_level = "high"

        return {
            "confidence_level": confidence_level,
            "risk_flags": risk_flags,
            "top1_top2_margin": margin,
            "is_review_needed": confidence_level != "high",
        }
