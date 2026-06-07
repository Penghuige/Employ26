"""打分与融合模块。"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import jieba

from .match_utils import min_max_normalize, normalize_compact, read_text_lines, unique_keep_order


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
        self.generic_terms = set(read_text_lines("dicts/job_generic_terms.txt", warn_missing=True))
        self.overlap_stopwords = set(read_text_lines("dicts/stopwords_recruitment_short.txt"))

    def normalize_scores(self, score_map: Dict[int, float]) -> Dict[int, float]:
        """对候选分数字典做 min-max 归一化到 [0, 1]。

        Args:
            score_map: {候选索引: 原始分数} 的字典。

        Returns:
            Dict[int, float]: 归一化后的 {候选索引: [0,1] 分数}。
        """
        return min_max_normalize(score_map)

    def compute_task_overlap(self, core_task_sentences: List[str], task_list: List[str]) -> float:
        """计算岗位职责句与大典任务条目的 token 重叠度。

        取所有句-任务对中 Jaccard 系数的最大值作为最终 overlap 分数。

        Args:
            core_task_sentences: 从 JD 抽取的核心职责句子列表。
            task_list: 职业大典条目的任务列表。

        Returns:
            float: [0, 1] 范围的最大 token 重叠度。
        """
        if not core_task_sentences or not task_list:
            return 0.0
        max_score = 0.0
        for sent in core_task_sentences:
            sent_tokens = self._task_tokens(sent)
            for task in task_list:
                task_tokens = self._task_tokens(task)
                if not sent_tokens or not task_tokens:
                    continue
                score = len(sent_tokens & task_tokens) / max(len(sent_tokens | task_tokens), 1)
                max_score = max(max_score, score)
        return max_score

    def _task_tokens(self, text: str) -> set[str]:
        """将职责文本切成较稳定的 token 集合，避免高频单字抬高相似度。

        Args:
            text: 待分词的职责文本。

        Returns:
            set[str]: 过滤停用词和单字后的 token 集合。
        """
        tokens: set[str] = set()
        for token in jieba.lcut(str(text or "")):
            token = token.strip()
            if not token or token in self.overlap_stopwords:
                continue
            if len(token) == 1 and not (token.isascii() and token.isalnum()):
                continue
            tokens.add(token.lower())
        return tokens

    def alias_exact_bonus(self, clean_title: str, title: str, aliases: Iterable[str]) -> float:
        """清洗后的岗位标题精确命中职业名称或别名时给予奖励分。

        Args:
            clean_title: 清洗后的岗位标题。
            title: 职业大典条目标题。
            aliases: 该职业的别名字列表。

        Returns:
            float: 命中则返回 alias_bonus 全量，紧凑规范化命中返回 0.85 倍，否则 0.0。
        """
        target = str(clean_title).strip()
        if not target:
            return 0.0
        all_names = unique_keep_order([title, *list(aliases)])
        if target in all_names:
            return self.alias_bonus
        normalized_target = normalize_compact(target)
        if normalized_target and any(normalize_compact(name) == normalized_target for name in all_names):
            return round(self.alias_bonus * 0.85, 6)
        return 0.0

    def compute_generic_penalty(self, clean_title: str, function_terms: List[str], object_terms: List[str]) -> float:
        """对只体现泛职能、缺少业务对象的标题施加降权。

        当岗位标题仅包含泛化职能词（如"经理""专员"）且无具体业务对象时，
        说明匹配信息不足，应施加惩罚以降低误匹配风险。

        Args:
            clean_title: 清洗后的岗位标题。
            function_terms: 从标题/JD 中抽取的职能词。
            object_terms: 从标题/JD 中抽取的业务对象词。

        Returns:
            float: 惩罚分数，有 object_terms 时减半，无且标题短时全量，否则按命中数累加。
        """
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
        """根据高风险冲突词对明显不匹配候选施加惩罚。

        当前覆盖的冲突场景：房地产候选 vs 电商冲突词、电商候选 vs 房地产冲突词。

        Args:
            conflict_terms: 从岗位文本中抽取的高风险冲突词列表。
            candidate_text: 候选职业的层级/描述/标题合并文本。

        Returns:
            float: 冲突惩罚分数，上限为 conflict_penalty * 1.5。
        """
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
        """根据 title 命中层级返回奖励分。

        exact（精确匹配）> normalized_exact（规范化后精确匹配）> fuzzy（模糊匹配）。

        Args:
            title_match_level: 命中层级，取值为 "exact" / "normalized_exact" / "fuzzy" / "none"。

        Returns:
            float: exact 返回全量 title_direct_bonus，normalized_exact 返回 0.85 倍，
                   fuzzy 返回 0.35 倍，none 返回 0.0。
        """
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
        """按照配置权重公式计算最终融合分数。

        公式: title_weight*title_score + task_weight*task_score + desc_weight*desc_score
              + hierarchy_weight*hierarchy_score + alias_bonus + title_direct_bonus
              + task_overlap_bonus*task_overlap_bonus - generic_penalty - conflict_penalty

        Args:
            title_score: 归一化后的 BM25 title 路分数。
            task_score: 归一化后的 BM25 task 路分数。
            desc_score: 归一化后的 n-gram 描述路分数。
            hierarchy_score: 归一化后的层级匹配分数。
            alias_bonus: 别名精确命中奖励分。
            title_direct_bonus: title 直接命中奖励分。
            task_overlap_bonus: 职责 token 重叠奖励分。
            generic_penalty: 泛标题惩罚分。
            conflict_penalty: 领域冲突惩罚分。

        Returns:
            float: 最终融合分数，上限不封顶但通常 < 1.5。
        """
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
        """根据候选分数与惩罚信息构造置信度标记。

        评估规则：无候选 → low；存在 domain_conflict_penalty / small_top1_top2_margin
        / title_signal_missing 任一 → low；仅有其他风险标记 → medium；否则 → high。

        Args:
            top_candidates: 已排序的 TopK 候选列表。

        Returns:
            Dict[str, Any]: 包含 confidence_level / risk_flags / top1_top2_margin / is_review_needed。
        """
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
