"""岗位描述解析模块。"""

from __future__ import annotations

from typing import Any, Dict, List
import re

from .match_utils import normalize_text, unique_keep_order


class JDParser:
    """从岗位描述中提取句子、核心职责句和领域关键词。"""

    def __init__(self, config: Dict[str, Any]):
        jd_cfg = config.get("jd_parsing", {})
        self.action_verbs: List[str] = jd_cfg.get("action_verbs", [])
        self.domain_dict: List[str] = jd_cfg.get("domain_keywords", [])

    def parse(self, job_description: str) -> Dict[str, Any]:
        """解析岗位描述，为后续检索与融合提供轻量特征。

        Args:
            job_description: 原始岗位描述文本。

        Returns:
            Dict[str, Any]: 包含 jd_clean（清洗后全文）, jd_sentences（句子列表）,
                            core_task_sentences（含动作动词的核心职责句）,
                            domain_keywords（领域关键词列表）。
        """
        jd_clean = normalize_text(job_description)
        jd_sentences = self._split_sentences(jd_clean)
        core_task_sentences = [
            sent for sent in jd_sentences if any(verb in sent for verb in self.action_verbs)
        ]
        domain_keywords = self._extract_domain_keywords(jd_clean, jd_sentences)
        return {
            "jd_clean": jd_clean,
            "jd_sentences": jd_sentences,
            "core_task_sentences": core_task_sentences,
            "domain_keywords": domain_keywords,
        }

    def _split_sentences(self, text: str) -> List[str]:
        """按中英文标点切句，去掉空句。"""
        if not text:
            return []
        parts = re.split(r"[。！？!?;；\n]+", text)
        sentences = [re.sub(r"^\d+[\.、]\s*", "", p).strip() for p in parts]
        return [s for s in sentences if s]

    def _extract_domain_keywords(self, jd_clean: str, sentences: List[str]) -> List[str]:
        """通过词典匹配抽取简单领域关键词。"""
        hits: List[str] = []
        for keyword in self.domain_dict:
            if keyword in jd_clean:
                hits.append(keyword)
        # 补充一些 2~6 字的高频候选短语（极简规则）
        for sentence in sentences:
            for frag in re.findall(r"[\u4e00-\u9fa5A-Za-z]{2,6}", sentence):
                if frag in self.domain_dict:
                    hits.append(frag)
        return unique_keep_order(hits)
