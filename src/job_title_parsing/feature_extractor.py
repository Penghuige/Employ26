"""岗位文本结构化特征提取模块。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .match_utils import PROJECT_ROOT, normalize_text, read_text_lines, unique_keep_order


class JobFeatureExtractor:
    """抽取岗位标题与 JD 的平台词、领域词、职能词、对象词与冲突词。"""

    def __init__(self, dict_dir: str | Path | None = None):
        base_dir = Path(dict_dir) if dict_dir else PROJECT_ROOT / "dicts"
        self.platform_terms = read_text_lines(base_dir / "job_platform_terms.txt")
        self.domain_terms = read_text_lines(base_dir / "job_domain_terms.txt")
        self.function_terms = read_text_lines(base_dir / "job_function_terms.txt")
        self.object_terms = read_text_lines(base_dir / "job_object_terms.txt")
        self.conflict_terms = read_text_lines(base_dir / "job_conflict_terms.txt")

    def extract(self, title: str, job_description: str) -> Dict[str, List[str]]:
        """抽取结构化特征。"""
        title_text = normalize_text(title)
        jd_text = normalize_text(job_description)
        merged_text = f"{title_text} {jd_text}".strip()

        return {
            "platform_terms": self._match_terms(merged_text, self.platform_terms),
            "domain_terms": self._match_terms(merged_text, self.domain_terms),
            "function_terms": self._match_terms(merged_text, self.function_terms),
            "object_terms": self._match_terms(merged_text, self.object_terms),
            "conflict_terms": self._match_terms(merged_text, self.conflict_terms),
            "title_terms": self._match_terms(title_text, unique_keep_order(
                self.platform_terms + self.domain_terms + self.function_terms + self.object_terms
            )),
            "jd_terms": self._match_terms(jd_text, unique_keep_order(
                self.platform_terms + self.domain_terms + self.function_terms + self.object_terms
            )),
        }

    def _match_terms(self, text: str, terms: List[str]) -> List[str]:
        """返回命中的词典词。"""
        if not text:
            return []
        hits = [term for term in terms if term and term in text]
        return unique_keep_order(hits)
