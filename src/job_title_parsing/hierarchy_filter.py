"""层级过滤模块。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .match_utils import PROJECT_ROOT


class HierarchyFilter:
    """通过关键词对职业大类做弱过滤。"""

    def __init__(self, config: Dict[str, Any]):
        hierarchy_cfg = config.get("hierarchy", {})
        self.keyword_to_major = dict(hierarchy_cfg.get("keyword_to_major", {}))
        dict_path = hierarchy_cfg.get("keyword_dict_path", "")
        if dict_path:
            self.keyword_to_major.update(self._load_keyword_dict(dict_path))

    def _load_keyword_dict(self, dict_path: str | Path) -> Dict[str, str]:
        """加载自动构建的层级关键词词典（keyword\tmajor）。"""
        target = Path(dict_path)
        if not target.is_absolute():
            target = PROJECT_ROOT / target
        if not target.exists():
            return {}

        mapping: Dict[str, str] = {}
        for raw in target.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            keyword = parts[0].strip()
            major = parts[1].strip()
            if keyword and major:
                mapping[keyword] = major
        return mapping

    def filter_candidates_by_hierarchy(self, job_text: str, catalog_df: pd.DataFrame) -> pd.DataFrame:
        """根据 job_text 判断粗层级，若无法判断则不过滤。"""
        inferred = self.infer_major(job_text)
        if not inferred:
            return catalog_df
        if "大类" not in catalog_df.columns:
            return catalog_df
        filtered = catalog_df[catalog_df["大类"].astype(str).str.contains(str(inferred), regex=False)]
        return filtered if not filtered.empty else catalog_df

    def infer_major(self, job_text: str) -> str:
        """根据关键词推断大类。"""
        text = str(job_text)
        for keyword, major in self.keyword_to_major.items():
            if keyword in text:
                return str(major)
        return ""

    def hierarchy_match_bonus(self, job_text: str, row: pd.Series) -> float:
        """若候选职业命中推断层级，则返回基础分。"""
        inferred = self.infer_major(job_text)
        if not inferred:
            return 0.0
        return 1.0 if inferred in str(row.get("大类", "")) else 0.0
