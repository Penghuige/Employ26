"""层级过滤模块。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import warnings

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
            warnings.warn(f"层级关键词词典不存在，将仅使用配置内置映射: {target}", RuntimeWarning, stacklevel=2)
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
        """根据岗位文本推断职业大类，过滤候选职业大典行。

        若 key_word_to_major 词典中无任何关键词命中，则不做过滤，原样返回全部候选。
        若过滤后为空，同样退回全量候选。

        Args:
            job_text: 合并了标题和 JD 的文本。
            catalog_df: 职业大典 DataFrame（需含"大类"列）。

        Returns:
            pd.DataFrame: 过滤后的职业大典子集（或原全集）。
        """
        inferred = self.infer_major(job_text)
        if not inferred:
            return catalog_df
        if "大类" not in catalog_df.columns:
            return catalog_df
        filtered = catalog_df[catalog_df["大类"].astype(str).str.contains(str(inferred), regex=False)]
        return filtered if not filtered.empty else catalog_df

    def infer_major(self, job_text: str) -> str:
        """根据关键词推断岗位所属的职业大类。

        Args:
            job_text: 岗位描述文本。

        Returns:
            str: 推断出的职业大类（如 "2"），无法判断时返回空字符串。
        """
        text = str(job_text)
        for keyword, major in self.keyword_to_major.items():
            if keyword in text:
                return str(major)
        return ""

    def hierarchy_match_bonus(self, job_text: str, row: pd.Series) -> float:
        """若候选职业的"大类"与推断层级一致，则返回基础奖励分。

        Args:
            job_text: 岗位描述文本。
            row: 职业大典单行记录。

        Returns:
            float: 命中返回 1.0，否则 0.0。
        """
        inferred = self.infer_major(job_text)
        if not inferred:
            return 0.0
        return 1.0 if inferred in str(row.get("大类", "")) else 0.0
