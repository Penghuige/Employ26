"""从职业大典自动构建层级关键词词典模块。"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import re

import jieba
import pandas as pd

from .match_utils import PROJECT_ROOT, normalize_text, read_text_lines


class HierarchyKeywordBuilder:
    """基于职业大典层级文本抽取关键词，并映射到大类。"""

    def __init__(self, stopwords_path: str | Path = "dicts/stopwords_recruitment_short.txt"):
        self.stopwords = set(read_text_lines(stopwords_path))

    def build_from_catalog(
        self,
        catalog_df: pd.DataFrame,
        output_path: str | Path = "dicts/hierarchy_keyword_to_major_auto.txt",
        top_n_per_major: int = 80,
        min_freq: int = 5,
    ) -> Path:
        """从职业大典 DataFrame 构建关键词->大类词典。"""
        required_cols = ["大类", "中类", "小类", "细类", "title", "desc", "tasks"]
        work_df = catalog_df.copy()
        for col in required_cols:
            if col not in work_df.columns:
                work_df[col] = ""

        major_tokens: Dict[str, Counter] = defaultdict(Counter)
        global_tokens: Counter = Counter()

        for _, row in work_df.iterrows():
            major = str(row.get("大类", "")).strip()
            if not major:
                continue
            merged = " ".join(
                [
                    str(row.get("中类", "")),
                    str(row.get("小类", "")),
                    str(row.get("细类", "")),
                    str(row.get("title", "")),
                    str(row.get("desc", "")),
                    str(row.get("tasks", "")),
                ]
            )
            tokens = self._tokenize(merged)
            major_tokens[major].update(tokens)
            global_tokens.update(tokens)

        rows: List[Tuple[str, str, int, float]] = []
        major_count = max(len(major_tokens), 1)

        for major, counter in major_tokens.items():
            for token, freq in counter.items():
                if freq < min_freq:
                    continue
                if self._is_noise_token(token):
                    continue
                # 简单区分度: 类内频次 / 全局频次
                distinctiveness = freq / max(global_tokens[token], 1)
                # 稳定优先：先按频次再按区分度
                rows.append((token, major, freq, distinctiveness))

        rows.sort(key=lambda x: (x[1], -x[2], -x[3], x[0]))

        selected: List[Tuple[str, str, int, float]] = []
        per_major_count: Dict[str, int] = defaultdict(int)
        for token, major, freq, distinctiveness in rows:
            if per_major_count[major] >= top_n_per_major:
                continue
            selected.append((token, major, freq, distinctiveness))
            per_major_count[major] += 1

        output_target = Path(output_path)
        if not output_target.is_absolute():
            output_target = PROJECT_ROOT / output_target
        output_target.parent.mkdir(parents=True, exist_ok=True)

        lines = ["# keyword\tmajor\tfreq\tdistinctiveness"]
        for token, major, freq, distinctiveness in selected:
            lines.append(f"{token}\t{major}\t{freq}\t{distinctiveness:.6f}")
        output_target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_target

    def _tokenize(self, text: str) -> List[str]:
        """分词并过滤噪声词。"""
        cleaned = normalize_text(text)
        if not cleaned:
            return []
        tokens = [w.strip() for w in jieba.lcut(cleaned) if w.strip()]
        return [t for t in tokens if t not in self.stopwords]

    def _is_noise_token(self, token: str) -> bool:
        """过滤无意义词。"""
        if not token:
            return True
        if len(token) <= 1:
            return True
        if token in {"负责", "进行", "相关", "工作", "岗位", "职业", "人员", "技术", "管理"}:
            return True
        if re.fullmatch(r"[\dA-Za-z]+", token):
            return True
        return False
