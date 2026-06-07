"""字符 n-gram 检索与相似度。"""

from __future__ import annotations

from typing import Iterable, List, Set

from .match_utils import normalize_text


def char_ngrams(text: str, n: int = 2) -> List[str]:
    """生成字符 n-gram。"""
    text = normalize_text(text).replace(" ", "")
    if len(text) < n:
        return [text] if text else []
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def overlap_score(query: str, doc: str, n: int = 2) -> float:
    """基于字符 n-gram 交集/并集的简易相似度，返回 [0, 1]。

    Args:
        query: 查询文本。
        doc: 文档文本。
        n: n-gram 的 n 值，默认 2。

    Returns:
        float: Jaccard 系数（交集 / 并集），范围 [0, 1]。
    """
    q_set: Set[str] = set(char_ngrams(query, n=n))
    d_set: Set[str] = set(char_ngrams(doc, n=n))
    if not q_set or not d_set:
        return 0.0
    inter = len(q_set & d_set)
    union = len(q_set | d_set)
    return inter / max(union, 1)
