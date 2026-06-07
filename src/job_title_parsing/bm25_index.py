"""BM25 检索模块。"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence
from collections import Counter, defaultdict
import heapq
import math

import jieba

from .match_utils import normalize_text


class SimpleBM25:
    """一个轻量本地 BM25 实现（倒排加速）。"""

    def __init__(self, tokenized_corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = tokenized_corpus
        self.k1 = k1
        self.b = b
        self.doc_count = len(tokenized_corpus)
        self.doc_lengths = [len(doc) for doc in tokenized_corpus]
        self.avgdl = sum(self.doc_lengths) / max(self.doc_count, 1)
        self.doc_freq: Dict[str, int] = defaultdict(int)
        self.term_freqs: List[Counter] = []
        self.postings: Dict[str, List[tuple[int, int]]] = defaultdict(list)

        for idx, doc in enumerate(tokenized_corpus):
            counter = Counter(doc)
            self.term_freqs.append(counter)
            for token, freq in counter.items():
                self.doc_freq[token] += 1
                self.postings[token].append((idx, freq))

    def _idf(self, token: str) -> float:
        """计算 BM25 idf。"""
        df = self.doc_freq.get(token, 0)
        if df <= 0:
            return 0.0
        return math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))

    def get_scores_sparse(self, query_tokens: Sequence[str]) -> Dict[int, float]:
        """仅对倒排命中的文档计算分数。"""
        scores: Dict[int, float] = defaultdict(float)
        if not query_tokens:
            return {}

        token_weights = Counter(query_tokens)
        for token, qtf in token_weights.items():
            posting_list = self.postings.get(token)
            if not posting_list:
                continue
            idf = self._idf(token)
            if idf <= 0:
                continue

            for doc_idx, freq in posting_list:
                dl = self.doc_lengths[doc_idx]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                if denom <= 0:
                    continue
                scores[doc_idx] += qtf * idf * (freq * (self.k1 + 1) / denom)

        return dict(scores)


class BM25Index:
    """对指定文本字段构建 BM25 索引。"""

    def __init__(self, documents: List[str], stopwords: Iterable[str] | None = None, k1: float = 1.5, b: float = 0.75):
        self.stopwords = set(stopwords or [])
        self.documents = [normalize_text(doc) for doc in documents]
        self.tokenized = [self.tokenize(doc) for doc in self.documents]
        self.bm25 = SimpleBM25(self.tokenized, k1=k1, b=b)

    def tokenize(self, text: str) -> List[str]:
        """jieba 分词并过滤停用词。"""
        return [w.strip() for w in jieba.lcut(text) if w.strip() and w.strip() not in self.stopwords]

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, float]]:
        """通用 BM25 检索接口。

        Args:
            query: 查询文本。
            top_k: 返回的文档数量。

        Returns:
            List[Dict[str, float]]: 按分数降序的检索结果，每项含 index（文档索引）和 score。
        """
        tokens = self.tokenize(query)
        if not tokens:
            return []

        sparse_scores = self.bm25.get_scores_sparse(tokens)
        if not sparse_scores:
            return []

        ranked = heapq.nlargest(top_k, sparse_scores.items(), key=lambda x: x[1])
        return [{"index": idx, "score": score} for idx, score in ranked if score > 0]

    def search_title(self, query: str, top_k: int = 10) -> List[Dict[str, float]]:
        """title/alias 检索接口。"""
        return self.search(query, top_k=top_k)

    def search_tasks(self, query: str, top_k: int = 10) -> List[Dict[str, float]]:
        """tasks 检索接口。"""
        return self.search(query, top_k=top_k)
