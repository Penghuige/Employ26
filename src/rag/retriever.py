import os
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import logging as hf_logging

from .config import RAGConfig

# 抑制 BertModel 加载报告（权重不完全匹配是微调模型的正常现象，不影响推理）
hf_logging.set_verbosity_error()

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "未安装 faiss。请先安装 faiss-cpu 或 faiss-gpu。"
    ) from exc


class OccupationRetriever:
    """职业知识库向量检索器。

    功能：
    - 用 BGE 对职业条目编码
    - 建立/保存/加载 FAISS 内积索引（配合归一化即余弦相似度）
    - 对 query 返回 TopK 候选
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self.embedding_model = SentenceTransformer(config.embedding_model_path)
        self.index = None

    def build_index(self, records: List[Dict[str, str]]) -> None:
        """基于知识库记录构建索引。"""
        if not records:
            raise ValueError("知识库记录为空，无法构建索引。")

        texts = [(r.get("search_text") or r.get("text") or "").strip() for r in records]
        if not any(texts):
            raise ValueError("索引文本为空，无法构建向量索引。")
        vectors = self.embedding_model.encode(
            texts,
            batch_size=self.config.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        vectors = np.asarray(vectors, dtype=np.float32)

        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)

    def save_index(self) -> None:
        """将 FAISS 索引保存到磁盘。"""
        if self.index is None:
            raise ValueError("索引尚未构建，无法保存。")

        os.makedirs(os.path.dirname(self.config.index_path), exist_ok=True)
        faiss.write_index(self.index, self.config.index_path)

    def load_index(self) -> None:
        """从磁盘加载 FAISS 索引。"""
        if not os.path.exists(self.config.index_path):
            raise FileNotFoundError(f"索引文件不存在: {self.config.index_path}")
        self.index = faiss.read_index(self.config.index_path)

    def search(self, query: str, records: List[Dict[str, str]], top_k: int = None) -> List[Dict]:
        """检索 query 的 TopK 候选。"""
        if self.index is None:
            raise ValueError("索引未加载，请先 build_index 或 load_index。")

        top_k = top_k or self.config.top_k
        qv = self.embedding_model.encode([query], normalize_embeddings=True)
        qv = np.asarray(qv, dtype=np.float32)

        scores, indices = self.index.search(qv, top_k)
        result: List[Dict] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(records):
                continue
            rec = records[idx]
            result.append(
                {
                    "rank": len(result) + 1,
                    "score": float(score),
                    "code": rec["code"],
                    "title": rec["title"],
                    "title_main": rec.get("title_main", rec["title"]),
                    "sub_titles": rec.get("sub_titles", []),
                    "title_flag": rec.get("title_flag", ""),
                    "desc": rec["desc"],
                    "tasks": rec["tasks"],
                    "task_items": rec.get("task_items", []),
                    "is_other_bucket": rec.get("is_other_bucket", False),
                    "search_text": rec["search_text"],
                }
            )

        return result
