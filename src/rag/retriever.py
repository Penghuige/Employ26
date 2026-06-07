"""向量检索器（v2）：双索引（definition + task chunk）+ 层级重排序。

变更:
- 为 definition 和 task 分别构建 FAISS 索引
- 检索时双路召回 + 加权融合
- 按职业代码去重后 → 层级感知重排序
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import logging as hf_logging

from .config import RAGConfig

hf_logging.set_verbosity_error()

try:
    import faiss
except ImportError as exc:
    raise ImportError("未安装 faiss。请先安装 faiss-cpu 或 faiss-gpu。") from exc


class OccupationRetriever:
    """双 chunk 向量检索器 + 层级重排序。

    架构:
    - def_index: definition chunk 的 FAISS 索引（用于标题/定义匹配）
    - task_index: task chunk 的 FAISS 索引（用于 JD/任务匹配）
    - 检索时双路并行召回，按职业代码去重融合，层级重排序输出
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self.embedding_model = SentenceTransformer(config.embedding_model_path)
        self.def_index: Optional[faiss.Index] = None
        self.task_index: Optional[faiss.Index] = None

    # ------------------------------------------------------------------
    # 索引构建
    # ------------------------------------------------------------------

    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """从 chunk 列表构建双索引。

        Args:
            chunks: build_chunks() 产出的所有 chunk。
        """
        def_texts = []
        task_texts = []

        for c in chunks:
            if c["chunk_type"] == "definition":
                def_texts.append(c["text"])
            elif c["chunk_type"] == "task":
                task_texts.append(c["text"])

        if not def_texts:
            raise ValueError("没有 definition chunk，无法构建索引")

        self.def_index = self._build_faiss_index(def_texts)
        self.task_index = self._build_faiss_index(task_texts) if task_texts else None

    def _build_faiss_index(self, texts: List[str]) -> faiss.Index:
        """对文本列表编码并构建 FAISS 内积索引。

        Args:
            texts: 待编码的文本列表。

        Returns:
            faiss.IndexFlatIP: 内积索引（等价余弦相似度）。
        """
        vectors = self.embedding_model.encode(
            texts,
            batch_size=self.config.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        return index

    # ------------------------------------------------------------------
    # 索引持久化
    # ------------------------------------------------------------------

    def save_index(self) -> None:
        """保存双索引到磁盘。"""
        os.makedirs(os.path.dirname(self.config.def_index_path), exist_ok=True)
        if self.def_index is None:
            raise ValueError("索引尚未构建")
        faiss.write_index(self.def_index, self.config.def_index_path)
        if self.task_index is not None:
            faiss.write_index(self.task_index, self.config.task_index_path)

    def load_index(self) -> None:
        """从磁盘加载双索引。"""
        if not os.path.exists(self.config.def_index_path):
            raise FileNotFoundError(f"索引文件不存在: {self.config.def_index_path}")
        self.def_index = faiss.read_index(self.config.def_index_path)
        task_path = self.config.task_index_path
        if os.path.exists(task_path):
            self.task_index = faiss.read_index(task_path)

    # ------------------------------------------------------------------
    # 双路检索 + 融合
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        records: List[Dict[str, Any]],
        top_k: int = None,
    ) -> List[Dict[str, Any]]:
        """双路检索 + 层级重排序。

        流程:
        1. 双路并行 FAISS 检索（def + task），每路取 pool_size 个
        2. 按 chunk→record 映射，合并为职业条目分数
        3. 加权融合（def_weight * def_score + task_weight * task_score）
        4. 层级感知排序（同大类加分，层级越接近加分越高）
        5. 输出 top_k 条职业记录

        Args:
            query: 查询文本（岗位名 + 描述）。
            chunks: 所有 chunk 列表。
            records: 原始职业大典记录列表。
            top_k: 返回数量。

        Returns:
            List[Dict]: 含 code/title/desc/tasks/score/hierarchy 的候选列表。
        """
        if self.def_index is None:
            raise ValueError("索引未加载，请先 build_index 或 load_index。")

        top_k = top_k or self.config.top_k
        pool_size = self.config.retrieval_pool_size

        # ---- Step 1: 编码 query ----
        qv = self.embedding_model.encode([query], normalize_embeddings=True)
        qv = np.asarray(qv, dtype=np.float32)

        # ---- Step 2: 双路召回 ----
        def_scores: Dict[int, float] = {}
        task_scores: Dict[int, float] = {}

        def_scores_arr, def_indices = self.def_index.search(qv, pool_size)
        for score, idx in zip(def_scores_arr[0], def_indices[0]):
            if 0 <= idx < len(chunks):
                c = chunks[idx]
                if c["chunk_type"] == "definition":
                    rec_idx = c.get("record_idx", idx)
                    def_scores[rec_idx] = max(def_scores.get(rec_idx, 0), float(score))

        if self.task_index is not None:
            task_scores_arr, task_indices = self.task_index.search(qv, pool_size)
            for score, idx in zip(task_scores_arr[0], task_indices[0]):
                if 0 <= idx < len(chunks):
                    c = chunks[idx]
                    rec_idx = c.get("record_idx", 0)
                    task_scores[rec_idx] = max(task_scores.get(rec_idx, 0), float(score))

        # ---- Step 3: 加权融合 ----
        dw = self.config.def_weight
        tw = self.config.task_weight
        all_rec_indices = set(def_scores) | set(task_scores)
        fused: Dict[int, float] = {}
        for ri in all_rec_indices:
            fused[ri] = dw * def_scores.get(ri, 0) + tw * task_scores.get(ri, 0)

        # ---- Step 4: 层级重排序 ----
        # 提取 query 中可能的大类关键词，对同大类候选加分
        scored = []
        for ri, score in sorted(fused.items(), key=lambda x: -x[1]):
            if ri >= len(records):
                continue
            rec = records[ri]
            # 同大类加分（简单规则：检索文本中出现"大类X"→同大类 +0.05）
            hierarchy_bonus = self._compute_hierarchy_bonus(query, rec)
            final_score = score + hierarchy_bonus
            scored.append((final_score, ri, rec))

        scored.sort(key=lambda x: -x[0])
        top = scored[:top_k]

        # ---- Step 5: 构建输出 ----
        results = []
        for rank, (score, ri, rec) in enumerate(top, 1):
            results.append({
                "rank": rank,
                "score": round(score, 4),
                "code": rec["code"],
                "title": rec["title"],
                "desc": rec.get("desc", ""),
                "tasks": rec.get("tasks", ""),
                "hierarchy": rec.get("hierarchy", {}),
                "hierarchy_text": rec.get("hierarchy_text", ""),
                "aliases": rec.get("aliases", []),
            })
        return results

    def _compute_hierarchy_bonus(self, query: str, record: Dict) -> float:
        """计算层级匹配奖励分。

        规则：如果 query 中出现大类关键词（如"技术""销售"等），
        且候选职业的大类匹配，则加分。

        Args:
            query: 用户查询文本。
            record: 职业记录。

        Returns:
            float: 奖励分。
        """
        hierarchy = record.get("hierarchy", {})
        major = hierarchy.get("大类", "")
        if not major:
            return 0.0

        # 大类关键词映射（简化版，可从 dicts 加载）
        major_keywords = {
            "专业技术人员": ["技术", "工程", "开发", "设计", "分析", "研发", "算法", "架构"],
            "办事人员和有关人员": ["行政", "人事", "财务", "会计", "审计", "法务", "合规"],
            "社会生产服务和生活服务人员": ["服务", "销售", "客服", "运营", "配送", "物流", "餐饮"],
            "生产制造及有关人员": ["生产", "制造", "加工", "装配", "质检", "操作"],
            "党的机关、国家机关、群众团体和社会组织、企事业单位负责人": ["经理", "总监", "主管", "总裁", "负责人"],
        }

        for category, keywords in major_keywords.items():
            if category in major and any(kw in query for kw in keywords):
                return 0.05
        return 0.0
