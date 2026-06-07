"""RAG v2 主流程：DuckDB 数据源 + 双 chunk 检索 + DeepSeek V4 Pro 生成。

支持两种模式:
- build: 从 DuckDB 构建双索引（definition + task）
- query: RAG 检索 + DeepSeek 生成，输出最佳匹配职业细类
- judge: 给定固定候选列表，DeepSeek 评判选出最佳（对齐 eval_annotation_quality）

用法:
    # 构建索引
    python -m src.rag.cli build

    # RAG 查询
    python -m src.rag.cli query --title "Java开发工程师" --requirements "Spring Boot, MySQL..."

    # Judge 模式
    python -m src.rag.cli judge --title "..." --requirements "..." --candidates-json "[{...}]"
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .config import RAGConfig
from .generator import DeepSeekGenerator
from .kb_builder import build_chunks, load_metadata, load_occupation_records, save_metadata
from .retriever import OccupationRetriever


class OccupationRAG:
    """职业细类匹配 RAG 主流程（v2）。

    流程:
    1) 索引构建: DuckDB → records → chunks → 双 FAISS 索引
    2) 检索: query → 双路召回 → 加权融合 → 层级重排序 → top_k 候选
    3) 生成: top_k 候选 → DeepSeek V4 Pro → 最佳匹配 JSON
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self.retriever = OccupationRetriever(config)
        self.generator: Optional[DeepSeekGenerator] = None
        self.records: List[Dict[str, Any]] = []
        self.chunks: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 索引构建
    # ------------------------------------------------------------------

    def build_index(self) -> None:
        """从 DuckDB 加载数据 → 构建双索引 → 保存。"""
        print("[RAG] 从 DuckDB 加载职业大典...")
        records = load_occupation_records(self.config)
        print(f"[RAG] 加载 {len(records)} 条职业记录")

        print("[RAG] 生成双 chunk...")
        chunks = build_chunks(self.config, records)
        def_cnt = sum(1 for c in chunks if c["chunk_type"] == "definition")
        task_cnt = sum(1 for c in chunks if c["chunk_type"] == "task")
        print(f"[RAG] definition chunks: {def_cnt}, task chunks: {task_cnt}")

        print("[RAG] 构建双 FAISS 索引...")
        self.retriever.build_index(chunks)
        self.retriever.save_index()
        print(f"[RAG] def_index → {self.config.def_index_path}")
        print(f"[RAG] task_index → {self.config.task_index_path}")

        save_metadata(self.config, records, chunks)
        print(f"[RAG] metadata → {self.config.metadata_path}")

        self.records = records
        self.chunks = chunks

    # ------------------------------------------------------------------
    # 运行时加载
    # ------------------------------------------------------------------

    def load(self) -> None:
        """加载索引和元数据（用于 query/judge 模式）。"""
        if not os.path.exists(self.config.metadata_path):
            raise FileNotFoundError(
                f"元数据不存在: {self.config.metadata_path}。请先执行 build。"
            )
        payload = load_metadata(self.config.metadata_path)
        self.records = payload["records"]
        self.chunks = payload["chunks"]
        self.retriever.load_index()
        print(f"[RAG] 已加载 {len(self.records)} 条记录, {len(self.chunks)} 个 chunk")

    @property
    def gen(self) -> DeepSeekGenerator:
        """惰性加载 DeepSeek 生成器。"""
        if self.generator is None:
            self.generator = DeepSeekGenerator(self.config)
        return self.generator

    # ------------------------------------------------------------------
    # RAG 查询模式
    # ------------------------------------------------------------------

    def query(
        self,
        job_title: str,
        job_requirements: str = "",
        top_k: int = None,
    ) -> Dict[str, Any]:
        """RAG 查询：检索 + 生成最佳匹配。

        Args:
            job_title: 岗位名称。
            job_requirements: 岗位要求描述（可选）。
            top_k: 返回候选数量。

        Returns:
            Dict: {"query": ..., "candidates": [...], "result": {...}}
        """
        query_text = f"{job_title} {job_requirements}".strip()
        candidates = self.retriever.search(
            query_text, self.chunks, self.records, top_k=top_k
        )
        result = self.gen.generate(
            query=query_text,
            job_title=job_title,
            job_requirements=job_requirements,
            candidates=candidates,
        )
        return {
            "query": query_text,
            "candidates": candidates,
            "result": result,
        }

    # ------------------------------------------------------------------
    # Judge 模式（对齐 eval_annotation_quality）
    # ------------------------------------------------------------------

    def judge(
        self,
        job_title: str,
        job_requirements: str,
        candidates: List[Dict],
    ) -> Dict[str, Any]:
        """Judge 模式：给定固定候选列表，DeepSeek 选出最佳匹配。

        输出格式对齐 eval_annotation_quality.py:
        {"best_candidate": "A"|"B"|...|"NONE", "confidence": float, "reasoning": str}

        Args:
            job_title: 岗位名称。
            job_requirements: 岗位要求描述。
            candidates: 候选列表，每个含 code/title/desc。

        Returns:
            Dict: 评判结果。
        """
        return self.gen.judge(job_title, job_requirements, candidates)
