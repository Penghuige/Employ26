import os
from typing import Dict, List

from .config import RAGConfig
from .generator import QwenGenerator
from .kb_builder import build_chunks, load_metadata, load_occupation_records, save_metadata
from .retriever import OccupationRetriever


class LocalOccupationRAG:
    """本地职业知识库 RAG 主流程。

    该类负责串联三个子模块：
    1) 知识库构建：Excel -> 结构化 records；
    2) 向量检索：BGE -> FAISS TopK；
    3) 结果生成：Qwen3-8B 根据候选给出最终判断。

    使用建议：
    - 首次使用：先调用 `build_knowledge_index()`；
    - 日常调用：直接 `query()`。
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self.retriever = OccupationRetriever(config)
        self.generator = None  # 惰性加载，避免仅构建索引时占用大显存
        self.records: List[Dict[str, str]] = []

    # -------------------------
    # 1) 索引构建阶段
    # -------------------------
    def build_knowledge_index(self) -> None:
        """构建并落盘本地知识库索引。

        执行内容：
        - 读取职业大典 Excel；
        - 生成用于检索的 `search_text`；
        - 计算向量并建立 FAISS 索引；
        - 保存 metadata 与 index 文件。
        """
        print("[RAG] 开始构建本地知识库索引...")

        records = load_occupation_records(self.config)
        if not records:
            raise ValueError("知识库为空，请检查 Excel 内容。")

        chunks = build_chunks(self.config, records)

        self.retriever.build_index(records)
        self.retriever.save_index()
        save_metadata(self.config, records, chunks=chunks)

        self.records = records

        print("[RAG] 知识库索引构建完成。")
        print(f"[RAG] 索引文件: {self.config.index_path}")
        print(f"[RAG] 元数据文件: {self.config.metadata_path}")
        print(f"[RAG] 条目数量: {len(records)}，分块数量: {len(chunks)}")

    # -------------------------
    # 2) 运行时加载阶段
    # -------------------------
    def load_runtime(self) -> None:
        """加载检索所需资源（索引 + 元数据）。"""
        if not os.path.exists(self.config.index_path):
            raise FileNotFoundError(
                f"索引文件不存在: {self.config.index_path}。请先执行 build_knowledge_index()。"
            )
        if not os.path.exists(self.config.metadata_path):
            raise FileNotFoundError(
                f"元数据文件不存在: {self.config.metadata_path}。请先执行 build_knowledge_index()。"
            )

        self.retriever.load_index()
        self.records = load_metadata(self.config.metadata_path)

    def _ensure_generator(self) -> None:
        """按需初始化 Qwen 生成器。"""
        if self.generator is None:
            self.generator = QwenGenerator(self.config)

    # -------------------------
    # 3) RAG 查询阶段
    # -------------------------
    def query(self, user_query: str, top_k: int = None) -> Dict:
        """对输入文本执行 RAG 查询。

        参数：
        - user_query: 用户待匹配文本（可为岗位名或“岗位名+描述”）
        - top_k: 覆盖默认检索数量（为空则使用 config.top_k）

        返回：
        {
          "query": 原输入,
          "top_k_candidates": 检索候选列表,
          "final_answer": Qwen 结构化结果
        }
        """
        if not user_query or not str(user_query).strip():
            raise ValueError("user_query 不能为空。")

        self.load_runtime()
        candidates = self.retriever.search(user_query, self.records, top_k=top_k)

        self._ensure_generator()
        final_answer = self.generator.generate(user_query, candidates)

        return {
            "query": user_query,
            "top_k_candidates": candidates,
            "final_answer": final_answer,
        }
