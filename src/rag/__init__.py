"""本地 RAG 流程模块（职业知识库场景）。"""

from .config import RAGConfig
from .pipeline import LocalOccupationRAG

__all__ = ["RAGConfig", "LocalOccupationRAG"]
