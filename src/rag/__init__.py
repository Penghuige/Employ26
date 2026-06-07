"""RAG v2 模块（DuckDB 数据源 + bge-large + DeepSeek V4 Pro）。"""

from .config import RAGConfig
from .pipeline import OccupationRAG

__all__ = ["RAGConfig", "OccupationRAG"]
