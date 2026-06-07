"""RAG v2 配置。"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class RAGConfig:
    """RAG 全局配置（v2：DuckDB 数据源 + bge-large + DeepSeek 生成）。

    变更:
    - 数据源从 Excel 切换为 DuckDB（recruit.main.*）
    - 嵌入模型从 bge-base-zh-finetuned 切换为 bge-large-zh-v1.5
    - 生成器从本地 Qwen3-8B 切换为 DeepSeek V4 Pro（API 调用）
    - 启用双 chunk（definition + task）检索 + 层级重排序
    """

    # ---- 数据源 ----
    duckdb_path: str = r"output\recruit.duckdb"
    catalog_table: str = "recruit.main.chinese_occupational_dictionary_joined_preprocessed"

    # ---- 模型路径 ----
    embedding_model_path: str = r"D:\model\bge-large-zh-v1.5"

    # ---- 索引产物 ----
    index_dir: str = r"src\rag\artifacts"
    def_index_path: str = r"src\rag\artifacts\occupation_def_index.faiss"
    task_index_path: str = r"src\rag\artifacts\occupation_task_index.faiss"
    metadata_path: str = r"src\rag\artifacts\occupation_metadata_v2.json"

    # ---- 检索参数 ----
    embedding_batch_size: int = 64
    top_k: int = 8
    retrieval_pool_size: int = 30  # 粗召回池大小
    def_weight: float = 0.6        # definition chunk 匹配权重
    task_weight: float = 0.4       # task chunk 匹配权重

    # ---- DeepSeek 生成参数 ----
    generator_model: str = "deepseek-v4-pro"
    max_tokens: int = 512
    temperature: float = 0.1

    # ---- 层级字段候选 ----
    hierarchy_fields: List[str] = field(default_factory=lambda: ["大类", "中类", "小类", "细类"])
