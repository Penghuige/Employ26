"""RAG v2 配置，项目路径统一从 `config.paths` 获取。"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from config.paths import get_project_paths


@dataclass
class RAGConfig:
    """RAG 全局配置（v2：DuckDB 数据源 + bge-large + DeepSeek 生成）。

    变更:
    - 数据源从 Excel 切换为 DuckDB（recruit.main.*）
    - 嵌入模型从 bge-base-zh-finetuned 切换为 bge-large-zh-v1.5
    - 生成器从本地 Qwen3-8B 切换为 DeepSeek V4 Pro（API 调用）
    - 启用双 chunk（definition + task）检索 + 层级重排序
    - 模型路径统一从 `ProjectPaths` 获取，支持环境变量覆盖
    """

    # ---- 数据源 ----
    duckdb_path: str = ""
    catalog_table: str = "recruit.main.chinese_occupational_dictionary_joined_preprocessed"

    # ---- 模型路径 ----
    embedding_model_path: str = ""

    # ---- 索引产物 ----
    index_dir: str = ""
    def_index_path: str = ""
    task_index_path: str = ""
    metadata_path: str = ""

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

    @classmethod
    def from_project_paths(cls, paths: ProjectPaths | None = None) -> "RAGConfig":
        """使用集中路径配置构建 RAGConfig，消除硬编码路径。

        参数:
            paths: 可选的 ProjectPaths 实例；为空时自动从环境变量/默认值构建。

        返回:
            RAGConfig: 包含完整路径的配置对象。
        """
        if paths is None:
            paths = get_project_paths()
        index_dir = paths.rag_index_dir
        return cls(
            duckdb_path=str(paths.duckdb_path),
            embedding_model_path=str(paths.bge_model_path),
            index_dir=str(index_dir),
            def_index_path=str(index_dir / "occupation_def_index.faiss"),
            task_index_path=str(index_dir / "occupation_task_index.faiss"),
            metadata_path=str(index_dir / "occupation_metadata_v2.json"),
        )
