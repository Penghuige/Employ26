"""集中化的项目路径管理。

本模块提供 `ProjectPaths` dataclass，将所有模型路径、PostgreSQL 连接参数
和输出路径集中到单一配置对象中，并通过环境变量支持跨环境覆盖。

用法:
    from config.paths import get_project_paths
    paths = get_project_paths()
    pg = paths.pg_connection_params  # dict: host, port, dbname, user, password
    model = paths.bge_model_path     # 读取环境变量或默认值

环境变量覆盖:
    EMPLOYDATA_PG_HOST     — PostgreSQL 主机地址
    EMPLOYDATA_PG_PORT     — PostgreSQL 端口
    EMPLOYDATA_PG_DBNAME   — 数据库名
    EMPLOYDATA_PG_USER     — 用户名
    EMPLOYDATA_PG_PASSWORD — 密码
    EMPLOYDATA_BGE_MODEL_PATH — BGE embedding 模型路径
    EMPLOYDATA_QWEN_MODEL_PATH — Qwen LLM 模型路径
    EMPLOYDATA_BERT_MODEL_PATH — BERT 模型路径
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    """返回项目根目录（config/ 的父目录）。"""
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ProjectPaths:
    """项目集中配置对象。

    所有模型路径和 Postgres 连接参数的唯一来源。
    每个字段都可通过同名环境变量覆盖，默认值基于项目根目录的相对路径。
    """

    project_root: Path
    pg_connection_params: dict = field(default_factory=dict)
    bge_model_path: Path = field(default_factory=Path)
    qwen_model_path: Path = field(default_factory=Path)
    bert_model_path: Path = field(default_factory=Path)
    output_dir: Path = field(default_factory=Path)
    dict_dir: Path = field(default_factory=Path)
    config_dir: Path = field(default_factory=Path)

    @property
    def pg_host(self) -> str:
        """PostgreSQL 主机地址。"""
        return self.pg_connection_params.get("host", "localhost")

    @property
    def pg_port(self) -> int:
        """PostgreSQL 端口。"""
        return int(self.pg_connection_params.get("port", 5432))

    @property
    def pg_dbname(self) -> str:
        """PostgreSQL 数据库名。"""
        return self.pg_connection_params.get("dbname", "employ26")

    @property
    def pg_user(self) -> str:
        """PostgreSQL 用户名。"""
        return self.pg_connection_params.get("user", "postgres")

    @property
    def pg_password(self) -> str:
        """PostgreSQL 密码。"""
        return self.pg_connection_params.get("password", "")

    @property
    def pg_schema(self) -> str:
        """PostgreSQL schema 前缀。"""
        return self.pg_connection_params.get("schema", "recruit")

    @property
    def skill_extraction_output_dir(self) -> Path:
        """技能抽取产物的输出目录。"""
        return self.output_dir / "skill_extraction"

    @property
    def cache_dir(self) -> Path:
        """中间产物缓存目录。"""
        return self.skill_extraction_output_dir / "cache"

    @property
    def report_dir(self) -> Path:
        """报告输出目录。"""
        return self.skill_extraction_output_dir / "reports"

    @property
    def prompt_dir(self) -> Path:
        """LLM prompt 模板目录。"""
        return self.skill_extraction_output_dir / "prompts"

    @property
    def rag_index_dir(self) -> Path:
        """RAG 向量索引产物目录。"""
        return self.project_root / "src" / "rag" / "artifacts"


def get_project_paths(
    *,
    pg_params: dict | None = None,
    bge_model_path: str | Path | None = None,
    qwen_model_path: str | Path | None = None,
    bert_model_path: str | Path | None = None,
) -> ProjectPaths:
    """构建 `ProjectPaths` 实例，优先使用显式参数，其次环境变量，最后默认值。

    Args:
        pg_params: 覆盖 PostgreSQL 连接参数（dict，键: host/port/dbname/user/password/schema）。
        bge_model_path: 覆盖 BGE embedding 模型路径。
        qwen_model_path: 覆盖 Qwen LLM 模型路径。
        bert_model_path: 覆盖 BERT 模型路径。

    Returns:
        ProjectPaths: 冻结的路径与连接配置对象。
    """
    root = _project_root()

    def _resolve(key: str, explicit: str | Path | None, default: str | Path) -> Path:
        """按 显式参数 > 环境变量 > 默认值 的优先级解析路径。"""
        if explicit is not None:
            return Path(explicit)
        env_val = os.getenv(key)
        if env_val:
            return Path(env_val)
        return Path(default)

    # Postgres 连接参数：显式 dict > 环境变量 > 默认值
    if pg_params is None:
        pg_params = {}
    pg_connection_params = {
        "host": pg_params.get("host") or os.getenv("EMPLOYDATA_PG_HOST", "localhost"),
        "port": pg_params.get("port") or int(os.getenv("EMPLOYDATA_PG_PORT", "5432")),
        "dbname": pg_params.get("dbname") or os.getenv("EMPLOYDATA_PG_DBNAME", "employ26"),
        "user": pg_params.get("user") or os.getenv("EMPLOYDATA_PG_USER", "postgres"),
        "password": pg_params.get("password") or os.getenv("EMPLOYDATA_PG_PASSWORD", ""),
        "schema": pg_params.get("schema") or os.getenv("EMPLOYDATA_PG_SCHEMA", "recruit"),
    }

    return ProjectPaths(
        project_root=root,
        pg_connection_params=pg_connection_params,
        bge_model_path=_resolve(
            "EMPLOYDATA_BGE_MODEL_PATH",
            bge_model_path,
            root / "models" / "bge-base-zh-v1.5",
        ),
        qwen_model_path=_resolve(
            "EMPLOYDATA_QWEN_MODEL_PATH",
            qwen_model_path,
            root / "models" / "Qwen3-8B",
        ),
        bert_model_path=_resolve(
            "EMPLOYDATA_BERT_MODEL_PATH",
            bert_model_path,
            root / "models" / "chinese-roberta-wwm-ext",
        ),
        output_dir=root / "output",
        dict_dir=root / "dicts",
        config_dir=root / "config",
    )
