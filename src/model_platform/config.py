"""模型运行时配置读取。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.paths import get_project_paths, load_database_yaml


@dataclass(frozen=True)
class ModelRuntimeConfig:
    """模型平台运行时配置。"""

    default_llm_backend: str
    fallback_llm_backend: str
    vllm_config_path: Path
    llm_env_file: Path
    llm_request_timeout: int
    llm_retry: int
    default_embedding_model: str
    embedding_batch_size: int
    normalize_embeddings: bool
    prefer_cuda: bool
    empty_cache_after_batch: bool


def _as_bool(value: Any, default: bool = False) -> bool:
    """宽松解析 YAML 中的布尔值。"""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_project_path(value: str | Path, root: Path) -> Path:
    """将相对路径解析到项目根目录。"""
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_model_runtime_config(
    config_path: str | Path | None = None,
) -> ModelRuntimeConfig:
    """加载 `config/model_runtime.yaml`。"""
    paths = get_project_paths()
    root = paths.project_root
    target = Path(config_path) if config_path else root / "config" / "model_runtime.yaml"
    raw = load_database_yaml(target)

    llm = raw.get("llm", {}) if isinstance(raw, dict) else {}
    embedding = raw.get("embedding", {}) if isinstance(raw, dict) else {}
    torch_cfg = raw.get("torch", {}) if isinstance(raw, dict) else {}

    return ModelRuntimeConfig(
        default_llm_backend=str(llm.get("default_backend", "wsl_vllm")),
        fallback_llm_backend=str(llm.get("fallback_backend", "external_api")),
        vllm_config_path=_resolve_project_path(llm.get("vllm_config", "config/vllm.toml"), root),
        llm_env_file=_resolve_project_path(llm.get("env_file", ".env.local"), root),
        llm_request_timeout=int(llm.get("request_timeout", 600)),
        llm_retry=max(1, int(llm.get("retry", 2))),
        default_embedding_model=str(embedding.get("default_model", "bge")),
        embedding_batch_size=int(embedding.get("default_batch_size", 128)),
        normalize_embeddings=_as_bool(embedding.get("normalize_embeddings"), True),
        prefer_cuda=_as_bool(torch_cfg.get("prefer_cuda"), True),
        empty_cache_after_batch=_as_bool(torch_cfg.get("empty_cache_after_batch"), False),
    )
