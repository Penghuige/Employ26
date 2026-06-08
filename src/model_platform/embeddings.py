"""Embedding 模型统一加载入口。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import load_model_runtime_config
from .torch_runtime import resolve_model_path, resolve_torch_device


@lru_cache(maxsize=8)
def _load_sentence_transformer(model_path: str, device: str):
    """缓存加载 SentenceTransformer。"""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_path, device=device)


def get_embedding_model(
    model_key: str | None = None,
    *,
    device: str | None = None,
) -> Any:
    """返回统一缓存的 embedding 模型。"""
    runtime = load_model_runtime_config()
    key = model_key or runtime.default_embedding_model
    model_path = resolve_model_path(key)
    target_device = device or resolve_torch_device(runtime.prefer_cuda)
    return _load_sentence_transformer(str(Path(model_path)), target_device)
