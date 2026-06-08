"""PyTorch 运行时与模型路径工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config.paths import get_project_paths


def resolve_torch_device(prefer_cuda: bool = True) -> str:
    """解析 PyTorch 推荐设备。"""
    if not prefer_cuda:
        return "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def empty_cuda_cache_safe() -> None:
    """安全清理 CUDA cache。"""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _lookup_nested(mapping: dict[str, Any], key: str) -> Any:
    """在嵌套模型清单中查找 key。"""
    if key in mapping:
        return mapping[key]
    for value in mapping.values():
        if isinstance(value, dict) and key in value:
            return value[key]
    return None


def resolve_model_path(model_key: str) -> Path:
    """从 `config/database.yaml` 的 model_paths 中解析模型路径。"""
    paths = get_project_paths()
    value = _lookup_nested(paths.model_paths, model_key)
    if value is None:
        legacy_attr = {
            "bge": paths.bge_model_path,
            "llm": paths.qwen_model_path,
            "qwen": paths.qwen_model_path,
            "bert": paths.bert_model_path,
        }.get(model_key)
        if legacy_attr is not None:
            return Path(legacy_attr)
        raise KeyError(f"未配置模型路径: {model_key}")
    return Path(str(value))
