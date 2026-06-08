"""统一模型平台层。

本包收口 LLM、embedding 和 PyTorch runtime 的配置与入口。
生成式 LLM 默认走 WSL vLLM；BGE/BERT 仍使用 PyTorch/Transformers。
"""

from .config import ModelRuntimeConfig, load_model_runtime_config
from .embeddings import get_embedding_model
from .llm import LLMClient, create_llm_client
from .torch_runtime import empty_cuda_cache_safe, resolve_model_path, resolve_torch_device

__all__ = [
    "LLMClient",
    "ModelRuntimeConfig",
    "create_llm_client",
    "empty_cuda_cache_safe",
    "get_embedding_model",
    "load_model_runtime_config",
    "resolve_model_path",
    "resolve_torch_device",
]
