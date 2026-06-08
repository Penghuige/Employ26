from pathlib import Path

from config.paths import get_project_paths
from src.model_platform.config import load_model_runtime_config


def test_model_runtime_defaults_to_wsl_vllm():
    config = load_model_runtime_config()
    assert config.default_llm_backend == "wsl_vllm"
    assert config.fallback_llm_backend == "external_api"
    assert config.vllm_config_path.name == "vllm.toml"
    assert config.prefer_cuda is True


def test_project_paths_exposes_model_inventory():
    paths = get_project_paths()
    assert "embedding" in paths.model_paths
    assert Path(paths.model_paths["embedding"]["bge_base_zh_v1_5"]).name == "bge-base-zh-v1.5"
