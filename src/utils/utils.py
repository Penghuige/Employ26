from dataclasses import dataclass
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
DEFAULT_DATABASE_CONFIG_PATH = PROJECT_ROOT / "config" / "database.yaml"

def safe_text(value: object) -> str:
    """安全转字符串。"""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def truncate_text(text: str, max_chars: int = 300) -> str:
    """截断上下文文本，控制序列长度。"""
    text = safe_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def load_database_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """加载数据库与表配置。"""
    target = Path(config_path) if config_path else DEFAULT_DATABASE_CONFIG_PATH
    if not target.exists() or not target.read_text(encoding="utf-8").strip():
        return {
            "database": {
                "duckdb_path": "output/recruit.duckdb",
                "duckdb_threads": 32,
            },
            "job_title_parsing": {
                "catalog_table": "recruit.main.chinese_occupational_dictionary_joined",
                "catalog_preprocessed_table": "recruit.main.chinese_occupational_dictionary_joined_preprocessed",
                "jobs_table": "recruit.main.jobs_sample",
                "match_result_table": "recruit.main.job_match_results",
            },
        }
    return simple_yaml_load(target)