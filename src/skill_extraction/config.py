"""
技能词典流程配置模块。

职责：
1. 统一读取 `config/database.yaml`；
2. 规范 `src/skill_extraction` 产物目录；
3. 规范职业细类匹配与职业技能词典的存储位置。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from src.job_title_parsing.match_utils import load_database_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _split_table_names(value: object) -> List[str]:
    """解析 jobs_table 配置，支持逗号分隔字符串或 YAML 列表。"""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def qualify_table_name(table_name: str, catalog: str = "recruit", schema: str = "main") -> str:
    """补齐未带 catalog/schema 前缀的 DuckDB 表名。"""
    normalized = str(table_name).strip()
    if not normalized:
        raise ValueError("table_name 不能为空")
    if normalized.count(".") >= 2:
        return normalized
    if normalized.count(".") == 1:
        return f"{catalog}.{normalized}"
    return f"{catalog}.{schema}.{normalized}"


@dataclass(frozen=True)
class SkillExtractionConfig:
    """技能词典流程配置。"""

    project_root: Path
    db_path: Path
    duckdb_threads: int
    embedding_model_path: Path
    embedding_device: str
    embedding_batch_size: int
    match_top_k: int
    catalog_table: str
    catalog_preprocessed_table: str
    jobs_tables: List[str]
    requirement_match_table: str
    output_dir: Path
    prompt_train_dir: Path
    prompt_supplement_dir: Path
    report_dir: Path
    cache_dir: Path
    dict_dir: Path
    dictionary_path: Path
    catalog_embedding_cache_path: Path
    training_manifest_path: Path
    training_requirements_path: Path
    validation_pool_path: Path
    category_summary_path: Path
    state_path: Path

    def ensure_dirs(self) -> None:
        """创建流程所需目录。"""
        for target in [
            self.output_dir,
            self.prompt_train_dir,
            self.prompt_supplement_dir,
            self.report_dir,
            self.cache_dir,
            self.dict_dir,
        ]:
            target.mkdir(parents=True, exist_ok=True)


def load_skill_extraction_config(database_config_path: str | Path | None = None) -> SkillExtractionConfig:
    """读取并构建技能词典流程配置。"""
    import torch

    raw_config = load_database_config(database_config_path)
    db_settings = raw_config.get("database", {})
    parsing_settings = raw_config.get("job_title_parsing", {})
    skill_settings = raw_config.get("skill_extraction", {})

    db_path = PROJECT_ROOT / db_settings.get("duckdb_path", "output/recruit.duckdb")
    jobs_tables = [
        qualify_table_name(table_name)
        for table_name in _split_table_names(parsing_settings.get("jobs_table"))
    ]

    if not jobs_tables:
        raise ValueError("config/database.yaml 未配置 job_title_parsing.jobs_table")

    output_dir = PROJECT_ROOT / "output" / "skill_extraction"
    prompt_dir = output_dir / "prompts"
    report_dir = output_dir / "reports"
    cache_dir = output_dir / "cache"
    dict_dir = PROJECT_ROOT / "dicts"

    config = SkillExtractionConfig(
        project_root=PROJECT_ROOT,
        db_path=db_path,
        duckdb_threads=max(1, int(db_settings.get("duckdb_threads", 8))),
        embedding_model_path=Path(r"D:\model\bge-base-zh-finetuned"),
        embedding_device="cuda" if torch.cuda.is_available() else "cpu",
        embedding_batch_size=128,
        match_top_k=5,
        catalog_table=qualify_table_name(
            parsing_settings.get(
                "catalog_table",
                "recruit.main.chinese_occupational_dictionary_joined",
            )
        ),
        catalog_preprocessed_table=qualify_table_name(
            parsing_settings.get(
                "catalog_preprocessed_table",
                "recruit.main.chinese_occupational_dictionary_joined_preprocessed",
            )
        ),
        jobs_tables=jobs_tables,
        requirement_match_table=qualify_table_name(
            skill_settings.get(
                "requirement_match_table",
                "recruit.main.skill_extraction_requirement_matches",
            )
        ),
        output_dir=output_dir,
        prompt_train_dir=prompt_dir / "train",
        prompt_supplement_dir=prompt_dir / "supplement",
        report_dir=report_dir,
        cache_dir=cache_dir,
        dict_dir=dict_dir,
        dictionary_path=dict_dir / "occupation_skill_dictionary.json",
        catalog_embedding_cache_path=cache_dir / "occupation_catalog_embeddings.npy",
        training_manifest_path=output_dir / "occupation_skill_training_manifest.csv",
        training_requirements_path=output_dir / "occupation_skill_training_requirements.csv",
        validation_pool_path=output_dir / "occupation_skill_validation_pool.csv",
        category_summary_path=output_dir / "occupation_skill_category_summary.csv",
        state_path=output_dir / "occupation_skill_iteration_state.json",
    )
    config.ensure_dirs()
    return config
