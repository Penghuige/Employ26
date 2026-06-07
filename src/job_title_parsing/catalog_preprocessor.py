"""职业大典预处理模块。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import re

import pandas as pd

from .alias_builder import AliasBuilder
from .match_utils import load_database_config, normalize_text, unique_keep_order


_DATABASE_CONFIG = load_database_config()
_DATABASE_SETTINGS = _DATABASE_CONFIG.get("database", {})
_JOB_TITLE_PARSING_SETTINGS = _DATABASE_CONFIG.get("job_title_parsing", {})


class CatalogPreprocessor:
    """读取、清洗并标准化《中国职业分类大典》数据。"""

    DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / _DATABASE_SETTINGS.get("duckdb_path", "output/recruit.duckdb")
    DEFAULT_TABLE_NAME = _JOB_TITLE_PARSING_SETTINGS.get(
        "catalog_table", "recruit.main.chinese_occupational_dictionary_joined"
    )

    REQUIRED_COLUMNS = [
        "code",
        "title",
        "desc",
        "tasks",
        "级别",
        "分类代码",
        "职业代码",
        "大类",
        "中类",
        "小类",
        "细类",
    ]

    def __init__(self, config: Dict[str, Any], alias_builder: AliasBuilder):
        self.config = config
        self.alias_builder = alias_builder

    def load_csv(self, csv_path: str | Path, encoding: str = "utf-8") -> pd.DataFrame:
        """读取职业大典 CSV。"""
        df = pd.read_csv(csv_path, encoding=encoding)
        return self.preprocess(df)

    def load_duckdb(
        self,
        db_path: str | Path | None = None,
        table_name: str | None = None,
        where_sql: str = "",
        limit: int | None = None,
    ) -> pd.DataFrame:
        """从 DuckDB 读取职业大典表并预处理。"""
        import duckdb

        db_target = Path(db_path) if db_path else self.DEFAULT_DB_PATH
        table_target = table_name or self.DEFAULT_TABLE_NAME

        if not db_target.exists():
            raise FileNotFoundError(f"DuckDB 文件不存在: {db_target}")

        query = f"SELECT * FROM {table_target}"
        if where_sql.strip():
            query += f" WHERE {where_sql.strip()}"
        if limit is not None and limit > 0:
            query += f" LIMIT {int(limit)}"

        conn = duckdb.connect(str(db_target), read_only=True)
        try:
            df = conn.execute(query).df()
        finally:
            conn.close()

        return self.preprocess(df)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化职业大典 DataFrame，生成检索所需的所有衍生字段。

        产出字段：task_list, task_text_joined, title_clean, desc_clean,
                 hierarchy_text, aliases, retrieval_title_text, retrieval_desc_text,
                 retrieval_task_text。

        Args:
            df: 原始职业大典 DataFrame，需包含 code / title / desc / tasks 等字段。

        Returns:
            pd.DataFrame: 添加了预处理字段的 DataFrame。
        """
        work_df = df.copy()
        for col in self.REQUIRED_COLUMNS:
            if col not in work_df.columns:
                work_df[col] = ""

        for col in self.REQUIRED_COLUMNS:
            if col == "tasks":
                work_df[col] = work_df[col].fillna("").map(lambda x: "" if x is None else str(x))
            else:
                work_df[col] = work_df[col].fillna("").map(normalize_text)

        work_df["task_list"] = work_df["tasks"].map(self.split_tasks)
        work_df["task_text_joined"] = work_df["task_list"].map(lambda x: " ".join(x))
        work_df["title_clean"] = work_df["title"].map(normalize_text)
        work_df["desc_clean"] = work_df["desc"].map(normalize_text)
        work_df["hierarchy_text"] = work_df.apply(self._build_hierarchy_text, axis=1)
        work_df["aliases"] = work_df["title_clean"].map(self.alias_builder.build_aliases)
        work_df["retrieval_title_text"] = work_df.apply(
            lambda row: " ".join(unique_keep_order([row["title_clean"], *row["aliases"]])),
            axis=1,
        )
        work_df["retrieval_desc_text"] = work_df["desc_clean"]
        work_df["retrieval_task_text"] = work_df.apply(
            lambda row: " ".join(
                unique_keep_order([
                    row["task_text_joined"],
                    row["desc_clean"],
                    row["hierarchy_text"],
                ])
            ),
            axis=1,
        )
        return work_df

    def save_duckdb(
        self,
        df: pd.DataFrame,
        db_path: str | Path,
        table_name: str,
    ) -> None:
        """保存预处理结果到 DuckDB。"""
        import duckdb

        conn = duckdb.connect(str(db_path))
        try:
            conn.register("catalog_df_view", df)
            conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM catalog_df_view")
        finally:
            conn.close()

    def split_tasks(self, tasks_text: Any) -> List[str]:
        """按换行、分号切分 tasks，清除 1. / 1、 等编号前缀。"""
        text = "" if tasks_text is None else str(tasks_text)
        text = text.replace("\u3000", " ")
        text = text.replace("；", ";")
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"\s+", " ", text)
        text = text.replace(" \n", "\n").replace("\n ", "\n").strip()
        if not text:
            return []

        parts = re.split(r"[\n;]+", text)
        results: List[str] = []
        for part in parts:
            cleaned = re.sub(r"^\d+[\.、]\s*", "", part).strip()
            cleaned = re.sub(r"^[-•]\s*", "", cleaned).strip()
            if cleaned:
                results.append(cleaned)
        return results

    def _build_hierarchy_text(self, row: pd.Series) -> str:
        """拼接层级信息。"""
        return " ".join(
            unique_keep_order([
                row.get("大类", ""),
                row.get("中类", ""),
                row.get("小类", ""),
                row.get("细类", ""),
            ])
        )
