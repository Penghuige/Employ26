"""将三家招聘平台 sample 表补充写入统一规范层。"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd
from sqlalchemy import text

from src.db.postgres import create_pg_engine
from src.db.recruitment_jobs_normalized import (
    DEFAULT_NORMALIZED_TABLE,
    build_normalized_rows_from_dataframe,
    ensure_recruitment_jobs_normalized_table,
    quote_table_name,
    upsert_recruitment_jobs_normalized,
)


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

DEFAULT_SOURCE_TABLES = [
    '"51job".sample',
    '"Liepin".sample',
    '"Zhilian".sample',
]


def _load_source_dataframe(
    source_table: str,
    normalized_table: str,
    only_missing: bool = False,
) -> pd.DataFrame:
    """读取单张 PostgreSQL source 表，并生成稳定 source_row_number。"""
    engine = create_pg_engine()
    try:
        with engine.connect() as connection:
            where_sql = "WHERE n.recruitment_record_id IS NULL" if only_missing else ""
            return pd.read_sql_query(
                text(
                    f"""
                    WITH src AS (
                        SELECT
                            row_number() OVER (ORDER BY ctid) AS __source_row_number,
                            *
                        FROM {source_table}
                    )
                    SELECT src.*
                    FROM src
                    LEFT JOIN {quote_table_name(normalized_table)} n
                      ON n.source_table = :source_table
                     AND n.source_row_number = src.__source_row_number
                    {where_sql}
                    ORDER BY src.__source_row_number
                    """
                ),
                connection,
                params={"source_table": source_table},
            )
    finally:
        engine.dispose()


def _count_missing_source_rows(source_table: str, normalized_table: str) -> dict[str, int]:
    """统计指定 source_table 中尚未进入统一规范层的记录数。"""
    engine = create_pg_engine()
    try:
        with engine.connect() as connection:
            row = connection.execute(
                text(
                    f"""
                    WITH src AS (
                        SELECT row_number() OVER (ORDER BY ctid) AS source_row_number
                        FROM {source_table}
                    )
                    SELECT
                        count(*) AS sample_rows,
                        sum(CASE WHEN n.recruitment_record_id IS NULL THEN 1 ELSE 0 END) AS missing_in_normalized,
                        sum(CASE WHEN n.recruitment_record_id IS NOT NULL THEN 1 ELSE 0 END) AS present_in_normalized
                    FROM src s
                    LEFT JOIN {quote_table_name(normalized_table)} n
                      ON n.source_table = :source_table
                     AND n.source_row_number = s.source_row_number
                    """
                ),
                {"source_table": source_table},
            ).mappings().one()
            return {key: int(value or 0) for key, value in row.items()}
    finally:
        engine.dispose()


def backfill_recruitment_jobs_normalized(
    source_tables: list[str] | None = None,
    normalized_table: str = DEFAULT_NORMALIZED_TABLE,
    dry_run: bool = False,
    only_missing: bool = True,
) -> dict[str, dict[str, int]]:
    """将三家平台 sample 表补充写入统一规范层。"""
    source_tables = list(source_tables or DEFAULT_SOURCE_TABLES)
    summary: dict[str, dict[str, int]] = {}

    if dry_run:
        for source_table in source_tables:
            summary[source_table] = _count_missing_source_rows(
                source_table=source_table,
                normalized_table=normalized_table,
            )
        return summary

    engine = create_pg_engine()
    try:
        with engine.begin() as connection:
            ensure_recruitment_jobs_normalized_table(connection, table_name=normalized_table)
    finally:
        engine.dispose()

    for source_table in source_tables:
        source_df = _load_source_dataframe(
            source_table=source_table,
            normalized_table=normalized_table,
            only_missing=only_missing,
        )
        rows = build_normalized_rows_from_dataframe(
            source_df,
            source_table=source_table,
        )
        written_count = upsert_recruitment_jobs_normalized(
            rows,
            table_name=normalized_table,
        )
        counts = _count_missing_source_rows(
            source_table=source_table,
            normalized_table=normalized_table,
        )
        counts["written_rows"] = int(written_count)
        summary[source_table] = counts

    return summary


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""
    parser = argparse.ArgumentParser(description="补齐三家 sample 表到 recruitment_jobs_normalized")
    parser.add_argument(
        "--source-table",
        nargs="*",
        default=None,
        help="可选：覆盖默认 sample 表列表",
    )
    parser.add_argument(
        "--normalized-table",
        default=DEFAULT_NORMALIZED_TABLE,
        help="目标 recruitment_jobs_normalized 表名",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只输出缺口统计，不写数据库",
    )
    parser.add_argument(
        "--all-source-rows",
        action="store_true",
        help="默认仅补 normalized 缺口；传入后改为重扫 source 全表并执行 upsert",
    )
    return parser


def main() -> None:
    """CLI 入口。"""
    args = build_parser().parse_args()
    summary = backfill_recruitment_jobs_normalized(
        source_tables=args.source_table,
        normalized_table=args.normalized_table,
        dry_run=bool(args.dry_run),
        only_missing=not bool(args.all_source_rows),
    )
    logger.info("backfill summary: %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
