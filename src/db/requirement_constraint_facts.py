"""`public.requirement_constraint_facts` 的建表与写入逻辑。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from sqlalchemy import text

from src.db.postgres import create_pg_engine, ensure_schema
from src.db.recruitment_jobs_normalized import quote_table_name, split_table_name


DEFAULT_FACTS_TABLE = "public.requirement_constraint_facts"
CONSTRAINT_TYPES = (
    "hard_gate",
    "preference",
    "range",
    "binary",
)
OPERATORS = (
    "eq",
    "ge",
    "le",
    "between",
    "contains",
    "allow",
    "exclude",
)


@dataclass(frozen=True)
class RequirementConstraintFactRow:
    """单条 requirement 约束事实。"""

    recruitment_record_id: str
    source_table: str
    source_row_number: int
    item_index: int
    item_text_raw: str
    item_text_normalized: str
    dimension_name: str
    constraint_type: str
    raw_value: str
    normalized_value: str
    operator: str
    value_min: float | None
    value_max: float | None
    unit: str
    evidence_text: str
    rule_id: int | None
    extractor_version: str


def _sql_in(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{value}'" for value in values)


def ensure_requirement_constraint_facts_table(
    connection,
    table_name: str = DEFAULT_FACTS_TABLE,
) -> None:
    """确保 requirement 约束事实表存在。"""
    schema_name, raw_table_name = split_table_name(table_name)
    ensure_schema(connection, schema_name)
    qualified_table = quote_table_name(table_name)
    object_prefix = raw_table_name.replace('"', "").replace(".", "_").strip("_") or "requirement_constraint_facts"

    connection.execute(
        text(
            f"""
            CREATE TABLE IF NOT EXISTS {qualified_table} (
                fact_id bigserial PRIMARY KEY,
                recruitment_record_id text NOT NULL,
                source_table text NOT NULL,
                source_row_number bigint NOT NULL,
                item_index integer NOT NULL,
                item_text_raw text NOT NULL,
                item_text_normalized text NOT NULL,
                dimension_name text NOT NULL,
                constraint_type text NOT NULL CHECK (constraint_type IN ({_sql_in(CONSTRAINT_TYPES)})),
                raw_value text NOT NULL DEFAULT '',
                normalized_value text NOT NULL,
                operator text NOT NULL CHECK (operator IN ({_sql_in(OPERATORS)})),
                value_min double precision,
                value_max double precision,
                unit text NOT NULL DEFAULT '',
                evidence_text text NOT NULL DEFAULT '',
                rule_id bigint,
                extractor_version text NOT NULL,
                created_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
    )
    connection.execute(
        text(
            f"""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_{object_prefix}_logical_unique
            ON {qualified_table} (
                recruitment_record_id,
                item_index,
                dimension_name,
                normalized_value,
                operator,
                constraint_type,
                extractor_version
            )
            """
        )
    )
    connection.execute(
        text(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{object_prefix}_record
            ON {qualified_table} (recruitment_record_id)
            """
        )
    )
    connection.execute(
        text(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{object_prefix}_dimension
            ON {qualified_table} (dimension_name, extractor_version)
            """
        )
    )
    connection.execute(
        text(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{object_prefix}_source_locator
            ON {qualified_table} (source_table, source_row_number)
            """
        )
    )


def ensure_requirement_constraint_facts_table_in_database(
    table_name: str = DEFAULT_FACTS_TABLE,
) -> None:
    """在数据库中创建 requirement 约束事实表。"""
    engine = create_pg_engine()
    try:
        with engine.begin() as connection:
            ensure_requirement_constraint_facts_table(connection, table_name=table_name)
    finally:
        engine.dispose()


def replace_requirement_constraint_facts(
    rows: list[RequirementConstraintFactRow],
    *,
    extractor_version: str,
    table_name: str = DEFAULT_FACTS_TABLE,
) -> int:
    """全量替换指定 extractor_version 的事实表结果。"""
    qualified_table = quote_table_name(table_name)
    engine = create_pg_engine()
    try:
        with engine.begin() as connection:
            ensure_requirement_constraint_facts_table(connection, table_name=table_name)
            connection.execute(
                text(
                    f"""
                    DELETE FROM {qualified_table}
                    WHERE extractor_version = :extractor_version
                    """
                ),
                {"extractor_version": extractor_version},
            )
            if rows:
                connection.execute(
                    text(
                        f"""
                        INSERT INTO {qualified_table} (
                            recruitment_record_id,
                            source_table,
                            source_row_number,
                            item_index,
                            item_text_raw,
                            item_text_normalized,
                            dimension_name,
                            constraint_type,
                            raw_value,
                            normalized_value,
                            operator,
                            value_min,
                            value_max,
                            unit,
                            evidence_text,
                            rule_id,
                            extractor_version
                        )
                        VALUES (
                            :recruitment_record_id,
                            :source_table,
                            :source_row_number,
                            :item_index,
                            :item_text_raw,
                            :item_text_normalized,
                            :dimension_name,
                            :constraint_type,
                            :raw_value,
                            :normalized_value,
                            :operator,
                            :value_min,
                            :value_max,
                            :unit,
                            :evidence_text,
                            :rule_id,
                            :extractor_version
                        )
                        """
                    ),
                    [row.__dict__ for row in rows],
                )
    finally:
        engine.dispose()
    return len(rows)


def load_requirement_constraint_facts_dataframe(
    *,
    extractor_version: str = "",
    table_name: str = DEFAULT_FACTS_TABLE,
):
    """读取 requirement 约束事实表。"""
    import pandas as pd

    qualified_table = quote_table_name(table_name)
    engine = create_pg_engine()
    try:
        with engine.connect() as connection:
            if extractor_version:
                return pd.read_sql_query(
                    text(
                        f"""
                        SELECT *
                        FROM {qualified_table}
                        WHERE extractor_version = :extractor_version
                        """
                    ),
                    connection,
                    params={"extractor_version": extractor_version},
                )
            return pd.read_sql_query(text(f"SELECT * FROM {qualified_table}"), connection)
    finally:
        engine.dispose()


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数。"""
    parser = argparse.ArgumentParser(description="requirement_constraint_facts 表管理工具")
    parser.add_argument(
        "--ensure-schema",
        action="store_true",
        help="在 PostgreSQL 中创建 public.requirement_constraint_facts",
    )
    return parser


def main() -> None:
    """CLI 入口。"""
    args = build_parser().parse_args()
    if args.ensure_schema:
        ensure_requirement_constraint_facts_table_in_database()


if __name__ == "__main__":
    main()
