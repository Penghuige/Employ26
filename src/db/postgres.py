"""统一的 PostgreSQL 连接与元数据访问层。"""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from config.paths import get_project_paths

PROJECT_PATHS = get_project_paths()


def build_pg_engine_options(
    *,
    pool_size: int | None = None,
    max_overflow: int | None = None,
    pool_recycle: int | None = None,
    pool_pre_ping: bool = True,
    application_name: str | None = None,
) -> dict[str, object]:
    """构建 SQLAlchemy PostgreSQL engine 参数。"""
    options: dict[str, object] = {
        "future": True,
        "pool_pre_ping": bool(pool_pre_ping),
    }
    if pool_size is not None:
        options["pool_size"] = int(pool_size)
    if max_overflow is not None:
        options["max_overflow"] = int(max_overflow)
    if pool_recycle is not None:
        options["pool_recycle"] = int(pool_recycle)
    if application_name:
        options["connect_args"] = {"application_name": str(application_name)}
    return options


@lru_cache(maxsize=8)
def resolve_pg_dbname(preferred: str | None = None) -> str:
    """解析当前机器上真实可连接的 PostgreSQL 数据库名。"""
    configured = str(preferred or PROJECT_PATHS.pg_connection_params["dbname"])
    candidate_names = [configured]
    if configured.lower() != configured:
        candidate_names.append(configured.lower())
    if configured.upper() != configured:
        candidate_names.append(configured.upper())
    if configured != "Employ26":
        candidate_names.append("Employ26")

    candidate_engine = create_engine(
        PROJECT_PATHS.pg_sqlalchemy_url(configured),
        future=True,
    )
    try:
        with candidate_engine.connect():
            return configured
    except OperationalError:
        pass
    finally:
        candidate_engine.dispose()

    admin_engine = create_engine(
        PROJECT_PATHS.pg_sqlalchemy_url("postgres"),
        future=True,
    )
    try:
        with admin_engine.connect() as conn:
            dbnames = {str(row[0]) for row in conn.execute(text("select datname from pg_database"))}
    finally:
        admin_engine.dispose()

    for name in candidate_names:
        if name in dbnames:
            return name
        for dbname in dbnames:
            if dbname.lower() == name.lower():
                return dbname

    raise RuntimeError(
        f"未找到可用 PostgreSQL 数据库。配置值={configured}，实际库列表={sorted(dbnames)}"
    )


def create_pg_engine(
    dbname: str | None = None,
    *,
    pool_size: int | None = None,
    max_overflow: int | None = None,
    pool_recycle: int | None = None,
    pool_pre_ping: bool = True,
    application_name: str | None = None,
):
    """创建 PostgreSQL SQLAlchemy engine。"""
    target_db = dbname or resolve_pg_dbname()
    return create_engine(
        PROJECT_PATHS.pg_sqlalchemy_url(target_db),
        **build_pg_engine_options(
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_recycle=pool_recycle,
            pool_pre_ping=pool_pre_ping,
            application_name=application_name,
        ),
    )


def ensure_schema(connection, schema_name: str) -> None:
    """确保目标 schema 存在。"""
    connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"'))


def table_exists(connection, schema_name: str, table_name: str) -> bool:
    """判断给定表是否存在。"""
    return connection.execute(
        text(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = :schema_name
              AND table_name = :table_name
            """
        ),
        {"schema_name": schema_name, "table_name": table_name},
    ).scalar_one_or_none() is not None


def get_table_columns(connection, schema_name: str, table_name: str) -> list[str]:
    """读取目标表列名。"""
    rows = connection.execute(
        text(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = :schema_name
              AND table_name = :table_name
            ORDER BY ordinal_position
            """
        ),
        {"schema_name": schema_name, "table_name": table_name},
    ).fetchall()
    return [str(row[0]) for row in rows]
