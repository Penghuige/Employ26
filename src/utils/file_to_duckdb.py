# -*- coding: utf-8 -*-
"""
文件导入 DuckDB 工具。

支持将 CSV / XLSX 文件导入 DuckDB，并可自定义目标表名。
支持可选的列名规范化（去空格/特殊符号、重名处理）。
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


DEFAULT_DB_PATH = r"D:\PythonProjects\Employ26\output\recruit.duckdb"


def normalize_column_name(column_name: str) -> str:
    """
    规范化单个列名。

    规则：
        1. 去除首尾空白并转小写
        2. 空白替换为下划线
        3. 非中英文/数字/下划线字符替换为下划线
        4. 合并连续下划线
        5. 去除首尾下划线
        6. 若为空或数字开头，自动补前缀 col_

    Args:
        column_name: 原始列名

    Returns:
        规范化后的列名
    """
    name = str(column_name).strip().lower()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^\u4e00-\u9fffa-z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")

    if not name:
        name = "col"
    if re.match(r"^\d", name):
        name = f"col_{name}"
    return name


def normalize_dataframe_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    规范化 DataFrame 所有列名，并处理重名冲突。

    重名处理策略：
        - 第一次出现：name
        - 第二次出现：name_2
        - 第三次出现：name_3

    Args:
        df: 原始 DataFrame

    Returns:
        (新 DataFrame, 列名映射字典 old->new)
    """
    new_df = df.copy()
    mapping: dict[str, str] = {}
    counter: dict[str, int] = {}
    normalized_cols: list[str] = []

    for col in new_df.columns:
        base = normalize_column_name(col)
        counter[base] = counter.get(base, 0) + 1
        if counter[base] == 1:
            new_col = base
        else:
            new_col = f"{base}_{counter[base]}"

        mapping[str(col)] = new_col
        normalized_cols.append(new_col)

    new_df.columns = normalized_cols
    return new_df, mapping


def import_file_to_duckdb(
    file_path: str,
    table_name: str,
    db_path: str = DEFAULT_DB_PATH,
    catalog: str = "recruit",
    schema: str = "main",
    if_exists: str = "replace",
    normalize_columns: bool = False,
) -> int:
    """
    将 CSV 或 XLSX 文件导入 DuckDB 目标表。

    Args:
        file_path: 输入文件路径（.csv/.xlsx/.xls）
        table_name: 目标表名（例如 chinese_occupational_dictionary）
        db_path: DuckDB 文件路径
        catalog: DuckDB catalog 名称，默认 recruit
        schema: DuckDB schema 名称，默认 main
        if_exists: 目标表存在时策略，支持 replace / fail
        normalize_columns: 是否规范化列名（默认 False）

    Returns:
        导入后的行数

    Raises:
        FileNotFoundError: 输入文件不存在
        ValueError: 文件类型不支持或参数非法
    """
    input_path = Path(file_path)
    if not input_path.exists():
        raise FileNotFoundError(f"文件不存在: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise ValueError(f"仅支持 CSV/XLSX/XLS 文件，当前为: {suffix}")

    if if_exists not in {"replace", "fail"}:
        raise ValueError("if_exists 仅支持 'replace' 或 'fail'")

    logger.info(f"读取文件: {input_path}")
    if suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        df = pd.read_excel(input_path)

    if normalize_columns:
        df, mapping = normalize_dataframe_columns(df)
        logger.info("已启用列名规范化。前10列映射如下：")
        for i, (old, new) in enumerate(mapping.items()):
            if i >= 10:
                break
            logger.info(f"  {old} -> {new}")

    full_table_name = f"{catalog}.{schema}.{table_name}"
    logger.info(f"连接数据库: {db_path}")
    conn = duckdb.connect(db_path)

    try:
        if if_exists == "fail":
            exists = conn.execute(
                f"""
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_catalog = '{catalog}'
                  AND table_schema = '{schema}'
                  AND table_name = '{table_name}'
                """
            ).fetchone()[0]
            if exists:
                raise ValueError(f"目标表已存在: {full_table_name}")
        else:
            conn.execute(f"DROP TABLE IF EXISTS {full_table_name}")

        conn.register("temp_import_df", df)
        conn.execute(
            f"CREATE TABLE {full_table_name} AS SELECT * FROM temp_import_df"
        )
        conn.unregister("temp_import_df")

        row_count = conn.execute(
            f"SELECT COUNT(*) FROM {full_table_name}"
        ).fetchone()[0]

        logger.info(f"导入完成: {full_table_name} ({row_count} 行)")
        return int(row_count)
    finally:
        conn.close()


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="CSV/XLSX 导入 DuckDB")
    parser.add_argument("--file", required=True, help="输入文件路径")
    parser.add_argument("--table", required=True, help="目标表名")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="DuckDB 文件路径")
    parser.add_argument("--catalog", default="recruit", help="catalog 名称")
    parser.add_argument("--schema", default="main", help="schema 名称")
    parser.add_argument(
        "--if-exists",
        choices=["replace", "fail"],
        default="replace",
        help="目标表存在时处理策略",
    )
    parser.add_argument(
        "--normalize-columns",
        action="store_true",
        help="启用列名规范化（推荐）",
    )
    args = parser.parse_args()

    import_file_to_duckdb(
        file_path=args.file,
        table_name=args.table,
        db_path=args.db,
        catalog=args.catalog,
        schema=args.schema,
        if_exists=args.if_exists,
        normalize_columns=args.normalize_columns,
    )


if __name__ == "__main__":
    main()
