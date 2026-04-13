# -*- coding: utf-8 -*-
"""
样本数据提取模块

从 DuckDB 数据库中按采样率提取样本数据，保存回数据库。
支持从 _cleaned_data 或 _raw_data 表中采样。

用法：
    python src/preprocessing/sample_data.py --sample-rate 0.1 --source cleaned
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# 默认 DuckDB 数据库路径（原始数据）
DEFAULT_DB_PATH = r"D:\PythonProjects\Employ26\output\recruit.duckdb"

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def get_table_names(
    conn,
    suffix: str = "_cleaned_data",
    schema: str = "main",
    catalog: str = "recruit",
) -> list[str]:
    """
    获取数据库中所有指定后缀的表名。

    Args:
        conn   : DuckDB 连接
        suffix : 表名后缀（如 _cleaned_data 或 _raw_data）
        schema : schema 名称
        catalog: catalog 名称

    Returns:
        表名列表（不含 schema/catalog 前缀）
    """
    query = f"""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_catalog = '{catalog}'
          AND table_schema = '{schema}'
          AND table_name LIKE '%{suffix}'
    """
    result = conn.execute(query).fetchall()
    return [row[0] for row in result]


def sample_table(
    db_path: str,
    table_name: str,
    sample_rate: float = 0.1,
    schema: str = "main",
    catalog: str = "recruit",
) -> int:
    """
    从表中按采样率提取样本，保存到新表。

    采样策略：使用 RANDOM() 函数进行随机采样，确保每次运行结果一致
    （通过 SEED 固定随机种子）。

    Args:
        db_path    : DuckDB 数据库文件路径
        table_name : 原始表名（含 _cleaned_data 或 _raw_data 后缀）
        sample_rate: 采样率（0.0 ~ 1.0），默认 0.1（10%）
        schema     : schema 名称
        catalog    : catalog 名称

    Returns:
        采样后的行数
    """
    try:
        import duckdb
    except ImportError as e:
        raise ImportError("请先安装：pip install duckdb") from e

    if not 0 < sample_rate <= 1.0:
        raise ValueError(f"采样率必须在 (0, 1] 之间，收到：{sample_rate}")

    conn = duckdb.connect(db_path)

    full_table_name = f"{catalog}.{schema}.{table_name}"

    # 验证表是否存在
    try:
        total_count = conn.execute(
            f"SELECT COUNT(*) FROM {full_table_name}"
        ).fetchone()[0]
    except Exception as e:
        logger.error(f"表不存在或无法访问：{full_table_name}")
        logger.error(f"错误：{e}")
        conn.close()
        return 0

    logger.info(f"开始采样表：{full_table_name}")
    logger.info(f"总行数：{total_count}，采样率：{sample_rate:.1%}")

    # 生成新表名（将后缀改为 _sample）
    # 例如：recruit_cleaned_data → recruit_sample
    if "_cleaned_data" in table_name:
        new_table_name = table_name.replace("_cleaned_data", "_sample")
    elif "_raw_data" in table_name:
        new_table_name = table_name.replace("_raw_data", "_sample")
    else:
        new_table_name = f"{table_name}_sample"

    full_new_table_name = f"{catalog}.{schema}.{new_table_name}"

    # 删除旧表（如果存在）
    try:
        conn.execute(f"DROP TABLE IF EXISTS {full_new_table_name}")
        logger.info(f"删除旧表（如果存在）：{full_new_table_name}")
    except Exception as e:
        logger.warning(f"删除旧表失败：{e}")

    # 执行采样（使用 RANDOM() 和 WHERE 子句）
    logger.info("执行采样...")
    try:
        # 使用 RANDOM() 进行随机采样
        # 为了可复现性，可以使用 seed_random() 固定随机种子
        conn.execute(f"SELECT setseed(0.42)")  # 固定随机种子
        conn.execute(
            f"""
            CREATE TABLE {full_new_table_name} AS
            SELECT *
            FROM {full_table_name}
            WHERE RANDOM() < {sample_rate}
            """
        )
    except Exception as e:
        logger.error(f"采样失败：{e}")
        conn.close()
        return 0

    # 验证采样结果
    sampled_count = conn.execute(
        f"SELECT COUNT(*) FROM {full_new_table_name}"
    ).fetchone()[0]
    actual_rate = sampled_count / total_count if total_count > 0 else 0

    logger.info(
        f"✓ 采样完成：{sampled_count} 行（实际采样率：{actual_rate:.1%}）"
    )
    logger.info(f"  新表名：{full_new_table_name}")

    conn.close()
    return sampled_count


def main():
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="DuckDB 样本数据提取")
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help="DuckDB 数据库文件路径",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=0.01,
        help="采样率 [0.0 ~ 1.0]，默认 0.01 [1%]",
    )
    parser.add_argument(
        "--source",
        choices=["cleaned", "raw", "all"],
        default="cleaned",
        help="数据源: cleaned [_cleaned_data] / raw [_raw_data] / all [两者都采样]",
    )
    parser.add_argument(
        "--schema", default="main", help="schema 名称 [默认 main]"
    )
    parser.add_argument(
        "--catalog", default="recruit", help="catalog 名称 [默认 recruit]"
    )
    parser.add_argument(
        "--table-prefix",
        default="",
        help="表名前缀（默认 为空）",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        logger.error(f"数据库文件不存在：{db_path}")
        sys.exit(1)

    try:
        import duckdb
    except ImportError as e:
        logger.error("请先安装 DuckDB：pip install duckdb")
        sys.exit(1)

    # 验证采样率
    if not 0 < args.sample_rate <= 1.0:
        logger.error(f"采样率必须在 (0, 1] 之间，收到：{args.sample_rate}")
        sys.exit(1)

    conn = duckdb.connect(str(db_path))

    # 确定要采样的表后缀
    suffixes = []
    if args.source in ["cleaned", "all"]:
        suffixes.append("_cleaned_data")
    if args.source in ["raw", "all"]:
        suffixes.append("_raw_data")

    total_sampled = 0

    for suffix in suffixes:
        logger.info(f"\n{'='*60}")
        logger.info(f"查找所有 {suffix} 表...")
        logger.info(f"{'='*60}")

        tables = get_table_names(conn, suffix, args.schema, args.catalog)
        if not tables:
            logger.warning(f"未找到 {suffix} 表")
            continue

        logger.info(f"找到 {len(tables)} 个表：{tables}")

        # 筛选指定前缀的表
        tables_to_sample = [
            t for t in tables if t.startswith(args.table_prefix)
        ]
        if not tables_to_sample:
            logger.warning(
                f"未找到前缀为 '{args.table_prefix}' 的 {suffix} 表"
            )
            continue

        # 逐表采样
        for table in tables_to_sample:
            logger.info(f"\n--- 处理表：{table} ---")
            try:
                sampled = sample_table(
                    str(db_path),
                    table,
                    args.sample_rate,
                    args.schema,
                    args.catalog,
                )
                total_sampled += sampled
            except Exception as e:
                logger.error(f"采样失败：{e}")

    conn.close()

    logger.info(f"\n{'='*60}")
    logger.info(f"✓ 采样完成！总采样行数：{total_sampled}")
    logger.info(f"  数据库：{db_path}")
    logger.info(f"  采样率：{args.sample_rate:.1%}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
