#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据迁移工具：CSV -> DuckDB + Parquet
将大型CSV文件迁移到更高效的存储格式

使用方法：
  python migrate_to_duckdb.py                    # 迁移 data/ 目录下所有CSV
  python migrate_to_duckdb.py --source output/integrated  # 迁移整合数据
  python migrate_to_duckdb.py --query            # 查询示例
"""

import re
import pandas as pd
import duckdb
from pathlib import Path
import logging
import argparse
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / 'output' / 'recruit.duckdb'
PARQUET_DIR = BASE_DIR / 'output' / 'parquet'

# 中文文件名 -> 英文表名映射（长词优先，防止短词提前截断）
_CHINESE_NAME_MAP = [
    ('广东省招聘数据', 'gd_recruit'),
    ('前程无忧',       'qcwy'),
    ('猎聘网',         'liepin'),
    ('猎聘',           'liepin'),
    ('智联招聘',       'zhilian'),
    ('智联',           'zhilian'),
    ('广东省',         'guangdong'),
    ('广东',           'guangdong'),
    ('招聘数据',       'recruit'),
    ('招聘',           'recruit'),
    ('数据',           'data'),
    ('整合',           'integrated'),
    ('解析',           'parsed'),
    ('处理',           'processed'),
]


def make_table_name(stem: str) -> str:
    """Convert filename (may contain Chinese) to a valid, readable DuckDB table name.

    Examples:
      gd_recruit_qcwy_202201_202506  (from qcwy source)
      zhilian_guangdong_202203_202506
      gd_recruit_liepin_202201_202506
    """
    name = stem
    for cn, en in _CHINESE_NAME_MAP:
        name = name.replace(cn, en)
    name = name.replace('-', '_').replace(' ', '_')
    safe = ''
    prev_us = False
    for ch in name:
        if ch.isascii() and (ch.isalnum() or ch == '_'):
            safe += ch
            prev_us = (ch == '_')
        else:
            if not prev_us and safe:
                safe += '_'
                prev_us = True
    safe = re.sub(r'_+', '_', safe).strip('_') or 'data'
    if safe[0].isdigit():
        safe = 't_' + safe
    return safe

def csv_to_parquet(csv_dir: str = 'data', pattern: str = '*.csv'):
    """
    将指定目录下的所有CSV文件转换为Parquet格式

    Args:
        csv_dir: CSV文件目录（相对于项目根目录）
        pattern: 文件匹配模式
    """
    src_dir = BASE_DIR / csv_dir
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    if not src_dir.exists():
        logger.error(f"❌ 目录不存在: {src_dir}")
        return

    csv_files = list(src_dir.glob(pattern))
    if not csv_files:
        logger.warning(f"⚠️  未找到CSV文件: {src_dir}/{pattern}")
        return

    logger.info(f"找到 {len(csv_files)} 个CSV文件，开始转换...")

    for csv_file in csv_files:
        t0 = time.time()
        out_file = PARQUET_DIR / (csv_file.stem + '.parquet')

        # 已存在则跳过
        if out_file.exists():
            logger.info(f"  ⏭️  已存在，跳过: {out_file.name}")
            continue

        try:
            logger.info(f"  转换: {csv_file.name}")
            chunks = []
            for chunk in pd.read_csv(csv_file, encoding='utf-8',
                                     low_memory=False, chunksize=100_000):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)

            df.to_parquet(out_file, index=False, compression='snappy', engine='pyarrow')

            csv_mb = csv_file.stat().st_size / 1024 / 1024
            pq_mb = out_file.stat().st_size / 1024 / 1024
            ratio = csv_mb / pq_mb if pq_mb > 0 else 0
            elapsed = time.time() - t0
            logger.info(f"  ✅ {csv_file.name}: {csv_mb:.1f}MB -> {pq_mb:.1f}MB Parquet "
                        f"(压缩比 {ratio:.1f}x, 耗时 {elapsed:.1f}s, {len(df):,} 行)")
        except Exception as e:
            logger.error(f"  ❌ 转换失败: {csv_file.name} - {e}")
            import traceback
            traceback.print_exc()

    logger.info(f"\nParquet 文件保存在: {PARQUET_DIR}")


def build_duckdb(csv_dir: str = 'data', pattern: str = '*.csv'):
    """
    将CSV文件直接导入DuckDB数据库

    Args:
        csv_dir: CSV文件目录
        pattern: 文件匹配模式
    """
    src_dir = BASE_DIR / csv_dir
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not src_dir.exists():
        logger.error(f"❌ 目录不存在: {src_dir}")
        return

    csv_files = list(src_dir.glob(pattern))
    if not csv_files:
        logger.warning(f"⚠️  未找到CSV文件")
        return

    logger.info(f"\n正在构建 DuckDB 数据库: {DB_PATH}")
    con = duckdb.connect(str(DB_PATH))

    try:
        for csv_file in csv_files:
            t0 = time.time()
            table_name = make_table_name(csv_file.stem)
            csv_path_str = str(csv_file).replace('\\', '/')

            logger.info(f"  导入: {csv_file.name} -> 表 [{table_name}]")

            # 表名用双引号包裹，避免数字开头或关键字冲突
            con.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            con.execute(
                f'CREATE TABLE "{table_name}" AS '
                f"SELECT * FROM read_csv_auto('{csv_path_str}', "
                f"header=true, all_varchar=false, ignore_errors=true)"
            )

            row_count = con.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
            elapsed = time.time() - t0
            logger.info(f"  ✅ {table_name}: {row_count:,} 行，耗时 {elapsed:.1f}s")

        # 显示所有表
        tables = con.execute("SHOW TABLES").fetchall()
        logger.info(f"\n数据库中共 {len(tables)} 张表:")
        for (t,) in tables:
            cnt = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            cols = len(con.execute(f'DESCRIBE "{t}"').fetchall())
            logger.info(f"  - {t}: {cnt:,} 行, {cols} 列")

    finally:
        con.close()

    logger.info(f"\n✅ DuckDB 数据库构建完成: {DB_PATH}")
    logger.info(f"   使用方法: python migrate_to_duckdb.py --query")


def build_duckdb_from_parquet():
    """
    从已生成的Parquet文件构建DuckDB（更快，Parquet已存在时推荐）
    """
    parquet_files = list(PARQUET_DIR.glob('*.parquet'))
    if not parquet_files:
        logger.warning(f"⚠️  未找到Parquet文件: {PARQUET_DIR}")
        return

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"\n从Parquet构建 DuckDB: {DB_PATH}")
    con = duckdb.connect(str(DB_PATH))

    try:
        for pq_file in parquet_files:
            t0 = time.time()
            table_name = make_table_name(pq_file.stem)
            pq_path_str = str(pq_file).replace('\\', '/')

            logger.info(f"  导入: {pq_file.name} -> 表 [{table_name}]")
            con.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            con.execute(
                f'CREATE TABLE "{table_name}" AS '
                f"SELECT * FROM read_parquet('{pq_path_str}')"
            )

            row_count = con.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
            elapsed = time.time() - t0
            logger.info(f"  ✅ {table_name}: {row_count:,} 行，耗时 {elapsed:.1f}s")

        tables = con.execute("SHOW TABLES").fetchall()
        logger.info(f"\n数据库中共 {len(tables)} 张表:")
        for (t,) in tables:
            cnt = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            cols = len(con.execute(f'DESCRIBE "{t}"').fetchall())
            logger.info(f"  - {t}: {cnt:,} 行, {cols} 列")
    finally:
        con.close()

    logger.info(f"\n✅ DuckDB 数据库构建完成: {DB_PATH}")


def query_examples():
    """展示DuckDB查询示例"""
    if not DB_PATH.exists():
        logger.error(f"❌ 数据库不存在: {DB_PATH}")
        logger.info("请先运行: python migrate_to_duckdb.py")
        return

    logger.info(f"\n连接数据库: {DB_PATH}")
    con = duckdb.connect(str(DB_PATH), read_only=True)

    try:
        tables = con.execute("SHOW TABLES").fetchall()
        if not tables:
            logger.warning("数据库中没有表")
            return

        logger.info(f"\n数据库中共 {len(tables)} 张表:")
        for (t,) in tables:
            cnt = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            logger.info(f"  - {t}: {cnt:,} 行")

        # 使用第一张表做示例
        table = tables[0][0]
        logger.info(f"\n以表 [{table}] 为例，展示查询方式：")

        cols_info = con.execute(f'DESCRIBE "{table}"').fetchall()
        col_names = [c[0] for c in cols_info]
        logger.info(f"\n字段列表 ({len(col_names)} 个):")
        for c in col_names:
            logger.info(f"  - {c}")

        logger.info(f"\n示例1：查看前5行")
        df = con.execute(f'SELECT * FROM "{table}" LIMIT 5').df()
        logger.info(f"\n{df.to_string()}")

        if '薪资水平' in col_names:
            logger.info(f"\n示例2：薪资水平分布（前10）")
            df2 = con.execute(
                f'SELECT 薪资水平, COUNT(*) AS 数量 FROM "{table}" '
                f'GROUP BY 薪资水平 ORDER BY 数量 DESC LIMIT 10'
            ).df()
            logger.info(f"\n{df2.to_string()}")

        if '工作城市' in col_names:
            logger.info(f"\n示例3：城市分布（前10）")
            df3 = con.execute(
                f'SELECT 工作城市, COUNT(*) AS 数量 FROM "{table}" '
                f'GROUP BY 工作城市 ORDER BY 数量 DESC LIMIT 10'
            ).df()
            logger.info(f"\n{df3.to_string()}")

        if '学历要求' in col_names:
            logger.info(f"\n示例4：学历要求分布")
            df4 = con.execute(
                f'SELECT 学历要求, COUNT(*) AS 数量 FROM "{table}" '
                f'GROUP BY 学历要求 ORDER BY 数量 DESC'
            ).df()
            logger.info(f"\n{df4.to_string()}")

    finally:
        con.close()


def get_connection(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """
    获取DuckDB连接（供其他模块调用）

    Args:
        read_only: 是否只读模式

    Returns:
        DuckDB连接对象

    Example:
        con = get_connection()
        df = con.execute('SELECT * FROM "qcwy_202201_202506" LIMIT 100').df()
        con.close()
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"数据库不存在: {DB_PATH}\n"
            f"请先运行: python migrate_to_duckdb.py"
        )
    return duckdb.connect(str(DB_PATH), read_only=read_only)


def main():
    parser = argparse.ArgumentParser(description='CSV -> DuckDB/Parquet 数据迁移工具')
    parser.add_argument('--source', type=str, default='data',
                        help='CSV源目录（默认：data/）')
    parser.add_argument('--pattern', type=str, default='*.csv',
                        help='文件匹配模式（默认：*.csv）')
    parser.add_argument('--parquet-only', action='store_true',
                        help='仅转换为Parquet，不导入DuckDB')
    parser.add_argument('--from-parquet', action='store_true',
                        help='从已有Parquet文件构建DuckDB（跳过CSV转换）')
    parser.add_argument('--query', action='store_true',
                        help='展示查询示例')
    args = parser.parse_args()

    if args.query:
        query_examples()
        return

    if args.from_parquet:
        logger.info("从已有Parquet文件构建DuckDB...")
        build_duckdb_from_parquet()
        return

    logger.info("=" * 70)
    logger.info("数据迁移工具：CSV -> DuckDB + Parquet")
    logger.info("=" * 70)
    logger.info(f"源目录: {BASE_DIR / args.source}")
    logger.info(f"DuckDB 路径: {DB_PATH}")
    logger.info(f"Parquet 目录: {PARQUET_DIR}")

    logger.info("\n【步骤1】转换为 Parquet 格式...")
    csv_to_parquet(args.source, args.pattern)

    if not args.parquet_only:
        logger.info("\n【步骤2】从Parquet构建 DuckDB 数据库...")
        build_duckdb_from_parquet()

    logger.info("\n" + "=" * 70)
    logger.info("✅ 迁移完成！")
    logger.info("=" * 70)
    logger.info("\n后续使用方式：")
    logger.info("  1. 查询示例：")
    logger.info("     python migrate_to_duckdb.py --query")
    logger.info("  2. 在代码中使用：")
    logger.info("     from migrate_to_duckdb import get_connection")
    logger.info("     con = get_connection()")
    logger.info('     df = con.execute(\'SELECT * FROM "表名" LIMIT 100\').df()')
    logger.info("     con.close()")
    logger.info("  3. 直接查询Parquet（无需DuckDB）：")
    logger.info("     df = pd.read_parquet('output/parquet/文件名.parquet')")


if __name__ == '__main__':
    main()
