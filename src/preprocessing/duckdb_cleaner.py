# -*- coding: utf-8 -*-
"""
DuckDB 招聘数据清洗模块

功能：
    1. 连接 DuckDB 数据库（D:\PythonProjects\Employ26\output\recruit.duckdb）
    2. 读取 main.recruit.{table_name}_raw_data 表
    3. 清洗"岗位描述"列：去除 HTML 标签、多余空白、特殊符号
    4. 保存到 main.recruit.{table_name}_cleaned_data 表

用法：
    python src/llm/duckdb_cleaner.py --table-prefix recruit
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def clean_job_description(text: Optional[str]) -> str:
    """
    清洗岗位描述文本。

    清洗步骤：
        1. 去除 HTML 标签（<p>, <br>, </br>, 等）
        2. 去除 HTML 实体（&nbsp;, &#8203;, 等）
        3. 去除多余空白（连续空格、制表符、换行符）
        4. 去除特殊符号（但保留中英文、数字、常见标点）
        5. 去除首尾空白

    Args:
        text: 原始文本

    Returns:
        清洁后的文本
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. 去除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)

    # 2. 去除 HTML 实体
    text = re.sub(r'&[a-zA-Z0-9#]+;', '', text)
    text = re.sub(r'&#\d+;', '', text)

    # 3. 去除多余空白（连续空格、制表符、换行符）
    # 先将所有空白符（包括换行、制表符）替换为单个空格
    text = re.sub(r'[\s\n\r\t]+', ' ', text)

    # 4. 去除特殊符号（保留中英文、数字、常见标点）
    # 保留：中文字符、英文字母、数字、常见标点（。，；：？！、）、空格
    text = re.sub(
        r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef'  # 中文字符范围
        r'a-zA-Z0-9'                                   # 英文字母和数字
        r'。，；：？！、\-\(\)（）\[\]【】\{\}《》'      # 常见中文标点
        r'.,;:?!\-\(\)\[\]\{\}<>"\'/\s]',             # 常见英文标点
        '',
        text
    )

    # 5. 再次清理多余空白（可能在去除符号后产生）
    text = re.sub(r'\s+', ' ', text)

    # 6. 去除首尾空白
    text = text.strip()

    return text


def get_table_names(conn, schema: str = "main", catalog: str = "recruit") -> list[str]:
    """
    获取数据库中所有以 _raw_data 结尾的表名。

    Args:
        conn    : DuckDB 连接
        schema  : schema 名称
        catalog : catalog 名称

    Returns:
        表名列表（不含 schema/catalog 前缀）
    """
    query = f"""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_catalog = '{catalog}'
          AND table_schema = '{schema}'
          AND table_name LIKE '%_raw_data'
    """
    result = conn.execute(query).fetchall()
    return [row[0] for row in result]


def clean_table(
    db_path: str,
    table_name: str,
    schema: str = "main",
    catalog: str = "recruit",
    description_col: str = "岗位描述",
) -> None:
    """
    清洗单个表的岗位描述列，保存到新表。

    Args:
        db_path          : DuckDB 数据库文件路径
        table_name       : 原始表名（含 _raw_data 后缀）
        schema           : schema 名称
        catalog          : catalog 名称
        description_col  : 岗位描述列名
    """
    try:
        import duckdb
    except ImportError as e:
        raise ImportError("请先安装：pip install duckdb") from e

    conn = duckdb.connect(db_path)

    # 验证表是否存在
    full_table_name = f"{catalog}.{schema}.{table_name}"
    try:
        conn.execute(f"SELECT 1 FROM {full_table_name} LIMIT 1")
    except Exception as e:
        logger.error(f"表不存在或无法访问：{full_table_name}")
        logger.error(f"错误：{e}")
        conn.close()
        return

    # 检查列是否存在
    col_query = f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_catalog = '{catalog}'
          AND table_schema = '{schema}'
          AND table_name = '{table_name}'
    """
    columns = [row[0] for row in conn.execute(col_query).fetchall()]
    if description_col not in columns:
        logger.error(
            f"列 '{description_col}' 不存在。可用列：{columns}"
        )
        conn.close()
        return

    logger.info(f"开始清洗表：{full_table_name}")
    logger.info(f"岗位描述列：{description_col}")

    # 读取原始数据
    logger.info("读取原始数据...")
    df = conn.execute(f"SELECT * FROM {full_table_name}").df()
    logger.info(f"共 {len(df)} 行")

    # 清洗岗位描述列
    logger.info(f"清洗 '{description_col}' 列...")
    df[description_col] = df[description_col].apply(clean_job_description)

    # 生成新表名
    new_table_name = table_name.replace("_raw_data", "_cleaned_data")
    full_new_table_name = f"{catalog}.{schema}.{new_table_name}"

    # 删除旧表（如果存在）
    try:
        conn.execute(f"DROP TABLE IF EXISTS {full_new_table_name}")
        logger.info(f"删除旧表（如果存在）：{full_new_table_name}")
    except Exception as e:
        logger.warning(f"删除旧表失败：{e}")

    # 写入新表
    logger.info(f"写入新表：{full_new_table_name}")
    conn.register("temp_df", df)
    conn.execute(f"CREATE TABLE {full_new_table_name} AS SELECT * FROM temp_df")
    conn.unregister("temp_df")

    # 验证
    count = conn.execute(f"SELECT COUNT(*) FROM {full_new_table_name}").fetchone()[0]
    logger.info(f"✓ 新表创建成功，共 {count} 行")

    # 显示清洗前后对比（前3条）
    logger.info("\n【清洗前后对比（前3条）】")
    df_orig = conn.execute(f"SELECT * FROM {full_table_name} LIMIT 3").df()
    df_clean = conn.execute(f"SELECT * FROM {full_new_table_name} LIMIT 3").df()
    for i in range(min(3, len(df_orig))):
        logger.info(f"\n--- 第 {i+1} 条 ---")
        logger.info(f"清洗前（{len(df_orig[description_col].iloc[i])} 字）：")
        logger.info(f"  {df_orig[description_col].iloc[i][:100]}...")
        logger.info(f"清洗后（{len(df_clean[description_col].iloc[i])} 字）：")
        logger.info(f"  {df_clean[description_col].iloc[i][:100]}...")

    conn.close()
    logger.info(f"\n✓ 表 {table_name} 清洗完成！")


def main():
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="DuckDB 招聘数据清洗")
    parser.add_argument(
        "--db",
        default=r"D:\PythonProjects\Employ26\output\recruit.duckdb",
        help="DuckDB 数据库文件路径",
    )
    parser.add_argument(
        "--schema", default="main", help="schema 名称（默认 main）"
    )
    parser.add_argument(
        "--catalog", default="recruit", help="catalog 名称（默认 recruit）"
    )
    parser.add_argument(
        "--table-prefix",
        default="recruit",
        help="表名前缀（默认 recruit，会查找 recruit_raw_data 等）",
    )
    parser.add_argument(
        "--description-col",
        default="岗位描述",
        help="岗位描述列名（默认 '岗位描述'）",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="清洗所有 _raw_data 表（默认只清洗指定前缀的表）",
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

    conn = duckdb.connect(str(db_path))

    # 获取所有 _raw_data 表
    all_tables = get_table_names(conn, args.schema, args.catalog)
    logger.info(f"找到 {len(all_tables)} 个 _raw_data 表：{all_tables}")

    # 筛选要处理的表
    if args.all:
        tables_to_clean = all_tables
    else:
        tables_to_clean = [t for t in all_tables if t.startswith(args.table_prefix)]
        if not tables_to_clean:
            logger.warning(
                f"未找到前缀为 '{args.table_prefix}' 的表。"
                f"可用表：{all_tables}"
            )
            logger.info("使用 --all 清洗所有表，或指定 --table-prefix")
            conn.close()
            return

    conn.close()

    # 逐表清洗
    for table in tables_to_clean:
        logger.info(f"\n{'='*60}")
        clean_table(
            str(db_path),
            table,
            args.schema,
            args.catalog,
            args.description_col,
        )

    logger.info(f"\n{'='*60}")
    logger.info(f"✓ 全部 {len(tables_to_clean)} 个表清洗完成！")


if __name__ == "__main__":
    main()

