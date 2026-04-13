#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DuckDB 数据查看工具
使用方式：python view_duckdb.py
"""

import duckdb
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).parent / 'output' / 'recruit.duckdb'


def main():
    print(f"连接数据库: {DB_PATH}")
    con = duckdb.connect(str(DB_PATH), read_only=True)

    # 列出所有表
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    print(f"\n数据库中共 {len(tables)} 张表:")
    for t in tables:
        cnt = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        cols = con.execute(f'DESCRIBE "{t}"').fetchall()
        print(f"  [{t}]  {cnt:,} 行 | {len(cols)} 列")

    print("\n" + "="*60)
    print("交互模式 - 输入 SQL 查询，输入 'quit' 退出")
    print("示例:")
    for t in tables:
        print(f'  SELECT * FROM "{t}" LIMIT 10')
        print(f'  SELECT 工作城市, COUNT(*) AS n FROM "{t}" GROUP BY 1 ORDER BY 2 DESC LIMIT 10')
        break
    print("="*60)

    while True:
        try:
            sql = input("\nSQL> ").strip()
            if not sql:
                continue
            if sql.lower() in ('quit', 'exit', 'q'):
                break
            if sql.lower() == 'tables':
                for t in tables:
                    cnt = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                    print(f"  {t}: {cnt:,} 行")
                continue
            if sql.lower().startswith('desc '):
                t = sql[5:].strip().strip('"')
                df = con.execute(f'DESCRIBE "{t}"').df()
                print(df.to_string(index=False))
                continue

            df = con.execute(sql).df()
            print(f"\n结果: {len(df)} 行 x {len(df.columns)} 列")
            pd.set_option('display.max_columns', 20)
            pd.set_option('display.width', 200)
            pd.set_option('display.max_colwidth', 30)
            print(df.to_string(index=False))
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"错误: {e}")

    con.close()
    print("已退出")


if __name__ == '__main__':
    main()
