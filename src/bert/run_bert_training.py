#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BERT 训练一键入口

用法:
    python src/bert/run_bert_training.py

流程:
    1. 数据准备（JSON -> DuckDB）
    2. BERT微调训练
    3. 评估结果写入 DuckDB
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def step1_prepare_data():
    """步骤1：解析JSON，写入DuckDB"""
    logger.info("=" * 60)
    logger.info("步骤1/2：数据准备")
    logger.info("=" * 60)
    from src.bert.prepare_data import main
    main()
    logger.info("步骤1 完成")


def step2_train():
    """步骤2：BERT微调训练"""
    logger.info("=" * 60)
    logger.info("步骤2/2：BERT微调训练")
    logger.info("=" * 60)
    from src.bert.train_bert import main
    main()
    logger.info("步骤2 完成")


def show_results():
    """展示DuckDB中的训练结果"""
    import duckdb
    db_path = Path(__file__).parent.parent.parent / "output" / "recruit.duckdb"
    con = duckdb.connect(str(db_path))

    logger.info("\n" + "=" * 60)
    logger.info("DuckDB 表结构")
    logger.info("=" * 60)
    tables = con.execute("SHOW TABLES").fetchall()
    for t in tables:
        cnt = con.execute(f"SELECT COUNT(*) FROM {t[0]}").fetchone()[0]
        logger.info(f"  {t[0]:30s}: {cnt:,} 行")

    logger.info("\n训练指标 (最近10条):")
    try:
        df = con.execute(
            "SELECT * FROM train_metrics ORDER BY ts DESC LIMIT 10"
        ).df()
        print(df.to_string(index=False))
    except Exception as e:
        logger.warning(f"读取训练指标失败: {e}")

    logger.info("\n测试集预测示例 (前10条):")
    try:
        df = con.execute(
            "SELECT row_id, true_category, pred_category FROM predictions LIMIT 10"
        ).df()
        print(df.to_string(index=False))
    except Exception as e:
        logger.warning(f"读取预测结果失败: {e}")

    con.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BERT职业类别分类训练")
    parser.add_argument("--skip-prepare", action="store_true",
                        help="跳过数据准备（DuckDB已存在时使用）")
    parser.add_argument("--prepare-only", action="store_true",
                        help="只做数据准备，不训练")
    args = parser.parse_args()

    if not args.skip_prepare:
        step1_prepare_data()

    if not args.prepare_only:
        step2_train()

    show_results()
    logger.info("\n全部完成！")
    logger.info(f"模型位置: output/models/bert_occ_category/")
    logger.info(f"数据库位置: output/recruit.duckdb")
