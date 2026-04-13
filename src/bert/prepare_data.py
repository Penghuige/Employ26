#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据准备：将 ls_jd_tasks.json 解析并写入 DuckDB

功能：
  1. 解析 JSON 标注数据（21547条）
  2. 提取训练所需字段
  3. 对 occ_category 做标签编码
  4. 过滤样本数不足的类别
  5. 划分 train/val/test（8:1:1）
  6. 写入 output/recruit.duckdb
"""

import json
import logging
from pathlib import Path

import duckdb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR  = Path(__file__).parent.parent.parent
JSON_PATH = BASE_DIR / "data" / "ls_jd_tasks.json"
DB_PATH   = BASE_DIR / "output" / "recruit.duckdb"


def parse_json(path):
    """解析 Label Studio JSON，提取训练字段"""
    logger.info("读取 %s ...", path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    logger.info("共 %d 条记录", len(raw))

    rows = []
    for idx, item in enumerate(raw):
        d = item.get("data", {})
        title   = str(d.get("clean_title") or d.get("job_title") or "").strip()
        snippet = str(d.get("jd_snippet") or "").strip()
        text    = (title + "[SEP]" + snippet)[:512]
        rows.append({
            "row_id":       int(d.get("row_id", idx)),
            "job_title":    d.get("job_title", ""),
            "clean_title":  title,
            "jd_snippet":   snippet,
            "text":         text,
            "occ_category": d.get("occ_category", ""),
            "occ_core":     d.get("occ_core", ""),
            "ai_level":     d.get("ai_level", ""),
            "ai_edu":       d.get("ai_edu", ""),
            "ai_exp":       d.get("ai_exp", ""),
            "hard_skills":  d.get("hard_skills", ""),
            "ner_status":   d.get("ner_status", ""),
        })

    df = pd.DataFrame(rows)
    logger.info("字段: %s", df.columns.tolist())
    return df


def prepare_labels(df):
    """过滤空标签和稀少类别，重新编码"""
    # 1. 过滤空标签
    df = df[df["occ_category"].notna() & (df["occ_category"] != "")].copy()
    logger.info("有效标签行数: %d", len(df))

    # 2. 统计各类别数量，过滤样本数 < 2 的（train_test_split 要求每类 >= 2）
    counts = df["occ_category"].value_counts()
    valid_cats = counts[counts >= 2].index
    dropped = counts[counts < 2]
    if len(dropped) > 0:
        logger.warning("过滤稀少类别: %s", dropped.to_dict())
    df = df[df["occ_category"].isin(valid_cats)].copy()

    # 3. 编码
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["occ_category"])
    label_map = {int(i): str(c) for i, c in enumerate(le.classes_)}
    logger.info("职业类别数: %d -> %s", len(label_map), list(label_map.values()))
    return df, le, label_map


def split_data(df):
    """8:1:1 划分。第一次分层，第二次不分层（tmp 里各类只有少量样本）"""
    train, tmp = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    # tmp 中部分类别可能只有1条，不做 stratify
    val, test = train_test_split(tmp, test_size=0.5, random_state=42)
    logger.info("train=%d, val=%d, test=%d", len(train), len(val), len(test))
    return train, val, test


def write_duckdb(db_path, df_all, train, val, test, label_map):
    """将数据写入 DuckDB"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))

    # jd_raw：全量原始数据
    con.execute("DROP TABLE IF EXISTS jd_raw")
    con.execute("CREATE TABLE jd_raw AS SELECT * FROM df_all")
    logger.info("jd_raw: %d 行", con.execute("SELECT COUNT(*) FROM jd_raw").fetchone()[0])

    # train_set
    con.execute("DROP TABLE IF EXISTS train_set")
    con.execute("""
        CREATE TABLE train_set AS
        SELECT row_id, text, occ_category, label, ai_level, ai_edu, ai_exp
        FROM train
    """)
    logger.info("train_set: %d 行", con.execute("SELECT COUNT(*) FROM train_set").fetchone()[0])

    # val_set
    con.execute("DROP TABLE IF EXISTS val_set")
    con.execute("""
        CREATE TABLE val_set AS
        SELECT row_id, text, occ_category, label, ai_level, ai_edu, ai_exp
        FROM val
    """)
    logger.info("val_set: %d 行", con.execute("SELECT COUNT(*) FROM val_set").fetchone()[0])

    # test_set
    con.execute("DROP TABLE IF EXISTS test_set")
    con.execute("""
        CREATE TABLE test_set AS
        SELECT row_id, text, occ_category, label, ai_level, ai_edu, ai_exp
        FROM test
    """)
    logger.info("test_set: %d 行", con.execute("SELECT COUNT(*) FROM test_set").fetchone()[0])

    # label_map
    label_df = pd.DataFrame([{"label_id": k, "occ_category": v} for k, v in label_map.items()])
    con.execute("DROP TABLE IF EXISTS label_map")
    con.execute("CREATE TABLE label_map AS SELECT * FROM label_df")
    logger.info("label_map: %d 个类别", con.execute("SELECT COUNT(*) FROM label_map").fetchone()[0])

    # train_metrics：训练指标（训练后写入）
    con.execute("DROP TABLE IF EXISTS train_metrics")
    con.execute("""
        CREATE TABLE train_metrics (
            epoch       INTEGER,
            split       VARCHAR,
            loss        DOUBLE,
            accuracy    DOUBLE,
            f1_macro    DOUBLE,
            ts          TIMESTAMP DEFAULT current_timestamp
        )
    """)

    # predictions：测试集预测结果（训练后写入）
    con.execute("DROP TABLE IF EXISTS predictions")
    con.execute("""
        CREATE TABLE predictions (
            row_id          INTEGER,
            true_label      INTEGER,
            pred_label      INTEGER,
            true_category   VARCHAR,
            pred_category   VARCHAR,
            confidence      DOUBLE
        )
    """)

    con.close()
    logger.info("DuckDB 写入完成: %s", db_path)


def main():
    df = parse_json(JSON_PATH)
    df, le, label_map = prepare_labels(df)
    train, val, test = split_data(df)
    write_duckdb(DB_PATH, df, train, val, test, label_map)
    logger.info("数据准备完成")


if __name__ == "__main__":
    main()
