"""从新 Label Studio 标注数据中提取训练特征。

数据源: data/project-4-at-2026-05-27-01-51-7cceb9ba.json (18611条)
目标: DuckDB 表 recruit.main.match_training_features

特征:
- 每条记录 5 个候选 × 4 个特征 = 20 维
- BGE 余弦相似度 (job_text vs candidate_text)
- BM25 召回排名 (1-5)
- 候选来源编码 (tier2_prefill_topN / 岗位职责 / ...)
- 候选层级大类型编码 (从职业代码第一段提取)

用法:
    python scripts/prepare_training_data.py
    python scripts/prepare_training_data.py --limit 500  # 调试
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(dotenv_path=str(PROJECT_ROOT / ".env.local"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("prepare_training")

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
INPUT_JSON = str(PROJECT_ROOT / "data" / "project-4-at-2026-05-27-01-51-7cceb9ba.json")
DUCKDB_PATH = str(PROJECT_ROOT / "output" / "recruit.duckdb")
OUTPUT_TABLE = "recruit.main.match_training_features"
EMBEDDING_MODEL_PATH = r"D:\model\bge-large-zh-v1.5"
BATCH_SIZE = 128

# 候选来源 → rank 映射
SOURCE_RANK_MAP = {
    "tier2_prefill_top1": 1,
    "tier2_prefill_top2": 2,
    "tier2_prefill_top3": 3,
    "tier2_prefill_top4": 4,
    "tier2_prefill_top5": 5,
    "tier3_prefill_top1": 1,
    "tier3_prefill_top2": 2,
    "tier3_prefill_top3": 3,
    "tier3_prefill_top4": 4,
    "tier3_prefill_top5": 5,
}


def load_data(json_path: str, limit: int = 0) -> List[Dict]:
    """加载 Label Studio 导出 JSON，返回标准化任务列表。"""
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    tasks = []
    for t in raw:
        tid = t["id"]
        td = t.get("data", {})

        job_title = td.get("job_title", "") or td.get("job_title_clean", "") or ""
        job_req = td.get("job_requirements_clean", "") or td.get("job_requirements", "") or ""

        # 提取 5 个候选
        candidates = []
        for letter in ["a", "b", "c", "d", "e"]:
            candidates.append({
                "letter": letter.upper(),
                "title": td.get(f"candidate_{letter}_title", "") or "",
                "desc": td.get(f"candidate_{letter}_desc", "") or "",
                "code": td.get(f"candidate_{letter}_code", "") or "",
                "source": td.get(f"candidate_{letter}_source", "") or "",
            })

        # 提取标注选择
        label = None
        for ann in t.get("annotations", []):
            for r in ann.get("result", []):
                choices = r.get("value", {}).get("choices", [])
                for c in choices:
                    c = c.strip()
                    if "都不属于" in c:
                        label = "NONE"
                    else:
                        c = c.replace("候选", "").strip()
                        if c in "ABCDE":
                            label = c
        if label is None:
            continue

        tasks.append({
            "task_id": tid,
            "job_title": job_title,
            "job_requirements": job_req,
            "candidates": candidates,
            "label": label,
            "sample_source": td.get("sample_source", ""),
            "is_validation": td.get("is_validation_sample", 0),
        })

    if limit > 0:
        tasks = tasks[:limit]

    logger.info("加载 %d 条任务", len(tasks))
    return tasks


def extract_source_features(candidates: List[Dict]) -> Tuple[List[int], List[str], List[int]]:
    """从 candidate_source 提取排名和来源类型。

    Returns:
        (ranks, source_types, category_ids):
        - ranks: [1-5] BM25 召回排名，0=未知
        - source_types: ['tier2_prefill', 'job_desc', ...]
        - category_ids: [1-9] 职业大类代码（从 code 第一段提取）
    """
    ranks = []
    source_types = []
    category_ids = []

    for c in candidates:
        src = c.get("source", "")
        ranks.append(SOURCE_RANK_MAP.get(src, 0))
        source_types.append(src.split("_")[0] if src else "unknown")

        code = c.get("code", "")
        cat_id = 0
        if code and "-" in code:
            try:
                cat_id = int(code.split("-")[0])
            except ValueError:
                pass
        category_ids.append(cat_id)

    return ranks, source_types, category_ids


def build_job_text(job_title: str, job_requirements: str) -> str:
    """构建用于 embedding 的岗位文本。"""
    parts = [job_title]
    if job_requirements:
        # 截断过长的 requirements
        parts.append(str(job_requirements)[:1500])
    return " ".join(p for p in parts if p)


def build_candidate_text(candidate: Dict) -> str:
    """构建用于 embedding 的候选文本。"""
    parts = [candidate.get("title", "")]
    desc = candidate.get("desc", "")
    if desc:
        parts.append(desc[:500])
    return " ".join(p for p in parts if p)


def compute_embeddings(
    model: SentenceTransformer, texts: List[str], desc: str = ""
) -> np.ndarray:
    """批量计算 embedding。"""
    if not desc:
        desc = "encoding"
    embeddings = model.encode(
        texts, normalize_embeddings=True, show_progress_bar=True,
        batch_size=BATCH_SIZE,
    )
    return np.asarray(embeddings, dtype=np.float32)


def compute_similarities(
    job_emb: np.ndarray, cand_embs: np.ndarray
) -> np.ndarray:
    """计算 job embedding 与 5 个 candidate embeddings 的余弦相似度。"""
    # job_emb: (1, dim), cand_embs: (5, dim)
    return np.dot(cand_embs, job_emb.T).flatten()


def main():
    parser = argparse.ArgumentParser(description="准备匹配训练数据")
    parser.add_argument("--limit", type=int, default=0, help="限制条数(0=全部)")
    parser.add_argument("--skip-embeddings", action="store_true", help="跳过BGE编码(调试)")
    args = parser.parse_args()

    # 1. 加载数据
    tasks = load_data(INPUT_JSON, args.limit)

    # 2. 构建文本
    job_texts = [build_job_text(t["job_title"], t["job_requirements"]) for t in tasks]
    cand_texts_flat = []
    for t in tasks:
        for c in t["candidates"]:
            cand_texts_flat.append(build_candidate_text(c))

    # 3. 计算 BGE embeddings
    if not args.skip_embeddings:
        model = SentenceTransformer(EMBEDDING_MODEL_PATH)
        logger.info("BGE 模型已加载: %s", EMBEDDING_MODEL_PATH)

        t0 = time.time()
        job_embs = compute_embeddings(model, job_texts, desc="job texts")
        cand_embs = compute_embeddings(model, cand_texts_flat, desc="candidate texts")
        logger.info("Embedding 完成 (%.1fs), job: %s, cand: %s",
                     time.time() - t0, job_embs.shape, cand_embs.shape)
    else:
        # 调试模式：随机生成 dummy embeddings
        rng = np.random.RandomState(42)
        job_embs = rng.randn(len(job_texts), 1024).astype(np.float32)
        cand_embs = rng.randn(len(cand_texts_flat), 1024).astype(np.float32)

    # 4. 构建特征表
    rows = []
    for i, t in enumerate(tqdm(tasks, desc="构建特征")):
        job_emb = job_embs[i:i + 1]
        cand_start = i * 5
        cand_emb = cand_embs[cand_start:cand_start + 5]

        # BGE 相似度
        sims = compute_similarities(job_emb, cand_emb)

        # 来源特征
        ranks, source_types, cat_ids = extract_source_features(t["candidates"])

        # 聚合特征（用于 NONE 检测）
        sims_rounded = [round(float(s), 6) for s in sims]
        sim_sorted = sorted(sims_rounded, reverse=True)
        max_sim = sim_sorted[0]
        min_sim = sim_sorted[-1]
        mean_sim = round(float(np.mean(sims_rounded)), 6)
        std_sim = round(float(np.std(sims_rounded)), 6)
        sim_gap = round(max_sim - sim_sorted[1], 6) if len(sim_sorted) > 1 else 0.0
        # top1/2/3 BGE 位置（哪个字母的 sim 最高）
        sim_with_letter = [(sims_rounded[j], chr(65 + j)) for j in range(5)]
        sim_with_letter.sort(key=lambda x: -x[0])
        top1_bge_pos = sim_with_letter[0][1]
        top2_bge_pos = sim_with_letter[1][1] if len(sim_with_letter) > 1 else ""
        top3_bge_pos = sim_with_letter[2][1] if len(sim_with_letter) > 2 else ""
        # 大类多样性（唯一大类的数量，越低越集中 → 越不可能 NONE）
        unique_cats = len(set(cat_ids) - {0}) if cat_ids else 0

        row = {
            "task_id": t["task_id"],
            "job_title": t["job_title"],
            "job_requirements": t["job_requirements"],
            "label": t["label"],
            "sample_source": t["sample_source"],
            "is_validation": int(t.get("is_validation", 0) or 0),
            # 聚合特征
            "agg_max_sim": max_sim,
            "agg_min_sim": min_sim,
            "agg_mean_sim": mean_sim,
            "agg_std_sim": std_sim,
            "agg_sim_gap": sim_gap,
            "agg_unique_cats": unique_cats,
            "agg_top1_bge_pos": top1_bge_pos,
        }

        for j, (letter, cand) in enumerate(
            zip(["A", "B", "C", "D", "E"], t["candidates"])
        ):
            row[f"cand_{letter}_code"] = cand["code"]
            row[f"cand_{letter}_title"] = cand["title"]
            row[f"cand_{letter}_desc"] = cand["desc"]
            row[f"cand_{letter}_source"] = cand["source"]
            row[f"cand_{letter}_bge_sim"] = round(float(sims[j]), 6)
            row[f"cand_{letter}_bm25_rank"] = ranks[j]
            row[f"cand_{letter}_cat_id"] = cat_ids[j]

        rows.append(row)

    df = pd.DataFrame(rows)

    # 5. 保存到 DuckDB
    conn = duckdb.connect(DUCKDB_PATH)
    try:
        conn.execute("CREATE SCHEMA IF NOT EXISTS recruit.main")
        conn.register("df_tmp", df)
        conn.execute(f"CREATE OR REPLACE TABLE {OUTPUT_TABLE} AS SELECT * FROM df_tmp")
        cnt = conn.execute(f"SELECT COUNT(*) FROM {OUTPUT_TABLE}").fetchone()[0]
        logger.info("DuckDB 表已保存: %s (%d 行)", OUTPUT_TABLE, cnt)
    finally:
        conn.close()

    # 6. 统计摘要
    print("\n" + "=" * 60)
    print("特征表构建完成")
    print("=" * 60)
    print(f"  总行数:     {len(df)}")
    print(f"  列数:       {len(df.columns)}")
    print(f"  输出表:     {OUTPUT_TABLE}")

    labels = df["label"].value_counts()
    print(f"\n  Label 分布:")
    for k, v in labels.items():
        print(f"    {k}: {v} ({v * 100 / len(df):.1f}%)")

    # NONE 的 BGE 聚合特征
    none_mask = df["label"] == "NONE"
    pick_mask = df["label"] != "NONE"
    print(f"\n  NONE vs PICK 聚合特征对比:")
    for col in ["agg_max_sim", "agg_mean_sim", "agg_std_sim", "agg_sim_gap", "agg_unique_cats"]:
        print(f"    {col}: NONE={df.loc[none_mask, col].mean():.4f}, "
              f"PICK={df.loc[pick_mask, col].mean():.4f}")

    print(f"\n  平均 BGE 相似度:")
    for letter in "ABCDE":
        col = f"cand_{letter}_bge_sim"
        print(f"    {letter}: all={df[col].mean():.4f}, "
              f"NONE={df.loc[none_mask, col].mean():.4f}, "
              f"PICK={df.loc[pick_mask, col].mean():.4f}")

    # BM25 top1 = 标注结果的比例
    top1_col = "cand_A_bm25_rank"
    top1_correct = (df[top1_col] == 1) & (df["label"] == "A")
    print(f"\n  BM25 top1 = 标注A 的准确率: {top1_correct.sum()}/{len(df)} (BM25 top1在A位)")
    cand_a_is_top1 = (df["cand_A_bm25_rank"] == 1).sum()
    print(f"  BM25 top1 出现在 A 位置: {cand_a_is_top1}/{len(df)}")

    # 各 candidate 位置上 top1 的分布
    for letter in "ABCDE":
        n_top1 = (df[f"cand_{letter}_bm25_rank"] == 1).sum()
        label_at_top1 = df[df[f"cand_{letter}_bm25_rank"] == 1]["label"].value_counts()
        print(f"  {letter}位=BM25 top1: {n_top1} 条, 标注分布: "
              + ", ".join(f"{k}={v}" for k, v in label_at_top1.head(3).items()))


if __name__ == "__main__":
    main()
