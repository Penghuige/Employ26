"""评估模块。"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import ast
import json
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from tqdm.auto import tqdm


def _parse_candidates(value: Any) -> List[dict]:
    """兼容 DataFrame 中 list 以及 CSV 读回后的字符串。"""
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    for loader in (json.loads, ast.literal_eval):
        try:
            obj = loader(text)
            return obj if isinstance(obj, list) else []
        except Exception:
            continue
    return []


def _evaluate_rows(rows: List[Tuple[str, str, Any]]) -> Tuple[int, int, int, int, int]:
    """评估一个分片，返回统计计数。"""
    total = len(rows)
    top1_hit = 0
    top3_hit = 0
    top5_hit = 0
    unmatched = 0

    for gold_code, top1_code, candidates_raw in rows:
        gold_code = str(gold_code or "").strip()
        top1_code = str(top1_code or "").strip()
        candidates = _parse_candidates(candidates_raw)
        candidate_codes = [str(item.get("code", "")) for item in candidates]

        if not candidates:
            unmatched += 1
        if gold_code and top1_code == gold_code:
            top1_hit += 1
        if gold_code and gold_code in candidate_codes[:3]:
            top3_hit += 1
        if gold_code and gold_code in candidate_codes[:5]:
            top5_hit += 1

    return total, top1_hit, top3_hit, top5_hit, unmatched


def evaluate_matches(result_df: pd.DataFrame, show_progress: bool = False) -> Dict[str, float]:
    """计算 top1 accuracy / top3 recall / top5 recall / unmatched rate。"""
    if result_df.empty:
        return {
            "top1_accuracy": 0.0,
            "top3_recall": 0.0,
            "top5_recall": 0.0,
            "unmatched_rate": 0.0,
            "sample_size": 0,
        }

    iterator = result_df.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=len(result_df), desc="Evaluating", unit="row")

    rows = [
        (
            row.get("gold_code", ""),
            row.get("top1_code", ""),
            row.get("candidates", []),
        )
        for _, row in iterator
    ]

    total, top1_hit, top3_hit, top5_hit, unmatched = _evaluate_rows(rows)
    return {
        "top1_accuracy": top1_hit / total,
        "top3_recall": top3_hit / total,
        "top5_recall": top5_hit / total,
        "unmatched_rate": unmatched / total,
        "sample_size": total,
    }


def evaluate_matches_parallel(
    result_df: pd.DataFrame,
    workers: int = 1,
    chunk_size: int = 20000,
    show_progress: bool = False,
) -> Dict[str, float]:
    """并行评估，适合超大样本。"""
    if workers <= 1 or len(result_df) <= chunk_size:
        return evaluate_matches(result_df, show_progress=show_progress)

    rows = [
        (
            row.get("gold_code", ""),
            row.get("top1_code", ""),
            row.get("candidates", []),
        )
        for _, row in result_df.iterrows()
    ]

    chunks: List[List[Tuple[str, str, Any]]] = [
        rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)
    ]

    totals = [0, 0, 0, 0, 0]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        mapped = ex.map(_evaluate_rows, chunks)
        if show_progress:
            mapped = tqdm(mapped, total=len(chunks), desc="Evaluating chunks", unit="chunk")
        for part in mapped:
            for i, val in enumerate(part):
                totals[i] += val

    total, top1_hit, top3_hit, top5_hit, unmatched = totals
    if total == 0:
        return {
            "top1_accuracy": 0.0,
            "top3_recall": 0.0,
            "top5_recall": 0.0,
            "unmatched_rate": 0.0,
            "sample_size": 0,
        }
    return {
        "top1_accuracy": top1_hit / total,
        "top3_recall": top3_hit / total,
        "top5_recall": top5_hit / total,
        "unmatched_rate": unmatched / total,
        "sample_size": total,
    }
