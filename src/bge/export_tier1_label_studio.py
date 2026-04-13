import argparse
import json
import os
import random
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.rag.config import RAGConfig
from src.rag.qc_utils import load_retriever, load_task_chunk_retriever, safe_str
from src.bge.format_label_studio_requirements import format_job_requirements


TIER2_FILE = r"src\bge\data5\Tier2_Matched_Data.csv"
TIER3_FILE = r"src\bge\data5\Tier3_Pending_Data.csv"
OUTPUT_JSON = r"src\bge\data5\Tier1_Matched_Data.label_studio.json"
OUTPUT_PREVIEW_CSV = r"src\bge\data5\Tier1_Matched_Data.label_studio.preview.csv"
OUTPUT_SHARD_DIR = r"src\bge\data5\Tier1_Matched_Data.label_studio_shards"

RAG_KB_EXCEL = r"data\中国职业大典.xlsx"
RAG_INDEX_PATH = r"src\rag\artifacts\occupation_index.faiss"
RAG_TASK_INDEX_PATH = r"src\rag\artifacts\occupation_task_index.faiss"
RAG_METADATA_PATH = r"src\rag\artifacts\occupation_metadata.json"
RAG_EMBEDDING_MODEL = r"D:\model\bge-base-zh-finetuned"

QUERY_BATCH_SIZE = 256
TASK_RETRIEVE_TOP_K = 20
RECORD_RETRIEVE_TOP_K = 20
CANDIDATE_COUNT = 5
VALIDATION_COUNT = 30
RAG_SCORE_TIE_DELTA = 0.005
SHARD_SIZE = 3000
MAX_DESC_LEN = 180

SOURCE_PRIORITY = {
    "岗位职责": 0,
    "任职要求": 1,
    "岗位描述_清洗": 2,
}

EXCLUDED_CODES = {
    "1-01-00-01",
    "1-01-00-02",
    "1-02-01-00",
    "1-02-02-00",
    "1-02-03-00",
    "1-02-04-00",
    "1-02-05-01",
    "7-01-00-00",
    "7-02-00-00",
    "7-03-00-00",
    "7-04-00-00",
    "1-02-05-02",
    "1-03-00-01",
    "1-03-00-02",
    "1-04-01-01",
    "1-04-01-03",
}


def _load_csv_with_fallback(file_path: str) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb2312", "latin-1"):
        try:
            return pd.read_csv(file_path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"无法解码文件: {file_path}")


def _normalize_code(value) -> str:
    code = safe_str(value)
    if code.startswith("'"):
        code = code[1:]
    return code


def _is_excluded_code(value) -> bool:
    return _normalize_code(value) in EXCLUDED_CODES


def _build_rag_cfg() -> RAGConfig:
    return RAGConfig(
        embedding_model_path=RAG_EMBEDDING_MODEL,
        kb_excel_path=RAG_KB_EXCEL,
        index_path=RAG_INDEX_PATH,
        task_index_path=RAG_TASK_INDEX_PATH,
        metadata_path=RAG_METADATA_PATH,
        top_k=TASK_RETRIEVE_TOP_K,
        task_chunk_mode="merged",
    )


def _pick_clean_requirements(row: pd.Series) -> str:
    for col in ("任职要求_items_text", "岗位职责_items_text", "岗位描述_清洗", "岗位描述"):
        text = safe_str(row.get(col, ""))
        if text:
            return text
    return ""


def _prepare_queries(df: pd.DataFrame) -> List[List[Dict]]:
    row_queries: List[List[Dict]] = []
    for _, row in df.iterrows():
        job_title = safe_str(row.get("岗位名称", ""))
        requirement_text = safe_str(row.get("任职要求_items_text", ""))
        duty_text = safe_str(row.get("岗位职责_items_text", ""))
        cleaned_desc = safe_str(row.get("岗位描述_清洗", "")) or safe_str(row.get("岗位描述", ""))

        queries: List[Dict] = []
        if requirement_text:
            queries.append(
                {
                    "source": "任职要求",
                    "query_text": f"{job_title} {requirement_text}".strip(),
                }
            )
        if duty_text:
            queries.append(
                {
                    "source": "岗位职责",
                    "query_text": f"{job_title} {duty_text}".strip(),
                }
            )
        queries.append(
            {
                "source": "岗位描述_清洗",
                "query_text": f"{job_title} {cleaned_desc}".strip() or job_title,
            }
        )
        row_queries.append(queries)
    return row_queries


def _encode_queries(retriever, queries: List[str], top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    vectors = retriever.embedding_model.encode(
        queries,
        batch_size=QUERY_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    return retriever.index.search(vectors, top_k)


def _build_record_candidate(record: Dict, score: float, source: str) -> Dict:
    return {
        "score": float(score),
        "source": source,
        "code": _normalize_code(record.get("code", "")),
        "title": safe_str(record.get("title_main", "")) or safe_str(record.get("title", "")),
        "desc": safe_str(record.get("desc", "")),
        "tasks": safe_str(record.get("tasks", "")),
        "task_items": record.get("task_items", []),
    }


def _aggregate_task_candidates(
    scores_row: np.ndarray,
    indices_row: np.ndarray,
    task_chunks: List[Dict],
    record_map: Dict[str, Dict],
    top_k: int,
    source: str,
) -> List[Dict]:
    best_hits_by_doc: Dict[str, Dict] = {}

    for score, chunk_idx in zip(scores_row, indices_row):
        if chunk_idx < 0 or chunk_idx >= len(task_chunks):
            continue
        chunk = task_chunks[chunk_idx]
        record = record_map.get(chunk["doc_id"])
        if not record:
            continue

        prev = best_hits_by_doc.get(chunk["doc_id"])
        if prev is None or float(score) > prev["score"]:
            best_hits_by_doc[chunk["doc_id"]] = {
                "score": float(score),
                "record": record,
            }

    hits = sorted(best_hits_by_doc.values(), key=lambda item: item["score"], reverse=True)[:top_k]
    return [_build_record_candidate(hit["record"], hit["score"], source=source) for hit in hits]


def _choose_best_query(query_results: List[Dict]) -> Dict:
    max_score = max(item["top1_score"] for item in query_results)
    near_best = [
        item for item in query_results
        if max_score - item["top1_score"] <= RAG_SCORE_TIE_DELTA
    ]
    return min(near_best, key=lambda item: SOURCE_PRIORITY.get(item["source"], 99))


def _format_candidate_desc(candidate: Dict) -> str:
    desc = safe_str(candidate.get("desc", ""))
    task_items = candidate.get("task_items", []) or []
    tasks = " | ".join(item for item in task_items if safe_str(item)) if task_items else safe_str(candidate.get("tasks", ""))

    # Label Studio 候选区只需要简短 desc，整段 tasks 会让导入包和页面都过重。
    if desc:
        return desc[:MAX_DESC_LEN]
    if tasks:
        return tasks[:MAX_DESC_LEN]
    return ""


def _fallback_record_candidates(retriever, records: List[Dict], query_text: str) -> List[Dict]:
    results = retriever.search(query_text, records, top_k=RECORD_RETRIEVE_TOP_K)
    candidates = []
    for result in results:
        candidates.append(
            {
                "score": float(result.get("score", -1.0)),
                "source": "record_fallback",
                "code": _normalize_code(result.get("code", "")),
                "title": safe_str(result.get("title_main", "")) or safe_str(result.get("title", "")),
                "desc": safe_str(result.get("desc", "")),
                "tasks": safe_str(result.get("tasks", "")),
                "task_items": result.get("task_items", []),
            }
        )
    return candidates


def _build_prefilled_tier2_candidates(row: pd.Series) -> List[Dict]:
    candidates = []
    for rank in range(1, CANDIDATE_COUNT + 1):
        code = _normalize_code(row.get(f"tier2_top{rank}_code", ""))
        title = safe_str(row.get(f"tier2_top{rank}_title", ""))
        desc = safe_str(row.get(f"tier2_top{rank}_desc", ""))
        tasks = safe_str(row.get(f"tier2_top{rank}_tasks", ""))
        score = row.get(f"tier2_top{rank}_score", "")
        if not code or not title:
            continue
        candidates.append(
            {
                "score": float(score) if safe_str(score) else -1.0,
                "source": f"tier2_prefill_top{rank}",
                "code": code,
                "title": title,
                "desc": desc,
                "tasks": tasks,
                "task_items": [],
            }
        )
    return candidates


def _filter_candidates(candidates: List[Dict]) -> List[Dict]:
    filtered: List[Dict] = []
    seen_codes = set()
    for candidate in candidates:
        code = _normalize_code(candidate.get("code", ""))
        if not code or code in seen_codes or code in EXCLUDED_CODES:
            continue
        candidate = dict(candidate)
        candidate["code"] = code
        filtered.append(candidate)
        seen_codes.add(code)
    return filtered


def _shuffle_candidates(candidates: List[Dict], seed_text: str) -> Dict[str, str]:
    shuffled = list(candidates)
    random.Random(seed_text).shuffle(shuffled)

    payload: Dict[str, str] = {}
    slots = ["a", "b", "c", "d", "e"]
    for slot, candidate in zip(slots, shuffled):
        payload[f"candidate_{slot}_title"] = candidate["title"]
        payload[f"candidate_{slot}_desc"] = _format_candidate_desc(candidate)
        payload[f"candidate_{slot}_code"] = candidate["code"]
        payload[f"candidate_{slot}_source"] = candidate["source"]

    for slot in slots[len(shuffled):]:
        payload[f"candidate_{slot}_title"] = ""
        payload[f"candidate_{slot}_desc"] = ""
        payload[f"candidate_{slot}_code"] = ""
        payload[f"candidate_{slot}_source"] = ""
    return payload


def _load_source_data() -> pd.DataFrame:
    tier2_df = _load_csv_with_fallback(TIER2_FILE).copy()
    tier2_df = tier2_df[~tier2_df["tier2_matched_code"].apply(_is_excluded_code)].copy()
    tier2_df = tier2_df.head(VALIDATION_COUNT).copy()
    tier2_df["__sample_source"] = "tier2_validation"

    tier3_df = _load_csv_with_fallback(TIER3_FILE).copy()
    tier3_df = tier3_df[~tier3_df["tier2_matched_code"].apply(_is_excluded_code)].copy()
    tier3_df["__sample_source"] = "tier3_main"

    combined_df = pd.concat([tier2_df, tier3_df], ignore_index=True)
    combined_df = combined_df.reset_index(drop=False).rename(columns={"index": "__row_index"})
    print(f"[Load] Tier2 验证集样本: {len(tier2_df)}")
    print(f"[Load] Tier3 主样本: {len(tier3_df)}")
    return combined_df


def _build_query_results(
    df: pd.DataFrame,
    task_retriever,
    task_chunks: List[Dict],
    record_map: Dict[str, Dict],
) -> Dict[int, List[Dict]]:
    row_queries = _prepare_queries(df)
    flat_queries: List[Dict] = []
    for row_pos, query_items in enumerate(row_queries):
        for query_item in query_items:
            flat_queries.append(
                {
                    "row_pos": row_pos,
                    "source": query_item["source"],
                    "query_text": query_item["query_text"],
                }
            )

    print(f"[RAG] 批量检索 {len(flat_queries)} 条 query...")
    scores_matrix, indices_matrix = _encode_queries(
        task_retriever,
        [item["query_text"] for item in flat_queries],
        top_k=TASK_RETRIEVE_TOP_K,
    )

    row_query_results: Dict[int, List[Dict]] = {row_pos: [] for row_pos in range(len(df))}
    for query_pos, query_meta in enumerate(flat_queries):
        candidates = _aggregate_task_candidates(
            scores_matrix[query_pos],
            indices_matrix[query_pos],
            task_chunks,
            record_map,
            top_k=TASK_RETRIEVE_TOP_K,
            source=query_meta["source"],
        )
        row_query_results[query_meta["row_pos"]].append(
            {
                "source": query_meta["source"],
                "query_text": query_meta["query_text"],
                "top1_score": float(candidates[0]["score"]) if candidates else -1.0,
                "candidates": candidates,
            }
        )

    return row_query_results


def _build_base_candidates(
    row: pd.Series,
) -> List[Dict]:
    sample_source = safe_str(row.get("__sample_source", ""))
    if sample_source in {"tier2_validation", "tier3_main"}:
        return _filter_candidates(_build_prefilled_tier2_candidates(row))
    return []


def _supplement_candidates(
    base_candidates: List[Dict],
    query_results: List[Dict],
    record_retriever,
    record_level_records: List[Dict],
) -> Tuple[List[Dict], Dict]:
    if not query_results:
        raise ValueError("query_results 不能为空")

    chosen_query = _choose_best_query(query_results)
    merged_candidates = list(base_candidates)
    seen_codes = {candidate["code"] for candidate in merged_candidates}

    ordered_query_results = [chosen_query] + [
        item for item in query_results
        if item["query_text"] != chosen_query["query_text"] or item["source"] != chosen_query["source"]
    ]

    for query_result in ordered_query_results:
        for candidate in _filter_candidates(query_result["candidates"]):
            if candidate["code"] in seen_codes:
                continue
            merged_candidates.append(candidate)
            seen_codes.add(candidate["code"])
            if len(merged_candidates) >= CANDIDATE_COUNT:
                return merged_candidates[:CANDIDATE_COUNT], chosen_query

    fallback_candidates = _fallback_record_candidates(
        record_retriever,
        record_level_records,
        chosen_query["query_text"],
    )
    for candidate in _filter_candidates(fallback_candidates):
        if candidate["code"] in seen_codes:
            continue
        merged_candidates.append(candidate)
        seen_codes.add(candidate["code"])
        if len(merged_candidates) >= CANDIDATE_COUNT:
            break

    return merged_candidates[:CANDIDATE_COUNT], chosen_query


def export_label_studio(
    output_json: str = OUTPUT_JSON,
    output_preview_csv: str = OUTPUT_PREVIEW_CSV,
    limit: int = 0,
) -> None:
    df = _load_source_data()
    if limit > 0:
        if limit < VALIDATION_COUNT:
            print(f"[WARN] limit={limit} 小于验证集数量 {VALIDATION_COUNT}，将仍保留前 {limit} 条输出。")
        df = df.head(limit).copy()
        print(f"[Load] 仅处理前 {len(df)} 条样本")

    cfg = _build_rag_cfg()
    print("[RAG] 加载 task-chunk 检索器...")
    task_retriever, records, task_chunks = load_task_chunk_retriever(cfg)
    print("[RAG] 加载 record 检索器（兜底）...")
    record_retriever, record_level_records = load_retriever(cfg)

    record_map = {record["doc_id"]: record for record in records}
    row_query_results = _build_query_results(df, task_retriever, task_chunks, record_map)

    tasks = []
    preview_rows = []

    for row_pos, (_, row) in enumerate(df.iterrows()):
        base_candidates = _build_base_candidates(row)
        if not base_candidates:
            continue

        candidates, chosen_query = _supplement_candidates(
            base_candidates=base_candidates,
            query_results=row_query_results[row_pos],
            record_retriever=record_retriever,
            record_level_records=record_level_records,
        )
        if not candidates:
            continue

        candidates = _filter_candidates(candidates)[:CANDIDATE_COUNT]
        if not candidates:
            continue

        top1_candidate = candidates[0]
        shuffled_payload = _shuffle_candidates(
            candidates,
            seed_text=f"{safe_str(row.get('__row_index', row_pos))}|{safe_str(row.get('岗位名称', ''))}|{safe_str(row.get('__sample_source', ''))}",
        )

        row_id = str(row.get("__row_index", row_pos))
        data = {
            "row_id": row_id,
            "sample_source": safe_str(row.get("__sample_source", "")),
            "job_title": safe_str(row.get("岗位名称", "")),
            "job_requirements_clean": format_job_requirements(_pick_clean_requirements(row)),
            "is_validation_sample": "1" if safe_str(row.get("__sample_source", "")) == "tier2_validation" else "0",
        }
        data.update(shuffled_payload)
        tasks.append({"data": data})

        preview_rows.append(
            {
                "row_id": row_id,
                "sample_source": data["sample_source"],
                "岗位名称": data["job_title"],
                "清洗后的岗位要求": data["job_requirements_clean"],
                "candidate_count": len(candidates),
                "rag_query_source": chosen_query["source"],
                "candidate_a_title": data["candidate_a_title"],
                "candidate_a_code": data["candidate_a_code"],
                "candidate_b_title": data["candidate_b_title"],
                "candidate_b_code": data["candidate_b_code"],
                "candidate_c_title": data["candidate_c_title"],
                "candidate_c_code": data["candidate_c_code"],
                "candidate_d_title": data["candidate_d_title"],
                "candidate_d_code": data["candidate_d_code"],
                "candidate_e_title": data["candidate_e_title"],
                "candidate_e_code": data["candidate_e_code"],
            }
        )

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    pd.DataFrame(preview_rows).to_csv(output_preview_csv, index=False, encoding="utf-8-sig")
    os.makedirs(OUTPUT_SHARD_DIR, exist_ok=True)
    shard_paths: List[str] = []
    for start in range(0, len(tasks), SHARD_SIZE):
        shard_tasks = tasks[start:start + SHARD_SIZE]
        shard_idx = start // SHARD_SIZE + 1
        shard_path = os.path.join(OUTPUT_SHARD_DIR, f"Tier1_Matched_Data.label_studio.part{shard_idx:03d}.json")
        with open(shard_path, "w", encoding="utf-8") as f:
            json.dump(shard_tasks, f, ensure_ascii=False, indent=2)
        shard_paths.append(shard_path)

    manifest_path = os.path.join(OUTPUT_SHARD_DIR, "manifest.txt")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("\n".join(shard_paths))

    print(f"[Done] Label Studio 合并 JSON 已生成: {output_json}")
    print(f"[Done] 预览 CSV 已生成: {output_preview_csv}")
    print(f"[Done] 分片目录已生成: {OUTPUT_SHARD_DIR}")
    print(f"[Done] 分片数量: {len(shard_paths)}，每片最多 {SHARD_SIZE} 条")
    print(f"[Done] 共导出 {len(tasks)} 条任务")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Label Studio 导入 JSON：前30条为 Tier2 验证集，其余来自 Tier3")
    parser.add_argument("--output_json", default=OUTPUT_JSON, help="输出 Label Studio JSON 路径")
    parser.add_argument("--output_preview_csv", default=OUTPUT_PREVIEW_CSV, help="输出预览 CSV 路径")
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 条，0 表示全量")
    args = parser.parse_args()

    export_label_studio(
        output_json=args.output_json,
        output_preview_csv=args.output_preview_csv,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
