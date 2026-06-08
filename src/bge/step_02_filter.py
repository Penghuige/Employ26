# =============================================================================
# 模块：D2_filter.py
# 功能：Tier-1 混合极速规则匹配 + RAG 知识库增强质检
#
# 流程：
#   Step 1~4：精确/子串/模糊规则匹配，初步分流
#   Step 5：在进入 RAG 前统一执行岗位描述切分预处理
#   Step 6：根据模式执行 legacy / parsed_adaptive_task / compare 检索与质检
#           is_correct=1 -> 写入 Tier1_Matched_Data（最终高置信）
#           is_correct=0 -> 降级至 Tier2_Pending_Data
# =============================================================================

import os
import re
import sys
import time
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from tqdm import tqdm
from tqdm.auto import tqdm as tqdm_auto

# 运行方式：从项目根目录执行 `python -m src.bge.step_02_filter`，
# 确保 src.* 包可通过标准 Python 模块搜索路径正确导入。
from ..preprocessing.parse_desc import parse_desc_df
from ..rag.config import RAGConfig
from ..rag.qc_utils import (
    batched_generate,
    build_qc_prompt,
    build_rag_context,
    extract_json,
    load_qwen_model,
    load_retriever,
    load_task_chunk_retriever,
    normalize_label,
    safe_str,
)
from ..rag.retriever import OccupationRetriever

# =============================================================================
# 配置区域
# =============================================================================

# ---- 1. 输入输出路径 ----
INPUT_DATA_FILE = r"src\bge\data5\Step1_Data_Deduplicated_Strict.csv"
STANDARD_DICT_FILE = r"data\中国职业大典.xlsx"
OUTPUT_MATCHED_FILE = r"src\bge\data5\Tier1_Matched_Data.csv"
OUTPUT_PENDING_FILE = r"src\bge\data5\Tier2_Pending_Data.csv"

# ---- 2. 规则层参数 ----
MATCH_THRESHOLD = 90
SENSITIVE_WORDS = ["不招", "助理", "实习生", "学徒", "兼职", "储备", "外包", "小白", "不用"]

# ---- 3. 岗位描述预处理参数 ----
PARSE_BATCH_SIZE = 2000
PARSE_NUM_WORKERS = max(1, min(8, max(1, (os.cpu_count() or 1) - 1)))

# ---- 4. RAG 检索参数 ----
RAG_KB_EXCEL = r"data\中国职业大典.xlsx"
RAG_INDEX_PATH = r"src\rag\artifacts\occupation_index.faiss"
RAG_TASK_INDEX_PATH = r"src\rag\artifacts\occupation_task_index.faiss"
RAG_METADATA_PATH = r"src\rag\artifacts\occupation_metadata.json"
from config.paths import get_project_paths
_paths = get_project_paths()
RAG_EMBEDDING_MODEL = str(_paths.bge_model_path)
RAG_TOP_K = 3
RAG_QUERY_BATCH_SIZE = 256
RAG_MATCH_MODE = "compare"  # legacy | parsed_adaptive_task | compare
RAG_SCORE_TIE_DELTA = 0.005

# ---- 5. Qwen3 质检参数（RTX 4090 24G）----
QWEN_MODEL_PATH = str(_paths.qwen_model_path)
QWEN_DTYPE = "bfloat16"
QWEN_DEVICE_MAP = "auto"
INFER_BATCH_SIZE = 16
MAX_NEW_TOKENS = 128
DO_SAMPLE = False

LEGACY_QUERY_SOURCE = "legacy_title_desc_200"
PARSED_SOURCE_PRIORITY = {
    "岗位职责": 0,
    "任职要求": 1,
    "岗位描述_清洗": 2,
}


# =============================================================================
# 规则层工具函数（原 D2 逻辑，保持不变）
# =============================================================================

def clean_for_exact_match(text: str) -> str:
    """去除非中文/字母数字字符，构建 O(1) 哈希精确匹配。"""
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^\w\u4e00-\u9fa5]", "", text)


def load_official_dictionary() -> Tuple[Dict, Dict]:
    """加载职业大典，构建常规映射表与精确匹配哈希表。"""
    print(f">>> 正在加载大典官方标准库: {STANDARD_DICT_FILE}")
    try:
        df_dict = pd.read_excel(STANDARD_DICT_FILE, engine="openpyxl")
    except Exception:
        df_dict = pd.read_csv(STANDARD_DICT_FILE)

    df_dict.fillna("", inplace=True)
    title_col = next((col for col in ["title", "职业名称", "name"] if col in df_dict.columns), None)
    code_col = next((col for col in ["code", "职业代码", "id"] if col in df_dict.columns), None)

    official_map: Dict = {}
    official_map_clean: Dict = {}

    for _, row in df_dict.iterrows():
        raw_title = str(row.get(title_col, "")).strip()
        code = str(row.get(code_col, "")).strip()
        title_base = re.split(r"[（(]", raw_title)[0].strip()
        if title_base and code:
            official_map[title_base] = {"code": code, "raw_title": raw_title}
            clean_base = clean_for_exact_match(title_base)
            if clean_base:
                official_map_clean[clean_base] = {"code": code, "raw_title": raw_title}

    print(f"成功加载并提取了 {len(official_map)} 个大典主词干。")
    return official_map, official_map_clean


def _build_rag_cfg() -> RAGConfig:
    """根据本文件配置区变量构建 RAGConfig。"""
    return RAGConfig(
        embedding_model_path=RAG_EMBEDDING_MODEL,
        generator_model_path=QWEN_MODEL_PATH,
        kb_excel_path=RAG_KB_EXCEL,
        index_path=RAG_INDEX_PATH,
        task_index_path=RAG_TASK_INDEX_PATH,
        metadata_path=RAG_METADATA_PATH,
        embedding_batch_size=128,
        top_k=RAG_TOP_K,
        task_chunk_mode="merged",
    )


# =============================================================================
# 检索层公共工具
# =============================================================================

def _encode_queries(retriever: OccupationRetriever, queries: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """批量编码并检索 query。"""
    if not queries:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.int64)

    query_vectors = retriever.embedding_model.encode(
        queries,
        batch_size=RAG_QUERY_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    query_vectors = np.asarray(query_vectors, dtype=np.float32)
    return retriever.index.search(query_vectors, RAG_TOP_K)


def _build_candidate_from_record(rank: int, score: float, record: Dict) -> Dict:
    """把 record 转成 build_rag_context 所需结构。"""
    return {
        "rank": rank,
        "score": float(score),
        "code": record["code"],
        "title": record["title"],
        "title_main": record.get("title_main", record["title"]),
        "sub_titles": record.get("sub_titles", []),
        "desc": record.get("desc", ""),
        "tasks": record.get("tasks", ""),
        "task_items": record.get("task_items", []),
    }


def _prepare_legacy_candidates(
    df_matched: pd.DataFrame,
    retriever: OccupationRetriever,
    records: List[Dict],
) -> List[Dict]:
    """构建旧版 record 检索候选。"""
    queries = [
        f"{safe_str(title)} {safe_str(desc)[:200]}".strip()
        for title, desc in zip(df_matched["岗位名称"], df_matched["岗位描述"])
    ]
    scores_matrix, indices_matrix = _encode_queries(retriever, queries)

    selected_infos: List[Dict] = []
    for row_idx in range(len(df_matched)):
        candidates = []
        for rank, (score, rec_idx) in enumerate(zip(scores_matrix[row_idx], indices_matrix[row_idx]), start=1):
            if rec_idx < 0 or rec_idx >= len(records):
                continue
            candidates.append(_build_candidate_from_record(rank, float(score), records[rec_idx]))

        selected_infos.append(
            {
                "query_text": queries[row_idx],
                "source": LEGACY_QUERY_SOURCE,
                "top1_score": float(candidates[0]["score"]) if candidates else -1.0,
                "candidates": candidates,
            }
        )
    return selected_infos


def _aggregate_task_candidates(
    scores_row: np.ndarray,
    indices_row: np.ndarray,
    task_chunks: List[Dict],
    record_map: Dict[str, Dict],
) -> List[Dict]:
    """把 chunk 命中聚合回职业记录级候选。"""
    best_hits_by_doc: Dict[str, Dict] = {}

    for score, chunk_idx in zip(scores_row, indices_row):
        if chunk_idx < 0 or chunk_idx >= len(task_chunks):
            continue
        chunk = task_chunks[chunk_idx]
        doc_id = chunk["doc_id"]
        record = record_map.get(doc_id)
        if not record:
            continue

        prev = best_hits_by_doc.get(doc_id)
        if prev is None or float(score) > prev["score"]:
            best_hits_by_doc[doc_id] = {
                "score": float(score),
                "record": record,
            }

    sorted_hits = sorted(best_hits_by_doc.values(), key=lambda item: item["score"], reverse=True)[:RAG_TOP_K]
    return [
        _build_candidate_from_record(rank, hit["score"], hit["record"])
        for rank, hit in enumerate(sorted_hits, start=1)
    ]


def _prepare_parsed_queries(df_matched: pd.DataFrame) -> List[List[Dict]]:
    """为每一行构造自适应 query 候选。"""
    all_queries: List[List[Dict]] = []

    for _, row in df_matched.iterrows():
        job_title = safe_str(row.get("岗位名称", ""))
        requirement_text = safe_str(row.get("任职要求_items_text", ""))
        duty_text = safe_str(row.get("岗位职责_items_text", ""))
        cleaned_desc = safe_str(row.get("岗位描述_清洗", ""))

        row_queries: List[Dict] = []
        if requirement_text:
            row_queries.append(
                {
                    "source": "任职要求",
                    "query_text": f"{job_title} {requirement_text}".strip(),
                }
            )
        if duty_text:
            row_queries.append(
                {
                    "source": "岗位职责",
                    "query_text": f"{job_title} {duty_text}".strip(),
                }
            )

        row_queries.append(
            {
                "source": "岗位描述_清洗",
                "query_text": f"{job_title} {cleaned_desc}".strip() or job_title,
            }
        )
        all_queries.append(row_queries)

    return all_queries


def _choose_best_parsed_query(row_query_results: List[Dict]) -> Dict:
    """按照 top1 分数自适应选择最佳 query。"""
    max_score = max(item["top1_score"] for item in row_query_results)
    near_best = [
        item for item in row_query_results
        if max_score - item["top1_score"] <= RAG_SCORE_TIE_DELTA
    ]
    return min(near_best, key=lambda item: PARSED_SOURCE_PRIORITY.get(item["source"], 99))


def _prepare_parsed_task_candidates(
    df_matched: pd.DataFrame,
    retriever: OccupationRetriever,
    records: List[Dict],
    task_chunks: List[Dict],
) -> List[Dict]:
    """构建新方案 task-chunk 检索候选，并按分数自适应选择 query。"""
    record_map = {record["doc_id"]: record for record in records}
    row_queries = _prepare_parsed_queries(df_matched)

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

    scores_matrix, indices_matrix = _encode_queries(
        retriever,
        [item["query_text"] for item in flat_queries],
    )

    row_query_results: Dict[int, List[Dict]] = {row_pos: [] for row_pos in range(len(df_matched))}
    for query_pos, query_meta in enumerate(flat_queries):
        candidates = _aggregate_task_candidates(
            scores_row=scores_matrix[query_pos],
            indices_row=indices_matrix[query_pos],
            task_chunks=task_chunks,
            record_map=record_map,
        )
        row_query_results[query_meta["row_pos"]].append(
            {
                "source": query_meta["source"],
                "query_text": query_meta["query_text"],
                "top1_score": float(candidates[0]["score"]) if candidates else -1.0,
                "candidates": candidates,
            }
        )

    selected_infos = []
    for row_pos in range(len(df_matched)):
        selected_infos.append(_choose_best_parsed_query(row_query_results[row_pos]))
    return selected_infos


# =============================================================================
# Step 5：RAG + Qwen3 二次质检
# =============================================================================

def _run_quality_check_with_selected_candidates(
    df_matched: pd.DataFrame,
    selected_infos: List[Dict],
    tokenizer,
    model,
    prefix: str,
) -> pd.DataFrame:
    """针对指定候选结果运行一轮质检，并写入带前缀的结果列。"""
    job_titles = df_matched["岗位名称"].fillna("").astype(str).str.strip().tolist()
    job_descs = (
        df_matched["岗位描述_清洗"].fillna("").astype(str).tolist()
        if "岗位描述_清洗" in df_matched.columns
        else df_matched["岗位描述"].fillna("").astype(str).tolist()
    )
    pred_titles = df_matched["tier1_matched_title"].fillna("").astype(str).tolist()
    pred_codes = df_matched["tier1_matched_code"].fillna("").astype(str).tolist()
    pred_scores = df_matched["tier1_match_score"].fillna("").astype(str).tolist()

    prompts = []
    for row_idx, info in enumerate(selected_infos):
        rag_context = build_rag_context(info["candidates"])
        prompts.append(
            build_qc_prompt(
                job_titles[row_idx],
                job_descs[row_idx],
                pred_titles[row_idx],
                pred_codes[row_idx],
                pred_scores[row_idx],
                rag_context,
            )
        )

    print(f"   开始批量 Qwen3 推理质检（{prefix}）...")
    total_batches = (len(prompts) + INFER_BATCH_SIZE - 1) // INFER_BATCH_SIZE
    qc_results: List[Dict] = []

    for start in tqdm_auto(
        range(0, len(prompts), INFER_BATCH_SIZE),
        total=total_batches,
        desc=f"{prefix}质检进度",
        unit="batch",
    ):
        batch_prompts = prompts[start: start + INFER_BATCH_SIZE]
        batch_pred_titles = pred_titles[start: start + INFER_BATCH_SIZE]
        outputs = batched_generate(
            tokenizer,
            model,
            batch_prompts,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
        )
        for raw_out, pred_t in zip(outputs, batch_pred_titles):
            parsed = extract_json(raw_out)
            qc_results.append(normalize_label(parsed, predicted_title=pred_t))

    prefix_name = f"{prefix}_"
    result_df = df_matched.copy()
    result_df[f"{prefix_name}RAG匹配文本"] = [info["query_text"] for info in selected_infos]
    result_df[f"{prefix_name}RAG匹配来源"] = [info["source"] for info in selected_infos]
    result_df[f"{prefix_name}RAG_top1_score"] = [info["top1_score"] for info in selected_infos]
    result_df[f"{prefix_name}qc_is_correct"] = [item["is_correct"] for item in qc_results]
    result_df[f"{prefix_name}qc_gold_title"] = [item["gold_title"] for item in qc_results]
    result_df[f"{prefix_name}qc_gold_code"] = [item["gold_code"] for item in qc_results]
    result_df[f"{prefix_name}qc_error_type"] = [item["error_type"] for item in qc_results]
    result_df[f"{prefix_name}qc_error_note"] = [item["error_note"] for item in qc_results]
    return result_df


def _apply_selected_strategy(df_result: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """把带前缀的方案结果映射到统一输出列。"""
    prefix_name = f"{prefix}_"
    output_df = df_result.copy()
    output_df["RAG匹配文本"] = output_df[f"{prefix_name}RAG匹配文本"]
    output_df["RAG匹配来源"] = output_df[f"{prefix_name}RAG匹配来源"]
    output_df["RAG_top1_score"] = output_df[f"{prefix_name}RAG_top1_score"]
    output_df["qc_is_correct"] = output_df[f"{prefix_name}qc_is_correct"]
    output_df["qc_gold_title"] = output_df[f"{prefix_name}qc_gold_title"]
    output_df["qc_gold_code"] = output_df[f"{prefix_name}qc_gold_code"]
    output_df["qc_error_type"] = output_df[f"{prefix_name}qc_error_type"]
    output_df["qc_error_note"] = output_df[f"{prefix_name}qc_error_note"]
    output_df["RAG最终采用方案"] = prefix
    return output_df


def _merge_compare_results(base_df: pd.DataFrame, *result_dfs: pd.DataFrame) -> pd.DataFrame:
    """把多个策略结果按新增列合并到同一 DataFrame。"""
    merged_df = base_df.copy()
    for result_df in result_dfs:
        for col in result_df.columns:
            if col in merged_df.columns:
                continue
            merged_df[col] = result_df[col]
    return merged_df


def _finalize_verified_and_pending(df_result: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """根据统一 qc 列生成最终 verified / downgraded DataFrame。"""
    df_verified = df_result[df_result["qc_is_correct"] == 1].copy()
    df_downgraded = df_result[df_result["qc_is_correct"] == 0].copy()
    df_downgraded = df_downgraded.drop(
        columns=[
            "tier1_matched_title",
            "tier1_matched_code",
            "tier1_match_score",
            "qc_is_correct",
            "qc_gold_title",
            "qc_gold_code",
            "qc_error_type",
            "qc_error_note",
        ],
        errors="ignore",
    )
    return df_verified, df_downgraded


def _summarize_selected_infos(name: str, selected_infos: List[Dict]) -> None:
    """输出检索 query 的基础统计。"""
    valid_scores = [info["top1_score"] for info in selected_infos if info["top1_score"] >= 0]
    avg_score = float(np.mean(valid_scores)) if valid_scores else -1.0
    source_counter = Counter(info["source"] for info in selected_infos)
    print(f"[{name}] 平均 top1 分数: {avg_score:.4f}")
    print(f"[{name}] 来源分布: {dict(source_counter)}")


def run_tier1_quality_check(
    df_matched: pd.DataFrame,
    cfg: RAGConfig,
    tokenizer,
    model,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """对规则层命中的 Tier1 候选执行 legacy / parsed_adaptive_task / compare 质检。"""
    print(f"\n[Step 6] 开始对 {len(df_matched)} 条 Tier1 候选执行 RAG 二次质检，模式: {RAG_MATCH_MODE}")
    if df_matched.empty:
        return df_matched.copy(), df_matched.copy()

    if RAG_MATCH_MODE == "legacy":
        legacy_retriever, records = load_retriever(cfg)
        legacy_selected = _prepare_legacy_candidates(df_matched, legacy_retriever, records)
        _summarize_selected_infos("legacy", legacy_selected)
        legacy_result = _run_quality_check_with_selected_candidates(
            df_matched=df_matched,
            selected_infos=legacy_selected,
            tokenizer=tokenizer,
            model=model,
            prefix="legacy",
        )
        final_result = _apply_selected_strategy(legacy_result, prefix="legacy")
        return _finalize_verified_and_pending(final_result)

    if RAG_MATCH_MODE == "parsed_adaptive_task":
        task_retriever, records, task_chunks = load_task_chunk_retriever(cfg)
        parsed_selected = _prepare_parsed_task_candidates(df_matched, task_retriever, records, task_chunks)
        _summarize_selected_infos("parsed_adaptive_task", parsed_selected)
        parsed_result = _run_quality_check_with_selected_candidates(
            df_matched=df_matched,
            selected_infos=parsed_selected,
            tokenizer=tokenizer,
            model=model,
            prefix="parsed",
        )
        final_result = _apply_selected_strategy(parsed_result, prefix="parsed")
        return _finalize_verified_and_pending(final_result)

    if RAG_MATCH_MODE != "compare":
        raise ValueError(f"不支持的 RAG_MATCH_MODE: {RAG_MATCH_MODE}")

    # compare：同时运行 old/new 两套逻辑，比较质检通过率
    legacy_retriever, legacy_records = load_retriever(cfg)
    task_retriever, task_records, task_chunks = load_task_chunk_retriever(cfg)

    legacy_selected = _prepare_legacy_candidates(df_matched, legacy_retriever, legacy_records)
    parsed_selected = _prepare_parsed_task_candidates(df_matched, task_retriever, task_records, task_chunks)
    _summarize_selected_infos("legacy", legacy_selected)
    _summarize_selected_infos("parsed_adaptive_task", parsed_selected)

    legacy_result = _run_quality_check_with_selected_candidates(
        df_matched=df_matched,
        selected_infos=legacy_selected,
        tokenizer=tokenizer,
        model=model,
        prefix="legacy",
    )
    parsed_result = _run_quality_check_with_selected_candidates(
        df_matched=df_matched,
        selected_infos=parsed_selected,
        tokenizer=tokenizer,
        model=model,
        prefix="parsed",
    )

    legacy_pass = int(legacy_result["legacy_qc_is_correct"].sum())
    parsed_pass = int(parsed_result["parsed_qc_is_correct"].sum())
    disagreement_count = int(
        (legacy_result["legacy_qc_is_correct"].values != parsed_result["parsed_qc_is_correct"].values).sum()
    )

    print("\n[Compare] 质检对比摘要")
    print(f"[Compare] legacy 通过数: {legacy_pass}")
    print(f"[Compare] parsed 通过数: {parsed_pass}")
    print(f"[Compare] 分歧样本数: {disagreement_count}")

    compare_result = _merge_compare_results(df_matched, legacy_result, parsed_result)
    chosen_prefix = "parsed" if parsed_pass >= legacy_pass else "legacy"
    print(f"[Compare] 最终采用方案: {chosen_prefix}")
    final_result = _apply_selected_strategy(compare_result, prefix=chosen_prefix)
    return _finalize_verified_and_pending(final_result)


# =============================================================================
# 主流程
# =============================================================================

def main_step2() -> None:
    """D2 主流程：规则匹配 + 统一描述切分 + RAG 二次质检 + 分流保存。"""
    if not os.path.exists(INPUT_DATA_FILE):
        print(f"找不到输入文件: {INPUT_DATA_FILE}，请先运行 Step 1。")
        return

    print("\n>>> 正在加载去重后的招聘数据...")
    df = pd.read_csv(INPUT_DATA_FILE, encoding="utf-8-sig")
    df["岗位名称"] = df["岗位名称"].fillna("").astype(str).str.strip()
    df["岗位描述"] = df["岗位描述"].fillna("").astype(str)
    print(f"待处理数据总量: {len(df)} 条")

    print("\n>>> 正在执行岗位描述结构化预处理...")
    df = parse_desc_df(
        df,
        desc_col="岗位描述",
        batch_size=PARSE_BATCH_SIZE,
        num_workers=PARSE_NUM_WORKERS,
    )
    print(
        f"预处理完成。解析列已追加，批大小={PARSE_BATCH_SIZE}，"
        f"worker={PARSE_NUM_WORKERS}"
    )

    official_map, official_map_clean = load_official_dictionary()
    official_base_titles = list(official_map.keys())
    sorted_official_titles = sorted(official_base_titles, key=len, reverse=True)

    print("\n>>> 开始执行 Tier-1 混合极速规则层比对...")
    start_time = time.time()
    unique_jobs = df["岗位名称"].unique()
    print(f"实际需要进行比对的独立岗位名称为: {len(unique_jobs)} 个")
    match_results_dict: Dict = {}

    for job in tqdm(unique_jobs, desc="匹配进度"):
        if not job:
            match_results_dict[job] = (None, None, 0)
            continue
        clean_job = clean_for_exact_match(job)

        if clean_job in official_map_clean:
            match_results_dict[job] = (
                official_map_clean[clean_job]["raw_title"],
                official_map_clean[clean_job]["code"],
                100,
            )
            continue

        if not any(sw in job for sw in SENSITIVE_WORDS):
            matched_flag = False
            for off_title in sorted_official_titles:
                if len(off_title) >= 2 and off_title in job:
                    match_results_dict[job] = (
                        official_map[off_title]["raw_title"],
                        official_map[off_title]["code"],
                        95,
                    )
                    matched_flag = True
                    break
            if matched_flag:
                continue

        best_match = process.extractOne(job, official_base_titles, scorer=fuzz.token_set_ratio)
        if best_match:
            matched_base, score = best_match[0], best_match[1]
            match_results_dict[job] = (
                official_map[matched_base]["raw_title"],
                official_map[matched_base]["code"],
                score,
            )
        else:
            match_results_dict[job] = (None, None, 0)

    print("\n>>> 正在将匹配结果合并至全量数据集...")
    df["tier1_matched_title"] = df["岗位名称"].map(lambda x: match_results_dict[x][0])
    df["tier1_matched_code"] = df["岗位名称"].map(lambda x: match_results_dict[x][1])
    df["tier1_match_score"] = df["岗位名称"].map(lambda x: match_results_dict[x][2])

    print("\n>>> 正在执行规则层阈值截断与初步分流...")
    df_rule_matched = df[df["tier1_match_score"] >= MATCH_THRESHOLD].copy()
    df_rule_pending = df[df["tier1_match_score"] < MATCH_THRESHOLD].copy()
    df_rule_pending = df_rule_pending.drop(
        columns=["tier1_matched_title", "tier1_matched_code", "tier1_match_score"],
        errors="ignore",
    )

    rule_end_time = time.time()
    print(
        f"规则层统计：总量={len(df)}，命中={len(df_rule_matched)}，"
        f"未命中={len(df_rule_pending)}，耗时={rule_end_time - start_time:.2f}s"
    )

    print("\n>>> 加载 Qwen3 模型...")
    cfg = _build_rag_cfg()
    tokenizer, model = load_qwen_model(QWEN_MODEL_PATH, QWEN_DTYPE, QWEN_DEVICE_MAP)

    df_verified, df_downgraded = run_tier1_quality_check(
        df_matched=df_rule_matched,
        cfg=cfg,
        tokenizer=tokenizer,
        model=model,
    )

    df_final_pending = pd.concat([df_rule_pending, df_downgraded], ignore_index=True)

    os.makedirs(os.path.dirname(OUTPUT_MATCHED_FILE), exist_ok=True)
    df_verified.to_csv(OUTPUT_MATCHED_FILE, index=False, encoding="utf-8-sig")
    df_final_pending.to_csv(OUTPUT_PENDING_FILE, index=False, encoding="utf-8-sig")

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("D2 完整流程统计结果")
    print("=" * 60)
    print(f"总耗时                 : {total_time:.2f} 秒")
    print(f"输入数据总量           : {len(df)} 条")
    print(f"规则层命中             : {len(df_rule_matched)} 条")
    print(f"质检通过 Tier1         : {len(df_verified)} 条 ({len(df_verified) / len(df) * 100:.2f}%)")
    print(f"降级+未命中 Tier2      : {len(df_final_pending)} 条")
    print(f"RAG_MATCH_MODE         : {RAG_MATCH_MODE}")
    print("=" * 60)
    print(f"Tier1 数据 -> {OUTPUT_MATCHED_FILE}")
    print(f"Tier2 数据 -> {OUTPUT_PENDING_FILE}")


if __name__ == "__main__":
    main_step2()
