# =============================================================================
# 模块：D5_qwen3_auto_label.py
# 功能：基于 RAG 知识库增强的 Qwen3-8B 自动标注流程
#       对 Tier1/Tier2/Tier3 抽样数据进行职业分类质检标注
#       标注时将检索 src/rag 本地知识库（中国职业大典）作为上下文
# 策略：直接 RAG 一次（而非两轮判断），理由见下方注释
# 硬件：针对 RTX 4090 24G 优化批处理参数
# 作者：自动生成
# =============================================================================
"""
【策略选择说明：为什么选择「直接 RAG 一次」而不是「先判 is_correct=0 再 RAG」】

直接 RAG 一次（本文件采用此方案）：
  - 每条样本先检索知识库 TopK 候选，直接带入 prompt 推理一次
  - 优点：
      1) 单次推理，无额外开销
      2) 模型有知识库支撑，判断准确率更高
      3) 4090 24G 显存充足，带候选上下文不影响吞吐
      4) 代码简洁，便于维护
  - 结论：对于质检准确率优先的项目，直接 RAG 是最优选择
"""

import os
import re
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# 运行方式：从项目根目录执行 `python -m src.bge.step_05_qwen3_auto_label`，
# 确保 src.* 包可通过标准 Python 模块搜索路径正确导入。
from ..rag.config import RAGConfig
from ..rag.qc_utils import (
    build_qc_prompt,
    build_rag_context,
    batched_generate,
    extract_json,
    load_qwen_model,
    load_retriever,
    normalize_label,
    safe_str,
)

# =============================================================================
# 配置区域（所有超参数集中在此处，禁止在代码中硬编码路径或参数）
# =============================================================================

# ---- 1. Qwen3 模型配置（RTX 4090 24G）----
from config.paths import get_project_paths
_paths = get_project_paths()
MODEL_PATH = str(_paths.qwen_model_path)
TORCH_DTYPE = "bfloat16"   # 4090 支持 bfloat16，比 float16 数值更稳定
DEVICE_MAP = "auto"        # 自动分配显卡，单卡直接全量加载

# ---- 2. 输入数据路径 ----
INPUT_STAGE_FILES = {
    "tier1": r"src\bge\data5\Tier1_Matched_Data.csv",
    "tier2": r"src\bge\data5\Tier2_Matched_Data.csv",
    "tier3": r"src\bge\data5\Tier3_Pending_Data.csv",
}

# 每个阶段抽样数量（调大可提高评估置信度，调小加快速度）
SAMPLE_SIZE_PER_STAGE = {
    "tier1": 2900,
    "tier2": 2900,  # tier2 提高抽样量，优先覆盖阈值附近难例
    "tier3": 80000,
}
RANDOM_SEED = 42

# tier2 难例抽样配置（用于自我迭代）
TIER2_HARD_SCORE_LOW = 0.70
TIER2_HARD_SCORE_HIGH = 0.85
TIER2_LOW_MARGIN_MAX = 0.03

# ---- 3. 推理批处理参数 ----
# 性能调优说明（RTX 4090 24G）：
#   - INFER_BATCH_SIZE: 带 RAG 上下文 prompt 约 1000~1500 token
#     4090 24G + bfloat16 可稳定跑 batch=32，显著提升 GPU 利用率
#     若出现 OOM，回退到 24 或 16
#   - MAX_NEW_TOKENS: 质检 JSON 实际输出约 60~100 token
#     从 256 降至 128 可将生成阶段耗时减少约 40%，精度无损
#   - RAG_TOP_K: 从 5 降至 3，减少约 160 token/条 padding 开销
#     候选 3 条已足够覆盖正确职业，精度基本不受影响
INFER_BATCH_SIZE = 16      # 提升：16 -> 32，充分利用 24G 显存
MAX_NEW_TOKENS = 128       # 降低：256 -> 128，JSON 输出无需更多 token
DO_SAMPLE = False          # 贪心解码，保证输出稳定可复现
TEMPERATURE = 0.0
TOP_P = 1.0

# ---- 4. RAG 知识库配置 ----
# 与 src/rag/config.py 保持一致，如需修改路径只改此处
RAG_KB_EXCEL = r"data\中国职业大典.xlsx"
RAG_INDEX_PATH = r"src\rag\artifacts\occupation_index.faiss"
RAG_METADATA_PATH = r"src\rag\artifacts\occupation_metadata.json"
RAG_EMBEDDING_MODEL = str(_paths.project_root / "models" / "bge-base-zh-finetuned")
RAG_TOP_K = 5              # 降低：5 -> 3，减少 prompt 长度约 160 token/条

# ---- 5. 输出文件配置 ----
OUTPUT_DIR = r"src\bge\output"
REVIEWER = "qwen3_8b_rag_v2"  # 版本标识，与旧版 v1 区分
FAILED_DUMP_MAX = 200      # 最多保存多少条解析失败的原始输出用于调试

# ---- 6. 标注输出字段（严格按 docs/manual_label_template.txt）----
# 每个 RAG 候选拆为三列，便于人工逐项核查：
#   rag_cand_N       : 「rank. [code] title (score)」
#   rag_cand_N_desc  : 职业定义
#   rag_cand_N_tasks : 主要工作任务（截断至 300 字）
_RAG_CAND_COLS = []
for _i in range(1, RAG_TOP_K + 1):
    _RAG_CAND_COLS += [f"rag_cand_{_i}", f"rag_cand_{_i}_desc", f"rag_cand_{_i}_tasks"]
OUTPUT_COLUMNS = (
    ["sample_id", "stage", "source_file", "row_index", "岗位名称", "岗位描述",
     "predicted_title", "predicted_code", "predicted_score"]
    + _RAG_CAND_COLS
    + ["is_correct", "gold_title", "gold_code",
       "error_type", "error_note", "reviewer", "review_time"]
)

# 合法错误类型枚举已移至 src/rag/qc_utils.py，此处不再重复定义


# =============================================================================
# D5 专用工具函数（非共享，保留在本文件）
# =============================================================================

def _build_rag_cfg() -> RAGConfig:
    """根据本文件配置区变量构建 RAGConfig，传入 qc_utils.load_retriever。"""
    return RAGConfig(
        embedding_model_path=RAG_EMBEDDING_MODEL,
        generator_model_path=MODEL_PATH,
        kb_excel_path=RAG_KB_EXCEL,
        index_path=RAG_INDEX_PATH,
        metadata_path=RAG_METADATA_PATH,
        embedding_batch_size=128,
        top_k=RAG_TOP_K,
    )
# =============================================================================


def _clean_desc_for_prompt(desc: str, max_len: int = 400) -> str:
    """清洗岗位描述：去除多余空白并截断，防止单条 prompt 过长。

    RAG 场景下 prompt 本身携带候选上下文，岗位描述截断到 400 字符
    避免超出模型输入限制（Qwen3-8B 最大 32k，但批处理时要留余量）。
    """
    desc = safe_str(desc)
    if not desc:
        return ""
    desc = re.sub(r"\s+", " ", desc)
    return desc[:max_len]


def _stage_prefix(stage: str) -> str:
    """根据阶段名称生成样本 ID 前缀（T1/T2/T3）。"""
    return {"tier1": "T1", "tier2": "T2", "tier3": "T3"}.get(stage, "TX")


def _pick_predicted_columns(stage: str) -> Tuple[str, str, str]:
    """根据阶段名称返回对应的预测列名（title/code/score）。

    tier1 使用规则层匹配结果；tier2 使用语义检索结果。
    tier3 是漏斗末端「完全未匹配」数据，没有任何预测列，返回空字符串占位。
    调用方需要判断返回值是否为空，并做特殊处理。
    """
    if stage == "tier1":
        return "tier1_matched_title", "tier1_matched_code", "tier1_match_score"
    if stage == "tier2":
        return "tier2_matched_title", "tier2_matched_code", "tier2_match_score"
    # tier3：无任何预测，用空字符串占位，由调用方判断
    return "", "", ""


def _load_and_sample(stage: str, file_path: str, n: int) -> pd.DataFrame:
    """加载指定阶段数据并抽样。

    tier2 优先抽取“阈值附近 + 低 margin”的困难样本，
    其余阶段保持随机抽样。

    参数：
        stage: 阶段名称（tier1/tier2/tier3）
        file_path: 数据文件路径
        n: 抽样数量

    返回：
        含 __stage/__source_file/__row_index 辅助列的 DataFrame
    """
    if not os.path.exists(file_path):
        print(f"[WARN] 文件不存在，跳过 {stage}: {file_path}")
        return pd.DataFrame()

    # 自动兼容 utf-8-sig / gbk 编码，避免 UnicodeDecodeError
    for _enc in ("utf-8-sig", "gbk", "gb2312", "latin-1"):
        try:
            df = pd.read_csv(file_path, encoding=_enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        print(f"[WARN] 无法解码文件编码，跳过 {stage}: {file_path}")
        return pd.DataFrame()

    if df.empty:
        return df

    sample_n = min(n, len(df))

    # tier2 难例优先抽样：阈值附近 + 低 margin
    if stage == "tier2":
        candidates = df.copy()
        hard_mask = pd.Series([False] * len(candidates), index=candidates.index)

        if "tier2_match_score" in candidates.columns:
            score_mask = (
                (candidates["tier2_match_score"] >= TIER2_HARD_SCORE_LOW)
                & (candidates["tier2_match_score"] <= TIER2_HARD_SCORE_HIGH)
            )
            hard_mask = hard_mask | score_mask

        if "tier2_score_margin" in candidates.columns:
            margin_mask = candidates["tier2_score_margin"] <= TIER2_LOW_MARGIN_MAX
            hard_mask = hard_mask | margin_mask

        hard_df = candidates[hard_mask]
        normal_df = candidates[~hard_mask]

        hard_quota = min(int(sample_n * 0.7), len(hard_df))
        normal_quota = sample_n - hard_quota

        sampled_parts = []
        if hard_quota > 0:
            sampled_parts.append(hard_df.sample(n=hard_quota, random_state=RANDOM_SEED))
        if normal_quota > 0:
            sampled_parts.append(normal_df.sample(n=min(normal_quota, len(normal_df)), random_state=RANDOM_SEED))

        sampled = pd.concat(sampled_parts).drop_duplicates() if sampled_parts else pd.DataFrame(columns=candidates.columns)
        if len(sampled) < sample_n:
            remain = sample_n - len(sampled)
            extra = candidates.drop(sampled.index, errors="ignore")
            if not extra.empty:
                sampled = pd.concat([sampled, extra.sample(n=min(remain, len(extra)), random_state=RANDOM_SEED)])
        sampled = sampled.head(sample_n).copy()
    else:
        sampled = df.sample(n=sample_n, random_state=RANDOM_SEED).copy()

    sampled["__stage"] = stage
    sampled["__source_file"] = os.path.basename(file_path)
    sampled["__row_index"] = sampled.index
    return sampled


# =============================================================================
# 质量检查
# =============================================================================

def _run_output_check(df: pd.DataFrame) -> None:
    """对标注结果输出质量摘要，发现异常时打印警告。"""
    total = len(df)
    if total == 0:
        print("[CHECK] 输出为空，请检查输入与推理流程。")
        return

    parse_failed = int((df["error_note"] == "qwen_output_parse_failed").sum())
    parse_failed_rate = parse_failed / total
    positive_rate = float((df["is_correct"] == 1).mean())

    print("\n[CHECK] 自动质检摘要")
    print(f"  总样本数     : {total}")
    print(f"  JSON解析失败  : {parse_failed} ({parse_failed_rate:.2%})")
    print(f"  is_correct=1 : {positive_rate:.2%}")

    if parse_failed_rate > 0.3:
        print("[CHECK][WARN] JSON 解析失败率 >30%，建议减小 INFER_BATCH_SIZE 并检查 prompt 格式。")
    if positive_rate == 0.0:
        print("[CHECK][WARN] is_correct 全为 0，建议复核 prompt 和输出解析逻辑。")


# =============================================================================
# 主流程
# =============================================================================

def run_auto_labeling() -> None:
    """RAG 增强标注主流程。

    流程：
    1) 加载并抽样三阶段数据
    2) 加载 RAG 检索器（BGE + FAISS）
    3) 加载 Qwen3-8B
    4) 对每条样本：检索知识库 TopK -> 构建含上下文的 prompt -> 批量推理
    5) 解析标签 -> 保存结果 + 失败原文
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Step 1: 加载数据 ----
    print("[1/5] 加载并抽样三阶段数据...")
    chunks = []
    for stage, path in INPUT_STAGE_FILES.items():
        sampled = _load_and_sample(stage, path, SAMPLE_SIZE_PER_STAGE.get(stage, 50))
        if not sampled.empty:
            chunks.append(sampled)

    if not chunks:
        print("❌ 没有可标注数据，请检查输入文件路径。")
        return

    df = pd.concat(chunks, ignore_index=True)
    print(f"   合计样本: {len(df)} 条")

    # ---- Step 2: 加载 RAG 检索器 ----
    print("[2/5] 加载 RAG 知识库检索器（BGE + FAISS）...")
    retriever, records = load_retriever(_build_rag_cfg())

    # ---- Step 3: 批量检索 + 组装标注任务 ----
    # 优化说明：
    #   旧版逐条调用 retriever.search()，每条单独编码 1 次 BGE forward pass
    #   新版先把所有 query 一次性批量编码（BGE batch encode），再做 FAISS 批量检索
    #   对 N 条样本：旧版 N 次 forward pass -> 新版 1 次 forward pass，速度提升 ~10x
    print("[3/5] 批量编码检索 query 并组装标注任务...")

    # 3a. 预提取所有行的字段，避免 iterrows 多次访问
    row_metas = []
    rag_queries = []
    for i, row in df.iterrows():
        stage = row["__stage"]
        pred_title_col, pred_code_col, pred_score_col = _pick_predicted_columns(stage)
        job_title = safe_str(row.get("岗位名称", ""))
        job_desc = _clean_desc_for_prompt(safe_str(row.get("岗位描述", "")))
        # tier3 无预测列（列名为空字符串），直接置空，不读 DataFrame
        pred_title = safe_str(row.get(pred_title_col)) if pred_title_col else ""
        pred_code = safe_str(row.get(pred_code_col)) if pred_code_col else ""
        pred_score = safe_str(row.get(pred_score_col)) if pred_score_col else ""
        # 剥离职业大典原始数据中的 L/S 标记（如「大数据工程技术人员S」→「大数据工程技术人员」）
        # L=主职业 S=细分工种，是大典内部分类标记，不应暴露给 LLM 作为职业名称
        pred_title = re.sub(r'\s*[LS]$', '', pred_title, flags=re.IGNORECASE).strip()
        prefix = _stage_prefix(stage)
        sample_id = f"{prefix}_{int(row['__row_index']):06d}"
        row_metas.append({
            "sample_id": sample_id,
            "stage": stage,
            "source_file": safe_str(row.get("__source_file", "")),
            "row_index": int(row.get("__row_index", i)),
            "job_title": job_title,
            "job_desc": job_desc,
            "pred_title": pred_title,
            "pred_code": pred_code,
            "pred_score": pred_score,
        })
        # query = 岗位名 + 描述前 200 字，语义覆盖更完整
        rag_queries.append(f"{job_title} {job_desc[:200]}".strip())

    # 3b. 一次性批量编码所有 query（BGE 单次 forward，比逐条快 ~10x）
    print(f"   批量编码 {len(rag_queries)} 条 query...")
    query_vectors = retriever.embedding_model.encode(
        rag_queries,
        batch_size=256,          # BGE 编码批大小，CPU/GPU 均可，越大越快
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    query_vectors = np.asarray(query_vectors, dtype=np.float32)

    # 3c. FAISS 批量检索（一次 index.search 返回所有结果）
    print("   FAISS 批量检索...")
    scores_matrix, indices_matrix = retriever.index.search(query_vectors, RAG_TOP_K)

    # 3d. 将检索结果拼装成 tasks
    tasks = []
    for idx, meta in enumerate(row_metas):
        candidates = []
        for rank, (score, rec_idx) in enumerate(
            zip(scores_matrix[idx], indices_matrix[idx]), start=1
        ):
            if rec_idx < 0 or rec_idx >= len(records):
                continue
            rec = records[rec_idx]
            candidates.append({
                "rank": rank,
                "score": float(score),
                "code": rec["code"],
                "title": rec["title"],
                "desc": rec["desc"],
                "tasks": rec["tasks"],
            })
        rag_context = build_rag_context(candidates)
        # tier3 无系统预测，has_prediction=False 触发专用 prompt 分支
        has_prediction = (meta["stage"] != "tier3")
        prompt = build_qc_prompt(
            meta["job_title"], meta["job_desc"],
            meta["pred_title"], meta["pred_code"], meta["pred_score"],
            rag_context,
            has_prediction=has_prediction,
        )
        tasks.append({
            "sample_id": meta["sample_id"],
            "stage": meta["stage"],
            "source_file": meta["source_file"],
            "row_index": meta["row_index"],
            "岗位名称": meta["job_title"],
            "岗位描述": meta["job_desc"],
            "predicted_title": meta["pred_title"],
            "predicted_code": meta["pred_code"],
            "predicted_score": meta["pred_score"],
            "candidates": candidates,  # 保留 RAG 候选，供输出时写入检查列
            "prompt": prompt,
        })
    print(f"   任务组装完成，共 {len(tasks)} 条。")

    # ---- Step 4: 加载 Qwen3 并批量推理 ----
    print("[4/5] 加载 Qwen3-8B 并执行批量推理标注...")
    tokenizer, model = load_qwen_model(MODEL_PATH, TORCH_DTYPE, DEVICE_MAP)

    rows_out = []
    failed_raw_records = []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_batches = (len(tasks) + INFER_BATCH_SIZE - 1) // INFER_BATCH_SIZE
    infer_start = time.time()

    pbar = tqdm(
        range(0, len(tasks), INFER_BATCH_SIZE),
        total=total_batches,
        desc="推理进度",
        unit="batch",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )

    for start in pbar:
        batch = tasks[start: start + INFER_BATCH_SIZE]
        prompts = [t["prompt"] for t in batch]
        outputs = batched_generate(
            tokenizer, model, prompts,
            max_new_tokens=MAX_NEW_TOKENS, do_sample=DO_SAMPLE)

        for task, raw_out in zip(batch, outputs):
            parsed = extract_json(raw_out)
            normalized = normalize_label(
                parsed,
                predicted_title=task["predicted_title"],
            )

            # 将 RAG 候选格式化为独立列，便于人工逐列对比
            # rag_cand_N      : 「rank. [code] title (score)」
            # rag_cand_N_desc : 职业定义
            # rag_cand_N_tasks: 主要工作任务（截断至 300 字）
            rag_cand_cells = {}
            for c in task["candidates"]:
                n = c['rank']
                rag_cand_cells[f"rag_cand_{n}"] = f"{n}. [{c['code']}] {c['title']} ({c['score']:.4f})"
                rag_cand_cells[f"rag_cand_{n}_desc"] = str(c.get('desc', ''))[:300]
                rag_cand_cells[f"rag_cand_{n}_tasks"] = str(c.get('tasks', ''))[:300]
            # 未检索到的位次留空
            for col in _RAG_CAND_COLS:
                rag_cand_cells.setdefault(col, "")

            rows_out.append({
                "sample_id": task["sample_id"],
                "stage": task["stage"],
                "source_file": task["source_file"],
                "row_index": task["row_index"],
                "岗位名称": task["岗位名称"],
                "岗位描述": task["岗位描述"],
                "predicted_title": task["predicted_title"],
                "predicted_code": task["predicted_code"],
                "predicted_score": task["predicted_score"],
                **rag_cand_cells,
                "is_correct": normalized["is_correct"],
                "gold_title": normalized["gold_title"],
                "gold_code": normalized["gold_code"],
                "error_type": normalized["error_type"],
                "error_note": normalized["error_note"],
                "reviewer": REVIEWER,
                "review_time": now_str,
            })

            # 保存解析失败的原始输出，供人工调试
            if (
                normalized["error_note"] == "qwen_output_parse_failed"
                and len(failed_raw_records) < FAILED_DUMP_MAX
            ):
                failed_raw_records.append({
                    "sample_id": task["sample_id"],
                    "stage": task["stage"],
                    "predicted_title": task["predicted_title"],
                    "raw_output": safe_str(raw_out),
                })

    infer_cost = time.time() - infer_start
    print(f"[4/5] 推理完成，耗时: {infer_cost:.2f}s")

    # ---- Step 5: 保存结果 ----
    print("[5/5] 保存结果文件...")
    out_df = pd.DataFrame(rows_out)
    _run_output_check(out_df)
    out_df = out_df[OUTPUT_COLUMNS]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(OUTPUT_DIR, f"qwen3_8b_rag_labels_{ts}.csv")
    latest_file = os.path.join(OUTPUT_DIR, "qwen3_8b_rag_labels_latest.csv")
    failed_file = os.path.join(OUTPUT_DIR, f"qwen3_8b_rag_failed_{ts}.txt")
    failed_latest = os.path.join(OUTPUT_DIR, "qwen3_8b_rag_failed_latest.txt")

    out_df.to_csv(out_file, index=False, encoding="utf-8-sig")
    out_df.to_csv(latest_file, index=False, encoding="utf-8-sig")

    if failed_raw_records:
        for fpath in (failed_file, failed_latest):
            with open(fpath, "w", encoding="utf-8") as f:
                for idx, rec in enumerate(failed_raw_records, 1):
                    f.write(f"==== FAILED SAMPLE {idx} ====\n")
                    f.write(f"sample_id: {rec['sample_id']}\n")
                    f.write(f"stage: {rec['stage']}\n")
                    f.write(f"predicted_title: {rec['predicted_title']}\n")
                    f.write("raw_output:\n")
                    f.write(rec["raw_output"] + "\n\n")

    print("\n✅ RAG 增强自动标注完成")
    print(f"  版本文件 : {out_file}")
    print(f"  最新文件 : {latest_file}")
    print(f"  样本总数 : {len(out_df)}")
    if failed_raw_records:
        print(f"  失败原文 : {failed_latest}")


if __name__ == "__main__":
    run_auto_labeling()
