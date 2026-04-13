import os
from datetime import datetime
from typing import Dict, List

import pandas as pd

# ================= 配置区域 =================
# 1) 输入文件
# D5 自动标注结果（抽样）
LABEL_FILE = r"src\bge\output\qwen3_8b_rag_labels_latest.csv"
# D4 全量缓存（用于计算 matched_rate）
SCORE_CACHE_FILE = r"src\bge\data5\Tier2_Intermediate_Cache.csv"

# 2) 阈值扫描范围
THRESHOLD_START = 0.65
THRESHOLD_END = 0.90
THRESHOLD_STEP = 0.01
TARGET_PRECISION = 0.85
MIN_MATCHED_RATE = 0.20

# 3) 模型/数据版本标识
MODEL_VERSION = "bge-base-zh-finetuned"
DATA_VERSION = "tier2_intermediate_cache"

# 4) 输出路径
OUTPUT_DIR = r"src\bge\output"

# 5) 阈值模板字段（严格按 docs/threshold_evaluation_template.txt）
THRESHOLD_COLUMNS = [
    "experiment_id", "model_version", "data_version", "threshold",
    "total_count", "matched_count", "pending_count", "matched_rate",
    "sample_size_eval", "precision_est", "noise_rate_est", "recall_proxy", "f1_proxy",
    "top_error_type_1", "top_error_type_2", "top_error_type_3", "decision", "notes"
]


def _gen_thresholds(start: float, end: float, step: float) -> List[float]:
    vals = []
    cur = start
    while cur <= end + 1e-9:
        vals.append(round(cur, 2))
        cur += step
    return vals


def _top3_error_types(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return ["", "", ""]
    err = df[df["is_correct"] == 0]
    if err.empty:
        return ["", "", ""]

    counts = err["error_type"].fillna("").astype(str).value_counts()
    top = counts.index.tolist()[:3]
    while len(top) < 3:
        top.append("")
    return top


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _make_decision(precision_est: float, matched_rate: float) -> str:
    if precision_est >= TARGET_PRECISION and matched_rate >= MIN_MATCHED_RATE:
        return "candidate"
    if precision_est >= TARGET_PRECISION:
        return "candidate"
    return "reject"


def build_threshold_report():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(LABEL_FILE):
        print(f"❌ 标注文件不存在: {LABEL_FILE}")
        return
    if not os.path.exists(SCORE_CACHE_FILE):
        print(f"❌ 分数缓存不存在: {SCORE_CACHE_FILE}")
        return

    labels = pd.read_csv(LABEL_FILE, encoding="utf-8-sig")
    cache = pd.read_csv(SCORE_CACHE_FILE, encoding="utf-8-sig")

    # 仅使用 tier2 样本估计阈值质量
    eval_df = labels[labels["stage"] == "tier2"].copy()
    if eval_df.empty:
        print("❌ 标注文件中没有 tier2 样本，无法生成阈值评估。")
        return

    # 统一分数列类型
    eval_df["predicted_score"] = pd.to_numeric(eval_df["predicted_score"], errors="coerce")
    eval_df = eval_df.dropna(subset=["predicted_score"])
    cache["tier2_match_score"] = pd.to_numeric(cache["tier2_match_score"], errors="coerce")
    cache = cache.dropna(subset=["tier2_match_score"])

    thresholds = _gen_thresholds(THRESHOLD_START, THRESHOLD_END, THRESHOLD_STEP)
    total_count = len(cache)

    rows_out: List[Dict] = []
    for thr in thresholds:
        # 全量侧统计
        matched_count = int((cache["tier2_match_score"] >= thr).sum())
        pending_count = total_count - matched_count
        matched_rate = _safe_div(matched_count, total_count)

        # 标注样本侧估计 precision
        eval_hit = eval_df[eval_df["predicted_score"] >= thr].copy()
        sample_size_eval = len(eval_hit)
        precision_est = float((eval_hit["is_correct"] == 1).mean()) if sample_size_eval > 0 else 0.0
        noise_rate_est = 1.0 - precision_est if sample_size_eval > 0 else 0.0

        top3 = _top3_error_types(eval_hit)
        decision = _make_decision(precision_est, matched_rate)
        notes = (
            f"precision={precision_est:.3f}, matched_rate={matched_rate:.3f}, sample={sample_size_eval}"
            if sample_size_eval > 0 else
            "该阈值在标注样本中无命中，建议扩大 D5 采样"
        )

        rows_out.append(
            {
                "experiment_id": f"thr_{thr:.2f}",
                "model_version": MODEL_VERSION,
                "data_version": DATA_VERSION,
                "threshold": thr,
                "total_count": total_count,
                "matched_count": matched_count,
                "pending_count": pending_count,
                "matched_rate": round(matched_rate, 6),
                "sample_size_eval": sample_size_eval,
                "precision_est": round(precision_est, 6),
                "noise_rate_est": round(noise_rate_est, 6),
                "recall_proxy": round(matched_rate, 6),
                "f1_proxy": "",
                "top_error_type_1": top3[0],
                "top_error_type_2": top3[1],
                "top_error_type_3": top3[2],
                "decision": decision,
                "notes": notes,
            }
        )

    out_df = pd.DataFrame(rows_out)
    out_df = out_df[THRESHOLD_COLUMNS]

    # 自动推荐：在 precision>=目标 下 matched_rate 最大
    candidates = out_df[
        (out_df["precision_est"] >= TARGET_PRECISION)
        & (out_df["sample_size_eval"] > 0)
    ].copy()

    if not candidates.empty:
        best_row = candidates.sort_values(
            by=["matched_rate", "threshold"], ascending=[False, True]
        ).iloc[0]
        print("\n✅ 推荐阈值（精度优先）")
        print(f"- threshold: {best_row['threshold']}")
        print(f"- precision_est: {best_row['precision_est']:.4f}")
        print(f"- matched_rate: {best_row['matched_rate']:.4f}")
    else:
        best_row = out_df.sort_values(
            by=["precision_est", "matched_rate"], ascending=[False, False]
        ).iloc[0]
        print("\n⚠️ 尚未达到目标精度，当前最优备选")
        print(f"- threshold: {best_row['threshold']}")
        print(f"- precision_est: {best_row['precision_est']:.4f}")
        print(f"- matched_rate: {best_row['matched_rate']:.4f}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(OUTPUT_DIR, f"threshold_evaluation_{ts}.csv")
    latest_file = os.path.join(OUTPUT_DIR, "threshold_evaluation_latest.csv")

    out_df.to_csv(out_file, index=False, encoding="utf-8-sig")
    out_df.to_csv(latest_file, index=False, encoding="utf-8-sig")

    print("\n✅ 阈值评估记录生成完成")
    print(f"- 版本文件: {out_file}")
    print(f"- 最新文件: {latest_file}")


if __name__ == "__main__":
    build_threshold_report()
