# =============================================================================
# 模块：D7_iterate_pipeline.py
# 功能：自我迭代一键流水线
#       按顺序执行：D4（重算分数）-> D5（边界样本标注）-> D3（困难样本微调）-> D6（阈值评估）
#       每轮结束后报告当前精度，未达目标时提示继续下一轮
# 用法：python src/bge/D7_iterate_pipeline.py [--rounds N] [--force-recalc]
# =============================================================================

import argparse
import os
import sys
import time
from datetime import datetime

import pandas as pd

# ---------------------------------------------------------------------------
# 将项目根目录加入 sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# 导入各步骤的主函数（复用已有模块，不重复写逻辑）
# ---------------------------------------------------------------------------
from src.bge.D4_T2match import main_tier2_retrieval          # D4
from src.bge.D5_qwen3_auto_label import run_auto_labeling    # D5
from src.bge.D3_finetune import train_finetuned_model        # D3
from src.bge.D6_threshold_eval import build_threshold_report  # D6

# ---------------------------------------------------------------------------
# 迭代控制配置
# ---------------------------------------------------------------------------
MAX_ROUNDS = 5              # 最多迭代轮次
TARGET_PRECISION = 0.85     # 精度目标
D4_FORCE_RECALCULATE = True # 每轮强制重算向量分数（使用新模型后必须开启）

OUTPUT_DIR = r"src\bge\output"
LABEL_FILE = r"src\bge\output\qwen3_8b_rag_labels_latest.csv"
THRESHOLD_REPORT = r"src\bge\output\threshold_evaluation_latest.csv"
ITER_LOG_FILE = os.path.join(OUTPUT_DIR, "iteration_log.csv")


def _banner(msg: str) -> None:
    """打印带分隔线的阶段标题。"""
    line = "=" * 60
    print(f"\n{line}")
    print(f"  {msg}")
    print(f"{line}")


def _read_best_precision() -> float:
    """从 D6 最新报告中读取推荐阈值行的 precision_est。"""
    if not os.path.exists(THRESHOLD_REPORT):
        return 0.0
    try:
        df = pd.read_csv(THRESHOLD_REPORT, encoding="utf-8-sig")
        candidates = df[
            (df["decision"] == "candidate")
            & (df["sample_size_eval"] > 0)
        ].copy()
        if candidates.empty:
            # 未达标时取最高精度行
            best = df.sort_values("precision_est", ascending=False).iloc[0]
        else:
            best = candidates.sort_values(
                by=["matched_rate", "threshold"], ascending=[False, True]
            ).iloc[0]
        return float(best["precision_est"])
    except Exception as e:
        print(f"[WARN] 读取精度报告失败: {e}")
        return 0.0


def _log_round(round_idx: int, precision: float, elapsed: float) -> None:
    """追加轮次结果到迭代日志文件。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    row = pd.DataFrame([{
        "round": round_idx,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "precision_est": round(precision, 6),
        "elapsed_sec": round(elapsed, 1),
        "target_met": precision >= TARGET_PRECISION,
    }])
    if os.path.exists(ITER_LOG_FILE):
        row.to_csv(ITER_LOG_FILE, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        row.to_csv(ITER_LOG_FILE, index=False, encoding="utf-8-sig")


def run_one_round(round_idx: int, force_recalc: bool) -> float:
    """执行单轮迭代：D4 -> D5 -> D3 -> D6，返回本轮结束后的 precision_est。"""
    round_start = time.time()

    # ---- Step 1: D4 全量打分（新模型后须强制重算）----
    _banner(f"Round {round_idx} | Step 1/4: D4 全量向量检索")
    # 临时覆盖 D4 模块级配置，使其强制重算
    import src.bge.D4_T2match as _d4
    _orig_force = _d4.FORCE_RECALCULATE
    _d4.FORCE_RECALCULATE = force_recalc
    try:
        main_tier2_retrieval()
    finally:
        _d4.FORCE_RECALCULATE = _orig_force

    # ---- Step 2: D5 边界样本标注 ----
    _banner(f"Round {round_idx} | Step 2/4: D5 Qwen3 自动标注")
    run_auto_labeling()

    # ---- Step 3: D3 困难样本微调 ----
    _banner(f"Round {round_idx} | Step 3/4: D3 BGE 微调")
    train_finetuned_model()

    # ---- Step 4: D6 阈值评估 ----
    _banner(f"Round {round_idx} | Step 4/4: D6 阈值评估")
    build_threshold_report()

    precision = _read_best_precision()
    elapsed = time.time() - round_start
    _log_round(round_idx, precision, elapsed)

    return precision


def main():
    parser = argparse.ArgumentParser(description="BGE 精度自迭代流水线")
    parser.add_argument(
        "--rounds", type=int, default=MAX_ROUNDS,
        help=f"最大迭代轮次（默认 {MAX_ROUNDS}）"
    )
    parser.add_argument(
        "--target", type=float, default=TARGET_PRECISION,
        help=f"目标精度，达到后自动停止（默认 {TARGET_PRECISION}）"
    )
    parser.add_argument(
        "--no-force-recalc", action="store_true",
        help="禁止强制重算向量（调试用，加快速度但不用新模型）"
    )
    args = parser.parse_args()

    target = args.target
    force_recalc = not args.no_force_recalc
    max_rounds = args.rounds

    _banner(f"自迭代流水线启动 | 目标精度 >= {target} | 最大轮次 {max_rounds}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_start = time.time()
    final_precision = 0.0

    for rnd in range(1, max_rounds + 1):
        print(f"\n{'#'*60}")
        print(f"#  开始第 {rnd} 轮迭代  (目标精度: {target:.2f})")
        print(f"{'#'*60}")

        try:
            precision = run_one_round(rnd, force_recalc=force_recalc)
        except Exception as e:
            print(f"\n❌ 第 {rnd} 轮发生异常，已跳过: {e}")
            import traceback
            traceback.print_exc()
            break

        final_precision = precision
        print(f"\n[Round {rnd}] precision_est = {precision:.4f} (目标: {target:.2f})")

        if precision >= target:
            print(f"\n✅ 已达到目标精度 {precision:.4f} >= {target:.2f}，停止迭代。")
            break
        else:
            remaining = max_rounds - rnd
            if remaining > 0:
                print(f"   尚未达标，继续第 {rnd + 1} 轮...")
            else:
                print(f"   已达到最大轮次 {max_rounds}，停止迭代。")

    total_elapsed = time.time() - total_start
    _banner("迭代流水线结束")
    print(f"  最终精度: {final_precision:.4f}")
    print(f"  总耗时  : {total_elapsed / 60:.1f} 分钟")
    print(f"  迭代日志: {ITER_LOG_FILE}")
    print(f"  阈值报告: {THRESHOLD_REPORT}")

    if final_precision >= target:
        print("\n✅ 精度达标，建议操作：")
        print("   1. 查看 D6 报告选择最优阈值")
        print("   2. 将该阈值写入 D4 的 MATCH_THRESHOLD")
        print("   3. 重新运行 D4 生成最终分流结果")
    else:
        print(f"\n⚠️ 精度未达标 ({final_precision:.4f} < {target:.2f})，建议：")
        print("   1. 增大 D5 的 SAMPLE_SIZE_PER_STAGE['tier2'] 扩大标注量")
        print("   2. 增大 D3 的 HARD_SAMPLE_REPEAT 提高困难样本权重")
        print("   3. 增大 D3 的 EPOCHS 延长训练")
        print("   4. 手动补充高质量 tier2 标注后重跑")


if __name__ == "__main__":
    main()
