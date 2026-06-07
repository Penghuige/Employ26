"""候选人提交评分脚本。

读取候选人提交的预测结果，与参考答案比对，生成排名和评分报告。

用法:
    python scripts/eval_submission.py --submission output/test_set/submissions/deepseek.json
    python scripts/eval_submission.py --submission output/test_set/submissions/*.json  # 批量评分
    python scripts/eval_submission.py --submit-dir output/test_set/submissions/        # 评分整个目录

提交格式 (JSON 数组):
[
  {"test_id": "TSK-46437", "prediction": "B", "confidence": 0.95, "reasoning": "..."},
  ...
]

输出:
    output/test_set/submissions/<name>_report.json   ← 单项详细报告
    output/test_set/leaderboard.json                  ← 排行榜（批量时）
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

TEST_DIR = PROJECT_ROOT / "output" / "test_set"
ANSWER_KEY = TEST_DIR / "answer_key.json"
CONFIG_FILE = TEST_DIR / "test_set_config.json"


def load_answer_key() -> List[Dict]:
    with open(ANSWER_KEY, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config() -> Dict:
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_submission(path: str) -> Tuple[str, List[Dict]]:
    """加载候选人提交。返回 (名称, 预测列表)。"""
    p = Path(path)
    name = p.stem  # 文件名作为候选人名称
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"{name}: 必须是 JSON 数组")
    return name, raw


def validate_submission(submission: List[Dict], answer_key: List[Dict]) -> List[str]:
    """校验提交完整性，返回错误列表。"""
    errors = []
    sub_ids = {s["test_id"] for s in submission}
    ans_ids = {a["test_id"] for a in answer_key}

    missing = ans_ids - sub_ids
    extra = sub_ids - ans_ids

    if missing:
        errors.append(f"缺少 {len(missing)} 条: {sorted(missing)[:5]}...")
    if extra:
        errors.append(f"多余 {len(extra)} 条: {sorted(extra)[:5]}...")

    # 校验 prediction 值
    valid = {"A", "B", "C", "D", "E", "NONE"}
    for s in submission:
        pred = s.get("prediction", "")
        if pred not in valid:
            errors.append(f"{s.get('test_id', '?')}: 无效 prediction='{pred}'")

    return errors


def bootstrap_ci(scores: np.ndarray, n_bootstrap: int = 2000) -> Tuple[float, float]:
    """Bootstrap 95% confidence interval."""
    rng = np.random.RandomState(42)
    means = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(scores), len(scores), replace=True)
        means.append(scores[idx].mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate(name: str, submission: List[Dict], answer_key: List[Dict]) -> Dict:
    """全面评分。"""
    ans_map = {a["test_id"]: a for a in answer_key}
    sub_map = {s["test_id"]: s for s in submission}

    tier_weights = {"gold": 3, "silver": 2, "bronze": 1}

    # 1. 按 tier 计算 weighted accuracy
    correct_by_tier = defaultdict(int)
    total_by_tier = defaultdict(int)
    weighted_score = 0.0
    weighted_total = 0.0

    for test_id, ans in ans_map.items():
        if test_id not in sub_map:
            continue
        pred = sub_map[test_id].get("prediction", "")
        label = ans["label"]
        tier = ans["tier"]
        weight = tier_weights.get(tier, 1)
        is_correct = 1 if pred == label else 0

        correct_by_tier[tier] += is_correct
        total_by_tier[tier] += 1
        weighted_score += is_correct * weight
        weighted_total += weight

    weighted_acc = weighted_score / weighted_total if weighted_total > 0 else 0

    # 2. Per-class metrics
    classes = ["A", "B", "C", "D", "E", "NONE"]
    per_class = {}
    for cls in classes:
        tp = sum(1 for aid, ans in ans_map.items()
                 if aid in sub_map and ans["label"] == cls
                 and sub_map[aid].get("prediction") == cls)
        fp = sum(1 for aid, ans in ans_map.items()
                 if aid in sub_map and ans["label"] != cls
                 and sub_map[aid].get("prediction") == cls)
        fn = sum(1 for aid, ans in ans_map.items()
                 if aid in sub_map and ans["label"] == cls
                 and sub_map[aid].get("prediction") != cls)
        support = sum(1 for ans in ans_map.values() if ans["label"] == cls)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        per_class[cls] = {"precision": round(precision, 4), "recall": round(recall, 4),
                          "f1": round(f1, 4), "support": support}

    # 3. NONE 专项
    none_stats = {}
    if "NONE" in per_class:
        none_stats = {
            "none_recall": per_class["NONE"]["recall"],
            "none_precision": per_class["NONE"]["precision"],
            "none_f1": per_class["NONE"]["f1"],
        }

    # 4. Bootstrap CI (per-tier correctness)
    all_correct = []
    for aid, ans in ans_map.items():
        if aid in sub_map:
            pred = sub_map[aid].get("prediction", "")
            label = ans["label"]
            tier = ans["tier"]
            weight = tier_weights.get(tier, 1)
            all_correct.append((1 if pred == label else 0, weight))

    scores = np.array([c for c, _ in all_correct])
    ci_low, ci_high = bootstrap_ci(scores)

    # 5. Confidence calibration (if available)
    confidences = [s.get("confidence", 0) for s in submission if "confidence" in s]
    avg_confidence = np.mean(confidences) if confidences else 0

    # 6. Per-tier breakdown
    tier_breakdown = {}
    for tier in ["gold", "silver", "bronze"]:
        if total_by_tier[tier] > 0:
            tier_breakdown[tier] = {
                "accuracy": round(correct_by_tier[tier] / total_by_tier[tier], 4),
                "correct": correct_by_tier[tier],
                "total": total_by_tier[tier],
            }

    return {
        "candidate": name,
        "n_total": len(sub_map),
        "weighted_accuracy": round(weighted_acc, 4),
        "overall_accuracy": round(np.mean(scores), 4),
        "ci_95": [round(ci_low, 4), round(ci_high, 4)],
        "avg_confidence": round(float(avg_confidence), 4),
        "macro_f1": round(float(np.mean([per_class[c]["f1"] for c in classes])), 4),
        "none_stats": none_stats,
        "per_class": per_class,
        "tier_breakdown": tier_breakdown,
        "has_errors": False,
        "errors": [],
    }


def format_report(result: Dict) -> str:
    """格式化为可读报告。"""
    lines = []
    lines.append("=" * 65)
    lines.append(f"  候选人: {result['candidate']}")
    lines.append("=" * 65)
    lines.append(f"  Weighted Accuracy: {result['weighted_accuracy']:.2%} "
                 f"(95% CI: {result['ci_95'][0]:.2%}-{result['ci_95'][1]:.2%})")
    lines.append(f"  Overall Accuracy:  {result['overall_accuracy']:.2%}")
    lines.append(f"  Macro F1:          {result['macro_f1']:.4f}")
    lines.append(f"  Avg Confidence:    {result['avg_confidence']:.4f}")
    lines.append("")

    if result["none_stats"]:
        ns = result["none_stats"]
        lines.append(f"  NONE Recall:    {ns['none_recall']:.2%}")
        lines.append(f"  NONE Precision: {ns['none_precision']:.2%}")
        lines.append(f"  NONE F1:        {ns['none_f1']:.4f}")
        lines.append("")

    # Tier breakdown
    lines.append("  Per Tier:")
    for tier, stats in result["tier_breakdown"].items():
        w = {"gold": 3, "silver": 2, "bronze": 1}.get(tier, 1)
        lines.append(f"    {tier:8s} (wt={w}): {stats['accuracy']:.2%} "
                     f"({stats['correct']}/{stats['total']})")

    # Per class
    lines.append("")
    lines.append(f"  {'Class':>6s} {'P':>7s} {'R':>7s} {'F1':>7s} {'Support':>7s}")
    for cls in ["A", "B", "C", "D", "E", "NONE"]:
        pc = result["per_class"][cls]
        lines.append(f"  {cls:>6s} {pc['precision']:>7.3f} {pc['recall']:>7.3f} "
                     f"{pc['f1']:>7.3f} {pc['support']:>7d}")

    return "\n".join(lines)


def update_leaderboard(reports: List[Dict]) -> None:
    """更新排行榜。"""
    sorted_reports = sorted(reports, key=lambda x: -x["weighted_accuracy"])

    leaderboard = {
        "updated_at": __import__("datetime").datetime.now().isoformat(),
        "test_set_version": "1.0",
        "rankings": [
            {
                "rank": i + 1,
                "candidate": r["candidate"],
                "weighted_accuracy": r["weighted_accuracy"],
                "ci_95": r["ci_95"],
                "macro_f1": r["macro_f1"],
                "none_recall": r["none_stats"].get("none_recall", 0),
                "none_precision": r["none_stats"].get("none_precision", 0),
            }
            for i, r in enumerate(sorted_reports)
        ],
    }

    lb_path = TEST_DIR / "leaderboard.json"
    with open(lb_path, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, ensure_ascii=False, indent=2)

    # 打印排行榜
    print(f"\n{'='*80}")
    print(f"  LEADERBOARD")
    print(f"{'='*80}")
    print(f"  {'Rank':<5s} {'Candidate':<20s} {'Weighted':>8s} {'95% CI':>16s} "
          f"{'MacroF1':>8s} {'NONE-R':>8s} {'NONE-P':>8s}")
    print(f"  {'-'*75}")
    for r in sorted_reports:
        ci = f"{r['ci_95'][0]:.2%}-{r['ci_95'][1]:.2%}"
        nr = r["none_stats"].get("none_recall", 0)
        np_ = r["none_stats"].get("none_precision", 0)
        print(f"  {sorted_reports.index(r)+1:>3d}. "
              f"{r['candidate']:<20s} {r['weighted_accuracy']:>7.2%} {ci:>16s} "
              f"{r['macro_f1']:>7.4f} {nr:>7.2%} {np_:>7.2%}")

    print(f"\n  Leaderboard saved: {lb_path}")


def main():
    parser = argparse.ArgumentParser(description="候选人提交评分")
    parser.add_argument("--submission", nargs="+", help="提交文件路径（可多个）")
    parser.add_argument("--submit-dir", help="批量评分整个目录")
    args = parser.parse_args()

    if not args.submission and not args.submit_dir:
        parser.error("需要 --submission 或 --submit-dir")

    # 加载答案
    answer_key = load_answer_key()
    config = load_config()
    print(f"答案加载: {len(answer_key)} 条")
    print(f"配置: MD5={config.get('checksum_md5', 'N/A')[:12]}...")

    # 收集提交
    submission_paths = []
    if args.submission:
        submission_paths = args.submission
    if args.submit_dir:
        import glob
        submission_paths = list(glob.glob(str(Path(args.submit_dir) / "*.json")))

    if not submission_paths:
        print("未找到提交文件")
        return

    reports = []
    for path in submission_paths:
        print(f"\n--- 评分: {path} ---")
        try:
            name, submission = load_submission(path)
        except Exception as e:
            print(f"  加载失败: {e}")
            continue

        # 校验
        errors = validate_submission(submission, answer_key)
        if errors:
            print(f"  校验错误 ({len(errors)}):")
            for e in errors[:5]:
                print(f"    - {e}")

        # 评分
        result = evaluate(name, submission, answer_key)
        result["has_errors"] = len(errors) > 0
        result["errors"] = errors

        # 输出报告
        print(format_report(result))

        # 保存单项报告
        report_path = TEST_DIR / "submissions" / f"{name}_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        reports.append(result)

    # 更新排行榜
    if len(reports) > 1:
        update_leaderboard(reports)

    print(f"\n评分完成: {len(reports)} 位候选人")


if __name__ == "__main__":
    main()
