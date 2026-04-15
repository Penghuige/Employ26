"""平面化硬技能抽取回归评测脚本。

该脚本用于给 ``flat_skill_dictionary.json`` 和匹配器建立稳定的自动回归门禁。
输入一份带标准答案的数据集，输出 precision / recall / F1 以及误报漏报明细。

支持的数据格式：

1. JSONL
   每行一个 JSON 对象，至少包含：
   - ``text`` 或任一岗位文本字段
   - ``gold_skills``: 标准技能名列表

2. CSV
   至少包含：
   - ``text`` 或任一岗位文本字段
   - ``gold_skills``: JSON 数组字符串，或使用 ``|`` / ``,` 分隔
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from .config import load_skill_extraction_config
from .match_flat_skills_to_duckdb import FlatHardSkillMatcher, load_flat_dictionary


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


DEFAULT_DATASET_PATH = "output/skill_extraction/regression/flat_skill_regression_dataset.jsonl"


@dataclass(frozen=True)
class RegressionRow:
    """单条回归评测样本。"""

    sample_id: str
    text: str
    gold_skills: List[str]


def _safe_text(value: object) -> str:
    """安全转字符串并去除空白。"""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _normalize_skill_name(name: str) -> str:
    """归一化技能名称，用于评测对齐。"""
    return _safe_text(name).casefold()


def _parse_skill_list(value: object) -> List[str]:
    """解析 gold_skills 字段。"""
    if value is None:
        return []
    if isinstance(value, list):
        return [_safe_text(item) for item in value if _safe_text(item)]

    text = _safe_text(value)
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        return [_safe_text(item) for item in parsed if _safe_text(item)]

    if "|" in text:
        return [_safe_text(item) for item in text.split("|") if _safe_text(item)]
    if "," in text:
        return [_safe_text(item) for item in text.split(",") if _safe_text(item)]
    return [text]


def _extract_eval_text(row: Dict[str, object]) -> str:
    """按优先级提取评测文本。"""
    for field in ["text", "任职要求_items_text", "岗位职责_items_text", "岗位描述_清洗"]:
        text = _safe_text(row.get(field, ""))
        if text:
            return text
    return ""


def _load_jsonl(path: Path) -> List[Dict]:
    """读取 JSONL 文件。"""
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_regression_dataset(path: str | Path) -> List[RegressionRow]:
    """加载回归评测数据集。"""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"评测数据集不存在: {dataset_path}")

    if dataset_path.suffix.lower() == ".jsonl":
        raw_rows = _load_jsonl(dataset_path)
    elif dataset_path.suffix.lower() == ".csv":
        raw_rows = pd.read_csv(dataset_path, encoding="utf-8").to_dict(orient="records")
    else:
        raise ValueError("仅支持 .jsonl 或 .csv 评测数据集")

    dataset: List[RegressionRow] = []
    for index, row in enumerate(raw_rows):
        sample_id = _safe_text(row.get("sample_id", "")) or f"sample_{index:06d}"
        text = _extract_eval_text(row)
        gold_skills = _parse_skill_list(row.get("gold_skills", row.get("gold_skill_name", [])))
        if not text:
            continue
        dataset.append(
            RegressionRow(
                sample_id=sample_id,
                text=text,
                gold_skills=gold_skills,
            )
        )

    return dataset


def _compute_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """根据 TP / FP / FN 计算指标。"""
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_regression_dataset(
    dataset_path: str | Path,
    dict_path: str | Path,
    output_dir: str | Path | None = None,
) -> Dict:
    """执行回归评测并输出报告。"""
    config = load_skill_extraction_config()
    dataset = load_regression_dataset(dataset_path)
    flat_dict = load_flat_dictionary(dict_path)
    matcher = FlatHardSkillMatcher(flat_dict)

    if not dataset:
        raise ValueError("评测数据集为空，无法执行回归评测")

    tp = 0
    fp = 0
    fn = 0
    exact_match_count = 0
    error_rows: List[Dict] = []

    for row in dataset:
        predicted = matcher.match_text(row.text)
        predicted_keys = {_normalize_skill_name(item): item for item in predicted}
        gold_keys = {_normalize_skill_name(item): item for item in row.gold_skills}

        predicted_set = set(predicted_keys.keys())
        gold_set = set(gold_keys.keys())

        true_positive = predicted_set & gold_set
        false_positive = predicted_set - gold_set
        false_negative = gold_set - predicted_set

        tp += len(true_positive)
        fp += len(false_positive)
        fn += len(false_negative)

        if not false_positive and not false_negative:
            exact_match_count += 1

        if false_positive or false_negative:
            error_rows.append(
                {
                    "sample_id": row.sample_id,
                    "text": row.text,
                    "predicted_skills": json.dumps(predicted, ensure_ascii=False),
                    "gold_skills": json.dumps(row.gold_skills, ensure_ascii=False),
                    "false_positives": json.dumps(
                        [predicted_keys[key] for key in sorted(false_positive)],
                        ensure_ascii=False,
                    ),
                    "false_negatives": json.dumps(
                        [gold_keys[key] for key in sorted(false_negative)],
                        ensure_ascii=False,
                    ),
                }
            )

    metrics = _compute_metrics(tp=tp, fp=fp, fn=fn)
    summary = {
        "evaluated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset_path": str(dataset_path),
        "dictionary_path": str(dict_path),
        "sample_count": len(dataset),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "exact_match_count": exact_match_count,
        "exact_match_rate": exact_match_count / max(len(dataset), 1),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "bert_model_path": str(config.bert_model_path),
        "llm_model_path": str(config.llm_model_path),
    }

    report_dir = Path(output_dir) if output_dir else (config.report_dir / "regression_eval")
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = report_dir / f"regression_summary_{timestamp}.json"
    error_path = report_dir / f"regression_errors_{timestamp}.csv"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(error_rows).to_csv(error_path, index=False, encoding="utf-8-sig")

    logger.info("回归评测完成: P=%.4f R=%.4f F1=%.4f", summary["precision"], summary["recall"], summary["f1"])
    logger.info("评测汇总: %s", summary_path)
    logger.info("误差明细: %s", error_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="平面化硬技能回归评测")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_PATH,
        help=f"回归评测数据集路径 (默认: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument(
        "--dictionary",
        default="dicts/flat_skill_dictionary.json",
        help="待评测的平面化技能词典路径",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="评测报告输出目录（默认写入 output/skill_extraction/reports/regression_eval）",
    )
    parser.add_argument(
        "--fail-under-f1",
        type=float,
        default=None,
        help="若 F1 低于该阈值，则进程返回非零退出码",
    )
    parser.add_argument(
        "--fail-under-precision",
        type=float,
        default=None,
        help="若 Precision 低于该阈值，则进程返回非零退出码",
    )
    return parser


def main() -> None:
    """CLI 入口。"""
    parser = build_parser()
    args = parser.parse_args()
    summary = evaluate_regression_dataset(
        dataset_path=args.dataset,
        dict_path=args.dictionary,
        output_dir=args.output_dir,
    )

    if args.fail_under_f1 is not None and summary["f1"] < float(args.fail_under_f1):
        raise SystemExit(1)
    if (
        args.fail_under_precision is not None
        and summary["precision"] < float(args.fail_under_precision)
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
