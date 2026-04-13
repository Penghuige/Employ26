"""新匹配系统评估脚本。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .matching_evaluator import evaluate_matches


def main() -> None:
    parser = argparse.ArgumentParser(description="评估岗位到职业大典匹配结果")
    parser.add_argument("--result-csv", required=True, help="包含 top1_code / candidates / gold_code 的结果文件")
    args = parser.parse_args()

    df = pd.read_csv(args.result_csv, encoding="utf-8")
    report = evaluate_matches(df)
    print(report)


if __name__ == "__main__":
    main()
