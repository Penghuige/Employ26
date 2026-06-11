"""活跃分析链路的统一 CLI。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable

from src.analysis.education_distribution_analysis import EducationDistributionAnalyzer
from src.analysis.generate_excel_summary import ExcelReportGenerator
from src.analysis.generate_standardized_tables import StandardizedTableGenerator
from src.analysis.industry_trend_analysis import IndustryTrendAnalyzer
from src.analysis.occupation_salary_analysis import OccupationSalaryAnalyzer
from src.analysis.requirement_text_analysis import (
    DEFAULT_EXTRACTOR_VERSION,
    DEFAULT_GROUP_SIZE,
    DEFAULT_MONTHLY_GROUP_SIZE,
    DEFAULT_TOP_N,
    AnalysisParams,
    analyze_requirement_texts,
    build_current_output_dir,
)
from src.analysis.structured_common import build_structured_output_dir, write_run_manifest
from src.analysis.structured_dimension_analysis import StructuredDimensionAnalyzer
from src.analysis.structured_pg_source import (
    build_structured_source_coverage,
    write_structured_source_coverage,
)


logger = logging.getLogger(__name__)


def _run_step(step_name: str, action: Callable[[], None]) -> None:
    """记录并运行一个分析步骤，保留原始异常方便排查。"""
    logger.info("运行分析步骤: %s", step_name)
    action()


def run_structured(args: argparse.Namespace) -> None:
    """运行结构化统计链路。"""
    base_dir = Path(args.base_dir) if args.base_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else build_structured_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    completed_steps: list[str] = []
    coverage = build_structured_source_coverage()
    write_structured_source_coverage(output_dir)
    logger.info(
        "结构化主输入覆盖率: normalized_rows=%s, matched_rows=%s, matched_share=%.4f",
        coverage["normalized_rows"],
        coverage["matched_rows"],
        coverage["matched_share"],
    )
    if float(coverage["matched_share"]) < 0.8:
        logger.warning(
            "职业匹配覆盖率偏低（matched_share=%.4f），职业维度相关报表可能不能代表总体数据。",
            coverage["matched_share"],
        )

    _run_step("occupation_salary_analysis", lambda: OccupationSalaryAnalyzer(base_dir=base_dir, output_dir=output_dir).run())
    completed_steps.append("occupation_salary_analysis")
    _run_step(
        "education_distribution_analysis",
        lambda: EducationDistributionAnalyzer(base_dir=base_dir, output_dir=output_dir).run(),
    )
    completed_steps.append("education_distribution_analysis")
    _run_step("industry_trend_analysis", lambda: IndustryTrendAnalyzer(base_dir=base_dir, output_dir=output_dir).run())
    completed_steps.append("industry_trend_analysis")
    _run_step(
        "structured_dimension_analysis",
        lambda: StructuredDimensionAnalyzer(base_dir=base_dir, output_dir=output_dir).run(),
    )
    completed_steps.append("structured_dimension_analysis")

    if not bool(args.skip_standardized):
        _run_step(
            "generate_standardized_tables",
            lambda: StandardizedTableGenerator(base_dir=base_dir, output_dir=output_dir).generate_all(),
        )
        completed_steps.append("generate_standardized_tables")

    if bool(args.with_excel):
        _run_step(
            "generate_excel_summary",
            lambda: ExcelReportGenerator(base_dir=base_dir, output_dir=output_dir).create_summary_report(),
        )
        completed_steps.append("generate_excel_summary")

    write_run_manifest(
        output_dir,
        workflow="structured_analysis",
        steps=completed_steps,
        params={
            "with_excel": bool(args.with_excel),
            "skip_standardized": bool(args.skip_standardized),
            "source": "postgres",
            "normalized_table": "public.recruitment_jobs_normalized",
            "occupation_match_table": "public.skill_extraction_requirement_matches",
        },
        input_files=[
            "postgres:public.recruitment_jobs_normalized",
            "postgres:public.skill_extraction_requirement_matches",
        ],
        output_files=sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    )


def run_requirements(args: argparse.Namespace) -> None:
    """运行 requirement text 约束抽取统计链路。"""
    extractor_version = str(args.extractor_version).strip() or DEFAULT_EXTRACTOR_VERSION
    analyze_requirement_texts(
        output_dir=build_current_output_dir(),
        params=AnalysisParams(
            top_n=int(args.top_n),
            min_group_size=int(args.min_group_size),
            min_monthly_group_size=int(args.min_monthly_group_size),
            extractor_version=extractor_version,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """构建统一分析 CLI 参数解析器。"""
    parser = argparse.ArgumentParser(description="Employ26 活跃分析链路")
    subparsers = parser.add_subparsers(dest="command", required=True)

    structured = subparsers.add_parser("structured", help="结构化统计链路（兼容: 直接运行）")
    structured_subparsers = structured.add_subparsers(dest="structured_command")
    structured_run = structured_subparsers.add_parser("run", help="运行结构化统计链路")
    _add_structured_args(structured)
    _add_structured_args(structured_run)
    structured.set_defaults(func=run_structured)
    structured_run.set_defaults(func=run_structured)

    requirements = subparsers.add_parser("requirements", help="requirement text 统计链路（兼容: 直接运行）")
    requirements_subparsers = requirements.add_subparsers(dest="requirements_command")
    requirements_run = requirements_subparsers.add_parser("run", help="运行 requirement text 约束统计")
    _add_requirements_args(requirements)
    _add_requirements_args(requirements_run)
    requirements.set_defaults(func=run_requirements)
    requirements_run.set_defaults(func=run_requirements)

    return parser


def _add_structured_args(parser: argparse.ArgumentParser) -> None:
    """为结构化统计命令添加公共参数。"""
    parser.add_argument(
        "--with-excel",
        action="store_true",
        help="生成最终 Excel 汇总",
    )
    parser.add_argument(
        "--skip-standardized",
        action="store_true",
        help="跳过规范化 CSV 汇总表",
    )
    parser.add_argument("--output-dir", default="", help="显式指定结构化统计输出目录")
    parser.add_argument("--base-dir", default="", help="显式指定项目根目录（兼容旧脚本）")


def _add_requirements_args(parser: argparse.ArgumentParser) -> None:
    """为 requirement text 命令添加公共参数。"""
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--min-group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--min-monthly-group-size", type=int, default=DEFAULT_MONTHLY_GROUP_SIZE)
    parser.add_argument("--extractor-version", default=DEFAULT_EXTRACTOR_VERSION)


def main() -> None:
    """CLI 入口。"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
