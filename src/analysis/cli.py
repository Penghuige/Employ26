"""Unified CLI for active analysis workflows."""

from __future__ import annotations

import argparse
import logging

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
from src.data_pipeline.occupation_integration import DataIntegrator


logger = logging.getLogger(__name__)


def _run_step(step_name: str, action) -> None:
    """Log and run one analysis step, letting original exceptions surface."""
    logger.info("运行分析步骤: %s", step_name)
    action()


def run_structured(args: argparse.Namespace) -> None:
    """Run the structured statistics workflow."""
    if bool(args.with_integration):
        _run_step(
            "occupation_integration",
            lambda: DataIntegrator(use_full_data=not bool(args.sample)).integrate_all(),
        )

    _run_step("occupation_salary_analysis", lambda: OccupationSalaryAnalyzer().run())
    _run_step("education_distribution_analysis", lambda: EducationDistributionAnalyzer().run())
    _run_step("industry_trend_analysis", lambda: IndustryTrendAnalyzer().run())

    if not bool(args.skip_standardized):
        _run_step("generate_standardized_tables", lambda: StandardizedTableGenerator().generate_all())

    if bool(args.with_excel):
        _run_step("generate_excel_summary", lambda: ExcelReportGenerator().create_summary_report())


def run_requirements(args: argparse.Namespace) -> None:
    """Run the requirement text constraint analysis workflow."""
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
    """Build the unified analysis CLI parser."""
    parser = argparse.ArgumentParser(description="Employ26 active analysis workflows")
    subparsers = parser.add_subparsers(dest="command", required=True)

    structured = subparsers.add_parser("structured", help="Run structured statistics reports")
    structured.add_argument(
        "--with-integration",
        action="store_true",
        help="Run occupation integration before analysis",
    )
    structured.add_argument(
        "--sample",
        action="store_true",
        help="Use sample data when --with-integration is enabled",
    )
    structured.add_argument(
        "--with-excel",
        action="store_true",
        help="Generate the final Excel summary after CSV and Markdown outputs",
    )
    structured.add_argument(
        "--skip-standardized",
        action="store_true",
        help="Skip standardized CSV table generation",
    )
    structured.set_defaults(func=run_structured)

    requirements = subparsers.add_parser("requirements", help="Run requirement text analysis")
    requirements.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    requirements.add_argument("--min-group-size", type=int, default=DEFAULT_GROUP_SIZE)
    requirements.add_argument("--min-monthly-group-size", type=int, default=DEFAULT_MONTHLY_GROUP_SIZE)
    requirements.add_argument("--extractor-version", default=DEFAULT_EXTRACTOR_VERSION)
    requirements.set_defaults(func=run_requirements)

    return parser


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
