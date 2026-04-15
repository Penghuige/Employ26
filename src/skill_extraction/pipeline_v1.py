"""v1 职业细类分层词典流水线入口。"""

from __future__ import annotations

import argparse

from .config import load_skill_extraction_config
from .history.occupation_skill_pipeline import OccupationSkillPipeline


def build_parser() -> argparse.ArgumentParser:
    """构建 v1 流水线命令行参数。"""
    parser = argparse.ArgumentParser(
        description="v1 职业细类分层技能词典流水线",
    )
    subparsers = parser.add_subparsers(dest="command")

    prepare = subparsers.add_parser(
        "prepare",
        help="生成训练样本、验证池和 prompt 文件",
    )
    prepare.add_argument("--train-size", type=int, default=100, help="每个职业细类的训练样本数")
    prepare.add_argument("--validation-batch-size", type=int, default=10, help="每轮验证每个细类抽取数量")
    prepare.add_argument("--seed", type=int, default=42, help="随机种子")
    prepare.add_argument("--limit-job-rows", type=int, default=None, help="调试用，限制每张招聘表读取行数")
    prepare.add_argument("--limit-categories", type=int, default=None, help="调试用，限制处理的职业细类数量")
    prepare.add_argument("--match-workers", type=int, default=4, help="岗位匹配并发数")
    prepare.add_argument("--match-chunk-size", type=int, default=256, help="岗位匹配分块大小")
    prepare.add_argument("--parse-workers", type=int, default=1, help="岗位描述切分并发数")
    prepare.add_argument("--progress", action="store_true", help="显示岗位匹配进度")

    iterate = subparsers.add_parser(
        "iterate",
        help="执行一轮覆盖率验证并生成补词 prompt",
    )
    iterate.add_argument("--validation-batch-size", type=int, default=10, help="每轮验证每个细类抽取数量")
    iterate.add_argument("--coverage-threshold", type=float, default=0.95, help="目标覆盖率阈值")
    iterate.add_argument("--limit-categories", type=int, default=None, help="调试用，限制验证的职业细类数量")
    iterate.add_argument("--parse-workers", type=int, default=1, help="岗位描述切分并发数")

    subparsers.add_parser("status", help="查看当前迭代状态")
    return parser


def main() -> None:
    """执行 v1 流水线入口分发。"""
    parser = build_parser()
    args = parser.parse_args()
    config = load_skill_extraction_config()
    pipeline = OccupationSkillPipeline(config)

    if args.command == "prepare":
        pipeline.prepare(
            train_size=args.train_size,
            validation_batch_size=args.validation_batch_size,
            seed=args.seed,
            limit_job_rows=args.limit_job_rows,
            limit_categories=args.limit_categories,
            match_workers=args.match_workers,
            match_chunk_size=args.match_chunk_size,
            parse_workers=args.parse_workers,
            show_progress=args.progress,
        )
        return

    if args.command == "iterate":
        pipeline.iterate(
            validation_batch_size=args.validation_batch_size,
            coverage_threshold=args.coverage_threshold,
            limit_categories=args.limit_categories,
            parse_workers=args.parse_workers,
        )
        return

    if args.command == "status":
        pipeline.status()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
