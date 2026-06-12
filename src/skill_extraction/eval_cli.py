"""软技能评估 CLI — 统一入口。

用法::

    python -m src.skill_extraction.eval_cli run
    python -m src.skill_extraction.eval_cli compare v1 v2
    python -m src.skill_extraction.eval_cli list
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _get_eval_dir() -> Path:
    """获取评估输出目录。"""
    from config.paths import get_project_paths

    return get_project_paths().project_root / "output" / "skill_extraction" / "eval"


def cmd_list(eval_dir: Optional[Path] = None) -> None:
    """列出所有评估记录。

    参数:
        eval_dir: 评估输出目录，为 None 时使用默认路径。
    """
    from ._eval_registry import load_registry

    registry_dir = eval_dir or _get_eval_dir()
    registry = load_registry(registry_dir)
    evaluations = registry.get("evaluations", [])

    if not evaluations:
        print("no eval records")
        return

    print(f"{'version':<8} {'evaluated_at':<22} {'soft_cov':<10} {'soft_prec':<10} {'hard_f1':<10}")
    print("-" * 60)
    for r in evaluations:
        soft = r.get("soft_skill_metrics", {})
        hard = r.get("hard_skill_metrics", {})
        print(
            f"{r.get('dict_version', '?'):<8} "
            f"{r.get('evaluated_at', '?')[:19]:<22} "
            f"{soft.get('coverage', 0):.4f}   "
            f"{soft.get('precision', 0):.4f}   "
            f"{hard.get('f1', 0):.4f}"
        )


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器。"""
    parser = argparse.ArgumentParser(
        description="soft skill eval CLI",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("run", help="run eval and write registry")
    cmp_parser = sub.add_parser("compare", help="compare two versions")
    cmp_parser.add_argument("version_a")
    cmp_parser.add_argument("version_b")
    sub.add_parser("list", help="list all eval records")

    return parser


def main() -> None:
    """CLI 入口。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "list":
        cmd_list()
    elif args.command == "run":
        logger.info("run command will be implemented in Task 5")
    elif args.command == "compare":
        logger.info("compare command will be implemented in Task 6")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
