"""
导入 LLM 返回的职业技能词典 JSON。

用法示例：
python -m src.skill_extraction.import_llm_results --input output/skill_extraction/llm_outputs
python -m src.skill_extraction.import_llm_results --input some_result.md --dry-run
"""

from __future__ import annotations

import argparse

from .config import load_skill_extraction_config
from .occupation_skill_pipeline import OccupationSkillPipeline


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""
    parser = argparse.ArgumentParser(description="导入 LLM 输出并合并回职业技能词典")
    parser.add_argument("--input", required=True, help="LLM 输出文件或目录")
    parser.add_argument("--non-recursive", action="store_true", help="目录导入时不递归子目录")
    parser.add_argument("--dry-run", action="store_true", help="只预演导入结果，不写回词典")
    return parser


def main() -> None:
    """CLI 入口。"""
    parser = build_parser()
    args = parser.parse_args()

    config = load_skill_extraction_config()
    pipeline = OccupationSkillPipeline(config)
    pipeline.import_llm_results(
        source_path=args.input,
        recursive=not args.non_recursive,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
