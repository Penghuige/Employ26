"""RAG v2 命令行入口。

命令:
    build    — 从 DuckDB 构建双 FAISS 索引
    query    — RAG 检索 + DeepSeek 生成最佳匹配职业细类
    judge    — 给定固定候选列表，DeepSeek 评判选出最佳（对齐 eval_annotation_quality）

用法:
    python -m src.rag.cli build
    python -m src.rag.cli query --title "Java开发工程师" --requirements "Spring Boot, MySQL..."
    python -m src.rag.cli judge --title "..." --requirements "..." --candidates-json "[{...}]"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from .config import RAGConfig
from .pipeline import OccupationRAG

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _parse_candidates(raw: str) -> List[Dict[str, Any]]:
    """解析候选 JSON 字符串或文件路径。

    Args:
        raw: JSON 字符串或 .json 文件路径。

    Returns:
        List[Dict]: 候选列表。
    """
    if not raw:
        return []
    # 尝试作为文件路径
    path = Path(raw)
    if path.exists() and path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    # 尝试作为 JSON 字符串
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError(f"无法解析候选数据: {raw[:100]}...")


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="python -m src.rag.cli",
        description="职业分类 RAG v2（DuckDB + bge-large + DeepSeek V4 Pro）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # build
    build_p = sub.add_parser("build", help="构建双 FAISS 索引")
    build_p.add_argument("--duckdb", default="output/recruit.duckdb")
    build_p.add_argument("--table", default="recruit.main.chinese_occupational_dictionary_joined_preprocessed")

    # query
    query_p = sub.add_parser("query", help="RAG 检索 + 生成")
    query_p.add_argument("--title", required=True, help="岗位名称")
    query_p.add_argument("--requirements", default="", help="岗位要求描述")
    query_p.add_argument("--top-k", type=int, default=8)
    from config.paths import get_project_paths
    query_p.add_argument("--embedding-model", default=str(get_project_paths().bge_model_path))

    # judge
    judge_p = sub.add_parser("judge", help="给定候选列表，DeepSeek 评判")
    judge_p.add_argument("--title", required=True, help="岗位名称")
    judge_p.add_argument("--requirements", default="", help="岗位要求描述")
    judge_p.add_argument("--candidates-json", default="", help="候选列表 JSON 字符串或 .json 文件路径")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build":
        cfg = RAGConfig(
            duckdb_path=args.duckdb,
            catalog_table=args.table,
        )
        rag = OccupationRAG(cfg)
        rag.build_index()
        print("索引构建完成。")
        return

    if args.command == "query":
        cfg = RAGConfig(
            embedding_model_path=args.embedding_model,
        )
        rag = OccupationRAG(cfg)
        rag.load()
        result = rag.query(
            job_title=args.title,
            job_requirements=args.requirements,
            top_k=args.top_k,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "judge":
        candidates = _parse_candidates(args.candidates_json)
        if not candidates:
            print("错误: --candidates-json 不能为空", file=sys.stderr)
            sys.exit(1)
        cfg = RAGConfig()
        rag = OccupationRAG(cfg)
        result = rag.judge(
            job_title=args.title,
            job_requirements=args.requirements,
            candidates=candidates,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    main()
