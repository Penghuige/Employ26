import argparse
import json

from .config import RAGConfig
from .pipeline import LocalOccupationRAG


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="python -m src.rag.cli",
        description="本地职业知识库 RAG（BGE 检索 + Qwen3-8B 生成）",
    )

    parser.add_argument(
        "--action",
        choices=["build", "query"],
        required=True,
        help="build=构建索引；query=执行查询",
    )

    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="当 action=query 时必填。示例：'财务会计，负责报税和凭证录入'",
    )

    parser.add_argument("--top_k", type=int, default=8, help="检索候选数")

    # 显式暴露路径参数，便于后续迁移环境时快速覆写
    parser.add_argument("--kb_excel_path", type=str, default=r"data\中国职业大典.xlsx")
    parser.add_argument("--embedding_model_path", type=str, default=r"D:\model\bge-base-zh-v1.5")
    parser.add_argument("--generator_model_path", type=str, default=r"D:\model\Qwen3-8B")
    parser.add_argument("--index_path", type=str, default=r"src\rag\artifacts\occupation_index.faiss")
    parser.add_argument("--metadata_path", type=str, default=r"src\rag\artifacts\occupation_metadata.json")

    return parser


def main() -> None:
    """CLI 主入口。"""
    parser = build_parser()
    args = parser.parse_args()

    cfg = RAGConfig(
        kb_excel_path=args.kb_excel_path,
        embedding_model_path=args.embedding_model_path,
        generator_model_path=args.generator_model_path,
        index_path=args.index_path,
        metadata_path=args.metadata_path,
        top_k=args.top_k,
    )

    app = LocalOccupationRAG(cfg)

    if args.action == "build":
        app.build_knowledge_index()
        return

    # action == query
    if not args.query.strip():
        raise ValueError("当 --action=query 时，--query 不能为空。")

    result = app.query(args.query, top_k=args.top_k)

    # 标准 JSON 输出，便于后续接 API 或批处理脚本
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
