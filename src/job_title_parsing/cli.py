"""岗位细类匹配命令行入口。

本文件只保留生产主路径需要的 DuckDB 参数：
1. `preprocess-catalog`：预处理《中国职业分类大典》原始表；
2. `build-hierarchy-dict`：从职业大典自动抽取“大类关键词 -> 大类”词典；
3. `match`：批量匹配招聘岗位到职业细类；
4. `evaluate`：评估带 `gold_code` 的匹配结果。

历史上的 CSV 临时入口和预留的 process 执行后端已移除，避免参数过多导致误用。
如需临时 CSV 调试，建议在 notebook/小脚本中调用 `MatchPipeline` 等底层类。
"""

from __future__ import annotations

import argparse
import json
import time
from os import cpu_count
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from .match_utils import load_config, load_database_config


DATABASE_CONFIG = load_database_config()
DATABASE_SETTINGS = DATABASE_CONFIG.get("database", {})
JOB_TITLE_PARSING_SETTINGS = DATABASE_CONFIG.get("job_title_parsing", {})

DEFAULT_DB_PATH = str(Path(__file__).resolve().parents[2] / DATABASE_SETTINGS.get("duckdb_path", "output/recruit.duckdb"))
DEFAULT_CATALOG_TABLE = JOB_TITLE_PARSING_SETTINGS.get(
    "catalog_table", "recruit.main.chinese_occupational_dictionary_joined"
)
DEFAULT_CATALOG_PREPROCESSED_TABLE = JOB_TITLE_PARSING_SETTINGS.get(
    "catalog_preprocessed_table", "recruit.main.chinese_occupational_dictionary_joined_preprocessed"
)
DEFAULT_JOBS_TABLE = JOB_TITLE_PARSING_SETTINGS.get("jobs_table", "recruit.main.jobs_sample")
DEFAULT_MATCH_RESULT_TABLE = JOB_TITLE_PARSING_SETTINGS.get("match_result_table", "recruit.main.job_match_results")
DEFAULT_WORKERS = max(1, min(32, (cpu_count() or 8) - 2))
DEFAULT_DUCKDB_THREADS = max(1, min(32, int(DATABASE_SETTINGS.get("duckdb_threads", cpu_count() or 8))))


class ChineseHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """同时显示默认值并保留中文多行说明的 argparse formatter。"""


def _add_help_argument(parser: argparse.ArgumentParser) -> None:
    """为 parser 添加中文 help 说明。"""
    parser.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS, help="显示帮助信息并退出")


def _log(message: str) -> None:
    """输出带时间戳的运行日志。"""
    print(f"[{time.strftime('%H:%M:%S')}] {message}")


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。

    设计原则：
    - 命令只面向 DuckDB 生产流程，不暴露 CSV 调试参数；
    - 每个参数都给出中文 help，运行 `-h` 时能直接看懂用途；
    - 只保留当前实现真正使用的参数，移除预留但未验证的选项。
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.job_title_parsing.cli",
        description="岗位到《中国职业分类大典》职业细类的批量匹配工具（DuckDB 主流程）。",
        epilog=(
            "常用示例:\n"
            "  python -m src.job_title_parsing.cli preprocess-catalog\n"
            "  python -m src.job_title_parsing.cli match --jobs-table recruit.main.jobs_sample --progress\n"
            "  python -m src.job_title_parsing.cli evaluate --result-table recruit.main.job_match_results\n"
        ),
        formatter_class=ChineseHelpFormatter,
        add_help=False,
    )
    _add_help_argument(parser)
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    # 预处理职业大典：将原始职业大典表标准化为可检索字段，并写入 DuckDB。
    prep = sub.add_parser(
        "preprocess-catalog",
        help="预处理职业大典表",
        description="读取职业大典 DuckDB 表，生成 title/task/desc 检索字段并写回 DuckDB。",
        formatter_class=ChineseHelpFormatter,
        add_help=False,
    )
    _add_help_argument(prep)
    prep.add_argument("--catalog-duckdb", default=DEFAULT_DB_PATH, help="职业大典输入 DuckDB 文件路径")
    prep.add_argument("--catalog-table", default=DEFAULT_CATALOG_TABLE, help="职业大典原始表名")
    prep.add_argument("--output-duckdb", default=DEFAULT_DB_PATH, help="预处理结果写入的 DuckDB 文件路径")
    prep.add_argument("--output-table", default=DEFAULT_CATALOG_PREPROCESSED_TABLE, help="预处理结果表名")
    prep.add_argument("--duckdb-threads", type=int, default=DEFAULT_DUCKDB_THREADS, help="DuckDB 查询/写入线程数")
    prep.add_argument("--config", default="", help="匹配配置文件路径；为空时使用 config/default.yaml")
    prep.add_argument("--alias-dict", default="", help="可选人工别名字典 JSON 路径")

    # 自动构建层级词典：用于粗粒度大类推断和层级加分。
    build_dict = sub.add_parser(
        "build-hierarchy-dict",
        help="构建层级关键词词典",
        description="从职业大典表中抽取关键词，生成 dicts/hierarchy_keyword_to_major_auto.txt。",
        formatter_class=ChineseHelpFormatter,
        add_help=False,
    )
    _add_help_argument(build_dict)
    build_dict.add_argument("--catalog-duckdb", default=DEFAULT_DB_PATH, help="职业大典输入 DuckDB 文件路径")
    build_dict.add_argument("--catalog-table", default=DEFAULT_CATALOG_TABLE, help="职业大典原始表名")
    build_dict.add_argument("--output-dict", default="dicts/hierarchy_keyword_to_major_auto.txt", help="输出词典路径")
    build_dict.add_argument("--top-n-per-major", type=int, default=80, help="每个大类最多保留的关键词数量")
    build_dict.add_argument("--min-freq", type=int, default=5, help="关键词进入词典的最低出现次数")
    build_dict.add_argument("--duckdb-threads", type=int, default=DEFAULT_DUCKDB_THREADS, help="DuckDB 读取线程数")

    # 批量匹配岗位：主业务命令，读取招聘岗位表并写入匹配结果表。
    match = sub.add_parser(
        "match",
        help="批量匹配岗位",
        description="读取招聘岗位 DuckDB 表，匹配到职业大典细类，并将 top_k 候选写入结果表。",
        formatter_class=ChineseHelpFormatter,
        add_help=False,
    )
    _add_help_argument(match)
    match.add_argument("--catalog-duckdb", default=DEFAULT_DB_PATH, help="职业大典 DuckDB 文件路径")
    match.add_argument("--catalog-table", default=DEFAULT_CATALOG_TABLE, help="职业大典表名，可使用原始表或预处理表")
    match.add_argument("--jobs-duckdb", default=DEFAULT_DB_PATH, help="招聘岗位输入 DuckDB 文件路径")
    match.add_argument("--jobs-table", default=DEFAULT_JOBS_TABLE, help="招聘岗位表名；支持逗号分隔多张表")
    match.add_argument("--output-duckdb", default=DEFAULT_DB_PATH, help="匹配结果写入的 DuckDB 文件路径")
    match.add_argument("--output-table", default=DEFAULT_MATCH_RESULT_TABLE, help="匹配结果表名")
    match.add_argument("--output-mode", choices=["replace", "append"], default="replace", help="结果表写入模式")
    match.add_argument("--job-title-col", default="岗位名称", help="岗位名称字段名")
    match.add_argument("--job-desc-col", default="岗位描述", help="岗位描述字段名")
    match.add_argument("--job-id-col", default="job_id", help="岗位唯一标识字段名；不存在时结果中为空")
    match.add_argument("--top-k", type=int, default=5, help="每条岗位输出的候选职业数量")
    match.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="匹配并发线程数")
    match.add_argument("--executor-backend", choices=["thread", "process"], default="process", help="并发后端：thread 或 process")
    match.add_argument("--chunk-size", type=int, default=256, help="每个线程任务处理的岗位行数")
    match.add_argument("--duckdb-threads", type=int, default=DEFAULT_DUCKDB_THREADS, help="DuckDB 读取/写入线程数")
    match.add_argument("--config", default="", help="匹配配置文件路径；为空时使用 config/default.yaml")
    match.add_argument("--alias-dict", default="", help="可选人工别名字典 JSON 路径")
    match.add_argument("--debug", action="store_true", help="输出 debug_info 字段，便于排查召回和打分")
    match.add_argument("--progress", action="store_true", help="显示表读取和岗位匹配进度条")

    # 评估匹配结果：要求结果表中包含 gold_code 字段。
    eval_cmd = sub.add_parser(
        "evaluate",
        help="评估匹配结果",
        description="读取匹配结果 DuckDB 表，计算 top1 accuracy、top3/top5 recall 和 unmatched rate。",
        formatter_class=ChineseHelpFormatter,
        add_help=False,
    )
    _add_help_argument(eval_cmd)
    eval_cmd.add_argument("--result-duckdb", default=DEFAULT_DB_PATH, help="匹配结果所在 DuckDB 文件路径")
    eval_cmd.add_argument("--result-table", default=DEFAULT_MATCH_RESULT_TABLE, help="匹配结果表名；支持逗号分隔多张表")
    eval_cmd.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="评估并发进程数")
    eval_cmd.add_argument("--chunk-size", type=int, default=20000, help="并行评估时每个任务处理的结果行数")
    eval_cmd.add_argument("--duckdb-threads", type=int, default=DEFAULT_DUCKDB_THREADS, help="DuckDB 读取线程数")
    eval_cmd.add_argument("--progress", action="store_true", help="显示表读取和评估进度条")
    eval_cmd.add_argument("--output-report", default="", help="可选评估报告 JSON 输出路径")

    return parser


def _split_table_names(table_name_expr: str) -> list[str]:
    """解析表名参数。

    `match` 和 `evaluate` 都允许一次读取多张 DuckDB 表。
    例如：`recruit.main.jobs_a,recruit.main.jobs_b`。
    """
    return [name.strip() for name in str(table_name_expr).split(",") if name.strip()]


def _load_table(
    db_path: str,
    table_name: str,
    duckdb_threads: int = DEFAULT_DUCKDB_THREADS,
    show_progress: bool = False,
) -> pd.DataFrame:
    """从 DuckDB 读取一张或多张表为 DataFrame。

    读取多表时会自动增加 `__source_table` 列，便于回溯每条记录来自哪张源表。
    """
    import duckdb

    table_names = _split_table_names(table_name)
    if not table_names:
        raise ValueError("table_name 不能为空")

    conn = duckdb.connect(db_path, read_only=True)
    try:
        conn.execute(f"PRAGMA threads={max(1, int(duckdb_threads))}")
        frames: list[pd.DataFrame] = []

        table_iter = table_names
        if show_progress and len(table_names) > 1:
            table_iter = tqdm(table_names, total=len(table_names), desc="Loading tables", unit="table")

        for tb in table_iter:
            _log(f"读取表: {tb}")
            df = conn.execute(f"SELECT * FROM {tb}").df()
            df["__source_table"] = tb
            frames.append(df)

        if len(frames) == 1:
            return frames[0]
        return pd.concat(frames, axis=0, ignore_index=True)
    finally:
        conn.close()


def _save_table(
    df: pd.DataFrame,
    db_path: str,
    table_name: str,
    output_mode: str = "replace",
    duckdb_threads: int = DEFAULT_DUCKDB_THREADS,
) -> None:
    """将 DataFrame 保存到 DuckDB。

    `replace` 用于重跑全量结果；`append` 用于分批追加新结果。
    """
    import duckdb

    mode = (output_mode or "replace").lower()
    if mode not in {"append", "replace"}:
        raise ValueError(f"不支持的 output_mode: {output_mode}")

    conn = duckdb.connect(db_path)
    try:
        conn.execute(f"PRAGMA threads={max(1, int(duckdb_threads))}")
        conn.register("tmp_result_df", df)

        if mode == "replace":
            conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM tmp_result_df")
            return

        try:
            conn.execute(f"INSERT INTO {table_name} SELECT * FROM tmp_result_df")
        except Exception:
            conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM tmp_result_df")
    finally:
        conn.close()


def main() -> None:
    """CLI 入口，根据子命令分发到对应流程。"""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "preprocess-catalog":
        from .alias_builder import AliasBuilder
        from .catalog_preprocessor import CatalogPreprocessor

        # 1. 初始化配置、别名构建器和职业大典预处理器。
        _log("初始化配置与预处理器")
        config = load_config(args.config or None)
        alias_builder = AliasBuilder(config, alias_dict_path=args.alias_dict or None)
        processor = CatalogPreprocessor(config, alias_builder)

        # 2. 从 DuckDB 读取职业大典原始表并标准化字段。
        _log("加载职业大典数据")
        df = processor.load_duckdb(
            db_path=args.catalog_duckdb,
            table_name=args.catalog_table,
        )

        # 3. 将预处理结果写回 DuckDB，供后续匹配复用。
        _log(f"写入 DuckDB 表: {args.output_table}")
        _save_table(
            df,
            db_path=args.output_duckdb,
            table_name=args.output_table,
            output_mode="replace",
            duckdb_threads=args.duckdb_threads,
        )

        print(f"catalog rows: {len(df)}")
        print(f"saved duckdb table: {args.output_table}")
        return

    if args.command == "build-hierarchy-dict":
        from .hierarchy_keyword_builder import HierarchyKeywordBuilder

        # 从职业大典中抽取高频且有区分度的关键词，作为层级过滤的弱信号。
        _log("加载职业大典数据用于构建层级词典")
        catalog_df = _load_table(
            args.catalog_duckdb,
            args.catalog_table,
            duckdb_threads=args.duckdb_threads,
            show_progress=False,
        )

        builder = HierarchyKeywordBuilder()
        output_path = builder.build_from_catalog(
            catalog_df,
            output_path=args.output_dict,
            top_n_per_major=args.top_n_per_major,
            min_freq=args.min_freq,
        )
        print(f"catalog rows: {len(catalog_df)}")
        print(f"saved hierarchy dict: {output_path}")
        return

    if args.command == "match":
        from .matching_pipeline import MatchPipeline

        start = time.time()
        # 1. 初始化匹配流程。配置负责控制清洗、召回、打分和置信标记等策略。
        _log("[1/5] 初始化匹配流程")
        pipeline = MatchPipeline(config_path=args.config or None, alias_dict_path=args.alias_dict or None)

        # 2. 加载职业大典并构建 title/tasks 两路 BM25 索引。
        _log("[2/5] 加载职业大典并构建索引")
        pipeline.load_catalog_duckdb(
            db_path=args.catalog_duckdb,
            table_name=args.catalog_table,
        )
        _log(f"职业大典条数: {len(pipeline.catalog_df)}")

        # 3. 读取岗位数据。`jobs-table` 支持逗号分隔多表，适合分批来源合并匹配。
        _log("[3/5] 加载岗位数据")
        jobs_df = _load_table(
            args.jobs_duckdb,
            args.jobs_table,
            duckdb_threads=args.duckdb_threads,
            show_progress=args.progress,
        )
        _log(f"岗位条数: {len(jobs_df)}")

        # 4. 批量匹配。支持线程池和进程池两种并发后端，默认使用进程池避免 GIL 竞争。
        _log("[4/5] 开始匹配")
        result_df = pipeline.match_batch(
            jobs_df,
            job_title_col=args.job_title_col,
            job_desc_col=args.job_desc_col,
            job_id_col=args.job_id_col,
            top_k=args.top_k,
            debug=args.debug,
            workers=args.workers,
            show_progress=args.progress,
            chunk_size=args.chunk_size,
            executor_backend=args.executor_backend,
        )

        # 5. DuckDB 不适合直接保存 Python list/dict，写入前转为 JSON 字符串。
        export_df = result_df.copy()
        export_df["candidates"] = export_df["candidates"].map(lambda x: json.dumps(x, ensure_ascii=False))
        export_df["debug_info"] = export_df["debug_info"].map(
            lambda x: json.dumps(x, ensure_ascii=False) if x is not None else ""
        )
        for column in [
            "platform_terms",
            "domain_terms",
            "function_terms",
            "object_terms",
            "conflict_terms",
            "risk_flags",
        ]:
            export_df[column] = export_df[column].map(lambda x: json.dumps(x, ensure_ascii=False))

        _log("[5/5] 写入匹配结果")
        _save_table(
            export_df,
            db_path=args.output_duckdb,
            table_name=args.output_table,
            output_mode=args.output_mode,
            duckdb_threads=args.duckdb_threads,
        )

        elapsed = time.time() - start
        print(f"matched rows: {len(export_df)}")
        print(f"saved duckdb table: {args.output_table} (mode={args.output_mode})")
        print(
            f"workers={args.workers}, chunk_size={args.chunk_size}, "
            f"duckdb_threads={args.duckdb_threads}, elapsed={elapsed:.1f}s"
        )
        return

    if args.command == "evaluate":
        from .matching_evaluator import evaluate_matches_parallel

        start = time.time()
        # 评估表需要包含 `top1_code`、`candidates`，若要计算准确率还需包含 `gold_code`。
        _log("[1/3] 加载评估数据")
        df = _load_table(
            args.result_duckdb,
            args.result_table,
            duckdb_threads=args.duckdb_threads,
            show_progress=args.progress,
        )
        _log(f"评估样本数: {len(df)}")

        _log("[2/3] 计算评估指标")
        report = evaluate_matches_parallel(
            df,
            workers=args.workers,
            chunk_size=args.chunk_size,
            show_progress=args.progress,
        )

        _log("[3/3] 输出报告")
        if args.output_report:
            Path(args.output_report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        elapsed = time.time() - start
        print(report)
        print(
            f"workers={args.workers}, chunk_size={args.chunk_size}, "
            f"duckdb_threads={args.duckdb_threads}, elapsed={elapsed:.1f}s"
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
