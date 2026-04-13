"""命令行工具（DuckDB 优先）。"""

from __future__ import annotations

import argparse
import json
import time
from os import cpu_count
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from .alias_builder import AliasBuilder
from .catalog_preprocessor import CatalogPreprocessor
from .hierarchy_keyword_builder import HierarchyKeywordBuilder
from .match_utils import load_config, load_database_config
from .matching_evaluator import evaluate_matches_parallel
from .matching_pipeline import MatchPipeline


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


def _log(message: str) -> None:
    """输出带时间戳的运行日志。"""
    print(f"[{time.strftime('%H:%M:%S')}] {message}")


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数。"""
    parser = argparse.ArgumentParser(description="岗位到《中国职业分类大典》初步匹配系统（DuckDB 优先）")
    sub = parser.add_subparsers(dest="command")

    prep = sub.add_parser("preprocess-catalog", help="预处理职业大典（默认读写 DuckDB）")
    prep.add_argument("--catalog-duckdb", default=DEFAULT_DB_PATH)
    prep.add_argument("--catalog-table", default=DEFAULT_CATALOG_TABLE)
    prep.add_argument("--catalog-csv", default="", help="仅小体量临时数据使用")
    prep.add_argument("--output-duckdb", default=DEFAULT_DB_PATH)
    prep.add_argument("--output-table", default=DEFAULT_CATALOG_PREPROCESSED_TABLE)
    prep.add_argument("--output-csv", default="", help="可选导出")
    prep.add_argument("--duckdb-threads", type=int, default=DEFAULT_DUCKDB_THREADS)
    prep.add_argument("--config", default="")
    prep.add_argument("--alias-dict", default="")

    build_dict = sub.add_parser("build-hierarchy-dict", help="从职业大典自动构建层级关键词词典")
    build_dict.add_argument("--catalog-duckdb", default=DEFAULT_DB_PATH)
    build_dict.add_argument("--catalog-table", default=DEFAULT_CATALOG_TABLE)
    build_dict.add_argument("--catalog-csv", default="", help="仅小体量临时数据使用")
    build_dict.add_argument("--output-dict", default="dicts/hierarchy_keyword_to_major_auto.txt")
    build_dict.add_argument("--top-n-per-major", type=int, default=80)
    build_dict.add_argument("--min-freq", type=int, default=5)
    build_dict.add_argument("--duckdb-threads", type=int, default=DEFAULT_DUCKDB_THREADS)

    match = sub.add_parser("match", help="批量匹配（默认读写 DuckDB）")
    match.add_argument("--catalog-duckdb", default=DEFAULT_DB_PATH)
    match.add_argument("--catalog-table", default=DEFAULT_CATALOG_TABLE)
    match.add_argument("--catalog-csv", default="", help="仅小体量临时数据使用")
    match.add_argument("--jobs-duckdb", default=DEFAULT_DB_PATH)
    match.add_argument("--jobs-table", default=DEFAULT_JOBS_TABLE, help="支持逗号分隔多张表")
    match.add_argument("--jobs-csv", default="", help="仅小体量临时数据使用")
    match.add_argument("--output-duckdb", default=DEFAULT_DB_PATH)
    match.add_argument("--output-table", default=DEFAULT_MATCH_RESULT_TABLE)
    match.add_argument("--output-mode", choices=["append", "replace"], default="replace")
    match.add_argument("--output-csv", default="", help="可选导出")
    match.add_argument("--job-title-col", default="岗位名称")
    match.add_argument("--job-desc-col", default="岗位描述")
    match.add_argument("--job-id-col", default="job_id")
    match.add_argument("--top-k", type=int, default=5)
    match.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="匹配并发线程数")
    match.add_argument("--chunk-size", type=int, default=256, help="匹配分块大小（每个并发任务处理的岗位数）")
    match.add_argument(
        "--executor-backend",
        choices=["thread", "process"],
        default="thread",
        help="并发执行后端（thread 默认；process 为后续优化预留）",
    )
    match.add_argument("--duckdb-threads", type=int, default=DEFAULT_DUCKDB_THREADS)
    match.add_argument("--config", default="")
    match.add_argument("--alias-dict", default="")
    match.add_argument("--debug", action="store_true")
    match.add_argument("--progress", action="store_true", help="显示匹配进度条")

    eval_cmd = sub.add_parser("evaluate", help="评估（支持 DuckDB 表或 CSV）")
    eval_cmd.add_argument("--result-duckdb", default=DEFAULT_DB_PATH)
    eval_cmd.add_argument("--result-table", default=DEFAULT_MATCH_RESULT_TABLE, help="支持逗号分隔多张表")
    eval_cmd.add_argument("--result-csv", default="")
    eval_cmd.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="评估并发进程数")
    eval_cmd.add_argument("--chunk-size", type=int, default=20000)
    eval_cmd.add_argument("--duckdb-threads", type=int, default=DEFAULT_DUCKDB_THREADS)
    eval_cmd.add_argument("--progress", action="store_true", help="显示评估进度条")
    eval_cmd.add_argument("--output-report", default="")

    return parser


def _split_table_names(table_name_expr: str) -> list[str]:
    """解析表名参数，支持逗号分隔多张表。"""
    return [name.strip() for name in str(table_name_expr).split(",") if name.strip()]


def _load_table(
    db_path: str,
    table_name: str,
    duckdb_threads: int = DEFAULT_DUCKDB_THREADS,
    show_progress: bool = False,
) -> pd.DataFrame:
    """从 DuckDB 读取一张或多张表为 DataFrame。"""
    import duckdb

    table_names = _split_table_names(table_name)
    if not table_names:
        raise ValueError("table_name 不能为空")

    conn = duckdb.connect(db_path)
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
    """将 DataFrame 保存到 DuckDB，支持 append/replace。"""
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
    """CLI 入口。"""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "preprocess-catalog":
        _log("初始化配置与预处理器")
        config = load_config(args.config or None)
        alias_builder = AliasBuilder(config, alias_dict_path=args.alias_dict or None)
        processor = CatalogPreprocessor(config, alias_builder)

        _log("加载职业大典数据")
        if args.catalog_csv:
            df = processor.load_csv(args.catalog_csv)
        else:
            df = processor.load_duckdb(
                db_path=args.catalog_duckdb,
                table_name=args.catalog_table,
            )

        _log(f"写入 DuckDB 表: {args.output_table}")
        _save_table(
            df,
            db_path=args.output_duckdb,
            table_name=args.output_table,
            output_mode="replace",
            duckdb_threads=args.duckdb_threads,
        )

        if args.output_csv:
            Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

        print(f"catalog rows: {len(df)}")
        print(f"saved duckdb table: {args.output_table}")
        return

    if args.command == "build-hierarchy-dict":
        _log("加载职业大典数据用于构建层级词典")
        if args.catalog_csv:
            catalog_df = pd.read_csv(args.catalog_csv, encoding="utf-8")
        else:
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
        start = time.time()
        _log("[1/5] 初始化匹配流程")
        pipeline = MatchPipeline(config_path=args.config or None, alias_dict_path=args.alias_dict or None)

        _log("[2/5] 加载职业大典并构建索引")
        if args.catalog_csv:
            pipeline.load_catalog_csv(args.catalog_csv)
        else:
            pipeline.load_catalog_duckdb(
                db_path=args.catalog_duckdb,
                table_name=args.catalog_table,
            )
        _log(f"职业大典条数: {len(pipeline.catalog_df)}")

        _log("[3/5] 加载岗位数据")
        if args.jobs_csv:
            jobs_df = pd.read_csv(args.jobs_csv, encoding="utf-8")
        else:
            jobs_df = _load_table(
                args.jobs_duckdb,
                args.jobs_table,
                duckdb_threads=args.duckdb_threads,
                show_progress=args.progress,
            )
        _log(f"岗位条数: {len(jobs_df)}")

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

        if args.output_csv:
            Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
            export_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

        elapsed = time.time() - start
        print(f"matched rows: {len(export_df)}")
        print(f"saved duckdb table: {args.output_table} (mode={args.output_mode})")
        print(
            f"workers={args.workers}, chunk_size={args.chunk_size}, executor_backend={args.executor_backend}, "
            f"duckdb_threads={args.duckdb_threads}, elapsed={elapsed:.1f}s"
        )
        return

    if args.command == "evaluate":
        start = time.time()
        _log("[1/3] 加载评估数据")
        if args.result_csv:
            df = pd.read_csv(args.result_csv, encoding="utf-8")
        else:
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
