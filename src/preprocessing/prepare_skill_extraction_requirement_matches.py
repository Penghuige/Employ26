"""
技能词典流程预处理脚本。

这个脚本负责把 `occupation_skill_pipeline.py` 中原本隐式完成的两步前置工作显式化，并落库:

1. 使用 `src.preprocessing.parse_desc.parse_desc_df` 从岗位描述里切出 `任职要求_items_text`
2. 以“任职要求优先，RAG 匹配文本兜底”的方式构造职业匹配查询文本
3. 使用本地 BGE 微调模型 `D:\\model\\bge-base-zh-finetuned` 做职业细类语义匹配
4. 把解析结果与职业匹配结果统一写回 DuckDB

写入表名来自 `config/database.yaml`:

```yaml
skill_extraction:
  requirement_match_table: recruit.main.skill_extraction_requirement_matches
```

表中会同时保存:
- 岗位唯一标识 `sample_row_id`
- 岗位描述切分结果
- 任职要求文本
- 用于职业匹配的查询文本与来源
- 匹配得到的职业 code / 细类 / 大类 / 中类 / 小类
- Top-K 候选结果

运行示例:
- `python -m src.preprocessing.prepare_skill_extraction_requirement_matches`
- `python -m src.preprocessing.prepare_skill_extraction_requirement_matches --limit-job-rows 1000`
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import duckdb
import pandas as pd

from src.preprocessing.parse_desc import parse_desc_df
from src.skill_extraction.bge_matcher import OccupationBGEMatcher
from src.skill_extraction.config import SkillExtractionConfig, load_skill_extraction_config
from src.skill_extraction.history.data_source import OccupationSampleBuilder


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def _build_requirement_query_columns(parsed_df: pd.DataFrame) -> pd.DataFrame:
    """补充职业匹配查询文本列。

    规则:
    - 优先使用 `任职要求_items_text`
    - 若任职要求为空，则回退到 `RAG匹配文本`

    这样可以尽量把语义匹配建立在“岗位要求”上，而不是整段岗位描述上。
    """
    required_columns = {"任职要求_items_text", "RAG匹配文本"}
    missing_columns = required_columns.difference(parsed_df.columns)
    if missing_columns:
        raise KeyError(f"缺少构造职业匹配查询文本所需列: {sorted(missing_columns)}")

    work_df = parsed_df.copy()
    requirement_series = work_df["任职要求_items_text"].fillna("").astype(str).str.strip()
    rag_series = work_df["RAG匹配文本"].fillna("").astype(str).str.strip()

    use_requirement_mask = requirement_series != ""
    work_df["职业匹配文本"] = requirement_series.where(use_requirement_mask, rag_series)
    work_df["职业匹配来源"] = pd.Series(
        ["任职要求_items_text" if flag else "RAG匹配文本" for flag in use_requirement_mask],
        index=work_df.index,
    )
    return work_df


def _load_jobs_for_skill_extraction(
    config: SkillExtractionConfig,
    limit_job_rows: int | None = None,
) -> pd.DataFrame:
    """复用技能词典流程的数据加载逻辑，统一生成 `sample_row_id`。"""
    builder = OccupationSampleBuilder(config)
    return builder.load_jobs(limit_job_rows=limit_job_rows)


def build_requirement_match_dataframe(
    config: SkillExtractionConfig,
    limit_job_rows: int | None = None,
    parse_workers: int = 1,
    parse_batch_size: int = 2000,
    top_k: int | None = None,
) -> pd.DataFrame:
    """构建“任职要求切分 + 职业细类匹配”结果表。

    返回的 DataFrame 可以直接落库，也可以被 `occupation_skill_pipeline.py`
    继续用于采样与 prompt 生成。
    """
    logger.info("开始构建技能词典流程预处理结果")
    jobs_df = _load_jobs_for_skill_extraction(config=config, limit_job_rows=limit_job_rows)
    logger.info("已加载招聘样本: %s 行", len(jobs_df))

    parse_input_df = jobs_df[["岗位名称", "岗位描述"]].copy()
    parsed_df = parse_desc_df(
        parse_input_df,
        desc_col="岗位描述",
        batch_size=max(1, int(parse_batch_size)),
        num_workers=max(1, int(parse_workers)),
    )
    parsed_df = _build_requirement_query_columns(parsed_df)

    matcher = OccupationBGEMatcher(config)
    match_input_df = pd.concat(
        [
            jobs_df[["sample_row_id", "__source_table", "__source_row_number", "岗位名称", "岗位描述"]].reset_index(drop=True),
            parsed_df[
                [
                    "岗位描述_清洗",
                    "岗位描述_切分JSON",
                    "任职要求_items_text",
                    "岗位职责_items_text",
                    "unclassified_text",
                    "sections_brief",
                    "RAG匹配文本",
                    "RAG匹配来源",
                    "职业匹配文本",
                    "职业匹配来源",
                ]
            ].reset_index(drop=True),
        ],
        axis=1,
    )

    matched_df = matcher.match_requirement_texts(
        requirement_df=match_input_df,
        query_col="职业匹配文本",
        top_k=top_k,
    )

    result_df = pd.concat(
        [
            match_input_df.reset_index(drop=True),
            matched_df.reset_index(drop=True),
        ],
        axis=1,
    )

    # 业务层更常用 occupation_code 这个名字，这里额外保留一份别名列。
    result_df["occupation_code"] = result_df["top1_code"].fillna("").astype(str)
    result_df["occupation_title"] = result_df["top1_title"].fillna("").astype(str)

    logger.info(
        "技能词典流程预处理完成: 总岗位=%s, 成功匹配到职业细类=%s",
        len(result_df),
        int(result_df["is_matched"].fillna(False).sum()),
    )
    return result_df


def write_requirement_match_table(
    result_df: pd.DataFrame,
    config: SkillExtractionConfig,
) -> None:
    """把预处理结果写入 DuckDB 指定表。

    这里使用 `CREATE OR REPLACE TABLE`，原因是:
    - 该表本质上是可重复生成的中间数据集
    - 每次重跑希望与当前解析规则、当前 BGE 匹配结果保持一致
    """
    if result_df.empty:
        raise ValueError("result_df 为空，无法写入 DuckDB")

    logger.info("准备写入 DuckDB: %s", config.requirement_match_table)
    with duckdb.connect(str(config.db_path)) as conn:
        conn.execute(f"PRAGMA threads={config.duckdb_threads}")
        conn.register("tmp_skill_extraction_requirement_matches", result_df)
        conn.execute(
            f"""
            CREATE OR REPLACE TABLE {config.requirement_match_table} AS
            SELECT * FROM tmp_skill_extraction_requirement_matches
            """
        )
        conn.unregister("tmp_skill_extraction_requirement_matches")

    logger.info("已写入 DuckDB 表: %s", config.requirement_match_table)


def prepare_requirement_matches_to_duckdb(
    config: SkillExtractionConfig | None = None,
    database_config_path: str | Path | None = None,
    limit_job_rows: int | None = None,
    parse_workers: int = 1,
    parse_batch_size: int = 2000,
    top_k: int | None = None,
) -> pd.DataFrame:
    """对外暴露的一站式入口。

    既可被 CLI 调用，也可被 `occupation_skill_pipeline.py` 直接复用。
    """
    config = config or load_skill_extraction_config(database_config_path=database_config_path)
    result_df = build_requirement_match_dataframe(
        config=config,
        limit_job_rows=limit_job_rows,
        parse_workers=parse_workers,
        parse_batch_size=parse_batch_size,
        top_k=top_k,
    )
    write_requirement_match_table(result_df=result_df, config=config)
    return result_df


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""
    parser = argparse.ArgumentParser(description="技能词典流程预处理: 任职要求切分 + 职业细类匹配 + DuckDB 入库")
    parser.add_argument(
        "--database-config",
        default=None,
        help="数据库配置文件路径，默认使用 config/database.yaml",
    )
    parser.add_argument(
        "--limit-job-rows",
        type=int,
        default=None,
        help="仅用于调试，限制每张招聘表读取的行数",
    )
    parser.add_argument(
        "--parse-workers",
        type=int,
        default=1,
        help="岗位描述切分并发数",
    )
    parser.add_argument(
        "--parse-batch-size",
        type=int,
        default=2000,
        help="岗位描述切分批大小",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="职业匹配保留的候选数量，默认使用 skill_extraction 配置中的 match_top_k",
    )
    return parser


def main() -> None:
    """CLI 入口。"""
    parser = build_parser()
    args = parser.parse_args()
    prepare_requirement_matches_to_duckdb(
        database_config_path=args.database_config,
        limit_job_rows=args.limit_job_rows,
        parse_workers=args.parse_workers,
        parse_batch_size=args.parse_batch_size,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
