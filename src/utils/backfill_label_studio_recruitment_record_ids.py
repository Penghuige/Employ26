"""为历史 Label Studio 任务回填 recruitment_record_id。"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text

from config.paths import get_project_paths
from src.db.postgres import create_pg_engine, ensure_schema
from src.db.recruitment_jobs_normalized import (
    RecruitmentNormalizedRow,
    build_dedupe_fingerprint,
    ensure_recruitment_jobs_normalized_table,
    quote_table_name,
    upsert_recruitment_jobs_normalized,
)


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

PROJECT_ROOT = get_project_paths().project_root
DEFAULT_TASK_TABLE = "annotations.label_studio_tasks_v2"
DEFAULT_NORMALIZED_TABLE = "public.recruitment_jobs_normalized"
DEFAULT_AUDIT_TABLE = "annotations.label_studio_task_rrid_backfill_audit"
BACKFILL_VERSION = "historical_annotation_rrid_backfill_v1"
MATCH_FIELDS = [
    "发布时间",
    "岗位名称",
    "工作城市",
    "薪资水平",
    "经验要求",
    "学历要求",
    "岗位描述",
    "公司名称",
    "公司规模",
    "公司行业",
]
SIMILARITY_AUTO_THRESHOLD = 0.97
SIMILARITY_MARGIN_THRESHOLD = 0.02


@dataclass(frozen=True)
class SnapshotResolution:
    """历史任务回放到导出快照后的结果。"""

    task_id: int
    historical_row_id: int
    snapshot_source: str
    snapshot_row_number: int
    payload: dict[str, Any]


@dataclass(frozen=True)
class SourceCandidate:
    """来自样本表的候选来源记录。"""

    source_table: str
    source_platform: str
    source_row_number: int
    payload: dict[str, Any]
    strict_key: tuple[str, ...]
    normalized_key: tuple[str, ...]


def normalize_text(value: object) -> str:
    """归一化文本以支持稳定比对。"""
    if value is None:
        return ""
    text_value = str(value).strip()
    if not text_value or text_value.lower() == "nan":
        return ""
    text_value = text_value.replace("\xa0", " ")
    text_value = re.sub(r"\s+", " ", text_value)
    return text_value


def normalize_for_key(value: object) -> str:
    """更强归一化，用于宽松唯一键。"""
    text_value = normalize_text(value).casefold()
    punctuation_map = str.maketrans({
        "：": ":",
        "，": ",",
        "；": ";",
        "（": "(",
        "）": ")",
    })
    text_value = text_value.translate(punctuation_map)
    text_value = re.sub(r"\s+", "", text_value)
    return text_value


def build_match_key(row: dict[str, Any], *, normalized: bool) -> tuple[str, ...]:
    """从招聘基础字段构建匹配键。"""
    builder = normalize_for_key if normalized else normalize_text
    return tuple(builder(row.get(field, "")) for field in MATCH_FIELDS)


def similarity_score(left: dict[str, Any], right: dict[str, Any]) -> float:
    """基于标题、公司和描述构建强文本相似度。"""
    left_text = "||".join(
        [
            normalize_for_key(left.get("岗位名称", "")),
            normalize_for_key(left.get("公司名称", "")),
            normalize_for_key(left.get("岗位描述", "")),
        ]
    )
    right_text = "||".join(
        [
            normalize_for_key(right.get("岗位名称", "")),
            normalize_for_key(right.get("公司名称", "")),
            normalize_for_key(right.get("岗位描述", "")),
        ]
    )
    return float(SequenceMatcher(None, left_text, right_text).ratio())


def load_csv_with_fallback(path: Path) -> pd.DataFrame:
    """容错读取历史 CSV。"""
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb2312", "latin-1"):
        try:
            return pd.read_csv(path, low_memory=False, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"无法解码文件: {path}")


def resolve_snapshot_row(
    task_id: int,
    historical_row_id: int,
    tier2_df: pd.DataFrame,
    tier3_df: pd.DataFrame,
) -> SnapshotResolution:
    """按历史导出规则回放任务对应的快照行。"""
    if historical_row_id < 0:
        raise ValueError(f"task_id={task_id} 的 historical_row_id 非法: {historical_row_id}")
    if historical_row_id < 30:
        if historical_row_id >= len(tier2_df):
            raise IndexError(f"task_id={task_id} 的 row_id 超出 Tier2 快照范围")
        payload = tier2_df.iloc[historical_row_id].to_dict()
        return SnapshotResolution(
            task_id=task_id,
            historical_row_id=historical_row_id,
            snapshot_source="tier2_validation",
            snapshot_row_number=historical_row_id,
            payload=payload,
        )

    tier3_row_number = historical_row_id - 30
    if tier3_row_number >= len(tier3_df):
        raise IndexError(f"task_id={task_id} 的 row_id 超出 Tier3 快照范围")
    payload = tier3_df.iloc[tier3_row_number].to_dict()
    return SnapshotResolution(
        task_id=task_id,
        historical_row_id=historical_row_id,
        snapshot_source="tier3_main",
        snapshot_row_number=tier3_row_number,
        payload=payload,
    )


def build_source_candidates(sample_frames: dict[str, pd.DataFrame]) -> list[SourceCandidate]:
    """将三家 sample 表预构建为可匹配候选。"""
    candidates: list[SourceCandidate] = []
    for source_table, dataframe in sample_frames.items():
        source_platform = source_table.split(".", 1)[0].strip('"')
        work_df = dataframe.copy().reset_index(drop=True)
        for row_number, (_, row) in enumerate(work_df.iterrows(), start=1):
            payload = row.to_dict()
            candidates.append(
                SourceCandidate(
                    source_table=source_table,
                    source_platform=source_platform,
                    source_row_number=row_number,
                    payload=payload,
                    strict_key=build_match_key(payload, normalized=False),
                    normalized_key=build_match_key(payload, normalized=True),
                )
            )
    return candidates


def choose_best_candidate(
    snapshot_payload: dict[str, Any],
    candidates: list[SourceCandidate],
) -> tuple[str, str, list[SourceCandidate], float | None, float | None]:
    """为快照行选择最佳来源候选。"""
    strict_key = build_match_key(snapshot_payload, normalized=False)
    strict_matches = [candidate for candidate in candidates if candidate.strict_key == strict_key]
    if len(strict_matches) == 1:
        return "AUTO_CONFIRMED", "exact_full_row_unique", strict_matches, None, None
    if len(strict_matches) > 1:
        source_tables = {candidate.source_table for candidate in strict_matches}
        if len(source_tables) == 1:
            selected = sorted(strict_matches, key=lambda candidate: candidate.source_row_number)[0]
            return "AUTO_CONFIRMED", "exact_duplicate_rows_same_source_table", [selected], None, None
        return "REVIEW_REQUIRED", "exact_full_row_ambiguous", strict_matches, None, None

    normalized_key = build_match_key(snapshot_payload, normalized=True)
    normalized_matches = [candidate for candidate in candidates if candidate.normalized_key == normalized_key]
    if len(normalized_matches) == 1:
        return "AUTO_CONFIRMED", "normalized_full_row_unique", normalized_matches, None, None
    if len(normalized_matches) > 1:
        return "REVIEW_REQUIRED", "normalized_full_row_ambiguous", normalized_matches, None, None

    narrowed = [
        candidate
        for candidate in candidates
        if normalize_for_key(candidate.payload.get("岗位名称", "")) == normalize_for_key(snapshot_payload.get("岗位名称", ""))
        and normalize_for_key(candidate.payload.get("公司名称", "")) == normalize_for_key(snapshot_payload.get("公司名称", ""))
    ]
    if not narrowed:
        return "UNMATCHED", "no_candidate", [], None, None

    scored = sorted(
        ((similarity_score(snapshot_payload, candidate.payload), candidate) for candidate in narrowed),
        key=lambda item: item[0],
        reverse=True,
    )
    best_score, best_candidate = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else None
    if best_score >= SIMILARITY_AUTO_THRESHOLD and (
        second_score is None or best_score - second_score >= SIMILARITY_MARGIN_THRESHOLD
    ):
        return "AUTO_CONFIRMED", "strong_text_similarity_unique", [best_candidate], best_score, second_score
    return "REVIEW_REQUIRED", "strong_text_similarity_ambiguous", [candidate for _, candidate in scored[:5]], best_score, second_score


def build_normalized_row(candidate: SourceCandidate) -> RecruitmentNormalizedRow:
    """将样本候选转换为统一规范层行。"""
    payload = candidate.payload
    return RecruitmentNormalizedRow(
        source_platform=candidate.source_platform,
        source_table=candidate.source_table,
        source_row_number=candidate.source_row_number,
        source_native_job_id="",
        dedupe_fingerprint=build_dedupe_fingerprint(
            source_platform=candidate.source_platform,
            company_name=normalize_text(payload.get("公司名称", "")),
            job_title=normalize_text(payload.get("岗位名称", "")),
            job_description_raw=normalize_text(payload.get("岗位描述", "")),
            publish_date=normalize_text(payload.get("发布时间", "")),
            work_city=normalize_text(payload.get("工作城市", "")),
        ),
        job_title=normalize_text(payload.get("岗位名称", "")),
        job_description_raw=normalize_text(payload.get("岗位描述", "")),
        work_city=normalize_text(payload.get("工作城市", "")),
        company_name=normalize_text(payload.get("公司名称", "")),
        publish_date=normalize_text(payload.get("发布时间", "")),
    )


def ensure_task_rrid_column(connection, task_table: str = DEFAULT_TASK_TABLE) -> None:
    """确保任务表存在 recruitment_record_id 列。"""
    connection.execute(
        text(
            f"""
            ALTER TABLE {task_table}
            ADD COLUMN IF NOT EXISTS recruitment_record_id text
            """
        )
    )


def ensure_audit_table(connection, audit_table: str = DEFAULT_AUDIT_TABLE) -> None:
    """确保审计表存在。"""
    ensure_schema(connection, "annotations")
    connection.execute(
        text(
            f"""
            CREATE TABLE IF NOT EXISTS {audit_table} (
                task_id integer PRIMARY KEY,
                historical_row_id integer NOT NULL,
                mapping_status text NOT NULL,
                mapping_rule text NOT NULL,
                confidence_tier text NOT NULL,
                source_table text,
                source_row_number bigint,
                recruitment_record_id text,
                candidate_count integer NOT NULL DEFAULT 0,
                best_similarity_score double precision,
                second_similarity_score double precision,
                evidence_summary text,
                backfill_version text NOT NULL,
                backfilled_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
    )


def load_sample_tables() -> dict[str, pd.DataFrame]:
    """从 PostgreSQL 读取三家 sample 表。"""
    engine = create_pg_engine()
    try:
        with engine.connect() as connection:
            result: dict[str, pd.DataFrame] = {}
            for source_table in ['"51job".sample', '"Liepin".sample', '"Zhilian".sample']:
                result[source_table] = pd.read_sql_query(text(f"select * from {source_table}"), connection)
            return result
    finally:
        engine.dispose()


def load_task_rows(task_table: str = DEFAULT_TASK_TABLE) -> list[dict[str, Any]]:
    """读取待回填任务。"""
    engine = create_pg_engine()
    try:
        with engine.connect() as connection:
            rows = connection.execute(
                text(
                    f"""
                    select id as task_id, row_id
                    from {task_table}
                    order by id
                    """
                )
            ).mappings()
            return [dict(row) for row in rows]
    finally:
        engine.dispose()


def build_evidence_summary(snapshot: SnapshotResolution, candidates: list[SourceCandidate]) -> str:
    """生成简短审计证据摘要。"""
    if not candidates:
        return f"snapshot={snapshot.snapshot_source}:{snapshot.snapshot_row_number}"
    top = candidates[0]
    return (
        f"snapshot={snapshot.snapshot_source}:{snapshot.snapshot_row_number}; "
        f"source={top.source_table}:{top.source_row_number}; "
        f"title={normalize_text(top.payload.get('岗位名称', ''))}; "
        f"company={normalize_text(top.payload.get('公司名称', ''))}"
    )


def run_backfill(
    *,
    dry_run: bool = False,
    task_table: str = DEFAULT_TASK_TABLE,
    normalized_table: str = DEFAULT_NORMALIZED_TABLE,
    audit_table: str = DEFAULT_AUDIT_TABLE,
) -> dict[str, int]:
    """执行历史任务 rrid 回填。"""
    tier2_df = load_csv_with_fallback(PROJECT_ROOT / "output" / "data5" / "Tier2_Matched_Data.csv")
    tier3_df = load_csv_with_fallback(PROJECT_ROOT / "output" / "data5" / "Tier3_Pending_Data.csv")
    sample_tables = load_sample_tables()
    candidates = build_source_candidates(sample_tables)
    task_rows = load_task_rows(task_table=task_table)

    audit_rows: list[dict[str, Any]] = []
    normalized_rows: list[RecruitmentNormalizedRow] = []
    auto_bindings: list[dict[str, Any]] = []

    for task in task_rows:
        task_id = int(task["task_id"])
        historical_row_id = int(task["row_id"])
        snapshot = resolve_snapshot_row(task_id, historical_row_id, tier2_df, tier3_df)
        mapping_status, mapping_rule, matched_candidates, best_score, second_score = choose_best_candidate(
            snapshot.payload,
            candidates,
        )

        recruitment_record_id = ""
        source_table = ""
        source_row_number: int | None = None
        confidence_tier = "LOW"

        if mapping_status == "AUTO_CONFIRMED" and matched_candidates:
            best_candidate = matched_candidates[0]
            normalized_rows.append(build_normalized_row(best_candidate))
            source_table = best_candidate.source_table
            source_row_number = best_candidate.source_row_number
            confidence_tier = "HIGH" if mapping_rule == "exact_full_row_unique" else "MEDIUM"
            auto_bindings.append(
                {
                    "task_id": task_id,
                    "source_table": source_table,
                    "source_row_number": source_row_number,
                }
            )
        elif mapping_status == "REVIEW_REQUIRED":
            confidence_tier = "MEDIUM"

        audit_rows.append(
            {
                "task_id": task_id,
                "historical_row_id": historical_row_id,
                "mapping_status": mapping_status,
                "mapping_rule": mapping_rule,
                "confidence_tier": confidence_tier,
                "source_table": source_table or None,
                "source_row_number": source_row_number,
                "recruitment_record_id": recruitment_record_id or None,
                "candidate_count": len(matched_candidates),
                "best_similarity_score": best_score,
                "second_similarity_score": second_score,
                "evidence_summary": build_evidence_summary(snapshot, matched_candidates),
                "backfill_version": BACKFILL_VERSION,
            }
        )

    if dry_run:
        return {
            "total_tasks": len(task_rows),
            "auto_confirmed": sum(1 for row in audit_rows if row["mapping_status"] == "AUTO_CONFIRMED"),
            "review_required": sum(1 for row in audit_rows if row["mapping_status"] == "REVIEW_REQUIRED"),
            "unmatched": sum(1 for row in audit_rows if row["mapping_status"] == "UNMATCHED"),
        }

    upsert_recruitment_jobs_normalized(normalized_rows, table_name=normalized_table)

    engine = create_pg_engine()
    try:
        with engine.begin() as connection:
            ensure_task_rrid_column(connection, task_table=task_table)
            ensure_audit_table(connection, audit_table=audit_table)
            ensure_recruitment_jobs_normalized_table(connection, table_name=normalized_table)

            locator_to_rrid = {
                (str(row[1]), int(row[2])): str(row[0])
                for row in connection.execute(
                    text(
                        f"""
                        select recruitment_record_id, source_table, source_row_number
                        from {quote_table_name(normalized_table)}
                        """
                    )
                ).fetchall()
            }

            for binding in auto_bindings:
                rrid = locator_to_rrid.get((binding["source_table"], int(binding["source_row_number"])))
                if not rrid:
                    continue
                connection.execute(
                    text(
                        f"""
                        update {task_table}
                        set recruitment_record_id = :recruitment_record_id
                        where id = :task_id
                        """
                    ),
                    {"recruitment_record_id": rrid, "task_id": binding["task_id"]},
                )
                for row in audit_rows:
                    if row["task_id"] == binding["task_id"]:
                        row["recruitment_record_id"] = rrid
                        break

            upsert_audit_sql = text(
                f"""
                insert into {audit_table} (
                    task_id,
                    historical_row_id,
                    mapping_status,
                    mapping_rule,
                    confidence_tier,
                    source_table,
                    source_row_number,
                    recruitment_record_id,
                    candidate_count,
                    best_similarity_score,
                    second_similarity_score,
                    evidence_summary,
                    backfill_version
                ) values (
                    :task_id,
                    :historical_row_id,
                    :mapping_status,
                    :mapping_rule,
                    :confidence_tier,
                    :source_table,
                    :source_row_number,
                    :recruitment_record_id,
                    :candidate_count,
                    :best_similarity_score,
                    :second_similarity_score,
                    :evidence_summary,
                    :backfill_version
                )
                on conflict (task_id) do update set
                    historical_row_id = excluded.historical_row_id,
                    mapping_status = excluded.mapping_status,
                    mapping_rule = excluded.mapping_rule,
                    confidence_tier = excluded.confidence_tier,
                    source_table = excluded.source_table,
                    source_row_number = excluded.source_row_number,
                    recruitment_record_id = excluded.recruitment_record_id,
                    candidate_count = excluded.candidate_count,
                    best_similarity_score = excluded.best_similarity_score,
                    second_similarity_score = excluded.second_similarity_score,
                    evidence_summary = excluded.evidence_summary,
                    backfill_version = excluded.backfill_version,
                    backfilled_at = now()
                """
            )
            for row in audit_rows:
                connection.execute(upsert_audit_sql, row)
    finally:
        engine.dispose()

    return {
        "total_tasks": len(task_rows),
        "auto_confirmed": sum(1 for row in audit_rows if row["mapping_status"] == "AUTO_CONFIRMED"),
        "review_required": sum(1 for row in audit_rows if row["mapping_status"] == "REVIEW_REQUIRED"),
        "unmatched": sum(1 for row in audit_rows if row["mapping_status"] == "UNMATCHED"),
    }


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""
    parser = argparse.ArgumentParser(description="回填历史 Label Studio 任务的 recruitment_record_id")
    parser.add_argument("--dry-run", action="store_true", help="只计算规则命中，不写数据库")
    return parser


def main() -> None:
    """CLI 入口。"""
    args = build_parser().parse_args()
    summary = run_backfill(dry_run=bool(args.dry_run))
    logger.info("回填完成: %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
