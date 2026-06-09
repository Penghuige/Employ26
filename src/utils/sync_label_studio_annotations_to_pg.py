#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""同步 Label Studio 标注数据到 PostgreSQL。

功能:
    1. 将 public schema 下名称中包含 `label` 或 `annotation` 的数据表迁移到
       `annotations` schema。
    2. 将 `project-4-at-2026-05-01-01-55-bca2dbbf.json` 导入
       `annotations.label_studio_annotations`。
       该表保留任务级原始 JSON 风格结构。
    3. 将 `project-4-at-2026-05-27-01-51-7cceb9ba.json` 导入
       `annotations.label_studio_tasks_v2` 与
       `annotations.label_studio_annotations_v2`。
       其中 `tasks_v2` 为任务级展开表，`annotations_v2` 为扁平标注表。
    4. 导入时基于主键去重，避免重复插入。

用法:
    python -m src.utils.sync_label_studio_annotations_to_pg
    python -m src.utils.sync_label_studio_annotations_to_pg --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from sqlalchemy import text

from config.paths import get_project_paths
from src.db.postgres import (
    create_pg_engine,
    ensure_schema,
    get_table_columns,
    resolve_pg_dbname,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

PROJECT_PATHS = get_project_paths()
PROJECT_ROOT = PROJECT_PATHS.project_root
ANNOTATIONS_SCHEMA = "annotations"

LEGACY_JSON_FILE = (
    PROJECT_ROOT / "data" / "annotations" / "project-4-at-2026-05-01-01-55-bca2dbbf.json"
)
V2_JSON_FILE = (
    PROJECT_ROOT / "data" / "annotations" / "project-4-at-2026-05-27-01-51-7cceb9ba.json"
)


def load_json_records(file_path: Path) -> list[dict[str, Any]]:
    """容错加载 Label Studio JSON 文件。"""
    raw_text = file_path.read_text(encoding="utf-8", errors="replace")

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        logger.warning("标准 JSON 解析失败，尝试容错恢复: %s", file_path)

        recovery_candidates: list[str] = []
        last_bracket = raw_text.rfind("]")
        if last_bracket >= 0:
            recovery_candidates.append(raw_text[: last_bracket + 1])

        last_task_boundary = raw_text.rfind('},{"id":')
        if last_task_boundary >= 0:
            recovery_candidates.append(raw_text[: last_task_boundary + 1] + "]")

        parsed = None
        last_error: Exception | None = exc
        for candidate in recovery_candidates:
            try:
                parsed = json.loads(candidate)
                break
            except json.JSONDecodeError as recovery_error:
                last_error = recovery_error

        if parsed is None:
            raise ValueError(f"容错恢复失败，无法解析文件: {file_path}") from last_error

    if not isinstance(parsed, list):
        raise ValueError(f"文件不是 JSON 数组格式: {file_path}")
    return [record for record in parsed if isinstance(record, dict)]
def move_public_tables_to_annotations(connection, dry_run: bool = False) -> list[str]:
    """将 public 中的标注相关表迁移到 annotations schema。"""
    rows = connection.execute(
        text(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_type = 'BASE TABLE'
              AND (
                lower(table_name) LIKE '%label%'
                OR lower(table_name) LIKE '%annotation%'
              )
            ORDER BY table_name
            """
        )
    ).fetchall()

    moved_tables: list[str] = []
    for row in rows:
        table_name = row[0]
        logger.info("准备迁移表: public.%s -> %s.%s", table_name, ANNOTATIONS_SCHEMA, table_name)
        moved_tables.append(table_name)
        if not dry_run:
            connection.execute(
                text(f'ALTER TABLE public."{table_name}" SET SCHEMA "{ANNOTATIONS_SCHEMA}"')
            )
    return moved_tables


def _compact_json(value: Any) -> str:
    """将任意 JSON 值压缩为紧凑字符串。"""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def normalize_legacy_annotations(record: dict[str, Any]) -> dict[str, Any]:
    """将旧版 JSON 记录转换为 legacy 表结构。"""
    return {
        "id": record.get("id"),
        "inner_id": record.get("inner_id"),
        "project": record.get("project"),
        "updated_by": record.get("updated_by"),
        "total_annotations": record.get("total_annotations"),
        "cancelled_annotations": record.get("cancelled_annotations"),
        "total_predictions": record.get("total_predictions"),
        "comment_count": record.get("comment_count"),
        "unresolved_comment_count": record.get("unresolved_comment_count"),
        "allow_skip": record.get("allow_skip"),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
        "last_comment_updated_at": record.get("last_comment_updated_at"),
        "file_upload": record.get("file_upload"),
        "annotations": _compact_json(record.get("annotations", [])),
        "data": _compact_json(record.get("data", {})),
        "meta": _compact_json(record.get("meta", {})),
        "drafts": _compact_json(record.get("drafts", [])),
        "predictions": _compact_json(record.get("predictions", [])),
        "comment_authors": _compact_json(record.get("comment_authors", [])),
    }


def filter_completed_annotations(annotations_raw: list[Any]) -> list[dict[str, Any]]:
    """过滤并保留已完成标注。"""
    completed: list[dict[str, Any]] = []
    for annotation in annotations_raw:
        if not isinstance(annotation, dict):
            continue
        if annotation.get("was_cancelled", False):
            continue
        result = annotation.get("result", [])
        if not result:
            continue
        has_effective_result = any(
            item.get("value", {}).get("choices") or item.get("value", {}).get("text")
            for item in result
            if isinstance(item, dict)
        )
        if not has_effective_result:
            continue
        if not annotation.get("created_at"):
            continue
        completed.append(annotation)
    return completed


def extract_flat_annotation(annotation: dict[str, Any], task_id: int) -> dict[str, Any]:
    """将单条 annotation 扁平化为 v2 标注表结构。"""
    best_candidate = ""
    soft_skill = ""
    reason = ""

    for result in annotation.get("result", []):
        if not isinstance(result, dict):
            continue
        from_name = result.get("from_name", "")
        value = result.get("value", {})
        if from_name == "best_candidate_choice":
            best_candidate = ",".join(value.get("choices", []))
        elif from_name == "no_softskill_choice":
            soft_skill = ",".join(value.get("choices", []))
        elif from_name == "choice_reason":
            text_values = value.get("text", [])
            if text_values:
                reason = str(text_values[0])

    return {
        "task_id": task_id,
        "annotation_id": annotation.get("id"),
        "annotator_id": annotation.get("completed_by"),
        "lead_time_sec": annotation.get("lead_time", 0),
        "is_completed": True,
        "best_candidate": best_candidate,
        "soft_skill": soft_skill,
        "reason": reason,
        "created_at": annotation.get("created_at", ""),
    }


def normalize_task_v2_record(record: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """将 v2 JSON 记录拆为任务表行和扁平标注表行。"""
    data = record.get("data", {}) if isinstance(record.get("data"), dict) else {}
    completed_annotations = filter_completed_annotations(record.get("annotations", []))

    task_row = {
        "id": record.get("id"),
        "total_annotations": record.get("total_annotations"),
        "comment_count": record.get("comment_count"),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
        "updated_by": record.get("updated_by"),
        "row_id": int(data.get("row_id", 0)) if str(data.get("row_id", "")).isdigit() else None,
        "recruitment_record_id": str(data.get("recruitment_record_id", "")),
        "sample_source": str(data.get("sample_source", "")),
        "job_title": str(data.get("job_title", "")),
        "job_requirements": str(data.get("job_requirements_clean", "")),
        "is_validation": str(data.get("is_validation_sample", "0")) == "1",
        "cand_a_code": str(data.get("candidate_a_code", "")),
        "cand_a_title": str(data.get("candidate_a_title", "")),
        "cand_a_source": str(data.get("candidate_a_source", "")),
        "cand_b_code": str(data.get("candidate_b_code", "")),
        "cand_b_title": str(data.get("candidate_b_title", "")),
        "cand_b_source": str(data.get("candidate_b_source", "")),
        "cand_c_code": str(data.get("candidate_c_code", "")),
        "cand_c_title": str(data.get("candidate_c_title", "")),
        "cand_c_source": str(data.get("candidate_c_source", "")),
        "cand_d_code": str(data.get("candidate_d_code", "")),
        "cand_d_title": str(data.get("candidate_d_title", "")),
        "cand_d_source": str(data.get("candidate_d_source", "")),
        "cand_e_code": str(data.get("candidate_e_code", "")),
        "cand_e_title": str(data.get("candidate_e_title", "")),
        "cand_e_source": str(data.get("candidate_e_source", "")),
        "annotations_completed": _compact_json(completed_annotations),
        "data_raw": _compact_json(data),
    }

    annotation_rows = [
        extract_flat_annotation(annotation, int(record["id"]))
        for annotation in completed_annotations
        if record.get("id") is not None and annotation.get("id") is not None
    ]

    return task_row, annotation_rows


def deduplicate_by_key(
    rows: list[dict[str, Any]],
    key_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    """按给定主键字段去重，保留最后一条。"""
    deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        if any(value is None for value in key):
            continue
        deduped[key] = row
    return list(deduped.values())


def resolve_table_schema(
    connection,
    preferred_schema: str,
    table_name: str,
) -> str:
    """解析目标表当前实际所在的 schema。"""
    for schema_name in [preferred_schema, "public"]:
        if get_table_columns(connection, schema_name, table_name):
            return schema_name
    raise ValueError(f"目标表不存在或无字段: {preferred_schema}.{table_name}")


def filter_rows_for_existing_columns(
    rows: list[dict[str, Any]],
    available_columns: list[str],
) -> list[dict[str, Any]]:
    """按现有表字段裁剪待插入行。"""
    return [
        {column: row.get(column) for column in available_columns}
        for row in rows
    ]


def get_existing_keys(
    connection,
    schema_name: str,
    table_name: str,
    key_fields: tuple[str, ...],
) -> set[tuple[Any, ...]]:
    """读取目标表当前已有的去重键集合。"""
    key_sql = ", ".join(f'"{field}"' for field in key_fields)
    rows = connection.execute(
        text(f'SELECT {key_sql} FROM "{schema_name}"."{table_name}"')
    ).fetchall()
    return {tuple(row) for row in rows}


def insert_rows(
    connection,
    schema_name: str,
    table_name: str,
    rows: list[dict[str, Any]],
    conflict_fields: tuple[str, ...],
    dry_run: bool = False,
) -> int:
    """向现有表批量插入数据，并按主键字段去重。"""
    if not rows:
        return 0

    actual_schema = resolve_table_schema(connection, schema_name, table_name)
    available_columns = get_table_columns(connection, actual_schema, table_name)

    existing_keys = get_existing_keys(connection, actual_schema, table_name, conflict_fields)
    candidate_rows = [
        row
        for row in rows
        if tuple(row.get(field) for field in conflict_fields) not in existing_keys
    ]
    prepared_rows = filter_rows_for_existing_columns(candidate_rows, available_columns)
    insert_columns = ", ".join(f'"{column}"' for column in available_columns)
    insert_values = ", ".join(f":{column}" for column in available_columns)

    if dry_run:
        return 0

    sql = text(
        f"""
        INSERT INTO "{actual_schema}"."{table_name}" ({insert_columns})
        VALUES ({insert_values})
        """
    )
    inserted_count = 0
    for row in prepared_rows:
        connection.execute(sql, row)
        inserted_count += 1
    return inserted_count


def sync_legacy_annotations(connection, dry_run: bool = False) -> tuple[int, int, int]:
    """同步旧版任务级标注表 `label_studio_annotations`。"""
    records = load_json_records(LEGACY_JSON_FILE)
    rows = [normalize_legacy_annotations(record) for record in records if record.get("id") is not None]
    rows = deduplicate_by_key(rows, ("id",))
    inserted = insert_rows(
        connection=connection,
        schema_name=ANNOTATIONS_SCHEMA,
        table_name="label_studio_annotations",
        rows=rows,
        conflict_fields=("id",),
        dry_run=dry_run,
    )
    return len(records), len(rows), inserted


def sync_v2_annotations(connection, dry_run: bool = False) -> tuple[int, int, int, int]:
    """同步 v2 任务表和扁平标注表。"""
    records = load_json_records(V2_JSON_FILE)

    task_rows: list[dict[str, Any]] = []
    annotation_rows: list[dict[str, Any]] = []
    for record in records:
        if record.get("id") is None:
            continue
        task_row, flat_rows = normalize_task_v2_record(record)
        task_rows.append(task_row)
        annotation_rows.extend(flat_rows)

    task_rows = deduplicate_by_key(task_rows, ("id",))
    annotation_rows = deduplicate_by_key(annotation_rows, ("task_id", "annotation_id"))

    inserted_tasks = insert_rows(
        connection=connection,
        schema_name=ANNOTATIONS_SCHEMA,
        table_name="label_studio_tasks_v2",
        rows=task_rows,
        conflict_fields=("id",),
        dry_run=dry_run,
    )
    inserted_annotations = insert_rows(
        connection=connection,
        schema_name=ANNOTATIONS_SCHEMA,
        table_name="label_studio_annotations_v2",
        rows=annotation_rows,
        conflict_fields=("task_id", "annotation_id"),
        dry_run=dry_run,
    )
    return len(records), len(task_rows), inserted_tasks, inserted_annotations


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="同步 Label Studio 标注数据到 PostgreSQL")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将要执行的迁移和导入，不真正写入数据库",
    )
    args = parser.parse_args()

    resolved_dbname = resolve_pg_dbname()
    logger.info("已连接目标数据库: %s", resolved_dbname)
    engine = create_pg_engine(resolved_dbname)

    with engine.begin() as connection:
        ensure_schema(connection, ANNOTATIONS_SCHEMA)
        moved_tables = move_public_tables_to_annotations(connection, dry_run=args.dry_run)
        if moved_tables:
            logger.info("迁移完成/待迁移表数: %s", len(moved_tables))

        legacy_source, legacy_deduped, legacy_inserted = sync_legacy_annotations(
            connection=connection,
            dry_run=args.dry_run,
        )
        logger.info(
            "legacy 导入结果 -> %s.label_studio_annotations: 原始=%s, 去重后=%s, 实际插入=%s",
            ANNOTATIONS_SCHEMA,
            legacy_source,
            legacy_deduped,
            legacy_inserted,
        )

        v2_source, v2_tasks, v2_inserted_tasks, v2_inserted_annotations = sync_v2_annotations(
            connection=connection,
            dry_run=args.dry_run,
        )
        logger.info(
            "v2 导入结果 -> %s.label_studio_tasks_v2 / label_studio_annotations_v2: 原始任务=%s, 去重后任务=%s, 实际插入任务=%s, 实际插入标注=%s",
            ANNOTATIONS_SCHEMA,
            v2_source,
            v2_tasks,
            v2_inserted_tasks,
            v2_inserted_annotations,
        )

    logger.info("脚本执行完成。")


if __name__ == "__main__":
    main()
