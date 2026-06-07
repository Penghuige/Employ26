"""Label Studio 人工标注数据集 → DuckDB 持久化脚本（v2 修正版）。

修正内容（相较于初版）:
- annotations 列：过滤 was_cancelled 和空 result 的草稿，仅保留已完成标注
- data 列：从 JSON 字符串展开为 25 个独立列（job_title / 5 candidates 等）
- 移除 12 个零信息/冗余列（meta, predictions, file_upload, allow_skip 等）
- 新增独立的扁平化标注表 annotations_v2（每行一条标注结果）

最终产出两张表：
    1. recruit.main.label_studio_tasks_v2      — 核心任务表（含展开的 data 列 + 已清洗的 annotations JSON）
    2. recruit.main.label_studio_annotations_v2 — 扁平化标注结果表（每行一条标注记录）

用法:
    .conda/python.exe src/test/test_json_to_db.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "project-4-at-2026-05-01-01-55-bca2dbbf.json"
DB_PATH = PROJECT_ROOT / "output" / "recruit.duckdb"
DB_PATH_FALLBACK = PROJECT_ROOT / "output" / "recruit_annotations.duckdb"

TARGET_TABLE_TASKS = "recruit.main.label_studio_tasks_v2"
TARGET_TABLE_ANNOTATIONS = "recruit.main.label_studio_annotations_v2"
TARGET_TABLE_TASKS_FB = "main.label_studio_tasks_v2"
TARGET_TABLE_ANNOTATIONS_FB = "main.label_studio_annotations_v2"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

# 核心任务表列（仅保留有实际业务价值的列 + 展开的 data 字段）
TASKS_COLUMNS: Dict[str, str] = {
    # 主键
    "id": "INTEGER",
    # 进度追踪
    "total_annotations": "INTEGER",
    "comment_count": "INTEGER",
    "created_at": "VARCHAR",
    "updated_at": "VARCHAR",
    "updated_by": "INTEGER",
    # ---- data JSON 展开为平铺列 ----
    "row_id": "INTEGER",
    "sample_source": "VARCHAR",
    "job_title": "VARCHAR",
    "job_requirements": "VARCHAR",
    "is_validation": "BOOLEAN",
    # 5 个候选: code / title / source（desc 字段过长且与被标注职业大典描述重复，省略）
    "cand_a_code": "VARCHAR",
    "cand_a_title": "VARCHAR",
    "cand_a_source": "VARCHAR",
    "cand_b_code": "VARCHAR",
    "cand_b_title": "VARCHAR",
    "cand_b_source": "VARCHAR",
    "cand_c_code": "VARCHAR",
    "cand_c_title": "VARCHAR",
    "cand_c_source": "VARCHAR",
    "cand_d_code": "VARCHAR",
    "cand_d_title": "VARCHAR",
    "cand_d_source": "VARCHAR",
    "cand_e_code": "VARCHAR",
    "cand_e_title": "VARCHAR",
    "cand_e_source": "VARCHAR",
    # ---- 清洗后的 annotations（仅含完成的标注）----
    "annotations_completed": "VARCHAR",
    # ---- 原始 data JSON（完整存档）----
    "data_raw": "VARCHAR",
}

TASKS_COLUMN_ORDER = list(TASKS_COLUMNS.keys())

# 扁平化标注表列
ANNOTATIONS_COLUMNS: Dict[str, str] = {
    "task_id": "INTEGER",
    "annotation_id": "INTEGER",
    "annotator_id": "INTEGER",
    "lead_time_sec": "DOUBLE",
    "is_completed": "BOOLEAN",
    "best_candidate": "VARCHAR",
    "soft_skill": "VARCHAR",
    "reason": "VARCHAR",
    "created_at": "VARCHAR",
}

ANNOTATIONS_COLUMN_ORDER = list(ANNOTATIONS_COLUMNS.keys())

# 应移除的列（对照：上轮分析中的零信息/冗余列）
REMOVED_COLUMNS = [
    "inner_id", "project", "allow_skip", "cancelled_annotations",
    "total_predictions", "unresolved_comment_count",
    "last_comment_updated_at", "file_upload",
    "meta", "drafts", "predictions", "comment_authors",
]


# ---------------------------------------------------------------------------
# 数据清洗
# ---------------------------------------------------------------------------

def _filter_completed_annotations(annotations_raw: list) -> list:
    """过滤标注列表，仅保留已完成的标注（排除草稿和取消的）。

    判定逻辑:
    - was_cancelled=False: 未被取消
    - result_count > 0 且 result 非空: 已填写标注内容
    - created_at 非空: 已提交（仅有 draft_created_at 的是草稿）

    Args:
        annotations_raw: Label Studio 导出的原始 annotations 列表。

    Returns:
        list: 仅含已完成标注的记录。
    """
    completed = []
    for a in annotations_raw:
        if a.get("was_cancelled", False):
            continue
        result = a.get("result", [])
        if not result:
            continue
        # 检查是否有有效的标注选择
        has_choice = any(
            r.get("value", {}).get("choices") or r.get("value", {}).get("text")
            for r in result
        )
        if not has_choice:
            continue
        # 已提交的标注有 created_at；仅有 draft_created_at 的是草稿
        if not a.get("created_at"):
            continue
        completed.append(a)
    return completed


def _extract_flat_annotation(annotation: dict, task_id: int) -> dict:
    """从单条标注记录中提取扁平化的标注结果。

    Args:
        annotation: 单条 Label Studio annotation 记录。
        task_id: 所属任务 ID。

    Returns:
        dict: 扁平化的标注结果（best_candidate, soft_skill, reason 等）。
    """
    best = ""
    soft = ""
    reason = ""
    for r in annotation.get("result", []):
        fn = r.get("from_name", "")
        val = r.get("value", {})
        if fn == "best_candidate_choice":
            best = ",".join(val.get("choices", []))
        elif fn == "no_softskill_choice":
            soft = ",".join(val.get("choices", []))
        elif fn == "choice_reason":
            text_arr = val.get("text", [])
            if text_arr:
                reason = str(text_arr[0])
    return {
        "task_id": task_id,
        "annotation_id": annotation.get("id"),
        "annotator_id": annotation.get("completed_by"),
        "lead_time_sec": annotation.get("lead_time", 0),
        "is_completed": True,
        "best_candidate": best,
        "soft_skill": soft,
        "reason": reason,
        "created_at": annotation.get("created_at", ""),
    }


def _flatten_data(data_raw: dict) -> dict:
    """将 data JSON 对象展开为 TASKS_COLUMNS 中定义的平铺字段。

    提取 5 个候选（A~E）的 code/title/source，跳过 desc（过长，且与职业大典重复）。

    Args:
        data_raw: Label Studio task 的 data 字段（dict）。

    Returns:
        dict: 展开后的平铺字段键值对。
    """
    flat: dict = {}
    flat["row_id"] = data_raw.get("row_id", 0)
    flat["sample_source"] = str(data_raw.get("sample_source", ""))
    flat["job_title"] = str(data_raw.get("job_title", ""))
    flat["job_requirements"] = str(data_raw.get("job_requirements_clean", ""))
    flat["is_validation"] = bool(data_raw.get("is_validation_sample", 0))

    for letter in ["a", "b", "c", "d", "e"]:
        flat[f"cand_{letter}_code"] = str(data_raw.get(f"candidate_{letter}_code", ""))
        flat[f"cand_{letter}_title"] = str(data_raw.get(f"candidate_{letter}_title", ""))
        flat[f"cand_{letter}_source"] = str(data_raw.get(f"candidate_{letter}_source", ""))

    return flat


# ---------------------------------------------------------------------------
# JSON 解析
# ---------------------------------------------------------------------------

def parse_records(input_path: Path) -> List[Dict[str, Any]]:
    """从 Label Studio JSON 导出文件中解析所有记录。

    使用手动大括号跟踪逐对象解析，兼容标准 json.load 失败的情况。

    Args:
        input_path: JSON 文件路径。

    Returns:
        List[Dict[str, Any]]: 解析出的所有 Label Studio task 记录。
    """
    logger.info("读取文件: %s", input_path)
    content = input_path.read_text(encoding="utf-8")
    total_chars = len(content)
    logger.info("文件大小: %.1f MB (%d 字符)", total_chars / 1024 / 1024, total_chars)

    records: List[Dict[str, Any]] = []
    depth = 0
    in_string = False
    escape_next = False
    obj_start = 0
    failed = 0

    for i in range(len(content)):
        c = content[i]
        if escape_next:
            escape_next = False
            continue
        if c == "\\":
            escape_next = True
            continue
        if c == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and obj_start > 0:
                obj_str = content[obj_start : i + 1]
                try:
                    records.append(json.loads(obj_str))
                except json.JSONDecodeError:
                    failed += 1

    logger.info("解析完成: %d 条有效记录, %d 条解析失败", len(records), failed)
    return records


# ---------------------------------------------------------------------------
# DuckDB 操作
# ---------------------------------------------------------------------------

def _connect_db() -> "duckdb.DuckDBPyConnection":
    """连接到 DuckDB，若目标文件被锁定则回退到独立文件。

    Returns:
        tuple: (连接, 实际数据库路径, 任务表全限定名, 标注表全限定名)。
    """
    import duckdb

    try:
        conn = duckdb.connect(str(DB_PATH))
        logger.info("已连接 DuckDB: %s", DB_PATH)
        return conn, DB_PATH, TARGET_TABLE_TASKS, TARGET_TABLE_ANNOTATIONS
    except duckdb.IOException:
        logger.warning("DuckDB 文件被锁定 (%s)，回退到独立文件", DB_PATH)
        fallback = DB_PATH_FALLBACK
        conn = duckdb.connect(str(fallback))
        logger.info("已连接 DuckDB (fallback): %s", fallback)
        conn.execute("CREATE SCHEMA IF NOT EXISTS main")
        return conn, fallback, TARGET_TABLE_TASKS_FB, TARGET_TABLE_ANNOTATIONS_FB


def create_tables(conn, tasks_table: str, annotations_table: str) -> None:
    """创建两张目标表（如已存在则先删除）。

    Args:
        conn: DuckDB 连接。
        tasks_table: 任务表全限定名。
        annotations_table: 标注表全限定名。
    """
    conn.execute("CREATE SCHEMA IF NOT EXISTS recruit.main")

    # 任务表
    col_defs = ", ".join(f'"{c}" {d}' for c, d in TASKS_COLUMNS.items())
    conn.execute(f"DROP TABLE IF EXISTS {tasks_table}")
    conn.execute(f"CREATE TABLE {tasks_table} ({col_defs})")

    # 标注表
    ann_col_defs = ", ".join(f'"{c}" {d}' for c, d in ANNOTATIONS_COLUMNS.items())
    conn.execute(f"DROP TABLE IF EXISTS {annotations_table}")
    conn.execute(f"CREATE TABLE {annotations_table} ({ann_col_defs})")

    logger.info("表创建成功: %s (%d列), %s (%d列)",
                 tasks_table, len(TASKS_COLUMNS),
                 annotations_table, len(ANNOTATIONS_COLUMNS))


def insert_all(conn, tasks_table: str, annotations_table: str, records: List[Dict[str, Any]]) -> tuple:
    """清洗并插入所有记录。

    Args:
        conn: DuckDB 连接。
        tasks_table: 任务表名。
        annotations_table: 标注表名。
        records: 原始解析记录。

    Returns:
        tuple: (任务表行数, 标注表行数, 总标注事件数, 完成标注数, 被过滤草稿数)。
    """
    task_rows: List[List[Any]] = []
    annotation_rows: List[List[Any]] = []
    total_ann_raw = 0
    completed_count = 0
    filtered_drafts = 0

    for rec in records:
        task_id = int(rec.get("id", 0))
        data_raw = rec.get("data", {})
        if isinstance(data_raw, str):
            try:
                data_raw = json.loads(data_raw)
            except json.JSONDecodeError:
                data_raw = {}

        # ---- 清洗 annotations ----
        anns_raw = rec.get("annotations", [])
        total_ann_raw += len(anns_raw)
        completed_anns = _filter_completed_annotations(anns_raw)
        completed_count += len(completed_anns)
        filtered_drafts += len(anns_raw) - len(completed_anns)

        # 生成扁平标注行
        for ann in completed_anns:
            flat = _extract_flat_annotation(ann, task_id)
            annotation_rows.append([flat[c] for c in ANNOTATIONS_COLUMN_ORDER])

        # ---- 构建任务行 ----
        flat_data = _flatten_data(data_raw)
        row = []
        for col in TASKS_COLUMN_ORDER:
            if col == "id":
                row.append(task_id)
            elif col in ("total_annotations", "comment_count"):
                row.append(rec.get(col, 0))
            elif col in ("created_at", "updated_at"):
                row.append(rec.get(col, ""))
            elif col == "updated_by":
                row.append(rec.get(col, 0))
            elif col == "annotations_completed":
                row.append(json.dumps(completed_anns, ensure_ascii=False) if completed_anns else "")
            elif col == "data_raw":
                row.append(json.dumps(data_raw, ensure_ascii=False))
            else:
                row.append(flat_data.get(col))
        task_rows.append(row)

    # ---- 写入 ----
    t_placeholders = ", ".join(["?"] * len(TASKS_COLUMN_ORDER))
    t_cols = ", ".join(f'"{c}"' for c in TASKS_COLUMN_ORDER)
    conn.executemany(f"INSERT INTO {tasks_table} ({t_cols}) VALUES ({t_placeholders})", task_rows)

    if annotation_rows:
        a_placeholders = ", ".join(["?"] * len(ANNOTATIONS_COLUMN_ORDER))
        a_cols = ", ".join(f'"{c}"' for c in ANNOTATIONS_COLUMN_ORDER)
        conn.executemany(f"INSERT INTO {annotations_table} ({a_cols}) VALUES ({a_placeholders})", annotation_rows)

    t_count = conn.execute(f"SELECT COUNT(*) FROM {tasks_table}").fetchone()[0]
    a_count = conn.execute(f"SELECT COUNT(*) FROM {annotations_table}").fetchone()[0]
    return t_count, a_count, total_ann_raw, completed_count, filtered_drafts


def verify(conn, tasks_table: str, annotations_table: str) -> None:
    """验证写入结果并打印统计摘要。

    Args:
        conn: DuckDB 连接。
        tasks_table: 任务表名。
        annotations_table: 标注表名。
    """
    print("\n" + "=" * 60)
    print("写入验证")
    print("=" * 60)

    # 任务表
    t_cnt = conn.execute(f"SELECT COUNT(*) FROM {tasks_table}").fetchone()[0]
    annotated_cnt = conn.execute(
        f"SELECT COUNT(*) FROM {tasks_table} WHERE annotations_completed != ''"
    ).fetchone()[0]
    val_cnt = conn.execute(
        f"SELECT COUNT(*) FROM {tasks_table} WHERE is_validation = true"
    ).fetchone()[0]
    print(f"\n[{tasks_table}]")
    print(f"  总任务: {t_cnt}")
    print(f"  已完成标注: {annotated_cnt}")
    print(f"  待标注: {t_cnt - annotated_cnt}")
    print(f"  验证集: {val_cnt}, 主集: {t_cnt - val_cnt}")

    # 标注表
    a_cnt = conn.execute(f"SELECT COUNT(*) FROM {annotations_table}").fetchone()[0]
    unique_tasks = conn.execute(
        f"SELECT COUNT(DISTINCT task_id) FROM {annotations_table}"
    ).fetchone()[0]
    print(f"\n[{annotations_table}]")
    print(f"  总标注记录: {a_cnt}")
    print(f"  涉及任务: {unique_tasks}")
    if a_cnt > 0:
        avg_lt = conn.execute(
            f"SELECT AVG(lead_time_sec) FROM {annotations_table} WHERE lead_time_sec > 0"
        ).fetchone()[0]
        print(f"  平均标注时长: {avg_lt:.1f}s" if avg_lt else "  平均标注时长: N/A")

    # 标注选择分布
    if a_cnt > 0:
        print(f"\n  best_candidate 分布:")
        for row in conn.execute(f"""
            SELECT best_candidate, COUNT(*) AS cnt
            FROM {annotations_table}
            WHERE best_candidate != ''
            GROUP BY 1 ORDER BY 2 DESC
        """).fetchall():
            print(f"    {row[0]}: {row[1]} 次")

    # 列信息
    print(f"\n列信息 [{tasks_table}]:")
    for row in conn.execute(f"DESCRIBE {tasks_table}").fetchall():
        print(f"  {row[0]:30s} {row[1]:12s}")

    print(f"\n列信息 [{annotations_table}]:")
    for row in conn.execute(f"DESCRIBE {annotations_table}").fetchall():
        print(f"  {row[0]:30s} {row[1]:12s}")

    # 样例
    print(f"\n样例任务 (前3条):")
    for row in conn.execute(f"""
        SELECT id, job_title, total_annotations,
               CASE WHEN annotations_completed != '' THEN '已标注' ELSE '待标注' END AS status
        FROM {tasks_table} LIMIT 3
    """).fetchall():
        print(f"  #{row[0]} {str(row[1])[:40]} ... ann={row[2]} [{row[3]}]")

    if a_cnt > 0:
        print(f"\n样例标注 (前3条):")
        for row in conn.execute(f"""
            SELECT task_id, annotator_id, best_candidate, soft_skill, lead_time_sec
            FROM {annotations_table} LIMIT 3
        """).fetchall():
            print(f"  task=#{row[0]} annotator=#{row[1]} best={row[2]} soft={row[3]} time={row[4]:.0f}s")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main() -> None:
    """主流程：解析 → 清洗 → 建表 → 写入 → 验证。"""
    if not INPUT_FILE.exists():
        logger.error("输入文件不存在: %s", INPUT_FILE)
        sys.exit(1)

    # 1. 解析
    records = parse_records(INPUT_FILE)
    if not records:
        logger.error("未解析到任何有效记录")
        sys.exit(1)

    # 2. 连接
    conn, actual_db, tasks_table, annotations_table = _connect_db()

    try:
        # 3. 建表
        create_tables(conn, tasks_table, annotations_table)

        # 4. 清洗并写入
        t_cnt, a_cnt, raw_ann, completed, drafts = insert_all(
            conn, tasks_table, annotations_table, records
        )

        # 5. 验证
        verify(conn, tasks_table, annotations_table)

        # 6. 数据质量报告
        print("\n" + "=" * 60)
        print("数据清洗报告")
        print("=" * 60)
        print(f"  原始标注记录总数: {raw_ann}")
        print(f"  已完成标注:        {completed}")
        print(f"  被过滤的草稿/取消:  {drafts} ({(drafts / max(raw_ann, 1)) * 100:.1f}%)")
        print(f"  移除的零信息列:    {len(REMOVED_COLUMNS)} 列")
        print(f"    ({', '.join(REMOVED_COLUMNS[:6])}...)")
        print(f"  任务表列数:        {len(TASKS_COLUMNS)} （初版 20 → 现 {len(TASKS_COLUMNS)}，含展开 data 列）")
        print(f"  标注表列数:        {len(ANNOTATIONS_COLUMNS)} （扁平化，每行一条标注）")

        # 7. 合并提示
        print(f"\n数据已持久化。")
        print(f"  任务表:     {tasks_table}")
        print(f"  标注表:     {annotations_table}")
        print(f"  数据库:     {actual_db}")
        if actual_db == DB_PATH_FALLBACK:
            print(f"\n  合并 SQL (关闭 VS Code DuckDB 连接后执行):")
            print(f"    ATTACH '{DB_PATH_FALLBACK}' AS src;")
            print(f"    INSERT INTO {TARGET_TABLE_TASKS} SELECT * FROM src.{TARGET_TABLE_TASKS_FB};")
            print(f"    INSERT INTO {TARGET_TABLE_ANNOTATIONS} SELECT * FROM src.{TARGET_TABLE_ANNOTATIONS_FB};")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
