"""知识库构建（v2）：DuckDB 数据源 + 完整层级 + 双 chunk。

从 DuckDB 的 `chinese_occupational_dictionary_joined_preprocessed` 表
加载职业大典数据，保留完整层级（大类/中类/小类/细类），
生成 definition + task 双类型 chunk 用于向量检索。
"""

from __future__ import annotations

import ast
import json
import os
import re
from typing import Any, Dict, List, Optional

from .config import RAGConfig


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    text = text.replace("　", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def load_occupation_records(config: RAGConfig) -> List[Dict[str, Any]]:
    """从 DuckDB 加载职业大典记录（含完整层级信息）。

    Args:
        config: RAG 配置对象。

    Returns:
        List[Dict]: 每条记录含 code/title/desc/tasks/层级/检索文本等字段。
    """
    import duckdb

    conn = duckdb.connect(config.duckdb_path, read_only=True)
    try:
        rows = conn.execute(
            f"SELECT * FROM {config.catalog_table}"
        ).fetchall()
        col_names = [desc[0] for desc in conn.description]
    finally:
        conn.close()

    records: List[Dict[str, Any]] = []
    for row in rows:
        rec = dict(zip(col_names, row))

        # 解析 task_list（可能是 VARCHAR[] 或 JSON 字符串）
        task_list = rec.get("task_list", [])
        if isinstance(task_list, str):
            try:
                task_list = ast.literal_eval(task_list)
            except (ValueError, SyntaxError):
                try:
                    task_list = json.loads(task_list)
                except (json.JSONDecodeError, TypeError):
                    task_list = []

        # 解析 aliases
        aliases = rec.get("aliases", [])
        if isinstance(aliases, str):
            try:
                aliases = ast.literal_eval(aliases)
            except (ValueError, SyntaxError):
                try:
                    aliases = json.loads(aliases)
                except (json.JSONDecodeError, TypeError):
                    aliases = []

        # 层级字段
        hierarchy = {}
        for field in config.hierarchy_fields:
            val = rec.get(field, "")
            hierarchy[field] = _normalize_text(val)

        records.append({
            "code": _normalize_text(rec.get("code", "")),
            "title": _normalize_text(rec.get("title_clean", rec.get("title", ""))),
            "desc": _normalize_text(rec.get("desc_clean", rec.get("desc", ""))),
            "tasks": _normalize_text(rec.get("tasks", "")),
            "task_list": task_list,
            "aliases": aliases,
            "hierarchy": hierarchy,
            "hierarchy_text": _normalize_text(rec.get("hierarchy_text", "")),
            "retrieval_title_text": _normalize_text(rec.get("retrieval_title_text", "")),
            "retrieval_desc_text": _normalize_text(rec.get("retrieval_desc_text", "")),
            "retrieval_task_text": _normalize_text(rec.get("retrieval_task_text", "")),
        })

    return records


def build_chunks(config: RAGConfig, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """生成 definition + task 双类型 chunk。

    definition chunk: 职业名 + 别名 + 层级 + 定义，用于标题和粗粒度匹配。
    task chunk: 每个 task_item 独立一个 chunk，用于细粒度 JD 语义匹配。

    Args:
        config: RAG 配置。
        records: 职业大典记录。

    Returns:
        List[Dict]: 所有 chunk，含 chunk_id/chunk_type/text/hierarchy。
    """
    chunks: List[Dict[str, Any]] = []

    for idx, rec in enumerate(records):
        hierarchy = rec.get("hierarchy", {})
        common_meta = {
            "record_idx": idx,
            "code": rec["code"],
            "title": rec["title"],
            "hierarchy": hierarchy,
        }

        # ---- definition chunk ----
        def_parts = [
            f"职业代码：{rec['code']}",
            f"职业名称：{rec['title']}",
        ]
        if rec.get("hierarchy_text"):
            def_parts.append(f"层级路径：{rec['hierarchy_text']}")
        if rec.get("aliases"):
            def_parts.append(f"别名：{'；'.join(rec['aliases'])}")
        if rec.get("desc"):
            def_parts.append(f"职业定义：{rec['desc']}")

        def_text = "。".join(def_parts).strip()
        if def_text:
            chunks.append({
                "chunk_id": f"def_{idx}_{rec['code']}",
                "chunk_type": "definition",
                "text": def_text,
                **common_meta,
            })

        # ---- task chunks（每项一个） ----
        for ti, task_item in enumerate(rec.get("task_list", [])):
            task_text = (
                f"职业代码：{rec['code']}。"
                f"职业名称：{rec['title']}。"
                f"层级路径：{rec.get('hierarchy_text', '')}。"
                f"工作任务：{_normalize_text(task_item)}"
            )
            chunks.append({
                "chunk_id": f"task_{idx}_{rec['code']}_{ti}",
                "chunk_type": "task",
                "text": task_text,
                "task_index": ti,
                **common_meta,
            })

        # 如果没有 task_list，回退到 retrieval_task_text 整体作为一个 chunk
        if not rec.get("task_list") and rec.get("retrieval_task_text"):
            chunks.append({
                "chunk_id": f"task_{idx}_{rec['code']}_full",
                "chunk_type": "task",
                "text": (
                    f"职业代码：{rec['code']}。"
                    f"职业名称：{rec['title']}。"
                    f"层级路径：{rec.get('hierarchy_text', '')}。"
                    f"工作任务：{rec['retrieval_task_text']}"
                ),
                "task_index": 0,
                **common_meta,
            })

    return chunks


def save_metadata(config: RAGConfig, records: List[Dict], chunks: List[Dict]) -> None:
    """保存元数据到 JSON 文件。

    Args:
        config: RAG 配置。
        records: 原始记录。
        chunks: 所有 chunk。
    """
    os.makedirs(os.path.dirname(config.metadata_path), exist_ok=True)
    payload = {
        "record_count": len(records),
        "chunk_count": len(chunks),
        "def_chunk_count": sum(1 for c in chunks if c["chunk_type"] == "definition"),
        "task_chunk_count": sum(1 for c in chunks if c["chunk_type"] == "task"),
        "records": records,
        "chunks": chunks,
    }
    with open(config.metadata_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_metadata(metadata_path: str) -> Dict[str, Any]:
    """加载元数据 payload。

    Args:
        metadata_path: JSON 文件路径。

    Returns:
        Dict: 含 records, chunks 的完整 payload。
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)
