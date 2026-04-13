import json
import os
import re
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .config import RAGConfig


_TITLE_FLAG_RE = re.compile(r"\s*(L/S|L|S)\s*$", re.IGNORECASE)
_TITLE_PAREN_RE = re.compile(r"^(?P<main>.+?)[（(](?P<subs>.+?)[）)]$")


def _pick_column(columns: List[str], candidates: Tuple[str, ...]) -> str:
    """从候选列名中挑选第一个存在的列。"""
    for name in candidates:
        if name in columns:
            return name
    return ""


def _normalize_text(value: Any) -> str:
    """统一清洗文本：去全角空格、统一换行、压缩多余空白。"""
    if value is None:
        return ""

    text = str(value)
    if text.lower() == "nan":
        return ""

    text = text.replace("\u3000", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def _normalize_code(value: Any) -> str:
    """职业代码标准化：去掉首尾空格、内部空白。"""
    code = _normalize_text(value)
    code = re.sub(r"\s+", "", code)
    return code


def _parse_title(title: str) -> Dict[str, Any]:
    """拆解职业标题。

    返回：
    - title: 清洗后的标题（去掉尾部 L/S 标记）
    - title_main: 主职业名
    - sub_titles: 括号中的细分工种/别名
    - title_flag: L / S / L/S
    """
    clean_title = _normalize_text(title)

    title_flag = ""
    m_flag = _TITLE_FLAG_RE.search(clean_title)
    if m_flag:
        title_flag = m_flag.group(1).upper()
        clean_title = clean_title[: m_flag.start()].strip()

    title_main = clean_title
    sub_titles: List[str] = []

    m_paren = _TITLE_PAREN_RE.match(clean_title)
    if m_paren:
        title_main = m_paren.group("main").strip()
        subs_raw = m_paren.group("subs").strip()
        sub_titles = [
            item.strip()
            for item in re.split(r"[、；;]", subs_raw)
            if item.strip()
        ]

    return {
        "title": clean_title,
        "title_main": title_main,
        "sub_titles": sub_titles,
        "title_flag": title_flag,
    }


def _split_task_items(tasks: str) -> List[str]:
    """把任务文本拆成条目。

    支持：
    - 1. xxx 2. xxx
    - 1、xxx 2、xxx
    - 多行任务
    若无法识别编号，则保留为单条。
    """
    clean_tasks = _normalize_text(tasks)
    if not clean_tasks:
        return []

    # 在编号前补换行，便于 split
    normalized = re.sub(r"\s*(\d+[\.、])\s*", r"\n\1 ", clean_tasks).strip()

    parts = []
    for seg in normalized.split("\n"):
        seg = seg.strip()
        if not seg:
            continue
        seg = re.sub(r"^\d+[\.、]\s*", "", seg).strip("；; ")
        if seg:
            parts.append(seg)

    return parts if parts else [clean_tasks]


def _is_other_bucket(title_main: str, desc: str) -> bool:
    """识别“其他类/未列入类”职业。"""
    if not title_main:
        return False

    if title_main.startswith("其他"):
        return True

    if "未列入" in title_main:
        return True

    if desc.startswith("指未列入") or "未列入" in desc:
        return True

    return False


def _build_record_search_text(
    code: str,
    title_main: str,
    title: str,
    desc: str,
    tasks: str,
    sub_titles: List[str],
    title_flag: str,
    is_other_bucket: bool,
) -> str:
    """构造记录级 search_text，用于粗粒度检索或调试。"""
    parts: List[str] = []

    if code:
        parts.append(f"职业代码：{code}")
    if title_main:
        parts.append(f"职业名称：{title_main}")
    elif title:
        parts.append(f"职业名称：{title}")

    if sub_titles:
        parts.append(f"细分工种：{'；'.join(sub_titles)}")

    if title_flag:
        parts.append(f"分类标记：{title_flag}")

    if is_other_bucket:
        parts.append("类别属性：其他类或未列入类")

    if desc:
        parts.append(f"职业定义：{desc}")

    if tasks:
        parts.append(f"主要任务：{tasks}")

    return "。".join(parts).strip()


def load_occupation_records(config: RAGConfig) -> List[Dict[str, Any]]:
    """加载 `中国职业大典.xlsx` 并转为结构化记录。

    每条记录包含：
    - code: 标准化后的职业代码
    - title: 清洗后的职业名称（不含尾部 L/S）
    - title_main: 主职业名
    - sub_titles: 括号中的细分工种/别名
    - title_flag: L / S / L/S
    - desc: 职业定义
    - tasks: 原始任务清洗文本
    - task_items: 拆分后的任务列表
    - is_other_bucket: 是否属于“其他/未列入类”
    - search_text: 用于粗粒度检索的拼接文本
    """
    if not os.path.exists(config.kb_excel_path):
        raise FileNotFoundError(f"知识库文件不存在: {config.kb_excel_path}")

    df = pd.read_excel(config.kb_excel_path, engine="openpyxl")
    df = df.fillna("")
    columns = [str(c).strip() for c in df.columns]
    df.columns = columns

    title_col = _pick_column(columns, config.title_candidates)
    code_col = _pick_column(columns, config.code_candidates)
    desc_col = _pick_column(columns, config.desc_candidates)
    task_col = _pick_column(columns, config.task_candidates)

    if not title_col or not code_col:
        raise ValueError(
            "职业大典缺少必要字段，至少需要职业名称和职业代码。"
            f"当前列: {columns}"
        )

    records: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        raw_title = row.get(title_col, "")
        raw_code = row.get(code_col, "")
        raw_desc = row.get(desc_col, "") if desc_col else ""
        raw_tasks = row.get(task_col, "") if task_col else ""

        code = _normalize_code(raw_code)
        desc = _normalize_text(raw_desc)
        tasks = _normalize_text(raw_tasks)

        parsed_title = _parse_title(str(raw_title))
        title = parsed_title["title"]
        title_main = parsed_title["title_main"]
        sub_titles = parsed_title["sub_titles"]
        title_flag = parsed_title["title_flag"]

        if not title or not code:
            continue

        task_items = _split_task_items(tasks)
        is_other_bucket = _is_other_bucket(title_main, desc)

        search_text = _build_record_search_text(
            code=code,
            title_main=title_main,
            title=title,
            desc=desc,
            tasks="；".join(task_items) if task_items else tasks,
            sub_titles=sub_titles,
            title_flag=title_flag,
            is_other_bucket=is_other_bucket,
        )

        records.append(
            {
                # 保持稳定，避免因过滤顺序变化导致 doc_id 漂移
                "doc_id": f"{code}__{idx}",
                "row_index": str(idx),
                "code": code,
                "title": title,
                "title_main": title_main,
                "sub_titles": sub_titles,
                "title_flag": title_flag,
                "desc": desc,
                "tasks": tasks,
                "task_items": task_items,
                "is_other_bucket": is_other_bucket,
                "search_text": search_text,
            }
        )

    return records


def build_chunks(config: RAGConfig, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把职业记录拆成两类 chunk：
    - definition: 用于职业名/定义/分类召回
    - task: 用于职责/JD 语义召回

    可选配置（不配也能跑）：
    - config.task_chunk_mode = "merged" | "item"
      - merged: 每个职业只生成一个 task chunk
      - item: 每个 task_item 生成一个 task chunk
    """
    task_chunk_mode = getattr(config, "task_chunk_mode", "merged")
    chunks: List[Dict[str, Any]] = []

    for record in records:
        common_meta = {
            "doc_id": record["doc_id"],
            "row_index": record["row_index"],
            "code": record["code"],
            "title": record["title"],
            "title_main": record["title_main"],
            "title_flag": record["title_flag"],
            "is_other_bucket": record["is_other_bucket"],
        }

        # 1) definition chunk
        definition_parts = [
            f"职业代码：{record['code']}",
            f"职业名称：{record['title_main'] or record['title']}",
        ]
        if record.get("sub_titles"):
            definition_parts.append(f"细分工种：{'；'.join(record['sub_titles'])}")
        if record.get("title_flag"):
            definition_parts.append(f"分类标记：{record['title_flag']}")
        if record.get("is_other_bucket"):
            definition_parts.append("类别属性：其他类或未列入类")
        if record.get("desc"):
            definition_parts.append(f"职业定义：{record['desc']}")

        definition_text = "。".join(part for part in definition_parts if part).strip()
        if definition_text:
            chunks.append(
                {
                    "chunk_id": f"{record['doc_id']}__def",
                    "chunk_type": "definition",
                    "text": definition_text,
                    "sub_titles": record.get("sub_titles", []),
                    **common_meta,
                }
            )

        # 2) task chunk
        task_items = record.get("task_items", [])
        if not task_items:
            continue

        if task_chunk_mode == "item":
            for i, task_item in enumerate(task_items):
                task_text = (
                    f"职业代码：{record['code']}。"
                    f"职业名称：{record['title_main'] or record['title']}。"
                    f"单项工作任务：{task_item}"
                )
                chunks.append(
                    {
                        "chunk_id": f"{record['doc_id']}__task_{i}",
                        "chunk_type": "task",
                        "task_index": i,
                        "text": task_text,
                        **common_meta,
                    }
                )
        else:
            # merged 模式：一个职业只生成一个任务 chunk
            task_text = (
                f"职业代码：{record['code']}。"
                f"职业名称：{record['title_main'] or record['title']}。"
                f"主要工作任务："
                + "；".join([f"{i + 1}. {item}" for i, item in enumerate(task_items)])
            )
            chunks.append(
                {
                    "chunk_id": f"{record['doc_id']}__task",
                    "chunk_type": "task",
                    "task_count": len(task_items),
                    "text": task_text,
                    **common_meta,
                }
            )

    return chunks


def _chunk_to_vectorstore_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """将 chunk 转成更适合向量库的 metadata。

    说明：
    - 向量库通常不希望 metadata 太大
    - list 类型在部分向量库里支持不好，因此转成字符串
    """
    metadata = {
        "chunk_id": str(chunk["chunk_id"]),
        "doc_id": str(chunk["doc_id"]),
        "row_index": str(chunk.get("row_index", "")),
        "chunk_type": str(chunk["chunk_type"]),
        "code": str(chunk.get("code", "")),
        "title": str(chunk.get("title", "")),
        "title_main": str(chunk.get("title_main", "")),
        "title_flag": str(chunk.get("title_flag", "")),
        "is_other_bucket": bool(chunk.get("is_other_bucket", False)),
    }

    if chunk.get("sub_titles"):
        metadata["sub_titles"] = "；".join(chunk["sub_titles"])

    if "task_index" in chunk:
        metadata["task_index"] = int(chunk["task_index"])

    if "task_count" in chunk:
        metadata["task_count"] = int(chunk["task_count"])

    return metadata


def prepare_embedding_payload(
    chunks: List[Dict[str, Any]],
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """把 chunks 转成向量库常见的三元组：
    - ids
    - texts
    - metadatas

    用法示例：
        ids, texts, metadatas = prepare_embedding_payload(chunks)
        vectors = embed_texts(texts, embed_fn)
        vector_store.add(ids=ids, documents=texts, embeddings=vectors, metadatas=metadatas)
    """
    ids: List[str] = []
    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for chunk in chunks:
        ids.append(str(chunk["chunk_id"]))
        texts.append(str(chunk["text"]))
        metadatas.append(_chunk_to_vectorstore_metadata(chunk))

    return ids, texts, metadatas


def embed_texts(
    texts: Sequence[str],
    embed_fn: Callable[[List[str]], List[List[float]]],
    batch_size: int = 64,
) -> List[List[float]]:
    """通用 embedding 批处理函数。

    embed_fn 约定：
        输入: List[str]
        输出: List[List[float]]

    这样你可以自由替换成任意 embedding 模型。
    """
    vectors: List[List[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        batch_vectors = embed_fn(batch)

        if len(batch_vectors) != len(batch):
            raise ValueError(
                "embedding 返回数量与输入数量不一致："
                f"输入 {len(batch)} 条，返回 {len(batch_vectors)} 条"
            )

        vectors.extend(batch_vectors)

    return vectors


def apply_metadata_filter(
    items: List[Dict[str, Any]],
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """对本地 records/chunks 做 metadata filter。

    支持三种写法：
    1. 精确匹配:
       {"chunk_type": "task"}

    2. 不等于:
       {"is_other_bucket": {"$ne": True}}

    3. in 集合:
       {"chunk_type": {"$in": ["definition", "task"]}}

    说明：
    - 这是一个本地 fallback 版本
    - 如果你用的是 Chroma / Milvus / Elasticsearch / pgvector，
      优先用它们自己的原生 filter
    """
    if not metadata_filter:
        return list(items)

    filtered: List[Dict[str, Any]] = []

    for item in items:
        matched = True

        for key, expected in metadata_filter.items():
            actual = item.get(key)

            if isinstance(expected, dict):
                if "$eq" in expected:
                    if actual != expected["$eq"]:
                        matched = False
                        break
                elif "$ne" in expected:
                    if actual == expected["$ne"]:
                        matched = False
                        break
                elif "$in" in expected:
                    if actual not in expected["$in"]:
                        matched = False
                        break
                else:
                    raise ValueError(
                        f"不支持的 metadata filter 操作: {expected}"
                    )
            else:
                if actual != expected:
                    matched = False
                    break

        if matched:
            filtered.append(item)

    return filtered


def save_metadata(
    config: RAGConfig,
    records: List[Dict[str, Any]],
    chunks: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """保存元数据，供检索阶段回查文本与结构化字段。"""
    os.makedirs(os.path.dirname(config.metadata_path), exist_ok=True)

    payload: Dict[str, Any] = {
        "config": asdict(config),
        "record_count": len(records),
        "records": records,
    }

    if chunks is not None:
        payload["chunk_count"] = len(chunks)
        payload["chunks"] = chunks

    with open(config.metadata_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_saved_payload(metadata_path: str) -> Dict[str, Any]:
    """加载完整 metadata payload。"""
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    return payload


def load_metadata(metadata_path: str) -> List[Dict[str, Any]]:
    """兼容旧接口：默认返回 records。"""
    payload = load_saved_payload(metadata_path)
    return payload.get("records", [])


def load_chunks(metadata_path: str) -> List[Dict[str, Any]]:
    """加载保存过的 chunks。"""
    payload = load_saved_payload(metadata_path)
    return payload.get("chunks", [])