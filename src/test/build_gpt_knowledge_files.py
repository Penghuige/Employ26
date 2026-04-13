#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_gpt_knowledge_files.py

把《中国职业大典.xlsx》整理成适合上传到 ChatGPT 自定义 GPT Knowledge 的 Markdown 文件。

功能：
1. 读取 Excel（默认首个 sheet）
2. 自动识别 code / title / desc / tasks 列
3. 清洗文本、标准化 code
4. 解析标题中的：
   - 主职业 title_main
   - 括号中的 sub_titles
   - 尾部标记 L / S / L/S
5. 把任务拆成 task_items
6. 识别“其他类/未列入类”
7. 生成 3 份知识文件 + 1 份 Instructions 模板 + 1 份 manifest
8. 自动按大小拆分成多个 part，避免单文件过大

运行示例：
python build_gpt_knowledge_files.py \
  --input /mnt/data/中国职业大典.xlsx \
  --output ./gpt_knowledge_out

依赖：
pip install pandas openpyxl
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


# ----------------------------
# 配置
# ----------------------------

@dataclass
class BuildConfig:
    input_excel_path: str
    output_dir: str

    # 如果你不确定列名，就保留默认候选
    title_candidates: Tuple[str, ...] = (
        "title", "职业名称", "职业名", "名称", "job_title"
    )
    code_candidates: Tuple[str, ...] = (
        "code", "职业代码", "代码", "职业编码", "job_code"
    )
    desc_candidates: Tuple[str, ...] = (
        "desc", "职业定义", "定义", "描述", "说明", "job_desc"
    )
    task_candidates: Tuple[str, ...] = (
        "tasks", "主要工作任务", "工作任务", "主要任务", "任务", "job_tasks"
    )

    # 控制每个导出 markdown 文件的近似最大字符数
    # 这是按字符数拆，不是 token 精确计数，但足够实用
    max_chars_per_file: int = 180_000

    # 是否导出“其他类/未列入类”
    include_other_bucket: bool = True

    # 是否导出 alias 文件
    export_alias_file: bool = True

    # markdown 标题级别
    top_heading_level: int = 1


# ----------------------------
# 文本清洗与解析
# ----------------------------

_TITLE_FLAG_RE = re.compile(r"\s*(L/S|L|S)\s*$", re.IGNORECASE)
_TITLE_PAREN_RE = re.compile(r"^(?P<main>.+?)[（(](?P<subs>.+?)[）)]$")
_TASK_PREFIX_RE = re.compile(r"^\d+[\.、]\s*")


def pick_column(columns: Sequence[str], candidates: Sequence[str]) -> str:
    for name in candidates:
        if name in columns:
            return name
    return ""


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value)
    if text.lower() == "nan":
        return ""

    text = text.replace("\u3000", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def normalize_code(value: Any) -> str:
    code = normalize_text(value)
    code = re.sub(r"\s+", "", code)
    return code


def parse_title(title: str) -> Dict[str, Any]:
    """
    拆分出：
    - title: 清洗后的职业名（去掉末尾 L/S 标记）
    - title_main: 主职业名
    - sub_titles: 括号中的细分工种/别名
    - title_flag: L / S / L/S
    """
    clean_title = normalize_text(title)

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


def split_task_items(tasks: str) -> List[str]:
    """
    把任务拆成列表，支持：
    - 1. xxx 2. xxx
    - 1、xxx 2、xxx
    - 多行任务
    """
    clean_tasks = normalize_text(tasks)
    if not clean_tasks:
        return []

    # 给编号前补换行
    normalized = re.sub(r"\s*(\d+[\.、])\s*", r"\n\1 ", clean_tasks).strip()

    parts: List[str] = []
    for seg in normalized.split("\n"):
        seg = seg.strip()
        if not seg:
            continue
        seg = _TASK_PREFIX_RE.sub("", seg).strip("；; ")
        if seg:
            parts.append(seg)

    return parts if parts else [clean_tasks]


def is_other_bucket(title_main: str, desc: str) -> bool:
    if not title_main:
        return False

    if title_main.startswith("其他"):
        return True
    if "未列入" in title_main:
        return True
    if desc.startswith("指未列入") or "未列入" in desc:
        return True

    return False


def escape_md(text: str) -> str:
    """
    轻量 Markdown 转义，避免标题和列表错乱。
    """
    if not text:
        return ""
    text = text.replace("\t", " ")
    return text


# ----------------------------
# 数据加载
# ----------------------------

def load_occupation_records(config: BuildConfig) -> List[Dict[str, Any]]:
    if not os.path.exists(config.input_excel_path):
        raise FileNotFoundError(f"输入文件不存在: {config.input_excel_path}")

    df = pd.read_excel(config.input_excel_path, engine="openpyxl")
    df = df.fillna("")

    columns = [str(c).strip() for c in df.columns]
    df.columns = columns

    title_col = pick_column(columns, config.title_candidates)
    code_col = pick_column(columns, config.code_candidates)
    desc_col = pick_column(columns, config.desc_candidates)
    task_col = pick_column(columns, config.task_candidates)

    if not title_col or not code_col:
        raise ValueError(
            "Excel 缺少必要字段，至少需要职业名称和职业代码。"
            f" 当前列为: {columns}"
        )

    records: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        raw_title = row.get(title_col, "")
        raw_code = row.get(code_col, "")
        raw_desc = row.get(desc_col, "") if desc_col else ""
        raw_tasks = row.get(task_col, "") if task_col else ""

        code = normalize_code(raw_code)
        desc = normalize_text(raw_desc)
        tasks = normalize_text(raw_tasks)

        parsed_title = parse_title(str(raw_title))
        title = parsed_title["title"]
        title_main = parsed_title["title_main"]
        sub_titles = parsed_title["sub_titles"]
        title_flag = parsed_title["title_flag"]

        if not code or not title:
            continue

        task_items = split_task_items(tasks)
        other_bucket = is_other_bucket(title_main, desc)

        if not config.include_other_bucket and other_bucket:
            continue

        records.append(
            {
                "doc_id": f"{code}__{idx}",
                "row_index": int(idx),
                "code": code,
                "title": title,
                "title_main": title_main,
                "sub_titles": sub_titles,
                "title_flag": title_flag,
                "desc": desc,
                "tasks": tasks,
                "task_items": task_items,
                "is_other_bucket": other_bucket,
            }
        )

    return records


# ----------------------------
# 渲染知识模板
# ----------------------------

def render_definition_record(record: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"## 职业：{escape_md(record['title_main'] or record['title'])}")
    lines.append("")
    lines.append(f"- 职业代码：{record['code']}")
    lines.append(f"- 标准职业名称：{escape_md(record['title_main'] or record['title'])}")

    if record.get("sub_titles"):
        lines.append(f"- 细分工种/别名：{'；'.join(escape_md(x) for x in record['sub_titles'])}")

    if record.get("title_flag"):
        lines.append(f"- 分类标记：{record['title_flag']}")

    lines.append(f"- 是否其他类：{'是' if record.get('is_other_bucket') else '否'}")

    if record.get("desc"):
        lines.append("")
        lines.append("### 职业定义")
        lines.append(escape_md(record["desc"]))

    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def render_task_record(record: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"## 职业：{escape_md(record['title_main'] or record['title'])}")
    lines.append("")
    lines.append(f"- 职业代码：{record['code']}")
    lines.append(f"- 标准职业名称：{escape_md(record['title_main'] or record['title'])}")

    if record.get("sub_titles"):
        lines.append(f"- 细分工种/别名：{'；'.join(escape_md(x) for x in record['sub_titles'])}")

    lines.append("")

    task_items = record.get("task_items", [])
    if task_items:
        lines.append("### 主要工作任务")
        for i, item in enumerate(task_items, start=1):
            lines.append(f"{i}. {escape_md(item)}")
    elif record.get("tasks"):
        lines.append("### 主要工作任务")
        lines.append(escape_md(record["tasks"]))
    else:
        lines.append("### 主要工作任务")
        lines.append("无")

    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def render_alias_record(record: Dict[str, Any]) -> Optional[str]:
    sub_titles = record.get("sub_titles", [])
    if not sub_titles:
        return None

    lines: List[str] = []
    lines.append(f"## 标准职业：{escape_md(record['title_main'] or record['title'])}")
    lines.append("")
    lines.append(f"- 职业代码：{record['code']}")
    lines.append(f"- 标准职业名称：{escape_md(record['title_main'] or record['title'])}")
    lines.append("- 可命中名称：")
    for alias in sub_titles:
        lines.append(f"  - {escape_md(alias)}")

    if record.get("desc"):
        lines.append("")
        lines.append("### 简要定义")
        lines.append(escape_md(record["desc"]))

    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def make_header(title: str, intro: str, heading_level: int = 1) -> str:
    marks = "#" * max(1, heading_level)
    return f"{marks} {title}\n\n{intro}\n\n---\n\n"


def split_sections_into_parts(
    header: str,
    sections: Iterable[str],
    max_chars_per_file: int,
) -> List[str]:
    """
    按记录边界拆分文件，避免切断单条职业记录。
    """
    parts: List[str] = []
    current = header

    for sec in sections:
        if not sec:
            continue

        # 如果单条记录本身过长，也至少单独成一个 part
        if len(current) + len(sec) > max_chars_per_file and current != header:
            parts.append(current)
            current = header + sec
        else:
            current += sec

    if current.strip():
        parts.append(current)

    return parts


# ----------------------------
# 文件写出
# ----------------------------

def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_markdown_parts(
    output_dir: Path,
    base_name: str,
    title: str,
    intro: str,
    sections: List[str],
    max_chars_per_file: int,
    heading_level: int,
) -> List[str]:
    header = make_header(title, intro, heading_level=heading_level)
    parts = split_sections_into_parts(header, sections, max_chars_per_file=max_chars_per_file)

    filenames: List[str] = []
    for i, part in enumerate(parts, start=1):
        filename = f"{base_name}_part{i:02d}.md"
        write_text(output_dir / filename, part)
        filenames.append(filename)

    return filenames


def build_instructions_template() -> str:
    return """# GPT Instructions 模板（可直接复制到自定义 GPT 的 Instructions）

你是“职业知识库助手”。回答时优先依据上传的知识文件，不要凭空补充。

当用户输入职业名称、职责描述、岗位 JD 或任职要求时，请按以下规则处理：

1. 先判断最可能对应的标准职业。
2. 优先参考“职业定义库”和“职业任务库”。
3. 输出：
   - 最可能匹配的标准职业名称
   - 职业代码
   - 匹配理由
4. 匹配理由尽量分成两部分：
   - 定义匹配：用户输入与职业定义的对应关系
   - 任务匹配：用户输入与主要工作任务的对应关系
5. 如果命中的是“其他类/未列入类”，必须明确提示：
   - “该结果属于兜底类职业，不一定是最精确的标准职业。”
6. 如果存在多个可能职业，最多给出 3 个候选，并说明差异。
7. 如果知识文件不足以支持判断，直接说“不确定”，并说明还缺什么信息。
8. 不要编造知识文件中不存在的职业代码、定义或任务。
9. 回答尽量使用知识文件中的原字段：
   - 职业代码
   - 标准职业名称
   - 细分工种/别名
   - 职业定义
   - 主要工作任务

建议回答格式：

【匹配结果】
- 标准职业名称：
- 职业代码：

【匹配理由】
- 定义匹配：
- 任务匹配：

【备注】
- 是否属于“其他类/未列入类”：
- 置信度：高 / 中 / 低
"""


def write_knowledge_files(config: BuildConfig, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) 定义文件
    definition_sections = [render_definition_record(r) for r in records]
    definition_files = write_markdown_parts(
        output_dir=output_dir,
        base_name="01_occupations_definition",
        title="职业定义知识库",
        intro=(
            "本文件用于提供标准职业名称、职业代码、职业定义、细分工种和分类标记。"
            "当需要理解某个职业是什么、属于哪类、别名有哪些时，优先参考本文件。"
        ),
        sections=definition_sections,
        max_chars_per_file=config.max_chars_per_file,
        heading_level=config.top_heading_level,
    )

    # 2) 任务文件
    task_sections = [render_task_record(r) for r in records]
    task_files = write_markdown_parts(
        output_dir=output_dir,
        base_name="02_occupations_tasks",
        title="职业任务知识库",
        intro=(
            "本文件用于提供各职业的主要工作任务。"
            "当需要根据岗位职责、JD、任职要求进行职业匹配时，优先参考本文件。"
        ),
        sections=task_sections,
        max_chars_per_file=config.max_chars_per_file,
        heading_level=config.top_heading_level,
    )

    # 3) 别名文件
    alias_files: List[str] = []
    if config.export_alias_file:
        alias_sections = []
        for r in records:
            sec = render_alias_record(r)
            if sec:
                alias_sections.append(sec)

        if alias_sections:
            alias_files = write_markdown_parts(
                output_dir=output_dir,
                base_name="03_occupations_aliases",
                title="职业别名与细分工种知识库",
                intro=(
                    "本文件用于把细分工种、别名、子职业映射回标准职业名称和职业代码。"
                    "当用户输入的不是标准职业名，而是具体工种或变体名称时，优先参考本文件。"
                ),
                sections=alias_sections,
                max_chars_per_file=config.max_chars_per_file,
                heading_level=config.top_heading_level,
            )

    # 4) Instructions 模板
    instruction_filename = "04_gpt_instructions_template.md"
    write_text(output_dir / instruction_filename, build_instructions_template())

    # 5) manifest
    manifest = {
        "config": asdict(config),
        "record_count": len(records),
        "generated_files": {
            "definition_files": definition_files,
            "task_files": task_files,
            "alias_files": alias_files,
            "instruction_template": instruction_filename,
        },
    }
    manifest_filename = "05_manifest.json"
    write_text(output_dir / manifest_filename, json.dumps(manifest, ensure_ascii=False, indent=2))

    return manifest


# ----------------------------
# 主函数
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将《中国职业大典.xlsx》整理成适合 GPT Knowledge 上传的 Markdown 文件。"
    )
    parser.add_argument(
        "--input",
        required=True,
        default=r"data\中国职业大典.xlsx",
        help="输入 Excel 路径，例如 /mnt/data/中国职业大典.xlsx",
    )
    parser.add_argument(
        "--output",
        required=True,
        default=r".\gpt_knowledge_out",
        help="输出目录，例如 ./gpt_knowledge_out",
    )
    parser.add_argument(
        "--max-chars-per-file",
        type=int,
        default=180000,
        help="单个输出 markdown 文件的近似最大字符数，默认 180000",
    )
    parser.add_argument(
        "--exclude-other-bucket",
        action="store_true",
        help="是否排除“其他类/未列入类”职业",
    )
    parser.add_argument(
        "--disable-alias-file",
        action="store_true",
        help="是否不导出别名/细分工种知识库",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = BuildConfig(
        input_excel_path=args.input,
        output_dir=args.output,
        max_chars_per_file=args.max_chars_per_file,
        include_other_bucket=not args.exclude_other_bucket,
        export_alias_file=not args.disable_alias_file,
    )

    records = load_occupation_records(config)
    manifest = write_knowledge_files(config, records)

    print("构建完成。")
    print(f"职业记录数: {len(records)}")
    print(f"输出目录: {config.output_dir}")
    print("生成文件:")
    for group_name, files in manifest["generated_files"].items():
        if isinstance(files, list):
            for f in files:
                print(f"  - [{group_name}] {f}")
        else:
            print(f"  - [{group_name}] {files}")


if __name__ == "__main__":
    main()