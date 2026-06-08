from __future__ import annotations

import json
import re
from pathlib import Path
import sys
from typing import Any, Dict, List

from config.paths import load_database_yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
DEFAULT_DATABASE_CONFIG_PATH = PROJECT_ROOT / "config" / "database.yaml"


def _parse_scalar(value: str) -> Any:
    """将简单 YAML 标量转换为 Python 值。"""
    text = value.strip()
    if text in {"", "null", "None"}:
        return ""
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return float(text)
    if (text.startswith("'") and text.endswith("'")) or (
        text.startswith('"') and text.endswith('"')
    ):
        return text[1:-1]
    return text


def simple_yaml_load(path: str | Path) -> Dict[str, Any]:
    """读取项目中使用的简单 YAML 配置。"""
    path = Path(path)
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    parsed_lines: List[tuple[int, str]] = []
    for raw in raw_lines:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        parsed_lines.append((indent, raw.strip()))

    root: Dict[str, Any] = {}
    stack: List[tuple[int, Any]] = [(-1, root)]

    for idx, (indent, line) in enumerate(parsed_lines):
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if line.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"YAML 列表缩进不合法: {line}")
            parent.append(_parse_scalar(line[2:]))
            continue

        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if value:
            parent[key] = _parse_scalar(value)
            continue

        next_is_list = False
        if idx + 1 < len(parsed_lines):
            next_indent, next_line = parsed_lines[idx + 1]
            next_is_list = next_indent > indent and next_line.startswith("- ")

        container: Any = [] if next_is_list else {}
        parent[key] = container
        stack.append((indent, container))

    return root


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """加载默认配置或指定配置文件。"""
    target = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    return simple_yaml_load(target)

def safe_text(value: object) -> str:
    """
    安全转字符串。
    
    输入: value: 任意输入值。
    返回：去除前后空白的字符串，如果输入为 None 或 "nan"（不区分大小写），返回空字符串。
    """
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def truncate_text(text: str, max_chars: int = 300) -> str:
    """
    截断上下文文本，控制序列长度。
    
    输入: text: 待处理文本；max_chars: 最大字符数，默认300。
    返回：如果文本长度超过 max_chars，返回前 max_chars 字符，否则返回原文本。
    """
    text = safe_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def load_database_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """
    加载数据库与表配置。
    
    输入: config_path: 可选的配置文件路径，默认为 None，此时使用默认路径。
    返回：包含数据库连接和表信息的字典。
    """
    loaded = load_database_yaml(config_path or DEFAULT_DATABASE_CONFIG_PATH)
    if not loaded:
        return {
            "database": {
                "host": "localhost",
                "port": 5432,
                "dbname": "Employ26",
                "user": "postgres",
                "password": "",
                "schema": "public",
                "duckdb_path": "output/recruit.duckdb",
                "duckdb_threads": 32,
            },
            "job_title_parsing": {
                "catalog_table": "public.occ_dict_detailed",
                "catalog_preprocessed_table": "public.occ_dict_pro",
                "jobs_table": [
                    '"Liepin".sample',
                    '"51job".sample',
                    '"Zhilian".sample',
                ],
                "match_result_table": "public.job_match_results",
            },
        }
    return loaded

def safe_print(value: object = "") -> None:
    """安全打印文本，避免 Windows 控制台编码限制导致程序崩溃。

    模型输出中可能包含当前终端代码页无法表示的字符。该函数会将无法编码的
    字符替换掉，而不是抛出 UnicodeEncodeError。

    参数：
        value: 任意需要转换为字符串并打印的对象。
    """
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, indent=2)
    else:
        text = str(value)
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="backslashreplace").decode(encoding))

