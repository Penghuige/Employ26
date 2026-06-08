"""job_title_parsing 配置与通用工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List
import json
import re
import warnings

from config.paths import load_database_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
DEFAULT_DATABASE_CONFIG_PATH = PROJECT_ROOT / "config" / "database.yaml"


def _parse_scalar(value: str) -> Any:
    """将 YAML 标量解析成 Python 值。"""
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
    """读取简单 YAML 配置。"""
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

        if value != "":
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


def load_database_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """加载数据库与表配置。"""
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


def normalize_text(text: Any) -> str:
    """基础文本清洗：统一空白、全角/半角常见符号。

    Args:
        text: 待清洗文本（支持 str/None）。

    Returns:
        str: 清洗后的文本。
    """
    text = "" if text is None else str(text)
    text = text.replace("\u3000", " ")
    text = text.replace("；", ";").replace("，", ",")
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("【", "[").replace("】", "]")
    text = re.sub(r"[\t\r\n]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_compact(text: Any) -> str:
    """用于精确比较的紧凑规范化文本，去除所有空白与分隔符后转小写。

    Args:
        text: 待处理文本。

    Returns:
        str: 紧凑规范化后的文本。
    """
    cleaned = normalize_text(text)
    cleaned = re.sub(r"[\s\-_/|]+", "", cleaned)
    return cleaned.lower()


def unique_keep_order(items: Iterable[str]) -> List[str]:
    """去重并保持原顺序。"""
    seen = set()
    result: List[str] = []
    for item in items:
        val = str(item).strip()
        if not val or val in seen:
            continue
        seen.add(val)
        result.append(val)
    return result


def read_json(path: str | Path) -> Dict[str, Any]:
    """读取 JSON 文件。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_text_lines(path: str | Path, warn_missing: bool = False) -> List[str]:
    """读取文本词典，忽略空行与注释。"""
    target = Path(path)
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    if not target.exists():
        if warn_missing:
            warnings.warn(f"词典文件不存在，将跳过加载: {target}", RuntimeWarning, stacklevel=2)
        return []
    results: List[str] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        results.append(line)
    return results


def load_stopwords(path: str | Path) -> List[str]:
    """读取停用词表。"""
    return read_text_lines(path)


def min_max_normalize(score_map: Dict[int, float]) -> Dict[int, float]:
    """对候选分数字典做 min-max 归一化到 [0, 1]。

    Args:
        score_map: {候选索引: 原始分数} 字典。

    Returns:
        Dict[int, float]: 归一化后的字典。若所有值相同则非零值映射为 1.0。
    """
    if not score_map:
        return {}
    values = list(score_map.values())
    min_v = min(values)
    max_v = max(values)
    if max_v == min_v:
        return {k: (1.0 if v > 0 else 0.0) for k, v in score_map.items()}
    return {k: (v - min_v) / (max_v - min_v) for k, v in score_map.items()}
