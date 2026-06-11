"""结构化统计链路的公共路径、输出和表结构工具。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config.paths import get_project_paths


RUN_DATE_FORMAT = "%m-%d"


@dataclass(frozen=True)
class StructuredAnalysisPaths:
    """结构化统计链路使用的输入输出目录。"""

    project_root: Path
    integrated_dir: Path
    output_dir: Path


def build_structured_output_dir(
    run_date: datetime | None = None,
    *,
    base_output_dir: Path | None = None,
) -> Path:
    """构建结构化统计批次输出目录。

    Args:
        run_date: 可选运行日期；为空时使用当前时间。
        base_output_dir: 可选 `output/reports` 根目录。

    Returns:
        Path: `structured_analysis_{mm-dd}` 批次目录。
    """
    current = run_date or datetime.now()
    reports_root = base_output_dir or (get_project_paths().output_dir / "reports")
    return reports_root / f"structured_analysis_{current.strftime(RUN_DATE_FORMAT)}"


def resolve_structured_paths(
    *,
    base_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> StructuredAnalysisPaths:
    """解析结构化统计链路的项目目录、整合数据目录和输出目录。

    Args:
        base_dir: 可选项目根目录；主要用于测试或兼容旧调用。
        output_dir: 可选显式输出目录。

    Returns:
        StructuredAnalysisPaths: 结构化统计路径集合。
    """
    project_root = Path(base_dir) if base_dir is not None else get_project_paths().project_root
    integrated_dir = project_root / "output" / "integrated"
    resolved_output_dir = Path(output_dir) if output_dir is not None else build_structured_output_dir(
        base_output_dir=project_root / "output" / "reports"
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    return StructuredAnalysisPaths(
        project_root=project_root,
        integrated_dir=integrated_dir,
        output_dir=resolved_output_dir,
    )


def list_integrated_files(integrated_dir: Path) -> list[Path]:
    """列出结构化统计输入文件。"""
    return sorted(integrated_dir.glob("*_整合_*.csv"))


def load_integrated_data(
    integrated_dir: Path,
    *,
    required_columns: set[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """加载全部整合数据，并校验必需字段。

    Args:
        integrated_dir: `output/integrated` 目录。
        required_columns: 调用方需要的字段集合。

    Returns:
        tuple[pd.DataFrame, list[str]]: 合并后的数据和输入文件名列表。

    Raises:
        FileNotFoundError: 未找到整合 CSV。
        ValueError: 输入文件缺少必需字段。
    """
    csv_files = list_integrated_files(integrated_dir)
    if not csv_files:
        raise FileNotFoundError(f"未找到整合数据文件: {integrated_dir}")

    frames: list[pd.DataFrame] = []
    missing_by_file: dict[str, list[str]] = {}
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, encoding="utf-8", low_memory=False)
        if required_columns:
            missing = sorted(required_columns - set(df.columns))
            if missing:
                missing_by_file[csv_file.name] = missing
        frames.append(df)

    if missing_by_file:
        details = "; ".join(
            f"{filename}: {', '.join(columns)}" for filename, columns in missing_by_file.items()
        )
        raise ValueError(f"整合数据缺少必需字段: {details}")

    return pd.concat(frames, ignore_index=True), [path.name for path in csv_files]


def write_run_manifest(
    output_dir: Path,
    *,
    workflow: str,
    steps: list[str],
    params: dict[str, Any],
    input_files: list[str],
    output_files: list[str],
) -> Path:
    """写入结构化统计运行清单。"""
    manifest = {
        "workflow": workflow,
        "run_timestamp": datetime.now().isoformat(),
        "steps": steps,
        "params": params,
        "input_files": input_files,
        "output_files": output_files,
    }
    output_path = output_dir / "run_manifest.json"
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def write_csv_with_legacy_copy(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    canonical_filename: str,
    legacy_filename: str | None = None,
) -> list[str]:
    """写入规范 CSV，并按需保留历史中文文件名副本。"""
    output_paths: list[str] = []
    canonical_path = output_dir / canonical_filename
    df.to_csv(canonical_path, index=False, encoding="utf-8-sig")
    output_paths.append(canonical_filename)
    if legacy_filename and legacy_filename != canonical_filename:
        legacy_path = output_dir / legacy_filename
        df.to_csv(legacy_path, index=False, encoding="utf-8-sig")
        output_paths.append(legacy_filename)
    return output_paths
