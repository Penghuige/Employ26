"""
职业细类采样数据访问模块。

职责：
1. 从 DuckDB 读取招聘样本；
2. 使用 BGE 模型将岗位匹配到职业细类；
3. 生成训练集和验证池清单；
4. 按需回查某一批样本的原始岗位描述。
"""

from __future__ import annotations

from hashlib import md5
import logging
from typing import Dict, List, Tuple

import duckdb
import pandas as pd

from ..bge_matcher import OccupationBGEMatcher
from ..config import SkillExtractionConfig


logger = logging.getLogger(__name__)


def _safe_filename(detail_path: str) -> str:
    """生成适合保存 prompt 的文件名。"""
    digest = md5(detail_path.encode("utf-8")).hexdigest()[:10]
    compact_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in detail_path)[:40]
    compact_name = compact_name.strip("_") or "detail"
    return f"{digest}_{compact_name}"


def _stable_seed(seed: int, detail_path: str) -> int:
    """为每个细类生成稳定的采样随机种子。"""
    digest = md5(detail_path.encode("utf-8")).hexdigest()[:8]
    return int(seed) + int(digest, 16)


class OccupationSampleBuilder:
    """职业细类样本构建器。"""

    def __init__(self, config: SkillExtractionConfig):
        self.config = config
        self._matcher: OccupationBGEMatcher | None = None

    @property
    def matcher(self) -> OccupationBGEMatcher:
        """按需初始化 BGE 匹配器。"""
        if self._matcher is None:
            self._matcher = OccupationBGEMatcher(self.config)
        return self._matcher

    def load_jobs(self, limit_job_rows: int | None = None) -> pd.DataFrame:
        """加载招聘样本表。"""
        logger.info("开始加载招聘样本表")
        frames: List[pd.DataFrame] = []

        with duckdb.connect(str(self.config.db_path)) as conn:
            conn.execute(f"PRAGMA threads={self.config.duckdb_threads}")
            for table_name in self.config.jobs_tables:
                query = f"""
                    SELECT
                        *,
                        ROW_NUMBER() OVER () AS __source_row_number
                    FROM {table_name}
                """
                if limit_job_rows is not None:
                    query += f" LIMIT {int(limit_job_rows)}"
                frame = conn.execute(query).df()
                frame["__source_table"] = table_name
                frames.append(frame)
                logger.info("已加载 %s: %s 行", table_name, len(frame))

        jobs_df = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()
        if jobs_df.empty:
            raise ValueError("未从 DuckDB 读取到招聘样本数据")

        jobs_df["sample_row_id"] = (
            jobs_df["__source_table"].astype(str)
            + ":"
            + jobs_df["__source_row_number"].astype(str)
        )
        return jobs_df

    def match_jobs(
        self,
        jobs_df: pd.DataFrame,
        match_workers: int = 1,
        match_chunk_size: int = 256,
        top_k: int = 5,
        show_progress: bool = False,
    ) -> pd.DataFrame:
        """把招聘样本匹配到职业细类。"""
        del match_workers
        del match_chunk_size
        del show_progress

        if jobs_df.empty:
            return pd.DataFrame()

        logger.info(
            "开始使用 BGE 职业细类匹配: model=%s, top_k=%s",
            self.config.embedding_model_path,
            top_k,
        )

        matched_df = self.matcher.match_jobs(
            jobs_df[["岗位名称", "岗位描述"]].copy(),
            top_k=top_k,
        )

        enriched_df = pd.concat(
            [
                jobs_df[
                    [
                        "sample_row_id",
                        "__source_table",
                        "__source_row_number",
                        "岗位名称",
                        "岗位描述",
                    ]
                ].reset_index(drop=True),
                matched_df.reset_index(drop=True),
            ],
            axis=1,
        )
        enriched_df["prompt_file_key"] = enriched_df["detail_path"].fillna("").astype(str).map(_safe_filename)
        logger.info("职业细类匹配完成: %s 条岗位，%s 条成功匹配到细类", len(enriched_df), int(enriched_df["is_matched"].sum()))
        return enriched_df

    def build_sampling_manifests(
        self,
        matched_df: pd.DataFrame,
        train_size: int,
        seed: int,
        limit_categories: int | None = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """按职业细类构建训练集与验证池。"""
        if matched_df.empty:
            raise ValueError("matched_df 为空，无法采样")

        work_df = matched_df.copy()
        work_df = work_df[work_df["is_matched"] == True].copy()  # noqa: E712
        work_df = work_df[work_df["detail_path"].fillna("").astype(str).str.strip() != ""].copy()
        if work_df.empty:
            raise ValueError("没有成功匹配到职业细类的数据，无法构建采样清单")

        summary_rows: List[Dict] = []
        manifest_rows: List[Dict] = []

        grouped = (
            work_df.groupby("detail_path", dropna=False)
            .size()
            .reset_index(name="available_count")
            .sort_values(["available_count", "detail_path"], ascending=[False, True])
        )
        if limit_categories is not None:
            grouped = grouped.head(int(limit_categories))

        selected_detail_paths = set(grouped["detail_path"].tolist())
        work_df = work_df[work_df["detail_path"].isin(selected_detail_paths)].copy()

        for detail_path, group in work_df.groupby("detail_path", sort=False):
            random_state = _stable_seed(seed=seed, detail_path=str(detail_path))
            shuffled = group.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

            train_count = min(int(train_size), len(shuffled))
            train_part = shuffled.iloc[:train_count].copy()
            validation_part = shuffled.iloc[train_count:].copy()

            detail_name = str(shuffled["detail_name"].iloc[0]).strip() or "未命名细类"
            major = str(shuffled["大类"].iloc[0]).strip()
            middle = str(shuffled["中类"].iloc[0]).strip()
            minor = str(shuffled["小类"].iloc[0]).strip()
            detail = str(shuffled["细类"].iloc[0]).strip()
            prompt_file_key = _safe_filename(str(detail_path))

            summary_rows.append(
                {
                    "detail_path": detail_path,
                    "detail_name": detail_name,
                    "大类": major,
                    "中类": middle,
                    "小类": minor,
                    "细类": detail,
                    "available_count": int(len(shuffled)),
                    "train_count": int(len(train_part)),
                    "validation_pool_count": int(len(validation_part)),
                    "prompt_file_key": prompt_file_key,
                }
            )

            for order, row in enumerate(train_part.to_dict(orient="records"), start=1):
                manifest_rows.append(
                    {
                        "sample_row_id": row["sample_row_id"],
                        "source_table": row["__source_table"],
                        "source_row_number": int(row["__source_row_number"]),
                        "岗位名称": row["岗位名称"],
                        "detail_path": detail_path,
                        "detail_name": detail_name,
                        "大类": major,
                        "中类": middle,
                        "小类": minor,
                        "细类": detail,
                        "top1_code": row["top1_code"],
                        "top1_title": row["top1_title"],
                        "top1_score": row["top1_score"],
                        "selected_candidate_rank": row.get("selected_candidate_rank", None),
                        "split": "train",
                        "sample_order": order,
                        "prompt_file_key": prompt_file_key,
                    }
                )

            for order, row in enumerate(validation_part.to_dict(orient="records"), start=1):
                manifest_rows.append(
                    {
                        "sample_row_id": row["sample_row_id"],
                        "source_table": row["__source_table"],
                        "source_row_number": int(row["__source_row_number"]),
                        "岗位名称": row["岗位名称"],
                        "detail_path": detail_path,
                        "detail_name": detail_name,
                        "大类": major,
                        "中类": middle,
                        "小类": minor,
                        "细类": detail,
                        "top1_code": row["top1_code"],
                        "top1_title": row["top1_title"],
                        "top1_score": row["top1_score"],
                        "selected_candidate_rank": row.get("selected_candidate_rank", None),
                        "split": "validation_pool",
                        "sample_order": order,
                        "prompt_file_key": prompt_file_key,
                    }
                )

        summary_df = pd.DataFrame(summary_rows).sort_values(
            ["available_count", "detail_name"], ascending=[False, True]
        )
        manifest_df = pd.DataFrame(manifest_rows).sort_values(
            ["detail_name", "split", "sample_order"], ascending=[True, True, True]
        )
        return summary_df, manifest_df

    def fetch_rows_by_manifest(self, manifest_df: pd.DataFrame) -> pd.DataFrame:
        """根据采样清单回查原始岗位描述。"""
        if manifest_df.empty:
            return pd.DataFrame()

        frames: List[pd.DataFrame] = []
        with duckdb.connect(str(self.config.db_path)) as conn:
            conn.execute(f"PRAGMA threads={self.config.duckdb_threads}")
            for source_table, group in manifest_df.groupby("source_table", sort=False):
                row_numbers = ",".join(str(int(value)) for value in group["source_row_number"].tolist())
                query = f"""
                    WITH numbered_jobs AS (
                        SELECT
                            *,
                            ROW_NUMBER() OVER () AS __source_row_number
                        FROM {source_table}
                    )
                    SELECT
                        *
                    FROM numbered_jobs
                    WHERE __source_row_number IN ({row_numbers})
                """
                frame = conn.execute(query).df()
                frame["source_table"] = source_table
                frames.append(frame)

        raw_df = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()
        if raw_df.empty:
            return raw_df

        raw_df["sample_row_id"] = (
            raw_df["source_table"].astype(str)
            + ":"
            + raw_df["__source_row_number"].astype(str)
        )
        return manifest_df.merge(
            raw_df,
            on="sample_row_id",
            how="left",
            suffixes=("", "_raw"),
        )
