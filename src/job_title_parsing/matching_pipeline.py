"""主匹配流程。"""

from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm.auto import tqdm

from .alias_builder import AliasBuilder
from .bm25_index import BM25Index
from .catalog_preprocessor import CatalogPreprocessor
from .feature_extractor import JobFeatureExtractor
from .hierarchy_filter import HierarchyFilter
from .jd_parser import JDParser
from .match_utils import load_config, load_stopwords, normalize_compact
from .ngram_retrieval import overlap_score
from .scoring import ScoreFusion
from .title_cleaner import JobTitleCleaner


_PROCESS_WORKER_PIPELINE: Any = None


def _init_process_worker(
    config_path: str | None,
    alias_dict_path: str | None,
    catalog_df: pd.DataFrame,
) -> None:
    """初始化进程池 worker 的全局匹配实例（每进程一次）。"""
    global _PROCESS_WORKER_PIPELINE
    pipeline = MatchPipeline(
        config_path=config_path,
        alias_dict_path=alias_dict_path,
    )
    pipeline.load_catalog_df(catalog_df)
    _PROCESS_WORKER_PIPELINE = pipeline


def _match_chunk_process_worker(
    chunk_rows: List[Dict[str, Any]],
    job_title_col: str,
    job_desc_col: str,
    job_id_col: str,
    top_k: Optional[int],
    debug: bool,
) -> List[Dict[str, Any]]:
    """使用进程池全局实例匹配分块。"""
    if _PROCESS_WORKER_PIPELINE is None:
        raise RuntimeError("进程 worker 尚未初始化，请检查 initializer 配置")

    return [
        _PROCESS_WORKER_PIPELINE.match_one(
            job_title=row.get(job_title_col, ""),
            job_description=row.get(job_desc_col, ""),
            job_id=row.get(job_id_col, None),
            top_k=top_k,
            debug=debug,
        )
        for row in chunk_rows
    ]


class MatchPipeline:
    """岗位到《中国职业分类大典》条目的初步匹配流程。"""

    def __init__(
        self,
        catalog_df: Optional[pd.DataFrame] = None,
        config_path: str | Path | None = None,
        alias_dict_path: str | Path | None = None,
    ):
        self.config_path = Path(config_path) if config_path else None
        self.alias_dict_path = Path(alias_dict_path) if alias_dict_path else None
        self.config = load_config(config_path)
        self.stopwords = load_stopwords(self.config.get("retrieval", {}).get("stopwords_path", ""))
        self.alias_builder = AliasBuilder(self.config, alias_dict_path=alias_dict_path)
        self.title_cleaner = JobTitleCleaner(self.config)
        self.jd_parser = JDParser(self.config)
        self.feature_extractor = JobFeatureExtractor()
        self.hierarchy_filter = HierarchyFilter(self.config)
        self.scorer = ScoreFusion(self.config)
        self.preprocessor = CatalogPreprocessor(self.config, self.alias_builder)
        self.catalog_df = pd.DataFrame()
        self.title_index: Optional[BM25Index] = None
        self.task_index: Optional[BM25Index] = None
        if catalog_df is not None:
            self.load_catalog_df(catalog_df)

    def load_catalog_csv(self, csv_path: str | Path, encoding: str = "utf-8") -> pd.DataFrame:
        """加载并预处理职业大典 CSV。"""
        df = self.preprocessor.load_csv(csv_path, encoding=encoding)
        self.load_catalog_df(df)
        return df

    def load_catalog_duckdb(
        self,
        db_path: str | Path | None = None,
        table_name: str | None = None,
        where_sql: str = "",
        limit: int | None = None,
    ) -> pd.DataFrame:
        """从 DuckDB 加载并预处理职业大典。"""
        df = self.preprocessor.load_duckdb(
            db_path=db_path,
            table_name=table_name,
            where_sql=where_sql,
            limit=limit,
        )
        self.load_catalog_df(df)
        return df

    def load_catalog_df(self, catalog_df: pd.DataFrame) -> None:
        """加载已预处理或原始 catalog_df，并建立检索索引。"""
        required = {"retrieval_title_text", "retrieval_task_text", "task_list", "aliases"}
        if not required.issubset(set(catalog_df.columns)):
            catalog_df = self.preprocessor.preprocess(catalog_df)
        self.catalog_df = catalog_df.reset_index(drop=True)
        self._build_indexes()

    def _build_indexes(self) -> None:
        """构建 title/alias 与 tasks 两路 BM25 索引。"""
        self.title_index = BM25Index(
            self.catalog_df["retrieval_title_text"].fillna("").tolist(),
            stopwords=self.stopwords,
            k1=float(self.config.get("retrieval", {}).get("bm25_k1", 1.5)),
            b=float(self.config.get("retrieval", {}).get("bm25_b", 0.75)),
        )
        self.task_index = BM25Index(
            self.catalog_df["retrieval_task_text"].fillna("").tolist(),
            stopwords=self.stopwords,
            k1=float(self.config.get("retrieval", {}).get("bm25_k1", 1.5)),
            b=float(self.config.get("retrieval", {}).get("bm25_b", 0.75)),
        )

    def _title_match_info(self, clean_title: str, row: pd.Series) -> Dict[str, Any]:
        """计算 title 三层匹配信息：exact / normalized_exact / fuzzy。"""
        if not clean_title:
            return {"level": "none", "fuzzy_score": 0.0, "matched_text": ""}

        raw_aliases = row.get("aliases", [])
        alias_list = raw_aliases if isinstance(raw_aliases, list) else []
        names = [row.get("title_clean", ""), *alias_list]
        names = [str(x).strip() for x in names if str(x).strip()]
        if not names:
            return {"level": "none", "fuzzy_score": 0.0, "matched_text": ""}

        if clean_title in names:
            return {"level": "exact", "fuzzy_score": 1.0, "matched_text": clean_title}

        compact_title = normalize_compact(clean_title)
        for name in names:
            if compact_title and compact_title == normalize_compact(name):
                return {"level": "normalized_exact", "fuzzy_score": 0.95, "matched_text": name}

        best_name = ""
        best_ratio = 0.0
        for name in names:
            token_ratio = SequenceMatcher(None, clean_title, name).ratio()
            partial_ratio = SequenceMatcher(None, clean_title, name[: len(clean_title)] if len(name) >= len(clean_title) else name).ratio()
            ratio = max(token_ratio, partial_ratio)
            if ratio > best_ratio:
                best_ratio = ratio
                best_name = name

        threshold = float(self.config.get("scoring", {}).get("title_fuzzy_threshold", 0.82))
        if best_ratio >= threshold:
            return {"level": "fuzzy", "fuzzy_score": round(best_ratio, 6), "matched_text": best_name}

        return {"level": "none", "fuzzy_score": round(best_ratio, 6), "matched_text": best_name}

    def _build_no_candidate_result(
        self,
        *,
        job_id: Any,
        job_title: str,
        clean_title: str,
        jd_info: Dict[str, Any],
        feature_info: Dict[str, Any],
        generic_penalty: float,
        filtered_count: int,
        debug: bool,
    ) -> Dict[str, Any]:
        """构造无可解释召回时的低置信结果，避免表顺序候选污染 top1。"""
        confidence_info = self.scorer.build_confidence_flags([])
        return {
            "job_id": job_id,
            "job_title": job_title,
            "clean_title": clean_title,
            "platform_terms": feature_info["platform_terms"],
            "domain_terms": feature_info["domain_terms"],
            "function_terms": feature_info["function_terms"],
            "object_terms": feature_info["object_terms"],
            "conflict_terms": feature_info["conflict_terms"],
            "confidence_level": confidence_info["confidence_level"],
            "risk_flags": confidence_info["risk_flags"],
            "top1_top2_margin": confidence_info["top1_top2_margin"],
            "is_review_needed": confidence_info["is_review_needed"],
            "top1_code": "",
            "top1_title": "",
            "top1_score": 0.0,
            "candidates": [],
            "debug_info": {
                "jd_clean": jd_info["jd_clean"],
                "jd_sentences": jd_info["jd_sentences"],
                "core_task_sentences": jd_info["core_task_sentences"],
                "domain_keywords": jd_info["domain_keywords"],
                "feature_info": feature_info,
                "generic_penalty": generic_penalty,
                "candidate_conflict_penalty": {},
                "confidence_info": confidence_info,
                "filtered_candidate_count": filtered_count,
                "no_candidate_reason": "title_and_task_retrieval_empty_after_hierarchy_filter",
            }
            if debug
            else None,
        }

    def match_one(
        self,
        job_title: str,
        job_description: str,
        job_id: Any = None,
        top_k: Optional[int] = None,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """对单条岗位执行完整匹配流程。

        流程：标题清洗 → 别名映射 → JD 解析 → 特征抽取 → 泛标题惩罚 →
              层级过滤 → title/task 双路 BM25 召回 → n-gram 描述打分 →
              候选遍历打分（别名奖励、title 直接命中、task overlap、冲突惩罚）→
              分数归一化与融合 → TopK 排序 → 置信度标记。

        Args:
            job_title: 原始岗位名称。
            job_description: 原始岗位描述（可为空字符串）。
            job_id: 岗位唯一标识（可选）。
            top_k: 返回的候选数量，默认从 config 读取。
            debug: 是否在结果中包含完整 debug_info。

        Returns:
            Dict[str, Any]: 包含 top1_code, top1_title, top1_score, candidates,
                            confidence_level, risk_flags, is_review_needed,
                            clean_title, 结构化特征词等字段的匹配结果字典。
        """
        if self.catalog_df.empty or self.title_index is None or self.task_index is None:
            raise ValueError("职业大典尚未加载，请先调用 load_catalog_csv/load_catalog_duckdb/load_catalog_df")

        top_k = top_k or int(self.config.get("scoring", {}).get("top_k", 5))
        candidate_pool = int(self.config.get("retrieval", {}).get("candidate_pool_size", 30))

        clean_title = self.title_cleaner.clean(job_title)
        clean_title = self.alias_builder.resolve_manual_alias(clean_title)
        jd_info = self.jd_parser.parse(job_description)
        feature_info = self.feature_extractor.extract(clean_title or job_title, job_description)
        generic_penalty = self.scorer.compute_generic_penalty(
            clean_title,
            feature_info["function_terms"],
            feature_info["object_terms"],
        )
        query_text = " ".join([clean_title, jd_info["jd_clean"], *jd_info["domain_keywords"]]).strip()
        filter_text = " ".join([clean_title, jd_info["jd_clean"]]).strip()

        filtered_df = self.hierarchy_filter.filter_candidates_by_hierarchy(filter_text, self.catalog_df)
        filtered_indices = set(filtered_df.index.tolist())

        title_hits = self.title_index.search_title(clean_title or job_title, top_k=candidate_pool)
        task_query = " ".join(jd_info["core_task_sentences"] or jd_info["jd_sentences"][:3] or [jd_info["jd_clean"]])
        task_hits = self.task_index.search_tasks(task_query, top_k=candidate_pool)

        title_scores_raw = {item["index"]: item["score"] for item in title_hits if item["index"] in filtered_indices}
        task_scores_raw = {item["index"]: item["score"] for item in task_hits if item["index"] in filtered_indices}

        candidate_indices = list(set(title_scores_raw) | set(task_scores_raw))
        if not candidate_indices:
            return self._build_no_candidate_result(
                job_id=job_id,
                job_title=job_title,
                clean_title=clean_title,
                jd_info=jd_info,
                feature_info=feature_info,
                generic_penalty=generic_penalty,
                filtered_count=len(filtered_df),
                debug=debug,
            )

        ngram_n = int(self.config.get("retrieval", {}).get("char_ngram_n", 2))
        desc_scores_raw: Dict[int, float] = {}
        hierarchy_scores_raw: Dict[int, float] = {}
        overlap_raw: Dict[int, float] = {}
        alias_bonus_raw: Dict[int, float] = {}
        title_direct_bonus_raw: Dict[int, float] = {}
        title_fuzzy_raw: Dict[int, float] = {}
        title_match_level_raw: Dict[int, str] = {}
        title_match_text_raw: Dict[int, str] = {}
        conflict_penalty_raw: Dict[int, float] = {}

        for idx in candidate_indices:
            row = self.catalog_df.loc[idx]
            desc_scores_raw[idx] = overlap_score(query_text, row.get("retrieval_desc_text", ""), n=ngram_n)
            hierarchy_scores_raw[idx] = self.hierarchy_filter.hierarchy_match_bonus(filter_text, row)
            overlap_raw[idx] = self.scorer.compute_task_overlap(
                jd_info["core_task_sentences"], row.get("task_list", [])
            )
            alias_bonus_raw[idx] = self.scorer.alias_exact_bonus(
                clean_title, row.get("title_clean", ""), row.get("aliases", [])
            )
            title_match_info = self._title_match_info(clean_title, row)
            title_match_level_raw[idx] = str(title_match_info["level"])
            title_fuzzy_raw[idx] = float(title_match_info["fuzzy_score"])
            title_match_text_raw[idx] = str(title_match_info["matched_text"])
            title_direct_bonus_raw[idx] = self.scorer.title_direct_match_bonus(title_match_level_raw[idx])
            candidate_text = " ".join(
                [
                    str(row.get("title", "")),
                    str(row.get("title_clean", "")),
                    str(row.get("hierarchy_text", "")),
                    str(row.get("desc_clean", "")),
                ]
            )
            conflict_penalty_raw[idx] = self.scorer.compute_conflict_penalty(
                feature_info["conflict_terms"],
                candidate_text,
            )

        title_scores = self.scorer.normalize_scores({idx: title_scores_raw.get(idx, 0.0) for idx in candidate_indices})
        task_scores = self.scorer.normalize_scores({idx: task_scores_raw.get(idx, 0.0) for idx in candidate_indices})
        desc_scores = self.scorer.normalize_scores(desc_scores_raw)
        hierarchy_scores = self.scorer.normalize_scores(hierarchy_scores_raw)

        candidates: List[Dict[str, Any]] = []
        for idx in candidate_indices:
            row = self.catalog_df.loc[idx]
            final_score = self.scorer.final_score(
                title_score=title_scores.get(idx, 0.0),
                task_score=task_scores.get(idx, 0.0),
                desc_score=desc_scores.get(idx, 0.0),
                hierarchy_score=hierarchy_scores.get(idx, 0.0),
                alias_bonus=alias_bonus_raw.get(idx, 0.0),
                title_direct_bonus=title_direct_bonus_raw.get(idx, 0.0),
                task_overlap_bonus=overlap_raw.get(idx, 0.0),
                generic_penalty=generic_penalty,
                conflict_penalty=conflict_penalty_raw.get(idx, 0.0),
            )
            evidence = {
                "title_hit": row.get("title_clean", "") if title_scores.get(idx, 0.0) > 0 else "",
                "title_match_level": title_match_level_raw.get(idx, "none"),
                "title_match_text": title_match_text_raw.get(idx, ""),
                "alias_hit": clean_title if alias_bonus_raw.get(idx, 0.0) > 0 else "",
                "task_hit": row.get("task_list", [])[:3],
                "hierarchy_hit": row.get("hierarchy_text", "") if hierarchy_scores.get(idx, 0.0) > 0 else "",
            }
            candidates.append(
                {
                    "code": row.get("code", ""),
                    "title": row.get("title", ""),
                    "final_score": round(final_score, 6),
                    "title_bm25_raw_score": round(title_scores_raw.get(idx, 0.0), 6),
                    "title_bm25_score": round(title_scores.get(idx, 0.0), 6),
                    "title_fuzzy_score": round(title_fuzzy_raw.get(idx, 0.0), 6),
                    "title_match_level": title_match_level_raw.get(idx, "none"),
                    "task_bm25_raw_score": round(task_scores_raw.get(idx, 0.0), 6),
                    "task_bm25_score": round(task_scores.get(idx, 0.0), 6),
                    "desc_ngram_score": round(desc_scores.get(idx, 0.0), 6),
                    "hierarchy_match_bonus": round(hierarchy_scores.get(idx, 0.0), 6),
                    "alias_exact_match_bonus": round(alias_bonus_raw.get(idx, 0.0), 6),
                    "title_direct_bonus": round(title_direct_bonus_raw.get(idx, 0.0), 6),
                    "task_overlap_bonus": round(overlap_raw.get(idx, 0.0), 6),
                    "generic_penalty": round(generic_penalty, 6),
                    "conflict_penalty": round(conflict_penalty_raw.get(idx, 0.0), 6),
                    "evidence": evidence,
                }
            )

        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        top_candidates = candidates[:top_k]
        top1 = top_candidates[0] if top_candidates else {}
        confidence_info = self.scorer.build_confidence_flags(top_candidates)

        return {
            "job_id": job_id,
            "job_title": job_title,
            "clean_title": clean_title,
            "platform_terms": feature_info["platform_terms"],
            "domain_terms": feature_info["domain_terms"],
            "function_terms": feature_info["function_terms"],
            "object_terms": feature_info["object_terms"],
            "conflict_terms": feature_info["conflict_terms"],
            "confidence_level": confidence_info["confidence_level"],
            "risk_flags": confidence_info["risk_flags"],
            "top1_top2_margin": confidence_info["top1_top2_margin"],
            "is_review_needed": confidence_info["is_review_needed"],
            "top1_code": top1.get("code", ""),
            "top1_title": top1.get("title", ""),
            "top1_score": top1.get("final_score", 0.0),
            "candidates": top_candidates,
            "debug_info": {
                "jd_clean": jd_info["jd_clean"],
                "jd_sentences": jd_info["jd_sentences"],
                "core_task_sentences": jd_info["core_task_sentences"],
                "domain_keywords": jd_info["domain_keywords"],
                "feature_info": feature_info,
                "generic_penalty": generic_penalty,
                "candidate_conflict_penalty": conflict_penalty_raw,
                "confidence_info": confidence_info,
                "filtered_candidate_count": len(filtered_df),
            }
            if debug
            else None,
        }

    def _chunk_rows(self, rows: List[Dict[str, Any]], chunk_size: int) -> List[List[Dict[str, Any]]]:
        """将批量岗位数据按块切分。"""
        size = max(1, int(chunk_size))
        return [rows[i : i + size] for i in range(0, len(rows), size)]

    def _match_row(
        self,
        row: Dict[str, Any],
        job_title_col: str,
        job_desc_col: str,
        job_id_col: str,
        top_k: Optional[int],
        debug: bool,
    ) -> Dict[str, Any]:
        """匹配单条记录（供分块执行复用）。"""
        return self.match_one(
            job_title=row.get(job_title_col, ""),
            job_description=row.get(job_desc_col, ""),
            job_id=row.get(job_id_col, None),
            top_k=top_k,
            debug=debug,
        )

    def _match_chunk(
        self,
        chunk_rows: List[Dict[str, Any]],
        job_title_col: str,
        job_desc_col: str,
        job_id_col: str,
        top_k: Optional[int],
        debug: bool,
    ) -> List[Dict[str, Any]]:
        """匹配一个分块。"""
        return [
            self._match_row(
                row,
                job_title_col=job_title_col,
                job_desc_col=job_desc_col,
                job_id_col=job_id_col,
                top_k=top_k,
                debug=debug,
            )
            for row in chunk_rows
        ]

    def _build_executor(self, backend: str, workers: int):
        """构造并发执行器（默认线程池，进程池支持 worker 初始化）。"""
        mode = (backend or "thread").lower()
        max_workers = max(1, int(workers))
        if mode == "process":
            return ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_process_worker,
                initargs=(
                    str(self.config_path) if self.config_path else None,
                    str(self.alias_dict_path) if self.alias_dict_path else None,
                    self.catalog_df,
                ),
            )
        return ThreadPoolExecutor(max_workers=max_workers)

    def match_batch(
        self,
        jobs_df: pd.DataFrame,
        job_title_col: str = "岗位名称",
        job_desc_col: str = "岗位描述",
        job_id_col: str = "job_id",
        top_k: Optional[int] = None,
        debug: bool = False,
        workers: int = 1,
        show_progress: bool = False,
        chunk_size: int = 256,
        executor_backend: str = "thread",
    ) -> pd.DataFrame:
        """批量岗位匹配，支持多线程/多进程并发。

        Args:
            jobs_df: 招聘岗位 DataFrame，至少包含岗位名称和岗位描述列。
            job_title_col: 岗位名称字段名。
            job_desc_col: 岗位描述字段名。
            job_id_col: 岗位唯一标识字段名。
            top_k: 每条岗位返回的候选职业数量。
            debug: 是否输出 debug_info 列。
            workers: 并发 worker 数（<=1 时单线程）。
            show_progress: 是否显示 tqdm 进度条。
            chunk_size: 每个并发任务处理的岗位行数。
            executor_backend: 并发后端，"thread" 或 "process"。

        Returns:
            pd.DataFrame: 每条岗位一行的匹配结果 DataFrame。
        """
        rows = list(jobs_df.to_dict(orient="records"))
        if not rows:
            return pd.DataFrame()

        # 单线程处理逻辑
        if workers <= 1:
            iterator = rows
            pbar = None
            if show_progress:
                pbar = tqdm(total=len(rows), desc="Matching jobs", unit="job", dynamic_ncols=True)
            results: List[Dict[str, Any]] = []
            for row in iterator:
                results.append(
                    self._match_row(
                        row,
                        job_title_col=job_title_col,
                        job_desc_col=job_desc_col,
                        job_id_col=job_id_col,
                        top_k=top_k,
                        debug=debug,
                    )
                )
                if pbar is not None:
                    pbar.update(1)
            if pbar is not None:
                pbar.close()
            return pd.DataFrame(results)

        row_chunks = self._chunk_rows(rows, chunk_size=chunk_size)
        results_by_chunk: List[List[Dict[str, Any]]] = [[] for _ in range(len(row_chunks))]

        use_process = (executor_backend or "thread").lower() == "process"

        with self._build_executor(executor_backend, workers=workers) as executor:
            if use_process:
                future_to_chunk_idx = {
                    executor.submit(
                        _match_chunk_process_worker,
                        chunk_rows,
                        job_title_col,
                        job_desc_col,
                        job_id_col,
                        top_k,
                        debug,
                    ): chunk_idx
                    for chunk_idx, chunk_rows in enumerate(row_chunks)
                }
            else:
                future_to_chunk_idx = {
                    executor.submit(
                        self._match_chunk,
                        chunk_rows,
                        job_title_col,
                        job_desc_col,
                        job_id_col,
                        top_k,
                        debug,
                    ): chunk_idx
                    for chunk_idx, chunk_rows in enumerate(row_chunks)
                }

            pbar = None
            if show_progress:
                pbar = tqdm(total=len(rows), desc="Matching jobs", unit="job", dynamic_ncols=True)

            for future in as_completed(future_to_chunk_idx):
                chunk_idx = future_to_chunk_idx[future]
                chunk_result = future.result()
                results_by_chunk[chunk_idx] = chunk_result
                if pbar is not None:
                    pbar.update(len(chunk_result))

            if pbar is not None:
                pbar.close()

        flat_results: List[Dict[str, Any]] = []
        for chunk_result in results_by_chunk:
            flat_results.extend(chunk_result)
        return pd.DataFrame(flat_results)
