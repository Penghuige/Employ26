#!/usr/bin/env python
"""RAG v2.0 批量职业细类匹配脚本。

功能:
    1. 从 DuckDB 读取岗位数据（job_title + job_requirements）
    2. 基于预构建的 FAISS 双索引进行 RAG 检索
    3. 调用 DeepSeek-V4-Pro 进行语义匹配与增强推理
    4. 输出 CSV + DuckDB 结果表
    5. 支持增量缓存、嵌入模型切换、结果验证

用法:
    # 完整批量运行（100 条）
    python src/rag/batch_rag_match.py --limit 100

    # 全部运行 + 输出到指定表
    python src/rag/batch_rag_match.py --output-table recruit.main.rag_match_results_v2

    # 干跑（仅检索，不调用 LLM）
    python src/rag/batch_rag_match.py --limit 20 --retrieval-only

    # 从缓存恢复继续
    python src/rag/batch_rag_match.py --resume

前置条件:
    1. 已运行 `python -m src.rag.cli build` 构建 FAISS 索引
    2. .env.local 中已配置 DEEPSEEK_API_KEY
    3. DuckDB 表 recruit.main.label_studio_tasks_v2 存在

可配置变量（脚本顶部 CONFIG 区域）:
    - 数据库连接、表名
    - 嵌入模型路径（支持切换为 bge-m3 / bge-base-zh-v1.5 / text2vec 等）
    - 检索参数（top_k, pool_size, 权重）
    - LLM 参数（model, temperature, max_tokens）
    - 缓存与输出路径
"""

from __future__ import annotations

import hashlib
import heapq
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# 运行方式：从项目根目录执行 `python -m src.rag.batch_rag_match`，
# 确保 src.* 包可通过标准 Python 模块搜索路径正确导入。
from config.paths import get_project_paths
_paths = get_project_paths()
PROJECT_ROOT = _paths.project_root
load_dotenv(dotenv_path=str(PROJECT_ROOT / ".env.local"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("batch_rag")


# ===================================================================
# 可配置变量区（按需修改）
# ===================================================================

@dataclass
class BatchConfig:
    """批量匹配配置。

    可通过命令行参数覆盖以下所有字段。
    """

    # ---- 数据库 ----
    duckdb_path: str = str(PROJECT_ROOT / "output" / "recruit.duckdb")
    input_table: str = "recruit.main.label_studio_tasks_v2"
    input_where: str = ""
    input_limit: int = 0

    # ---- 嵌入模型 ----
    embedding_model_path: str = str(_paths.bge_model_path)

    # ---- 检索参数 ----
    top_k: int = 8
    retrieval_pool_size: int = 30   # BM25 粗召回池
    # BM25 参数
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # ---- LLM ----
    llm_model: str = "deepseek-v4-pro"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 512
    api_sleep: float = 0.3

    # ---- 输出 ----
    output_csv: str = str(PROJECT_ROOT / "output" / "rag_match_results.csv")
    output_table: str = "recruit.main.rag_match_results_v2"

    # ---- 缓存 ----
    cache_path: str = str(PROJECT_ROOT / "output" / "rag_match_cache.jsonl")
    enable_cache: bool = True

    # ---- 模式 ----
    retrieval_only: bool = False
    resume: bool = False

    # ---- 索引 ----
    def_index_path: str = str(PROJECT_ROOT / "src" / "rag" / "artifacts" / "occupation_def_index.faiss")
    task_index_path: str = str(PROJECT_ROOT / "src" / "rag" / "artifacts" / "occupation_task_index.faiss")
    metadata_path: str = str(PROJECT_ROOT / "src" / "rag" / "artifacts" / "occupation_metadata_v2.json")


# ===================================================================
# 轻量 BM25（内嵌，避免循环导入）
# ===================================================================

class _SimpleBM25:
    """内存 BM25 实现。"""

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = len(corpus)
        self.doc_lengths = [len(d) for d in corpus]
        self.avgdl = sum(self.doc_lengths) / max(self.doc_count, 1)

        from collections import Counter
        self.doc_freq: Dict[str, int] = defaultdict(int)
        self.postings: Dict[str, List[tuple]] = defaultdict(list)

        for idx, doc in enumerate(corpus):
            for token, freq in Counter(doc).items():
                self.doc_freq[token] += 1
                self.postings[token].append((idx, freq))

    def search(self, query: str, top_k: int = 30) -> List[Dict]:
        import jieba
        tokens = [w.strip() for w in jieba.lcut(query) if len(w.strip()) >= 2]
        if not tokens:
            return []
        scores: Dict[int, float] = defaultdict(float)
        for token in set(tokens):
            df = self.doc_freq.get(token, 0)
            if df == 0:
                continue
            idf = math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))
            for doc_idx, freq in self.postings.get(token, []):
                dl = self.doc_lengths[doc_idx]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                if denom > 0:
                    scores[doc_idx] += idf * (freq * (self.k1 + 1) / denom)
        ranked = heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])
        return [{"index": idx, "score": s} for idx, s in ranked if s > 0]


# ===================================================================
# RAG 引擎：BM25 召回 + BGE 重排 + JD 预处理
# ===================================================================

class RAGEngine:
    """增强 RAG 引擎：BM25 召回 + BGE 重排 + JD 预处理。

    检索流程:
    1. JD 预处理: 提取含动作动词的核心职责句，拼接到 query
    2. BM25 粗召回: 对 occupation 的 retrieval 文本做 BM25，取 top pool_size
    3. BGE 精排: 对 BM25 召回候选做 BGE 重排序，取 top_k
    """

    # JD 核心职责句提取的动作动词（复用 jd_parser 逻辑）
    ACTION_VERBS = {
        "负责", "参与", "制定", "维护", "推进", "撰写", "分析", "跟进",
        "开发", "设计", "测试", "部署", "管理", "协调", "审核", "实施",
        "优化", "配置", "搭建", "处理", "解决", "支持", "运营", "执行",
        "编写", "架构", "调试", "监控", "交付", "验收", "评估", "调研",
        "销售", "采购", "核算", "统筹", "沟通", "接待", "配送",
    }

    def __init__(self, config: BatchConfig):
        self.config = config

        from sentence_transformers import SentenceTransformer
        logger.info("加载嵌入模型: %s", config.embedding_model_path)
        self.embedding_model = SentenceTransformer(config.embedding_model_path)

        self.def_index: Any = None
        self.task_index: Any = None
        self.bm25_index: Any = None
        self.records: List[Dict] = []
        self.chunks: List[Dict] = []

        # DuckDB 只读连接（用于关键词直搜）
        self._db_conn: Any = None

        self._llm_client: Any = None
        self._cache: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # 索引加载
    # ------------------------------------------------------------------

    def load_index(self) -> None:
        """加载 FAISS + BM25 索引与元数据。"""
        import faiss

        if not os.path.exists(self.config.def_index_path):
            raise FileNotFoundError(
                f"索引文件不存在: {self.config.def_index_path}\n"
                f"请先执行: python -m src.rag.cli build"
            )

        self.def_index = faiss.read_index(self.config.def_index_path)
        logger.info("已加载 def_index: %s", self.config.def_index_path)

        task_path = self.config.task_index_path
        if os.path.exists(task_path):
            self.task_index = faiss.read_index(task_path)
            logger.info("已加载 task_index: %s", task_path)

        with open(self.config.metadata_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.records = payload.get("records", [])
        self.chunks = payload.get("chunks", [])
        logger.info("已加载元数据: %d records, %d chunks", len(self.records), len(self.chunks))

        # 构建 BM25 索引（内存中，无需持久化）
        self._build_bm25()
        logger.info("BM25 索引已就绪: %d docs", len(self.records))

    def _build_bm25(self) -> None:
        """基于 records 的检索文本构建 BM25 索引。"""
        import jieba

        texts = []
        for rec in self.records:
            # 拼接: 职业名 + 别名 + 定义 + 任务（优先用 retrieval 字段）
            parts = [
                rec.get("retrieval_title_text", ""),
                rec.get("title", ""),
                rec.get("retrieval_task_text", ""),
                rec.get("desc", ""),
            ]
            texts.append(" ".join(p for p in parts if p))

        tokenized = []
        for text in texts:
            tokens = [w.strip() for w in jieba.lcut(text) if len(w.strip()) >= 2]
            tokenized.append(tokens)

        self.bm25_index = _SimpleBM25(
            tokenized,
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
        )

    # ------------------------------------------------------------------
    # JD 预处理
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_core_tasks(job_requirements: str) -> str:
        """从 JD 中提取含动作动词的核心职责句。

        提取逻辑: 按句号/分号/换行切分 → 保留含动作动词的句子 → 拼接去重。

        Args:
            job_requirements: 原始岗位要求文本。

        Returns:
            str: 核心职责句拼接文本。
        """
        import re
        if not job_requirements:
            return ""
        # 切句
        sentences = re.split(r"[。！？!?;；\n|]+", str(job_requirements))
        core = []
        for sent in sentences:
            sent = sent.strip()
            if not sent or len(sent) < 6:
                continue
            # 保留含动作动词的句子
            if any(verb in sent for verb in RAGEngine.ACTION_VERBS):
                core.append(sent)
        # 去重、截断
        seen = set()
        unique = []
        for s in core:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        return "。".join(unique[:5])

    def _build_query(self, job_title: str, job_requirements: str) -> str:
        """构建增强查询文本：title + 核心职责句。

        Args:
            job_title: 岗位名称。
            job_requirements: 原始岗位要求。

        Returns:
            str: 优化后的查询文本。
        """
        core = self._extract_core_tasks(job_requirements)
        parts = [job_title]
        if core:
            parts.append(core)
        # 也保留原始 requirements 中的关键部分（去学历/经验噪声）
        import re
        cleaned = re.sub(
            r"(本科|硕士|博士|全日制|及以上|以上学历|\d+年.*经验|五险一金|带薪年假)",
            "", str(job_requirements or "")
        )
        if cleaned.strip():
            parts.append(cleaned[:300])
        return " ".join(parts)

    # ------------------------------------------------------------------
    # DuckDB 关键词直搜（第三条召回通路）
    # ------------------------------------------------------------------

    def _search_catalog_keywords(self, job_title: str) -> List[int]:
        """在 DuckDB 职业大典中按关键词搜索，返回匹配的 record 索引列表。

        用 jieba 分词提取实义词，在 title/desc/tasks 中做 LIKE OR 搜索。

        Args:
            job_title: 岗位名称。

        Returns:
            List[int]: 匹配的 record 索引（在 self.records 中的位置）。
        """
        if self._db_conn is None:
            import duckdb
            self._db_conn = duckdb.connect(self.config.duckdb_path, read_only=True)

        import jieba
        title = str(job_title).strip()
        # 去括号内内容（如"（乳腺癌产品）"）
        import re
        title_clean = re.sub(r'[（(][^）)]*[）)]', '', title).strip()

        # jieba 分词 → 取长度 >=2 的实义词（去单字和纯数字/英文）
        all_words = [w.strip() for w in jieba.lcut(title_clean) if len(w.strip()) >= 2]
        # 去重保序，取前 6 个
        seen = set()
        keywords = []
        for w in all_words:
            if w not in seen and any('一' <= c <= '鿿' for c in w):
                seen.add(w)
                keywords.append(w)
        keywords = keywords[:6]

        if not keywords:
            return []

        # 构建 OR LIKE 条件
        safe_kws = [kw.replace("'", "''") for kw in keywords]
        like_clauses = " OR ".join(
            f"(title_clean LIKE '%{kw}%' OR desc_clean LIKE '%{kw}%' OR tasks LIKE '%{kw}%')"
            for kw in safe_kws
        )
        try:
            rows = self._db_conn.execute(f"""
                SELECT code FROM recruit.main.chinese_occupational_dictionary_joined_preprocessed
                WHERE {like_clauses}
                LIMIT 20
            """).fetchall()
        except Exception:
            return []

        # 映射回 record 索引
        code_to_idx = {rec.get("code", ""): i for i, rec in enumerate(self.records)}
        return [code_to_idx[c] for (c,) in rows if c in code_to_idx]

    # ------------------------------------------------------------------
    # 检索（BM25 召回 + DuckDB 直搜 + BGE 重排）
    # ------------------------------------------------------------------

    def search(self, job_title: str, job_requirements: str, use_full_jd: bool = False) -> List[Dict[str, Any]]:
        """BM25 召回 → BGE 重排。

        流程:
        1. JD 预处理 → 构建增强 query
        2. BM25 粗召回 top pool_size
        3. BGE 精排: 对召回候选的 definition chunk 编码, 计算余弦相似度
        4. 返回 top_k

        Args:
            job_title: 岗位名称。
            job_requirements: 岗位要求描述。
            use_full_jd: True=使用完整原始 JD（回退模式），False=预处理裁剪。

        Returns:
            List[Dict]: top_k 候选。
        """
        import numpy as np

        if use_full_jd:
            query_text = f"{job_title} {str(job_requirements)[:2000]}".strip()
        else:
            query_text = self._build_query(job_title, job_requirements)
        pool = self.config.retrieval_pool_size
        top_k = self.config.top_k

        # ---- Step 1: BM25 粗召回 + DuckDB 关键词直搜 ----
        if self.bm25_index is None:
            self._build_bm25()

        bm25_results = self.bm25_index.search(query_text, top_k=pool)
        candidate_indices = [r["index"] for r in bm25_results if r["score"] > 0]

        # 第三条召回通路：DuckDB 关键词直搜
        db_indices = self._search_catalog_keywords(job_title)
        if db_indices:
            candidate_indices = list(dict.fromkeys(candidate_indices + db_indices))  # 去重保序
            if len(db_indices) > 0:
                pass  # 已在上面合并

        if not candidate_indices:
            return self._search_bge_only(query_text)

        # ---- Step 2: 纯 BM25 排序（取消 BGE 精排，避免引入语义噪声） ----
        # BGE embedding 在职业细类匹配任务上的语义区分度不足，
        # 实证: BGE 曾将"法务经理"排到"电解精炼工"之后。
        # 直接用 BM25 关键词匹配 + DB 召回分数排序更可靠。
        bm25_lookup: Dict[int, float] = {r["index"]: r["score"] for r in bm25_results}
        bm25_max = max(bm25_lookup.values()) if bm25_lookup else 1.0

        scored = []
        for ri in candidate_indices:
            if ri >= len(self.records):
                continue
            bm25 = bm25_lookup.get(ri, 0.0)
            score = bm25 / max(bm25_max, 1e-9)
            # DuckDB 直搜命中的候选给基础分
            if ri in db_indices and bm25 == 0:
                score = 0.3
            scored.append((score, ri))

        scored.sort(key=lambda x: -x[0])
        top_indices = scored[:top_k]

        results = []
        for rank, (score, ri) in enumerate(top_indices, 1):
            rec = self.records[ri]
            results.append({
                "rank": rank,
                "score": round(score, 6),
                "code": rec.get("code", ""),
                "title": rec.get("title", ""),
                "desc": rec.get("desc", ""),
                "tasks": rec.get("tasks", ""),
                "hierarchy": rec.get("hierarchy", {}),
                "hierarchy_text": rec.get("hierarchy_text", ""),
                "aliases": rec.get("aliases", []),
            })
        return results

    def _search_bge_only(self, query_text: str) -> List[Dict[str, Any]]:
        """纯 BGE 检索（BM25 回退）。"""
        import numpy as np
        qv = self.embedding_model.encode(
            [query_text], normalize_embeddings=True, show_progress_bar=False
        )
        qv = np.asarray(qv, dtype=np.float32)
        pool = self.config.retrieval_pool_size
        ds_arr, ds_idx = self.def_index.search(qv, pool)

        results = []
        seen = set()
        for score, idx in zip(ds_arr[0], ds_idx[0]):
            if 0 <= idx < len(self.chunks):
                ri = self.chunks[idx].get("record_idx", idx)
                if ri in seen:
                    continue
                seen.add(ri)
                if ri < len(self.records):
                    rec = self.records[ri]
                    results.append({
                        "rank": len(results) + 1,
                        "score": round(float(score), 6),
                        "code": rec.get("code", ""),
                        "title": rec.get("title", ""),
                        "desc": rec.get("desc", ""),
                        "tasks": rec.get("tasks", ""),
                        "hierarchy": rec.get("hierarchy", {}),
                        "hierarchy_text": rec.get("hierarchy_text", ""),
                        "aliases": rec.get("aliases", []),
                    })
                if len(results) >= self.config.top_k:
                    break
        return results


    # ------------------------------------------------------------------
    # DeepSeek-V4-Pro 调用
    # ------------------------------------------------------------------

    @property
    def llm_client(self):
        """惰性初始化 DeepSeek 客户端。"""
        if self._llm_client is None:
            from openai import OpenAI

            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "DEEPSEEK_API_KEY 未在 .env.local 中设置。\n"
                    "请在 .env.local 中添加: DEEPSEEK_API_KEY=sk-xxx"
                )
            self._llm_client = OpenAI(
                api_key=api_key, base_url="https://api.deepseek.com"
            )
        return self._llm_client

    def generate(
        self,
        job_title: str,
        job_requirements: str,
        candidates: List[Dict],
    ) -> Dict[str, Any]:
        """调用 DeepSeek-V4-Pro 进行语义匹配。

        Args:
            job_title: 岗位名称。
            job_requirements: 岗位要求描述。
            candidates: 检索候选列表。

        Returns:
            Dict: {"best_candidate", "best_code", "best_title", "confidence", "reasoning", "evidence"}
        """
        if not candidates:
            return self._empty_result("无候选可供判断")

        # 构建 prompt（v2: 减少误判 NONE）
        candidates_text = self._build_candidates_text(candidates)
        system_prompt = (
            "你是《中华人民共和国职业分类大典》（2022年版）的资深分类专家。\n"
            "你的任务是根据招聘岗位的标题和描述，从检索到的候选职业细类中选择最匹配的一个。\n\n"
            "评判原则：\n"
            "1. 以实际工作内容（job_requirements）为主要判断依据，不要只看岗位名称。\n"
            "2. 充分利用候选职业的层级路径（大类→中类→小类→细类）作为约束。\n"
            "3. 代码结构为 X-XX-XX-XX，代表大类-中类-小类-细类。\n"
            "4. 优先选最接近的候选，而非轻易判 NONE。\n"
            "   仅在候选与岗位分属完全不同的行业大类时才选 NONE。\n"
            "5. 输出必须是严格的 JSON，不要附带任何解释性文字。"
        )
        user_prompt = (
            f"请从以下 {len(candidates)} 个候选职业中，选择与招聘岗位最匹配的一个。\n\n"
            f"【招聘岗位】\n"
            f"岗位名称：{job_title}\n"
            f"岗位要求：\n{job_requirements[:3000]}\n\n"
            f"【候选职业】\n{candidates_text}\n\n"
            f"请输出 JSON：\n"
            f'{{"best_candidate": "1"~"{len(candidates)}" 或 "NONE", '
            f'"best_code": "职业代码", "best_title": "职业名称", '
            f'"confidence": 0.0-1.0, "reasoning": "理由(30字内)", '
            f'"evidence": "引用的关键职业大典内容(30字内)"}}'
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
            )
            raw = response.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("LLM 调用失败: %s", exc)
            return self._empty_result(f"API错误: {exc}")

        parsed = self._parse_json(raw)
        raw_confidence = float(parsed.get("confidence", 0))

        # 两层回退：LLM 判 NONE 时，用完整原始 JD 再试一次
        if parsed.get("best_candidate", "") == "NONE" and job_requirements:
            fallback_candidates = self.search(
                job_title, job_requirements, use_full_jd=True
            )
            if fallback_candidates and fallback_candidates != candidates:
                logger.info("  NONE 回退检索 → 用完整 JD 重试")
                fb_text = self._build_candidates_text(fallback_candidates)
                fb_prompt = (
                    f"请从以下 {len(fallback_candidates)} 个候选职业中，选择与招聘岗位最匹配的一个。\n\n"
                    f"【招聘岗位】\n"
                    f"岗位名称：{job_title}\n"
                    f"岗位要求：\n{job_requirements[:3000]}\n\n"
                    f"【候选职业】\n{fb_text}\n\n"
                    f"请输出 JSON：\n"
                    f'{{"best_candidate": "1"~"{len(fallback_candidates)}" 或 "NONE", '
                    f'"best_code": "职业代码", "best_title": "职业名称", '
                    f'"confidence": 0.0-1.0, "reasoning": "理由(30字内)", '
                    f'"evidence": "引用的关键职业大典内容(30字内)"}}'
                )
                try:
                    fb_resp = self.llm_client.chat.completions.create(
                        model=self.config.llm_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": fb_prompt},
                        ],
                        temperature=self.config.llm_temperature,
                        max_tokens=self.config.llm_max_tokens,
                    )
                    fb_raw = fb_resp.choices[0].message.content or ""
                    fb_parsed = self._parse_json(fb_raw)
                    if fb_parsed.get("best_candidate", "NONE") != "NONE":
                        raw = fb_raw
                        parsed = fb_parsed
                        raw_confidence = float(parsed.get("confidence", 0))
                except Exception:
                    pass

        result = {
            "best_candidate": parsed.get("best_candidate", "NONE"),
            "best_code": parsed.get("best_code", ""),
            "best_title": parsed.get("best_title", ""),
            "confidence": self._calibrate_confidence(raw_confidence, parsed.get("best_candidate", "")),
            "reasoning": str(parsed.get("reasoning", ""))[:200],
            "evidence": str(parsed.get("evidence", ""))[:200],
            "raw_response": raw,
        }
        return result

    def _calibrate_confidence(self, raw_conf: float, best_candidate: str) -> float:
        """基于人工标注一致性数据校准置信度。

        校准逻辑（基于 31 条 multi-annotator 验证集分析）：
        - 人工标注完全一致率（3/3 以上一致）约 65%
        - LLM 原始 confidence 集中在 0.90-1.0，明显偏高
        - 校准映射: raw 0.95+ → calibrated ~0.70; raw 0.80-0.95 → ~0.60

        Args:
            raw_conf: LLM 原始置信度。
            best_candidate: LLM 选择（NONE=强制0）。

        Returns:
            float: 校准后的置信度。
        """
        if best_candidate == "NONE" or raw_conf <= 0:
            return 0.0
        # 分段线性校准: 将 [0.7, 1.0] 映射到 [0.35, 0.75]
        if raw_conf >= 0.90:
            calibrated = 0.60 + (raw_conf - 0.90) * 1.5  # 0.90→0.60, 0.98→0.72
        elif raw_conf >= 0.70:
            calibrated = 0.35 + (raw_conf - 0.70) * 1.25  # 0.70→0.35, 0.90→0.60
        else:
            calibrated = raw_conf * 0.5  # 0.50→0.25
        return round(min(calibrated, 0.85), 4)

    def _build_candidates_text(self, candidates: List[Dict]) -> str:
        """构建候选列表文本。"""
        parts = []
        for c in candidates:
            hier = c.get("hierarchy", {})
            path = " > ".join(hier.get(f, "") for f in ["大类", "中类", "小类", "细类"] if hier.get(f))
            parts.append(
                f"候选{c['rank']}: [{c['code']}] {c['title']}\n"
                f"  层级: {path}\n"
                f"  定义: {c.get('desc', '')[:200]}\n"
                f"  任务: {c.get('tasks', '')[:200]}\n"
                f"  分数: {c['score']:.4f}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _empty_result(reason: str = "") -> Dict[str, Any]:
        return {
            "best_candidate": "NONE", "best_code": "", "best_title": "",
            "confidence": 0.0, "reasoning": reason, "evidence": "", "raw_response": "",
        }

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        text = (raw or "").strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:]) if len(lines) > 1 else text
            text = text.rsplit("```", 1)[0].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        import re
        m = re.search(r"\{[^{}]*\}", text)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return {}

    # ------------------------------------------------------------------
    # 缓存
    # ------------------------------------------------------------------

    def _cache_key(self, job_title: str, job_requirements: str) -> str:
        return hashlib.md5(f"{job_title}||{job_requirements}".encode()).hexdigest()

    def load_cache(self) -> None:
        """加载已有缓存。"""
        if not os.path.exists(self.config.cache_path):
            self._cache = {}
            return
        with open(self.config.cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    key = self._cache_key(
                        entry.get("job_title", ""),
                        entry.get("job_requirements", ""),
                    )
                    self._cache[key] = entry
                except json.JSONDecodeError:
                    continue
        logger.info("已加载缓存: %d 条", len(self._cache))

    def write_cache(self, entry: Dict[str, Any]) -> None:
        """写入一条缓存（即时追加到文件）。"""
        if not self.config.enable_cache:
            return
        with open(self.config.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        key = self._cache_key(entry.get("job_title", ""), entry.get("job_requirements", ""))
        self._cache[key] = entry


# ===================================================================
# 批量处理
# ===================================================================

def load_tasks(config: BatchConfig) -> pd.DataFrame:
    """从 DuckDB 加载待匹配岗位。

    Args:
        config: 批量配置。

    Returns:
        pd.DataFrame: 含 id, job_title, job_requirements 的 DataFrame。
    """
    conn = duckdb.connect(config.duckdb_path, read_only=True)
    try:
        sql = f"SELECT id, job_title, job_requirements FROM {config.input_table}"
        if config.input_where:
            sql += f" WHERE {config.input_where}"
        if config.input_limit > 0:
            sql += f" LIMIT {config.input_limit}"
        df = conn.execute(sql).df()
        logger.info("加载 %d 条岗位数据", len(df))
        return df
    finally:
        conn.close()


def process_batch(
    config: BatchConfig,
    engine: RAGEngine,
    tasks: pd.DataFrame,
    skipped_ids: set,
) -> pd.DataFrame:
    """批量处理岗位匹配。

    Args:
        config: 配置。
        engine: RAG 引擎。
        tasks: 待处理岗位 DataFrame。
        skipped_ids: 已缓存的任务 ID 集合。

    Returns:
        pd.DataFrame: 匹配结果。
    """
    results: List[Dict[str, Any]] = []
    cached_count = 0

    iterator = tqdm(
        tasks.iterrows(), total=len(tasks),
        desc="RAG 匹配", unit="job", dynamic_ncols=True,
    )

    for _, row in iterator:
        task_id = row["id"]
        job_title = str(row.get("job_title", "") or "")
        job_requirements = str(row.get("job_requirements", "") or "")

        # 检查缓存
        cache_key = engine._cache_key(job_title, job_requirements)
        if config.resume and task_id in skipped_ids:
            cached = engine._cache.get(cache_key)
            if cached:
                results.append({
                    "task_id": task_id, "job_title": job_title,
                    "job_requirements": job_requirements,
                    "best_code": cached.get("best_code", ""),
                    "best_title": cached.get("best_title", ""),
                    "confidence": cached.get("confidence", 0),
                    "reasoning": cached.get("reasoning", ""),
                    "evidence": cached.get("evidence", ""),
                    "top1_retrieval_code": cached.get("top1_retrieval_code", ""),
                    "top1_retrieval_title": cached.get("top1_retrieval_title", ""),
                    "top1_retrieval_score": cached.get("top1_retrieval_score", 0),
                    "candidates_json": cached.get("candidates_json", ""),
                    "from_cache": True,
                })
                cached_count += 1
                continue

        # 1) 检索（BM25 召回 + BGE 重排 + JD 预处理）
        candidates = engine.search(job_title, job_requirements)

        top1_code = candidates[0]["code"] if candidates else ""
        top1_title = candidates[0]["title"] if candidates else ""
        top1_score = candidates[0]["score"] if candidates else 0

        # 2) LLM 推理
        if config.retrieval_only:
            llm_result = engine._empty_result("仅检索模式")
        else:
            llm_result = engine.generate(job_title, job_requirements, candidates)
            time.sleep(config.api_sleep)

        entry = {
            "task_id": task_id,
            "job_title": job_title,
            "job_requirements": job_requirements,
            "best_code": llm_result["best_code"],
            "best_title": llm_result["best_title"],
            "confidence": llm_result["confidence"],
            "reasoning": llm_result["reasoning"],
            "evidence": llm_result["evidence"],
            "top1_retrieval_code": top1_code,
            "top1_retrieval_title": top1_title,
            "top1_retrieval_score": top1_score,
            "candidates_json": json.dumps(candidates, ensure_ascii=False),
            "from_cache": False,
        }
        results.append(entry)

        # 写入缓存
        engine.write_cache(entry)

        # 更新进度条描述
        if llm_result["best_code"]:
            iterator.set_postfix_str(
                f"{job_title[:15]} → {llm_result['best_title'][:20]} ({llm_result['confidence']:.2f})"
            )

    if cached_count:
        logger.info("从缓存恢复 %d 条", cached_count)
    return pd.DataFrame(results)


def save_results(df: pd.DataFrame, config: BatchConfig) -> None:
    """保存结果到 CSV 和 DuckDB。

    Args:
        df: 结果 DataFrame。
        config: 配置。
    """
    # CSV
    csv_path = Path(config.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_csv = df.drop(columns=["candidates_json"], errors="ignore")
    df_csv.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info("CSV 已保存: %s (%d 行)", csv_path, len(df))

    # DuckDB
    conn = duckdb.connect(config.duckdb_path)
    try:
        conn.execute("CREATE SCHEMA IF NOT EXISTS recruit.main")
        conn.register("result_df", df)
        table = config.output_table
        conn.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM result_df")
        cnt = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        logger.info("DuckDB 表已保存: %s (%d 行)", table, cnt)
    finally:
        conn.close()


def print_summary(df: pd.DataFrame, config: BatchConfig) -> None:
    """打印结果摘要。

    Args:
        df: 结果 DataFrame。
        config: 配置。
    """
    total = len(df)
    with_result = (df["best_code"] != "").sum()
    high_conf = (df["confidence"] >= 0.7).sum()
    from_cache = df["from_cache"].sum() if "from_cache" in df.columns else 0
    avg_conf = df[df["confidence"] > 0]["confidence"].mean()

    print("\n" + "=" * 60)
    print("批量匹配完成")
    print("=" * 60)
    print(f"  总任务数:        {total}")
    print(f"  有匹配结果:      {with_result} ({with_result * 100 / max(total, 1):.1f}%)")
    print(f"  高置信度(>=0.7): {high_conf} ({high_conf * 100 / max(total, 1):.1f}%)")
    print(f"  平均置信度:      {avg_conf:.3f}" if not pd.isna(avg_conf) else "  平均置信度: N/A")
    print(f"  从缓存恢复:      {from_cache}")
    print(f"  CSV 输出:        {config.output_csv}")
    print(f"  DuckDB 表:       {config.output_table}")

    # Top 匹配
    print("\n  Top-5 高频匹配职业:")
    top5 = df[df["best_code"] != ""]["best_title"].value_counts().head(5)
    for title, cnt in top5.items():
        print(f"    {title[:50]}: {cnt} 次")

    # 无匹配
    unmatched = df[df["best_code"] == ""]
    if len(unmatched) > 0:
        print(f"\n  无匹配样本 ({len(unmatched)} 条):")
        for _, row in unmatched.head(5).iterrows():
            print(f"    [{row['task_id']}] {str(row['job_title'])[:40]}")


# ===================================================================
# CLI
# ===================================================================

def build_argparser() -> "argparse.ArgumentParser":
    import argparse
    p = argparse.ArgumentParser(
        description="RAG v2.0 批量职业细类匹配",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/batch_rag_match.py --limit 100
  python scripts/batch_rag_match.py --retrieval-only --limit 20
  python scripts/batch_rag_match.py --resume --output-table recruit.main.rag_v2
        """,
    )
    p.add_argument("--limit", type=int, default=0, help="处理条数限制（0=全部）")
    p.add_argument("--retrieval-only", action="store_true", help="仅检索不调用 LLM")
    p.add_argument("--resume", action="store_true", help="从缓存恢复已完成任务")
    p.add_argument("--no-cache", action="store_true", help="禁用缓存")
    p.add_argument("--output-table", default="", help="输出 DuckDB 表名（覆盖默认）")
    p.add_argument("--output-csv", default="", help="输出 CSV 路径（覆盖默认）")
    p.add_argument("--where", default="", help="输入表 WHERE 子句（如: is_validation=true）")
    p.add_argument("--top-k", type=int, default=8, help="检索候选数")
    p.add_argument("--embedding-model", default="", help="嵌入模型路径（覆盖默认）")
    p.add_argument("--min-score", type=float, default=0.15, help="检索最低相似度阈值")
    return p


def main():
    import argparse as _argparse
    parser = build_argparser()
    args = parser.parse_args()

    # 构建配置
    config = BatchConfig()
    if args.limit > 0:
        config.input_limit = args.limit
    if args.where:
        config.input_where = args.where
    config.retrieval_only = args.retrieval_only
    config.resume = args.resume
    config.enable_cache = not args.no_cache
    if args.output_table:
        config.output_table = args.output_table
    if args.output_csv:
        config.output_csv = args.output_csv
    if args.top_k:
        config.top_k = args.top_k
    if args.min_score:
        pass  # BM25 模式下 min_score 阈值由 BM25 自身 >0 过滤替代
    if args.embedding_model:
        config.embedding_model_path = args.embedding_model

    # 1. 初始化引擎
    engine = RAGEngine(config)
    engine.load_index()
    if config.resume or config.enable_cache:
        engine.load_cache()

    # 2. 加载数据
    tasks = load_tasks(config)
    if tasks.empty:
        logger.error("无待处理数据")
        sys.exit(1)

    # 收集已缓存的 task_id
    skipped_ids = set()
    if config.resume:
        for entry in engine._cache.values():
            skipped_ids.add(entry.get("task_id"))

    # 3. 批量处理
    logger.info("开始批量匹配 (%d 条)...", len(tasks))
    start = time.time()
    results_df = process_batch(config, engine, tasks, skipped_ids)
    elapsed = time.time() - start
    logger.info("耗时: %.1f 秒 (%.1f 秒/条)", elapsed, elapsed / max(len(tasks), 1))

    # 4. 保存
    save_results(results_df, config)
    print_summary(results_df, config)


if __name__ == "__main__":
    main()
