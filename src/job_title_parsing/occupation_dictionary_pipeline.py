"""职业词典自动迭代流程。"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd

from src.job_title_parsing.match_utils import load_database_config, normalize_compact
from src.skill_extraction.config import PROJECT_ROOT
from src.utils.llm_router import (
    LLMRouter,
    build_json_system_prompt,
    extract_json_from_response,
    is_noisy_job_title,
    score_candidate_margin,
    should_escalate,
)

logger = logging.getLogger(__name__)

CHEAP_DECISION_PROMPT = build_json_system_prompt(
    "你是职业词典归一化助手。任务是判断一个岗位标题是否应映射到给定候选职业，或者进入 review。\n"
    "规则：\n"
    "1. 只处理职业名，不把技能、职责、部门、行业、福利写成职业。\n"
    "2. 如果标题只是候选 canonical 的常见简称、别名、级别变体，可映射。\n"
    "3. 不确定时不要强行归并，输出 review。\n"
    "4. 如果是明显新职业名且无法安全归并，输出 new_canonical。\n"
    "JSON 格式: {\"decision\":\"map_alias|new_canonical|review|reject\",\"canonical_name\":\"\",\"confidence\":0.0,\"reason\":\"\",\"evidence\":[\"\"]}"
)

STRONG_DECISION_PROMPT = build_json_system_prompt(
    "你是高精度职业词典仲裁器。只在难例上做最终判断。\n"
    "规则：\n"
    "1. 优先 precision，不确定就 review。\n"
    "2. 只可依据标题、极短上下文和候选列表判断。\n"
    "3. 不要创造大规模 alias，不要把职责或技能误归为职业。\n"
    "JSON 格式同前。"
)


@dataclass(frozen=True)
class OccupationDictionaryConfig:
    """职业词典迭代配置。"""

    project_root: Path
    db_path: Path
    duckdb_threads: int
    jobs_tables: List[str]
    output_dir: Path
    cache_dir: Path
    review_dir: Path
    report_dir: Path
    state_path: Path
    dictionary_path: Path
    env_file: Path
    batch_size: int
    top_k: int
    cheap_confidence_threshold: float
    candidate_margin_threshold: float
    max_context_chars: int
    min_title_count: int

    def ensure_dirs(self) -> None:
        for path in [self.output_dir, self.cache_dir, self.review_dir, self.report_dir, self.dictionary_path.parent]:
            path.mkdir(parents=True, exist_ok=True)


def load_occupation_dictionary_config() -> OccupationDictionaryConfig:
    raw = load_database_config()
    db_settings = raw.get("database", {})
    parsing_settings = raw.get("job_title_parsing", {})
    occ_settings = raw.get("occupation_dictionary", {})
    jobs_table_value = parsing_settings.get(
        "jobs_table",
        ['"Liepin".sample', '"51job".sample', '"Zhilian".sample'],
    )
    if isinstance(jobs_table_value, list):
        jobs_tables = [str(item).strip() for item in jobs_table_value if str(item).strip()]
    else:
        jobs_tables = [item.strip() for item in str(jobs_table_value).split(",") if item.strip()]

    output_dir = PROJECT_ROOT / "output" / "occupation_dictionary"
    config = OccupationDictionaryConfig(
        project_root=PROJECT_ROOT,
        db_path=PROJECT_ROOT / db_settings.get("duckdb_path", "output/recruit.duckdb"),
        duckdb_threads=max(1, int(db_settings.get("duckdb_threads", 8))),
        jobs_tables=jobs_tables,
        output_dir=output_dir,
        cache_dir=output_dir / "cache",
        review_dir=output_dir / "review",
        report_dir=output_dir / "reports",
        state_path=output_dir / "iteration_state.json",
        dictionary_path=PROJECT_ROOT / occ_settings.get("dictionary_path", "dicts/occupation_dictionary.json"),
        env_file=PROJECT_ROOT / ".env.local",
        batch_size=max(1, int(occ_settings.get("batch_size", 20))),
        top_k=max(1, int(occ_settings.get("top_k", 8))),
        cheap_confidence_threshold=float(occ_settings.get("cheap_confidence_threshold", 0.82)),
        candidate_margin_threshold=float(occ_settings.get("candidate_margin_threshold", 0.08)),
        max_context_chars=max(80, int(occ_settings.get("max_context_chars", 140))),
        min_title_count=max(1, int(occ_settings.get("min_title_count", 2))),
    )
    config.ensure_dirs()
    return config


def normalize_job_title(title: str) -> str:
    text = str(title or "").strip()
    if not text:
        return ""
    text = re.sub(r"[（(][^）)]*[）)]", " ", text)
    text = re.sub(r"\b\d+[kKwW]?(-\d+[kKwW]?)?\b", " ", text)
    text = re.sub(r"\b[0-9]{4}\b", " ", text)
    text = re.sub(r"[【\[][^】\]]*[】\]]", " ", text)
    text = re.sub(r"(急聘|直聘|高薪|诚聘|双休|五险一金|包吃住)", " ", text)
    text = re.sub(r"[\-|_/|]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def title_hash(title: str) -> str:
    return hashlib.sha1(normalize_compact(title).encode("utf-8")).hexdigest()[:16]


def load_dictionary(path: str | Path) -> Dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {"canonical_occupations": [], "metadata": {"created_at": datetime.now().isoformat()}}
    return json.loads(target.read_text(encoding="utf-8"))


def save_dictionary(data: Dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_dictionary_shape(data: Dict[str, Any]) -> Dict[str, Any]:
    canonical = data.get("canonical_occupations")
    if isinstance(canonical, list):
        return data
    skills = data.get("occupations") if isinstance(data.get("occupations"), list) else []
    return {
        "canonical_occupations": skills,
        "metadata": data.get("metadata", {}),
    }


def build_dictionary_index(dictionary: Dict[str, Any]) -> List[Dict[str, Any]]:
    index_rows: List[Dict[str, Any]] = []
    for item in dictionary.get("canonical_occupations", []):
        canonical_name = str(item.get("name") or item.get("canonical_name") or "").strip()
        if not canonical_name:
            continue
        aliases = [str(alias).strip() for alias in (item.get("aliases") or []) if str(alias).strip()]
        alias_set = sorted(set([canonical_name, *aliases]))
        index_rows.append(
            {
                "canonical_name": canonical_name,
                "aliases": alias_set,
                "normalized_name": normalize_compact(canonical_name),
                "normalized_aliases": [normalize_compact(alias) for alias in alias_set],
                "raw": item,
            }
        )
    return index_rows


def retrieve_top_candidates(title: str, dictionary: Dict[str, Any], top_k: int = 8) -> List[Dict[str, Any]]:
    normalized_title = normalize_compact(title)
    if not normalized_title:
        return []
    rows = build_dictionary_index(dictionary)
    scored: List[Dict[str, Any]] = []
    title_tokens = set(re.findall(r"[a-z]+|[\u4e00-\u9fa5]+", title.lower()))
    for row in rows:
        best_score = 0.0
        best_match = row["canonical_name"]
        for alias, normalized_alias in zip(row["aliases"], row["normalized_aliases"]):
            score = 0.0
            if normalized_alias == normalized_title:
                score = 1.0
            elif normalized_alias and normalized_alias in normalized_title:
                score = max(score, 0.93)
            elif normalized_title in normalized_alias:
                score = max(score, 0.88)
            overlap = _token_overlap_score(title_tokens, alias)
            score = max(score, overlap)
            char_overlap = _char_ngram_overlap(normalized_title, normalized_alias)
            score = max(score, char_overlap)
            if score > best_score:
                best_score = score
                best_match = alias
        if best_score <= 0:
            continue
        scored.append(
            {
                "canonical_name": row["canonical_name"],
                "matched_alias": best_match,
                "score": round(best_score, 4),
            }
        )
    scored.sort(key=lambda item: (-item["score"], len(item["canonical_name"])))
    return scored[:top_k]


def _token_overlap_score(title_tokens: set[str], alias: str) -> float:
    alias_tokens = set(re.findall(r"[a-z]+|[\u4e00-\u9fa5]+", alias.lower()))
    if not title_tokens or not alias_tokens:
        return 0.0
    overlap = len(title_tokens & alias_tokens)
    if overlap == 0:
        return 0.0
    return overlap / max(len(alias_tokens), len(title_tokens))


def _char_ngram_overlap(left: str, right: str, n: int = 2) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    left_grams = {left[i : i + n] for i in range(max(1, len(left) - n + 1))}
    right_grams = {right[i : i + n] for i in range(max(1, len(right) - n + 1))}
    if not left_grams or not right_grams:
        return 0.0
    return len(left_grams & right_grams) / max(len(left_grams | right_grams), 1)


def load_state(path: str | Path) -> Dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {"processed_hashes": [], "runs": []}
    return json.loads(target.read_text(encoding="utf-8"))


def save_state(state: Dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def load_title_candidates(config: OccupationDictionaryConfig, limit: int | None = None) -> pd.DataFrame:
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("缺少 duckdb 依赖，无法从数据库加载岗位标题") from exc

    union_queries = []
    for table in config.jobs_tables:
        union_queries.append(
            f"SELECT 岗位名称 AS job_title, 岗位描述 AS job_description, '{table}' AS source_table FROM {table} WHERE 岗位名称 IS NOT NULL"
        )
    sql = " UNION ALL ".join(union_queries)
    if limit:
        sql += f" LIMIT {int(limit)}"
    with duckdb.connect(str(config.db_path), read_only=True) as conn:
        conn.execute(f"PRAGMA threads={config.duckdb_threads}")
        return conn.execute(sql).df()


def prepare_pilot_batch_from_records(
    records: Sequence[Dict[str, Any]],
    config: OccupationDictionaryConfig,
    pilot_size: int | None = None,
) -> List[Dict[str, Any]]:
    if not records:
        return []
    frame = pd.DataFrame(records)
    if "job_title" not in frame.columns:
        raise ValueError("records 缺少 job_title 字段")
    if "job_description" not in frame.columns:
        frame["job_description"] = ""
    if "source_table" not in frame.columns:
        frame["source_table"] = "memory"
    frame["job_title"] = frame["job_title"].fillna("").astype(str)
    frame["job_description"] = frame["job_description"].fillna("").astype(str)
    frame["normalized_title"] = frame["job_title"].map(normalize_job_title)
    frame = frame[frame["normalized_title"] != ""]
    frame["title_hash"] = frame["normalized_title"].map(title_hash)
    grouped = (
        frame.groupby(["title_hash", "normalized_title"], as_index=False)
        .agg(
            freq=("job_title", "size"),
            sample_title=("job_title", "first"),
            sample_description=("job_description", "first"),
            source_table=("source_table", "first"),
        )
        .sort_values(["freq", "normalized_title"], ascending=[False, True])
    )
    grouped = grouped[grouped["freq"] >= config.min_title_count]
    if pilot_size:
        grouped = grouped.head(int(pilot_size))
    else:
        grouped = grouped.head(config.batch_size)
    return grouped.to_dict(orient="records")


def prepare_pilot_batch(config: OccupationDictionaryConfig, pilot_size: int | None = None) -> List[Dict[str, Any]]:
    source_df = load_title_candidates(config)
    return prepare_pilot_batch_from_records(source_df.to_dict(orient="records"), config, pilot_size=pilot_size)


def classify_title(
    item: Dict[str, Any],
    dictionary: Dict[str, Any],
    router: LLMRouter | None,
    config: OccupationDictionaryConfig,
    cache: Dict[str, Any],
) -> Dict[str, Any]:
    normalized_title = item["normalized_title"]
    if normalized_title in cache:
        cached = dict(cache[normalized_title])
        cached["from_cache"] = True
        return cached

    candidates = retrieve_top_candidates(normalized_title, dictionary, top_k=config.top_k)
    candidate_margin = score_candidate_margin(candidates)
    is_new_title = not candidates or candidates[0]["score"] < 0.72
    noisy_title = is_noisy_job_title(item["sample_title"])
    context = str(item.get("sample_description") or "")[: config.max_context_chars]
    context_conflict = False
    if candidates and context:
        top_name = candidates[0]["canonical_name"]
        if any(token in context.lower() for token in ["实习", "兼职"]) and not any(token in top_name for token in ["实习", "兼职"]):
            context_conflict = True
    has_conflicting_candidates = len(candidates) >= 2 and candidate_margin < config.candidate_margin_threshold

    result = {
        "title": item["sample_title"],
        "normalized_title": normalized_title,
        "title_hash": item["title_hash"],
        "freq": int(item["freq"]),
        "source_table": item.get("source_table", ""),
        "candidates": candidates,
        "route": "rule_only",
        "decision": "review",
        "canonical_name": "",
        "confidence": 0.0,
        "reason": "",
        "evidence": [],
    }

    if candidates and candidates[0]["score"] >= 0.97 and candidate_margin >= 0.12 and not noisy_title:
        result.update(
            {
                "decision": "map_alias",
                "canonical_name": candidates[0]["canonical_name"],
                "confidence": float(min(0.98, candidates[0]["score"])),
                "reason": "high_precision_programmatic_match",
                "evidence": [candidates[0]["matched_alias"]],
            }
        )
        cache[normalized_title] = result
        return result

    if router is None or not router.is_configured():
        result["reason"] = "llm_not_configured"
        cache[normalized_title] = result
        return result

    cheap_payload = _run_llm_decision(
        router=router,
        prompt=CHEAP_DECISION_PROMPT,
        item=item,
        candidates=candidates,
        strength="cheap",
    )
    cheap_conf = float(cheap_payload.get("confidence", 0.0) or 0.0)
    escalate = should_escalate(
        cheap_confidence=cheap_conf,
        candidate_margin=candidate_margin,
        is_new_title=is_new_title,
        noisy_title=noisy_title,
        context_conflict=context_conflict,
        has_conflicting_candidates=has_conflicting_candidates,
        cheap_threshold=config.cheap_confidence_threshold,
        margin_threshold=config.candidate_margin_threshold,
    )
    final_payload = cheap_payload
    route = "cheap"
    if escalate:
        route = "strong"
        final_payload = _run_llm_decision(
            router=router,
            prompt=STRONG_DECISION_PROMPT,
            item=item,
            candidates=candidates,
            strength="strong",
        )

    result.update(
        {
            "route": route,
            "decision": str(final_payload.get("decision", "review")),
            "canonical_name": str(final_payload.get("canonical_name", "") or ""),
            "confidence": float(final_payload.get("confidence", 0.0) or 0.0),
            "reason": str(final_payload.get("reason", "") or ""),
            "evidence": list(final_payload.get("evidence", []) or []),
        }
    )
    cache[normalized_title] = result
    return result


def _run_llm_decision(
    *,
    router: LLMRouter,
    prompt: str,
    item: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    strength: str,
) -> Dict[str, Any]:
    candidate_payload = [
        {"canonical_name": c["canonical_name"], "matched_alias": c["matched_alias"], "score": c["score"]}
        for c in candidates[:8]
    ]
    user_prompt = json.dumps(
        {
            "job_title": item["sample_title"],
            "normalized_title": item["normalized_title"],
            "context": str(item.get("sample_description") or "")[:140],
            "candidates": candidate_payload,
        },
        ensure_ascii=False,
    )
    try:
        parsed = router.complete_json(
            system_prompt=prompt,
            user_prompt=user_prompt,
            strength=strength,
            max_output_tokens=320,
            reasoning_effort="low" if strength == "cheap" else "medium",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM decision failed for title=%s strength=%s error=%s", item.get("sample_title", ""), strength, exc)
        return {"decision": "review", "canonical_name": "", "confidence": 0.0, "reason": "llm_json_error", "evidence": []}
    if not isinstance(parsed, dict):
        return {"decision": "review", "canonical_name": "", "confidence": 0.0, "reason": "invalid_json", "evidence": []}
    return parsed


def apply_iteration_results(dictionary: Dict[str, Any], results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    dictionary = ensure_dictionary_shape(dictionary)
    canonical_map = {
        str(item.get("name") or item.get("canonical_name") or "").strip(): item
        for item in dictionary.get("canonical_occupations", [])
        if str(item.get("name") or item.get("canonical_name") or "").strip()
    }
    for result in results:
        if result["decision"] == "map_alias" and result["canonical_name"] in canonical_map and result["confidence"] >= 0.9:
            item = canonical_map[result["canonical_name"]]
            aliases = set(item.get("aliases") or [])
            aliases.add(result["normalized_title"])
            item["aliases"] = sorted(aliases)
            item.setdefault("sources", []).append(
                {
                    "title": result["title"],
                    "source_table": result["source_table"],
                    "confidence": result["confidence"],
                    "evidence": result["evidence"],
                    "at": datetime.now().isoformat(),
                }
            )
        elif result["decision"] == "new_canonical" and result["confidence"] >= 0.93:
            name = result["canonical_name"] or result["normalized_title"]
            if name not in canonical_map:
                canonical_map[name] = {
                    "name": name,
                    "aliases": [result["normalized_title"]] if result["normalized_title"] != name else [],
                    "sources": [
                        {
                            "title": result["title"],
                            "source_table": result["source_table"],
                            "confidence": result["confidence"],
                            "evidence": result["evidence"],
                            "at": datetime.now().isoformat(),
                        }
                    ],
                }
    dictionary["canonical_occupations"] = sorted(canonical_map.values(), key=lambda x: x["name"])
    metadata = dictionary.setdefault("metadata", {})
    metadata["updated_at"] = datetime.now().isoformat()
    return dictionary


def split_results(results: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets = {
        "accepted_new_canonical": [],
        "accepted_alias_mapping": [],
        "review": [],
        "rejected": [],
    }
    for result in results:
        if result["decision"] == "new_canonical" and result["confidence"] >= 0.93:
            buckets["accepted_new_canonical"].append(result)
        elif result["decision"] == "map_alias" and result["confidence"] >= 0.9:
            buckets["accepted_alias_mapping"].append(result)
        elif result["decision"] == "reject":
            buckets["rejected"].append(result)
        else:
            buckets["review"].append(result)
    return buckets


def build_eval_report(results: Sequence[Dict[str, Any]], buckets: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    reasons = Counter(result.get("reason") or "" for result in results)
    routes = Counter(result.get("route") or "" for result in results)
    return {
        "generated_at": datetime.now().isoformat(),
        "total": len(results),
        "route_stats": dict(routes),
        "decision_stats": {key: len(value) for key, value in buckets.items()},
        "error_type_stats": {key: count for key, count in reasons.items() if key},
        "avg_confidence": round(sum(float(item.get("confidence", 0.0)) for item in results) / max(len(results), 1), 4),
    }


def run_iteration(
    config: OccupationDictionaryConfig,
    *,
    pilot_size: int | None = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    router = LLMRouter.from_env(config.env_file)
    dictionary = ensure_dictionary_shape(load_dictionary(config.dictionary_path))
    state = load_state(config.state_path)
    processed_hashes = set(state.get("processed_hashes", []))
    cache_path = config.cache_dir / "classification_cache.json"
    cache = load_dictionary(cache_path)

    batch = [item for item in prepare_pilot_batch(config, pilot_size=pilot_size) if item["title_hash"] not in processed_hashes]
    batch = batch[: (pilot_size or config.batch_size)]
    results: List[Dict[str, Any]] = []
    for item in batch:
        results.append(classify_title(item, dictionary, router, config, cache))

    buckets = split_results(results)
    report = build_eval_report(results, buckets)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    review_path = config.review_dir / f"pilot_review_{timestamp}.jsonl"
    report_path = config.report_dir / f"pilot_report_{timestamp}.json"
    _write_jsonl(review_path, buckets["review"])
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save_dictionary(cache, cache_path)

    if not dry_run:
        updated_dictionary = apply_iteration_results(dictionary, results)
        save_dictionary(updated_dictionary, config.dictionary_path)
        processed_hashes.update(item["title_hash"] for item in batch)
        state["processed_hashes"] = sorted(processed_hashes)
        state.setdefault("runs", []).append(
            {
                "at": datetime.now().isoformat(),
                "batch_size": len(batch),
                "report_path": str(report_path),
                "review_path": str(review_path),
            }
        )
        save_state(state, config.state_path)

    return {
        "batch_size": len(batch),
        "report_path": str(report_path),
        "review_path": str(review_path),
        "report": report,
        "router_configured": router.is_configured(),
        "dry_run": dry_run,
    }


def _write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="职业词典自动迭代")
    parser.add_argument("run", nargs="?", default="run")
    parser.add_argument("--pilot-size", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_parser().parse_args()
    config = load_occupation_dictionary_config()
    summary = run_iteration(config, pilot_size=args.pilot_size, dry_run=args.dry_run)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
