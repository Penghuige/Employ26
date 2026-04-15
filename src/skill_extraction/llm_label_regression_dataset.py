"""
Automatically build a regression dataset with local LLM labeling.

使用方法；
    python -m src.skill_extraction.llm_label_regression_dataset --sample-size 400 --num-votes 3
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import logging
from math import ceil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .config import load_skill_extraction_config
from .llm_labeling_utils import (
    extract_json_from_response,
    load_requirement_match_rows,
    normalize_skill_key,
    prepare_labeling_frame,
    run_prompt_pairs,
    safe_text,
    stratified_sample_frame,
    write_jsonl,
)


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


DEFAULT_OUTPUT_PATH = "output/skill_extraction/regression/flat_skill_regression_dataset.jsonl"
DEFAULT_SAMPLE_SIZE = 400

SYSTEM_PROMPT = """\
你是招聘硬技能标注专家。请从岗位文本中抽取可以直接落入技能词典的硬技能。

标注规则：
1. 只保留硬技能：编程语言、框架、数据库、工具软件、办公软件、设备、工艺方法、具体证书/资质。
2. 排除软技能、学历、年限、岗位职责、福利待遇、泛化容器词。
3. 不要输出“测试”“办公软件”“数据分析工具”“资格证书”“专业知识”这类过泛名称。
4. evidence 必须是岗位原文中的连续子串，不能编造。
5. name 需要标准化；如果文本里出现别名，也要统一成更规范的技能名。

只输出 JSON：
{"skills":[{"name":"标准技能名","evidence":"原文证据","skill_type":"技能类别"}]}
"""

USER_TEMPLATE = """\
样本ID: {sample_id}
岗位名称: {job_title}
职业中类: {occupation_title}
职业编码: {occupation_code}

岗位文本：
{text}
"""


def _clean_skill_item(item: Dict, text: str) -> Dict | None:
    """Keep only high-confidence skills whose evidence is grounded in text."""
    name = safe_text(item.get("name", ""))
    evidence = safe_text(item.get("evidence", ""))
    skill_type = safe_text(item.get("skill_type", ""))
    if not name or not evidence or evidence not in text:
        return None
    return {
        "name": name,
        "evidence": evidence,
        "skill_type": skill_type,
    }


def _aggregate_votes(
    vote_results: Sequence[Sequence[Dict]],
    min_vote_support: int,
) -> List[Dict]:
    """Merge multiple LLM votes into a stable gold-skill list."""
    counts: Counter[str] = Counter()
    items_by_key: Dict[str, List[Dict]] = defaultdict(list)

    for vote in vote_results:
        seen_in_vote: set[str] = set()
        for item in vote:
            key = normalize_skill_key(item.get("name", ""))
            if not key or key in seen_in_vote:
                continue
            seen_in_vote.add(key)
            counts[key] += 1
            items_by_key[key].append(item)

    merged: List[Dict] = []
    for key, vote_count in counts.items():
        if vote_count < min_vote_support:
            continue
        candidates = items_by_key[key]
        name_counter = Counter(safe_text(item.get("name", "")) for item in candidates)
        evidence_counter = Counter(safe_text(item.get("evidence", "")) for item in candidates)
        type_counter = Counter(safe_text(item.get("skill_type", "")) for item in candidates)
        merged.append(
            {
                "name": name_counter.most_common(1)[0][0],
                "evidence": evidence_counter.most_common(1)[0][0],
                "skill_type": type_counter.most_common(1)[0][0],
                "vote_count": vote_count,
            }
        )

    merged.sort(key=lambda item: (-int(item["vote_count"]), item["name"].casefold()))
    return merged


def build_regression_dataset(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    source_table: str | None = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    limit: int | None = None,
    seed: int = 42,
    model_path: str | None = None,
    batch_size: int = 16,
    max_text_chars: int = 900,
    min_text_chars: int = 20,
    num_votes: int = 1,
    min_vote_support: int | None = None,
    gpu_memory_utilization: float = 0.80,
    max_model_len: int = 8192,
    max_num_seqs: int = 48,
) -> Dict:
    """Sample JD rows and let the local LLM generate regression gold labels."""
    config = load_skill_extraction_config()
    resolved_model_path = str(model_path or config.llm_model_path)
    raw_df = load_requirement_match_rows(config=config, source_table=source_table, limit=limit)
    prepared_df = prepare_labeling_frame(
        raw_df,
        max_text_chars=max_text_chars,
        min_text_chars=min_text_chars,
    )
    sampled_df = stratified_sample_frame(prepared_df, sample_size=sample_size, seed=seed)
    sampled_rows = sampled_df.to_dict(orient="records")
    if not sampled_rows:
        raise ValueError("No valid rows were found for regression dataset labeling.")

    vote_support = int(min_vote_support or ceil(max(1, int(num_votes)) / 2))
    prompt_pairs: List[Tuple[str, str]] = []
    task_rows: List[Tuple[str, int]] = []
    sample_lookup = {row["sample_id"]: row for row in sampled_rows}

    for row in sampled_rows:
        for vote_index in range(max(1, int(num_votes))):
            prompt_pairs.append(
                (
                    SYSTEM_PROMPT,
                    USER_TEMPLATE.format(
                        sample_id=row["sample_id"],
                        job_title=row["job_title"] or "未知岗位",
                        occupation_title=row["occupation_title"] or "未知职业",
                        occupation_code=row["occupation_code"] or "",
                        text=row["text"],
                    ),
                )
            )
            task_rows.append((row["sample_id"], vote_index))

    llm_outputs = run_prompt_pairs(
        model_path=resolved_model_path,
        prompt_pairs=prompt_pairs,
        batch_size=batch_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )

    votes_by_sample: Dict[str, List[List[Dict]]] = defaultdict(list)
    parse_success = 0
    for (sample_id, _vote_index), output_text in zip(task_rows, llm_outputs):
        parsed = extract_json_from_response(output_text) or {}
        cleaned_items = [
            cleaned
            for cleaned in (
                _clean_skill_item(item, sample_lookup[sample_id]["text"])
                for item in parsed.get("skills", [])
            )
            if cleaned is not None
        ]
        votes_by_sample[sample_id].append(cleaned_items)
        if parsed:
            parse_success += 1

    output_rows: List[Dict] = []
    total_gold_skills = 0
    for row in sampled_rows:
        aggregated_items = _aggregate_votes(
            votes_by_sample.get(row["sample_id"], []),
            min_vote_support=vote_support,
        )
        gold_skills = [item["name"] for item in aggregated_items]
        total_gold_skills += len(gold_skills)
        output_rows.append(
            {
                "sample_id": row["sample_id"],
                "text": row["text"],
                "job_title": row["job_title"],
                "occupation_title": row["occupation_title"],
                "occupation_code": row["occupation_code"],
                "gold_skills": gold_skills,
                "gold_skill_items": aggregated_items,
                "llm_vote_count": int(num_votes),
                "llm_min_vote_support": vote_support,
            }
        )

    write_jsonl(output_path, output_rows)
    summary = {
        "output_path": str(output_path),
        "source_table": source_table or config.requirement_match_table,
        "sample_count": len(output_rows),
        "parse_success": parse_success,
        "prompt_count": len(prompt_pairs),
        "avg_gold_skills_per_sample": total_gold_skills / max(len(output_rows), 1),
        "model_path": resolved_model_path,
        "num_votes": int(num_votes),
        "min_vote_support": vote_support,
    }
    logger.info(
        "Regression dataset written to %s (%d samples, avg skills %.2f)",
        output_path,
        len(output_rows),
        summary["avg_gold_skills_per_sample"],
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Use local LLM to build a regression dataset.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output JSONL path.")
    parser.add_argument("--source-table", default=None, help="DuckDB source table.")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Number of rows to label.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit before sampling.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--model", default=None, help="LLM model path; defaults to config/database.yaml.")
    parser.add_argument("--batch-size", type=int, default=16, help="vLLM prompt batch size.")
    parser.add_argument("--max-text-chars", type=int, default=900, help="Maximum chars kept per sample.")
    parser.add_argument("--min-text-chars", type=int, default=20, help="Minimum chars required per sample.")
    parser.add_argument("--num-votes", type=int, default=1, help="Number of LLM votes per sample.")
    parser.add_argument("--min-vote-support", type=int, default=None, help="Minimum votes required to keep a skill.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80, help="vLLM GPU memory utilization.")
    parser.add_argument("--max-model-len", type=int, default=8192, help="vLLM max model length.")
    parser.add_argument("--max-num-seqs", type=int, default=48, help="vLLM max concurrent sequences.")
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()
    summary = build_regression_dataset(
        output_path=args.output,
        source_table=args.source_table,
        sample_size=args.sample_size,
        limit=args.limit,
        seed=args.seed,
        model_path=args.model,
        batch_size=args.batch_size,
        max_text_chars=args.max_text_chars,
        min_text_chars=args.min_text_chars,
        num_votes=args.num_votes,
        min_vote_support=args.min_vote_support,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
