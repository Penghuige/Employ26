"""为基线方法生成测试集提交文件。

在 output/test_set/test_data.json 上运行各基线方法，
生成标准格式的提交 JSON，供 eval_submission.py 评分。

用法:
    python scripts/generate_baseline_submissions.py
    python scripts/generate_baseline_submissions.py --skip-deepseek
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(dotenv_path=str(PROJECT_ROOT / ".env.local"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gen_baselines")

TEST_DATA = str(PROJECT_ROOT / "output" / "test_set" / "test_data.json")
SUBMIT_DIR = PROJECT_ROOT / "output" / "test_set" / "submissions"
EMBEDDING_PATH = r"D:\model\bge-large-zh-v1.5"


def load_test_data() -> List[Dict]:
    with open(TEST_DATA, "r", encoding="utf-8") as f:
        return json.load(f)


# ===================================================================
# Random baseline
# ===================================================================
def gen_random(items: List[Dict]) -> List[Dict]:
    rng = np.random.RandomState(42)
    results = []
    for item in items:
        results.append({
            "test_id": item["test_id"],
            "prediction": rng.choice(["A", "B", "C", "D", "E", "NONE"]),
            "confidence": 0.17,
            "reasoning": "随机选择",
        })
    return results


# ===================================================================
# BM25 baseline
# ===================================================================
def gen_bm25(items: List[Dict]) -> List[Dict]:
    """选 BM25 召回排名第 1 的候选。

    BM25 排名从原始数据中提取（candidate_source 字段），
    不依赖候选在列表中的位置顺序。
    """
    # 从原始 JSON 中加载 BM25 排名映射
    orig_file = str(PROJECT_ROOT / "data" / "project-4-at-2026-05-27-01-51-7cceb9ba.json")
    with open(orig_file, "r", encoding="utf-8") as f:
        orig = json.load(f)

    # task_id -> {letter -> bm25_rank}
    bm25_map = {}
    for t in orig:
        tid = t["id"]
        td = t.get("data", {})
        ranks = {}
        for letter in ["a", "b", "c", "d", "e"]:
            src = td.get(f"candidate_{letter}_source", "")
            rank = 99
            for n in range(1, 6):
                if f"top{n}" in src:
                    rank = n
                    break
            ranks[letter.upper()] = rank
        bm25_map[tid] = ranks

    results = []
    for item in items:
        tid = int(item["test_id"].replace("TSK-", ""))
        ranks = bm25_map.get(tid, {})
        best_letter = min(ranks, key=ranks.get, default="A")
        best_rank = ranks.get(best_letter, 99)
        results.append({
            "test_id": item["test_id"],
            "prediction": best_letter,
            "confidence": round(1.0 / best_rank, 2) if best_rank < 99 else 0.2,
            "reasoning": f"BM25召回排名第{best_rank}",
        })
    return results


# ===================================================================
# BGE baseline
# ===================================================================
def gen_bge(items: List[Dict]) -> List[Dict]:
    model = SentenceTransformer(EMBEDDING_PATH)
    results = []

    for item in tqdm(items, desc="BGE encoding"):
        job_text = f"{item['job_title']} {item['job_requirements'][:1500]}"
        job_emb = model.encode([job_text], normalize_embeddings=True, show_progress_bar=False)

        best_sim = -1
        best_letter = None
        for c in item["candidates"]:
            cand_text = f"{c['title']} {c.get('desc', '')[:500]}"
            cand_emb = model.encode([cand_text], normalize_embeddings=True, show_progress_bar=False)
            sim = float(np.dot(cand_emb, job_emb.T).item())
            if sim > best_sim:
                best_sim = sim
                best_letter = c["letter"]

        results.append({
            "test_id": item["test_id"],
            "prediction": best_letter or "A",
            "confidence": round(best_sim, 4),
            "reasoning": f"BGE相似度最高 {best_sim:.3f}",
        })

    return results


# ===================================================================
# DeepSeek baseline
# ===================================================================
def gen_deepseek(items: List[Dict]) -> List[Dict]:
    from openai import OpenAI
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY 未设置")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    system = (
        "你是《中华人民共和国职业分类大典》（2022年版）的资深分类专家。\n"
        "你的任务是根据招聘岗位的标题和描述，从 5 个候选职业中选择最匹配的一个。\n"
        "评判原则：\n"
        "1. 以岗位描述中的实际工作内容为主要判断依据，不要只看岗位名称。\n"
        "2. 如果你认为5个候选都不合适，请选择 NONE。\n"
        "3. 输出必须是严格的 JSON，不要附带任何解释性文字。"
    )

    results = []
    for item in tqdm(items, desc="DeepSeek judging"):
        cands = item["candidates"]
        user = (
            f"【招聘岗位】\n岗位名称：{item['job_title']}\n"
            f"岗位要求：\n{item['job_requirements'][:3000]}\n\n"
            f"【候选职业】\n"
            f"候选A: [{cands[0]['code']}] {cands[0]['title']}\n"
            f"候选B: [{cands[1]['code']}] {cands[1]['title']}\n"
            f"候选C: [{cands[2]['code']}] {cands[2]['title']}\n"
            f"候选D: [{cands[3]['code']}] {cands[3]['title']}\n"
            f"候选E: [{cands[4]['code']}] {cands[4]['title']}\n\n"
            '请输出 JSON：{"best_candidate":"A"|"B"|"C"|"D"|"E"|"NONE","confidence":0.0,"reasoning":"30字内"}'
        )
        try:
            resp = client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                response_format={"type": "json_object"},
                temperature=0.0, max_tokens=5120, timeout=60,
            )
            msg = resp.choices[0].message
            raw = (getattr(msg, "content", None) or "").strip()
            if not raw:
                raw = (getattr(msg, "reasoning_content", None) or "").strip()
            parsed = _parse_json(raw)
            results.append({
                "test_id": item["test_id"],
                "prediction": parsed.get("best_candidate", "NONE"),
                "confidence": float(parsed.get("confidence", 0)),
                "reasoning": str(parsed.get("reasoning", ""))[:100],
            })
        except Exception as e:
            logger.warning("DS error on %s: %s", item["test_id"], e)
            results.append({
                "test_id": item["test_id"],
                "prediction": "NONE",
                "confidence": 0.0,
                "reasoning": f"API_ERROR: {e}",
            })
        time.sleep(0.3)

    return results


def _parse_json(raw: str) -> Dict:
    text = raw.strip()
    for m in ("```json", "```"):
        text = text.replace(m, "")
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    m = re.search(r'"best_candidate"\s*:\s*"([^"]+)"', text)
    if m:
        return {"best_candidate": m.group(1), "confidence": 0, "reasoning": text[:100]}
    return {}


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-deepseek", action="store_true")
    args = parser.parse_args()

    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    items = load_test_data()
    logger.info("测试集: %d 条", len(items))

    generators = [
        ("random", gen_random),
        ("bm25", gen_bm25),
        ("bge", gen_bge),
    ]
    if not args.skip_deepseek:
        generators.append(("deepseek", gen_deepseek))

    for name, gen_fn in generators:
        logger.info("生成: %s", name)
        t0 = time.time()
        preds = gen_fn(items)
        elapsed = time.time() - t0
        path = SUBMIT_DIR / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(preds, f, ensure_ascii=False, indent=2)
        logger.info("  保存: %s (%.1fs)", path, elapsed)

    logger.info("完成，运行评分: python scripts/eval_submission.py --submit-dir %s", SUBMIT_DIR)


if __name__ == "__main__":
    main()
