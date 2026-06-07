"""A/B 测试：DeepSeek 匹配准确率 —— 有/无 RAG 候选定义。

在 30 条验证集 (inner_id 1-30, 每条 17-20 位标注员) 上：
- Version A (baseline): 只给代码+名称
- Version B (with RAG): 给代码+名称+定义+层级

比较两者与人工多数票的一致率。

用法:
    python scripts/eval_deepseek_rag_ab.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import duckdb
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(dotenv_path=str(PROJECT_ROOT / ".env.local"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("eval_ab")

INPUT_JSON = str(PROJECT_ROOT / "data" / "project-4-at-2026-05-27-01-51-7cceb9ba.json")
DUCKDB_PATH = str(PROJECT_ROOT / "output" / "recruit.duckdb")


# ===================================================================
# Data loading
# ===================================================================
def load_validation_tasks() -> List[Dict]:
    """加载 30 条验证集任务 (inner_id 1-30)。"""
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = []
    for t in data:
        inner = t.get("inner_id", 0)
        if inner < 1 or inner > 30:
            continue
        td = t.get("data", {})
        candidates = []
        for letter in ["a", "b", "c", "d", "e"]:
            candidates.append({
                "letter": letter.upper(),
                "code": td.get(f"candidate_{letter}_code", "") or "",
                "title": td.get(f"candidate_{letter}_title", "") or "",
                "desc": td.get(f"candidate_{letter}_desc", "") or "",
                "source": td.get(f"candidate_{letter}_source", "") or "",
            })

        # 已有多数票（不是 DeepSeek 的，是人工标注员的）
        choices = Counter()
        for ann in t.get("annotations", []):
            for r in ann.get("result", []):
                for c in r.get("value", {}).get("choices", []):
                    c = c.strip()
                    if "都不属于" in c:
                        choices["NONE"] += 1
                    else:
                        c = c.replace("候选", "").strip()
                        if c in "ABCDE":
                            choices[c] += 1

        human_majority = choices.most_common(1)[0][0] if choices else "?"
        human_agreement = choices.most_common(1)[0][1] / max(choices.total(), 1) if choices else 0

        tasks.append({
            "task_id": t["id"],
            "inner_id": inner,
            "job_title": td.get("job_title", "") or "",
            "job_requirements": td.get("job_requirements_clean", "") or "",
            "candidates": candidates,
            "human_majority": human_majority,
            "human_agreement": human_agreement,
            "human_votes": dict(choices),
            "num_annotators": len(t.get("annotations", [])),
        })

    tasks.sort(key=lambda x: x["inner_id"])
    logger.info("加载 %d 条验证集 (annotators per task: %s)",
                 len(tasks), [t["num_annotators"] for t in tasks])
    return tasks


# ===================================================================
# Hierarchy lookup
# ===================================================================
def load_hierarchy_lookup() -> Dict[str, str]:
    """从 DuckDB 加载职业代码 → 层级路径的映射。"""
    try:
        conn = duckdb.connect(DUCKDB_PATH, read_only=True)
        rows = conn.execute("""
            SELECT code, hierarchy_text
            FROM recruit.main.chinese_occupational_dictionary_joined_preprocessed
            WHERE code IS NOT NULL
        """).fetchall()
        conn.close()
        lookup = {row[0]: (row[1] or "") for row in rows if row[0] and row[1]}
        logger.info("层级字典: %d 个代码", len(lookup))
        return lookup
    except Exception:
        logger.warning("无法加载层级字典，将跳过层级信息")
        return {}


# ===================================================================
# Prompt builders (A vs B)
# ===================================================================
JUDGE_SYSTEM_PROMPT = """你是《中华人民共和国职业分类大典》（2022年版）的资深分类专家。
你的任务是根据招聘岗位的标题和描述，从 5 个候选职业中选择最匹配的一个。

评判原则：
1. 以岗位描述中的实际工作内容为主要判断依据，不要只看岗位名称。
2. 岗位名称中的英文缩写（如 LED、CNC、CAD、SQE、PLC、MES等）应作为技术关键词保留原意。
3. 如果你认为5个候选都不合适，请选择 "NONE"。
4. 输出必须是严格的 JSON，不要附带任何解释性文字。"""


def build_prompt_a(task: Dict) -> str:
    """Version A: 只给代码+名称（原版）。"""
    c = task["candidates"]
    parts = [
        f"【招聘岗位】\n岗位名称：{task['job_title']}\n岗位要求：\n{task['job_requirements'][:3000]}\n",
        f"【候选职业】",
        f"候选A: [{c[0]['code']}] {c[0]['title']}",
        f"候选B: [{c[1]['code']}] {c[1]['title']}",
        f"候选C: [{c[2]['code']}] {c[2]['title']}",
        f"候选D: [{c[3]['code']}] {c[3]['title']}",
        f"候选E: [{c[4]['code']}] {c[4]['title']}",
        "",
        '请输出 JSON：{"best_candidate":"A"|"B"|"C"|"D"|"E"|"NONE","confidence":0.0,"reasoning":"30字内"}',
    ]
    return "\n".join(parts)


def build_prompt_b(task: Dict, hierarchy_lookup: Dict[str, str]) -> str:
    """Version B: 代码+名称+定义+层级（RAG增强版）。"""
    c = task["candidates"]
    cand_texts = []
    for cand in c:
        code = cand["code"]
        hier = hierarchy_lookup.get(code, "")
        parts = [
            f"候选{cand['letter']}: [{code}] {cand['title']}",
        ]
        if hier:
            parts.append(f"  层级: {hier}")
        if cand["desc"]:
            parts.append(f"  定义: {cand['desc'][:300]}")
        cand_texts.append("\n".join(parts))

    parts = [
        f"【招聘岗位】\n岗位名称：{task['job_title']}\n岗位要求：\n{task['job_requirements'][:3000]}\n",
        f"【候选职业】",
        "\n\n".join(cand_texts),
        "",
        '请输出 JSON：{"best_candidate":"A"|"B"|"C"|"D"|"E"|"NONE","confidence":0.0,"reasoning":"30字内"}',
    ]
    return "\n".join(parts)


# ===================================================================
# DeepSeek API call
# ===================================================================
class DeepSeekJudge:
    def __init__(self):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY 未设置")
        self._api_key = api_key
        self._base_url = "https://api.deepseek.com"

    def judge(self, system_prompt: str, user_prompt: str,
              timeout_sec: int = 60) -> Dict[str, Any]:
        client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        try:
            resp = client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=5120,
                timeout=timeout_sec,
            )
            message = resp.choices[0].message
            raw = (getattr(message, "content", None) or "").strip()
            if not raw:
                raw = (getattr(message, "reasoning_content", None) or "").strip()
        except Exception as exc:
            return {"best_candidate": "API_ERROR", "confidence": 0,
                    "reasoning": str(exc)[:100], "raw_response": ""}

        parsed = self._parse_json(raw)
        return {
            "best_candidate": parsed.get("best_candidate", "PARSE_ERROR"),
            "confidence": float(parsed.get("confidence", 0)),
            "reasoning": str(parsed.get("reasoning", ""))[:200],
            "raw_response": raw or "",
        }

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        text = raw.strip()
        for marker in ("```json", "```"):
            text = text.replace(marker, "")
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
    tasks = load_validation_tasks()
    hierarchy_lookup = load_hierarchy_lookup()
    judge = DeepSeekJudge()

    results_a = []  # 无 RAG
    results_b = []  # 有 RAG

    for i, task in enumerate(tasks):
        tid = task["task_id"]
        jt = task["job_title"][:30]
        hm = task["human_majority"]
        print(f"\n[{i+1:2d}/30] task={tid} title={jt}")
        print(f"  人工多数票={hm} (agreement={task['human_agreement']:.1%}, "
              f"annotators={task['num_annotators']})")

        # Version A
        prompt_a = build_prompt_a(task)
        result_a = judge.judge(JUDGE_SYSTEM_PROMPT, prompt_a)
        match_a = "OK" if result_a["best_candidate"] == hm else "XX"
        results_a.append({"task": task, "result": result_a, "match": result_a["best_candidate"] == hm})
        print(f"  A(-RAG):   {result_a['best_candidate']:>6s} conf={result_a['confidence']:.2f} "
              f"reason={result_a['reasoning'][:40]} {match_a}")

        # Version B
        prompt_b = build_prompt_b(task, hierarchy_lookup)
        result_b = judge.judge(JUDGE_SYSTEM_PROMPT, prompt_b)
        match_b = "OK" if result_b["best_candidate"] == hm else "XX"
        results_b.append({"task": task, "result": result_b, "match": result_b["best_candidate"] == hm})
        print(f"  B(+RAG):   {result_b['best_candidate']:>6s} conf={result_b['confidence']:.2f} "
              f"reason={result_b['reasoning'][:40]} {match_b}")

        time.sleep(0.3)

    # Summary
    acc_a = sum(1 for r in results_a if r["match"]) / len(results_a)
    acc_b = sum(1 for r in results_b if r["match"]) / len(results_b)

    print(f"\n{'='*60}")
    print(f"  A/B 对比结果 (n={len(tasks)})")
    print(f"{'='*60}")
    print(f"  Version A (无RAG):   {acc_a:.1%}")
    print(f"  Version B (+RAG):    {acc_b:.1%}")
    print(f"  提升:                {acc_b-acc_a:+.1%}")

    # 逐条对比
    print(f"\n  逐条差异:")
    changed = 0
    for i, (ra, rb) in enumerate(zip(results_a, results_b)):
        t = ra["task"]
        hm = t["human_majority"]
        da = ra["result"]["best_candidate"]
        db = rb["result"]["best_candidate"]
        if da != db:
            changed += 1
            improv = "↑" if (rb["match"] and not ra["match"]) else "↓"
            print(f"  [{improv}] #{t['inner_id']:2d} {t['job_title'][:25]:25s} "
                  f"human={hm} A→{da} B→{db}")
    print(f"  策略变化数: {changed}/{len(tasks)}")

    # 按类别细分 NONE
    for name, results in [("A (无RAG)", results_a), ("B (+RAG)", results_b)]:
        none_human = sum(1 for r in results if r["task"]["human_majority"] == "NONE")
        none_pred = sum(1 for r in results if r["result"]["best_candidate"] == "NONE")
        none_hit = sum(1 for r in results
                       if r["task"]["human_majority"] == "NONE"
                       and r["result"]["best_candidate"] == "NONE")
        print(f"\n  {name} NONE 检测: 人工={none_human}, 预测={none_pred}, "
              f"命中={none_hit} (recall={none_hit/max(none_human,1):.0%})")


if __name__ == "__main__":
    main()
