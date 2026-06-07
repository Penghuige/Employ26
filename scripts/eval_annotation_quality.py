"""标注质量评估脚本 — 方案二：DeepSeek V4 Pro 作为独立评判器。

从标注数据集中抽取验证集任务，用 DeepSeek V4 Pro 独立判断 best_candidate，
与标注员多数投票结果对比，计算一致性指标并输出分歧案例供人工仲裁。

用法:
    .conda/python.exe scripts/eval_annotation_quality.py [--limit 30] [--output output/reports/annotation_quality.md]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import duckdb
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(dotenv_path=str(PROJECT_ROOT / ".env.local"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DeepSeek V4 Pro 评估 Prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """你是《中国职业分类大典》的资深分类专家。
你的任务是根据招聘岗位的标题和描述，从 5 个候选职业中选择最匹配的一个。

评判原则：
1. 以岗位描述（job_requirements）中的实际工作内容为主要判断依据，不要只看岗位名称。
2. 如果岗位名称与描述不一致，以描述为准。
3. 候选职业的代码（code）提供了职业大类信息，大類相同但细类不同时优先考虑工作内容重叠度。
4. 如果你认为5个候选都不合适，请选择 "NONE"。
5. 输出必须是严格的 JSON，不要附带任何解释性文字。"""

JUDGE_USER_TEMPLATE = """请从以下 5 个候选职业中，选择与招聘岗位最匹配的一个。

【招聘岗位】
岗位名称：{job_title}
岗位要求：
{job_requirements}

【候选职业】
候选A: [{code_a}] {title_a}
  描述: {desc_a}

候选B: [{code_b}] {title_b}
  描述: {desc_b}

候选C: [{code_c}] {title_c}
  描述: {desc_c}

候选D: [{code_d}] {title_d}
  描述: {desc_d}

候选E: [{code_e}] {title_e}
  描述: {desc_e}

请输出 JSON：
{{"best_candidate": "A"|"B"|"C"|"D"|"E"|"NONE", "confidence": 0.0-1.0, "reasoning": "简短理由(30字内)"}}"""


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_tasks_with_candidates(
    db_path: str, where: str = "is_validation = true", limit: int = 0
) -> List[Dict[str, Any]]:
    """从 DuckDB 加载任务及其 5 个候选的完整信息。

    Args:
        db_path: DuckDB 数据库路径。
        where: 筛选条件。
        limit: 限制条数（0=不限）。

    Returns:
        List[Dict]: 含 job_title, job_requirements, 5 候选 code/title/desc 的任务列表。
    """
    conn = duckdb.connect(db_path, read_only=True)
    try:
        sql = f"""
            SELECT id, job_title, job_requirements, data_raw,
                   cand_a_code, cand_a_title, cand_a_source,
                   cand_b_code, cand_b_title, cand_b_source,
                   cand_c_code, cand_c_title, cand_c_source,
                   cand_d_code, cand_d_title, cand_d_source,
                   cand_e_code, cand_e_title, cand_e_source,
                   annotations_completed
            FROM recruit.main.label_studio_tasks_v2
            WHERE {where}
            ORDER BY id
        """
        if limit > 0:
            sql += f" LIMIT {limit}"

        tasks = []
        for row in conn.execute(sql).fetchall():
            task = {
                "id": row[0],
                "job_title": row[1],
                "job_requirements": row[2],
                "candidates": {},
            }
            # 从 data_raw 提取 desc
            data_raw = {}
            if row[3]:
                try:
                    data_raw = json.loads(row[3])
                except json.JSONDecodeError:
                    pass
            for i, letter in enumerate(["A", "B", "C", "D", "E"]):
                base = 4 + i * 3
                code = row[base]
                title = row[base + 1]
                source = row[base + 2]
                desc = data_raw.get(f"candidate_{letter.lower()}_desc", "")
                task["candidates"][letter] = {
                    "code": code, "title": title, "source": source, "desc": desc,
                }
            task["annotations_completed"] = row[19]
            tasks.append(task)
        return tasks
    finally:
        conn.close()


def load_annotator_votes(
    db_path: str, task_ids: List[int]
) -> Dict[int, List[Dict[str, Any]]]:
    """加载指定任务的标注员投票记录。

    Args:
        db_path: DuckDB 数据库路径。
        task_ids: 任务 ID 列表。

    Returns:
        Dict[int, List[Dict]]: {task_id: [{annotator_id, best_candidate, reason}, ...]}。
    """
    conn = duckdb.connect(db_path, read_only=True)
    try:
        placeholders = ",".join("?" * len(task_ids))
        rows = conn.execute(
            f"""SELECT task_id, annotator_id, best_candidate, reason, lead_time_sec
                FROM recruit.main.label_studio_annotations_v2
                WHERE task_id IN ({placeholders})
                  AND best_candidate != ''
                ORDER BY task_id, annotator_id""",
            task_ids,
        ).fetchall()

        votes: Dict[int, List[Dict]] = {}
        for row in rows:
            tid = row[0]
            if tid not in votes:
                votes[tid] = []
            votes[tid].append({
                "annotator_id": row[1],
                "best_candidate": row[2],
                "reason": row[3] or "",
                "lead_time_sec": row[4],
            })
        return votes
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# DeepSeek V4 Pro 评判
# ---------------------------------------------------------------------------

def build_judge_prompt(task: Dict[str, Any]) -> str:
    """为单个任务构建评估 prompt。

    Args:
        task: 含 job_title, job_requirements, candidates 的任务字典。

    Returns:
        str: 完整的 user prompt。
    """
    c = task["candidates"]
    def _desc(letter: str) -> str:
        d = c[letter].get("desc", "")
        return d[:300] if d else "(描述缺失，请根据标题和代码判断)"

    return JUDGE_USER_TEMPLATE.format(
        job_title=task["job_title"],
        job_requirements=task["job_requirements"][:3000],
        code_a=c["A"]["code"], title_a=c["A"]["title"], desc_a=_desc("A"),
        code_b=c["B"]["code"], title_b=c["B"]["title"], desc_b=_desc("B"),
        code_c=c["C"]["code"], title_c=c["C"]["title"], desc_c=_desc("C"),
        code_d=c["D"]["code"], title_d=c["D"]["title"], desc_d=_desc("D"),
        code_e=c["E"]["code"], title_e=c["E"]["title"], desc_e=_desc("E"),
    )


def judge_single(
    client: OpenAI,
    task: Dict[str, Any],
    model: str = "deepseek-v4-pro",
) -> Dict[str, Any]:
    """用 DeepSeek 对单个任务进行评判。

    Args:
        client: OpenAI 客户端。
        task: 任务字典。
        model: 模型名称。

    Returns:
        Dict: {"best_candidate": str, "confidence": float, "reasoning": str, "raw_response": str}。
    """
    user_prompt = build_judge_prompt(task)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        raw = response.choices[0].message.content or ""
    except Exception as exc:
        logger.error("Task #%d API 调用失败: %s", task["id"], exc)
        return {"best_candidate": "ERROR", "confidence": 0, "reasoning": str(exc), "raw_response": ""}

    # 解析 JSON 输出
    try:
        # 去掉可能的 markdown 代码块
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = json.loads(cleaned)
        return {
            "best_candidate": parsed.get("best_candidate", "ERROR"),
            "confidence": float(parsed.get("confidence", 0)),
            "reasoning": str(parsed.get("reasoning", ""))[:200],
            "raw_response": raw,
        }
    except json.JSONDecodeError:
        logger.warning("Task #%d JSON 解析失败, raw: %s", task["id"], raw[:200])
        # 尝试从 raw text 中直接提取 A/B/C/D/E/NONE
        for letter in ["A", "B", "C", "D", "E", "NONE"]:
            if f'"{letter}"' in raw or f"'{letter}'" in raw or f"候选{letter}" in raw:
                return {"best_candidate": letter, "confidence": 0.5, "reasoning": "从原始文本提取", "raw_response": raw}
        return {"best_candidate": "PARSE_ERROR", "confidence": 0, "reasoning": raw[:200], "raw_response": raw}


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------

def majority_vote(votes: List[Dict]) -> Tuple[str, int, int]:
    """计算多数投票结果。

    Args:
        votes: 标注员投票列表。

    Returns:
        tuple: (多数选择, 票数, 总票数)。
    """
    choices = [v["best_candidate"] for v in votes]
    counter = Counter(choices)
    top = counter.most_common(1)[0]
    return top[0], top[1], len(choices)


def compute_agreement(
    tasks: List[Dict],
    votes_map: Dict[int, List[Dict]],
    judge_results: Dict[int, Dict],
) -> Dict[str, Any]:
    """计算 DeepSeek 与标注员多数投票的一致性指标。

    Args:
        tasks: 任务列表。
        votes_map: 标注员投票字典。
        judge_results: DeepSeek 评判结果字典。

    Returns:
        Dict: 含 kappa, accuracy, agreement_rate, disagreements 等指标。
    """
    total = 0
    judge_majority_agree = 0
    judge_any_agree = 0
    categories = set()
    disagreements = []

    for task in tasks:
        tid = task["id"]
        votes = votes_map.get(tid, [])
        if not votes or tid not in judge_results:
            continue
        jr = judge_results[tid]
        if jr["best_candidate"] in ("ERROR", "PARSE_ERROR"):
            continue

        majority, maj_count, total_votes = majority_vote(votes)
        # 统一格式: "候选A" → "A", DeepSeek 返回 "A" 保持
        majority_normalized = majority.replace("候选", "").strip()
        judge_normalized = jr["best_candidate"].replace("候选", "").strip()

        categories.add(majority_normalized)
        categories.add(judge_normalized)
        total += 1

        if judge_normalized == majority_normalized:
            judge_majority_agree += 1

        # 判断 DeepSeek 的选择是否至少有一个标注员同意
        any_votes_for_judge = sum(
            1 for v in votes
            if v["best_candidate"].replace("候选", "").strip() == judge_normalized
        )
        if any_votes_for_judge > 0:
            judge_any_agree += 1

        # 记录分歧
        if judge_normalized != majority_normalized:
            disagreements.append({
                "task_id": tid,
                "job_title": task["job_title"],
                "majority_vote": majority_normalized,
                "majority_count": maj_count,
                "total_votes": total_votes,
                "judge_choice": judge_normalized,
                "judge_confidence": jr["confidence"],
                "judge_reasoning": jr["reasoning"],
                "vote_distribution": dict(Counter(
                    v["best_candidate"].replace("候选", "").strip() for v in votes
                )),
            })

    if total == 0:
        return {"error": "无可评估的有效任务", "total": 0}

    # 计算 Cohen's Kappa
    # 简化: 将 judge 视为第二个标注员，计算与多数投票的 kappa
    p_o = judge_majority_agree / total  # observed agreement

    # expected agreement: sum(P_category_majority * P_category_judge)
    all_cats = list(categories)
    p_e = 0.0
    for cat in all_cats:
        p_maj = sum(1 for t in tasks if t["id"] in votes_map and t["id"] in judge_results
                    and majority_vote(votes_map[t["id"]])[0].replace("候选", "").strip() == cat) / total
        p_judge = sum(1 for t in tasks if t["id"] in judge_results
                      and judge_results[t["id"]]["best_candidate"].replace("候选", "").strip() == cat) / total
        p_e += p_maj * p_judge

    kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 0

    return {
        "total_tasks": total,
        "judge_majority_agree": judge_majority_agree,
        "judge_any_agree": judge_any_agree,
        "agreement_rate": judge_majority_agree / total,
        "any_agreement_rate": judge_any_agree / total,
        "cohens_kappa": round(kappa, 4),
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
        "kappa_interpretation": _interpret_kappa(kappa),
    }


def _interpret_kappa(kappa: float) -> str:
    """解释 Kappa 值。"""
    if kappa > 0.8:
        return "几乎完美一致 — 标注质量很高，或 DeepSeek 与标注员判断高度吻合"
    if kappa > 0.6:
        return "高度一致 — 标注质量较好，少数分歧需人工仲裁"
    if kappa > 0.4:
        return "中等一致 — 标注存在一定主观性，需抽样复审"
    if kappa > 0.2:
        return "低度一致 — 标注可能存在系统性问题，建议扩大复审范围"
    return "轻微一致 — 标注与模型判断几乎随机，需重新审视标注指南"


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------

def generate_report(
    metrics: Dict[str, Any],
    tasks: List[Dict],
    votes_map: Dict[int, List[Dict]],
    judge_results: Dict[int, Dict],
    output_path: Path | None = None,
) -> str:
    """生成 Markdown 格式的评估报告。

    Args:
        metrics: 一致性指标。
        tasks: 任务列表。
        votes_map: 标注员投票。
        judge_results: DeepSeek 评判结果。
        output_path: 可选的输出文件路径。

    Returns:
        str: Markdown 格式报告。
    """
    lines = [
        "# 标注质量评估报告 — 方案二：DeepSeek V4 Pro 独立评判",
        "",
        f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"评估模型: deepseek-v4-pro",
        f"评估任务数: {metrics.get('total_tasks', 0)}",
        "",
        "## 一、一致性指标",
        "",
        f"| 指标 | 数值 |",
        f"|------|------|",
        f"| DeepSeek 与多数票一致 | {metrics['judge_majority_agree']}/{metrics['total_tasks']} = **{metrics['agreement_rate']:.1%}** |",
        f"| DeepSeek 被至少一位标注员认可 | {metrics['judge_any_agree']}/{metrics['total_tasks']} = **{metrics['any_agreement_rate']:.1%}** |",
        f"| Cohen's Kappa | **{metrics['cohens_kappa']:.4f}** |",
        f"| Kappa 解读 | {metrics['kappa_interpretation']} |",
        f"| 分歧任务数 | **{metrics['disagreement_count']}** |",
        "",
        "## 二、分歧案例（需人工仲裁）",
        "",
    ]

    disagreements = metrics.get("disagreements", [])
    if disagreements:
        lines.append(f"共 {len(disagreements)} 条 DeepSeek 与多数票不一致的案例：")
        lines.append("")
        lines.append("| # | Task ID | 岗位名称 | 多数票 | 票数 | DeepSeek | 置信度 | DeepSeek 理由 |")
        lines.append("|---|---------|----------|--------|------|----------|--------|---------------|")
        for i, d in enumerate(disagreements[:30], 1):
            title = str(d["job_title"])[:25]
            lines.append(
                f"| {i} | {d['task_id']} | {title} | {d['majority_vote']} "
                f"| {d['majority_count']}/{d['total_votes']} "
                f"| {d['judge_choice']} | {d['judge_confidence']:.2f} "
                f"| {d['judge_reasoning'][:60]} |"
            )
    else:
        lines.append("无分歧案例。")

    lines.extend([
        "",
        "## 三、逐任务详细结果",
        "",
        "| Task ID | 岗位名称 | 多数票 | DeepSeek | 一致? | DeepSeek 置信度 |",
        "|---------|----------|--------|----------|-------|-----------------|",
    ])
    for task in tasks:
        tid = task["id"]
        votes = votes_map.get(tid, [])
        if not votes or tid not in judge_results:
            continue
        jr = judge_results[tid]
        majority, maj_cnt, total_v = majority_vote(votes)
        maj_norm = majority.replace("候选", "").strip()
        jr_norm = jr["best_candidate"].replace("候选", "").strip()
        agree = "Y" if jr_norm == maj_norm else "N"
        lines.append(
            f"| {tid} | {str(task['job_title'])[:30]} | {maj_norm} ({maj_cnt}/{total_v}) "
            f"| {jr_norm} | {agree} | {jr['confidence']:.2f} |"
        )

    report = "\n".join(lines)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        logger.info("报告已保存至: %s", output_path)
    return report


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="标注质量评估 — DeepSeek V4 Pro 独立评判")
    parser.add_argument("--limit", type=int, default=30, help="评估任务数上限")
    parser.add_argument("--output", type=str, default="", help="报告输出路径")
    args = parser.parse_args()

    db_path = str(PROJECT_ROOT / "output" / "recruit.duckdb")

    # 1. 加载验证集任务
    logger.info("加载验证集任务...")
    tasks = load_tasks_with_candidates(db_path, where="is_validation = true", limit=args.limit)
    task_ids = [t["id"] for t in tasks]
    logger.info("加载 %d 个验证集任务", len(tasks))

    # 2. 加载标注员投票
    logger.info("加载标注员投票...")
    votes_map = load_annotator_votes(db_path, task_ids)
    for tid in task_ids:
        n = len(votes_map.get(tid, []))
        logger.info("  Task #%d: %d 条标注", tid, n)

    # 3. 初始化 DeepSeek 客户端
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        logger.error("DEEPSEEK_API_KEY 未在 .env.local 中设置")
        sys.exit(1)
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    logger.info("DeepSeek 客户端已初始化")

    # 4. 逐任务评判
    logger.info("开始 DeepSeek V4 Pro 评判 (%d 条)...", len(tasks))
    judge_results: Dict[int, Dict[str, Any]] = {}
    for i, task in enumerate(tasks, 1):
        tid = task["id"]
        logger.info("[%d/%d] Task #%d: %s", i, len(tasks), tid, task["job_title"][:40])
        result = judge_single(client, task)
        judge_results[tid] = result
        logger.info("  → %s (confidence=%.2f) %s",
                     result["best_candidate"], result["confidence"], result["reasoning"][:80])
        if i < len(tasks):
            time.sleep(0.3)

    # 5. 计算一致性
    metrics = compute_agreement(tasks, votes_map, judge_results)
    logger.info("一致性计算完成: agreement=%.1f%%, kappa=%.4f",
                 metrics["agreement_rate"] * 100, metrics["cohens_kappa"])

    # 6. 生成报告
    output_path = None
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = PROJECT_ROOT / "output" / "reports" / "annotation_quality_deepseek.md"

    report = generate_report(metrics, tasks, votes_map, judge_results, output_path)
    print(report)

    # 7. 保存详细 JSON
    detail_path = output_path.with_suffix(".json")
    detail_data = {
        "metrics": {k: v for k, v in metrics.items() if k != "disagreements"},
        "disagreements": metrics.get("disagreements", []),
        "judge_results": {
            str(tid): jr for tid, jr in judge_results.items()
        },
    }
    detail_path.write_text(json.dumps(detail_data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("详细数据已保存至: %s", detail_path)


if __name__ == "__main__":
    main()
