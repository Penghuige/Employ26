"""LLM 标注质量审计脚本。

对已标注任务，用 DeepSeek V4 Pro judge 模式独立评判，
与原始标注员选择对比，发现潜在标注错误。

用法:
    # 审计 100 条
    python scripts/audit_annotations.py --limit 100

    # 审计全部 + 输出报告
    python scripts/audit_annotations.py --output output/reports/annotation_audit.md

    # 只看验证集
    python scripts/audit_annotations.py --validation-only
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
from typing import Any, Dict, List

import duckdb
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(dotenv_path=str(PROJECT_ROOT / ".env.local"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audit")

# ---------------------------------------------------------------------------
# DeepSeek Judge Prompt（复用 eval_annotation_quality 的验证过的 prompt）
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """你是《中华人民共和国职业分类大典》（2022年版）的资深分类专家。
你的任务是根据招聘岗位的标题和描述，从 5 个候选职业中选择最匹配的一个。

评判原则：
1. 以岗位描述中的实际工作内容为主要判断依据，不要只看岗位名称。
2. 候选职业的代码提供了职业大类信息，大类相同但细类不同时优先考虑工作内容重叠度。
3. 如果你认为5个候选都不合适，请选择 "NONE"。
4. 输出必须是严格的 JSON，不要附带任何解释性文字。"""

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
{{"best_candidate": "A"|"B"|"C"|"D"|"E"|"NONE", "confidence": 0.0-1.0, "reasoning": "简短理由(30字内)"}}"""  # noqa: E501


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_annotated_tasks(
    db_path: str, validation_only: bool = False, limit: int = 0,
) -> List[Dict[str, Any]]:
    """加载已标注任务（含候选人信息 + 标注员选择）。

    Args:
        db_path: DuckDB 路径。
        validation_only: 只看验证集。
        limit: 条数限制。

    Returns:
        List[Dict]: 任务列表。
    """
    conn = duckdb.connect(db_path, read_only=True)
    try:
        where = "a.best_candidate != ''"
        if validation_only:
            where += " AND t.is_validation = true"

        sql = f"""
            SELECT DISTINCT a.task_id, a.best_candidate AS annotator_choice,
                   a.annotator_id, a.reason AS annotator_reason,
                   t.job_title, t.job_requirements,
                   t.cand_a_code, t.cand_a_title,
                   t.cand_b_code, t.cand_b_title,
                   t.cand_c_code, t.cand_c_title,
                   t.cand_d_code, t.cand_d_title,
                   t.cand_e_code, t.cand_e_title,
                   t.is_validation, t.sample_source
            FROM recruit.main.label_studio_annotations_v2 a
            JOIN recruit.main.label_studio_tasks_v2 t ON a.task_id = t.id
            WHERE {where}
            ORDER BY a.task_id, a.annotator_id
        """
        if limit > 0:
            sql += f" LIMIT {limit}"

        rows = conn.execute(sql).fetchall()
        tasks = []
        for row in rows:
            tasks.append({
                "task_id": row[0],
                "annotator_choice": row[1].replace("候选", "").strip(),
                "annotator_id": row[2],
                "annotator_reason": row[3] or "",
                "job_title": row[4] or "",
                "job_requirements": row[5] or "",
                "candidates": {
                    "A": {"code": row[6] or "", "title": row[7] or ""},
                    "B": {"code": row[8] or "", "title": row[9] or ""},
                    "C": {"code": row[10] or "", "title": row[11] or ""},
                    "D": {"code": row[12] or "", "title": row[13] or ""},
                    "E": {"code": row[14] or "", "title": row[15] or ""},
                },
                "is_validation": bool(row[16]),
                "sample_source": row[17] or "",
            })
        return tasks
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# DeepSeek 评判
# ---------------------------------------------------------------------------

class AnnotationAuditor:
    """用 DeepSeek 审计标注质量。"""

    def __init__(self):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY 未设置")
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    def judge(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """对单条任务进行独立评判。

        Args:
            task: 含 job_title, job_requirements, candidates 的任务。

        Returns:
            Dict: {"best_candidate": str, "confidence": float, "reasoning": str}。
        """
        c = task["candidates"]
        user_prompt = JUDGE_USER_TEMPLATE.format(
            job_title=task["job_title"],
            job_requirements=str(task["job_requirements"])[:3000],
            code_a=c["A"]["code"], title_a=c["A"]["title"], desc_a="",
            code_b=c["B"]["code"], title_b=c["B"]["title"], desc_b="",
            code_c=c["C"]["code"], title_c=c["C"]["title"], desc_c="",
            code_d=c["D"]["code"], title_d=c["D"]["title"], desc_d="",
            code_e=c["E"]["code"], title_e=c["E"]["title"], desc_e="",
        )

        try:
            resp = self.client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1, max_tokens=256,
            )
            raw = (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("Task #%d API 失败: %s", task["task_id"], exc)
            return {"best_candidate": "ERROR", "confidence": 0, "reasoning": str(exc)}

        return self._parse_response(raw)

    @staticmethod
    def _parse_response(raw: str) -> Dict[str, Any]:
        import re
        text = raw.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        m = re.search(r'\{[^{}]*\}', text)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return {"best_candidate": "PARSE_ERROR", "confidence": 0, "reasoning": raw[:100]}


# ---------------------------------------------------------------------------
# 审计逻辑
# ---------------------------------------------------------------------------

def run_audit(
    tasks: List[Dict], auditor: AnnotationAuditor, sleep: float = 0.3,
) -> List[Dict[str, Any]]:
    """逐任务审计。

    Args:
        tasks: 任务列表。
        auditor: DeepSeek 评判器。
        sleep: API 调用间隔。

    Returns:
        List[Dict]: 审计结果。
    """
    results = []
    for task in tqdm(tasks, desc="审计标注", unit="task"):
        judge_result = auditor.judge(task)
        judge_choice = judge_result.get("best_candidate", "ERROR")
        annotator_choice = task["annotator_choice"]

        # 判定一致性
        agreement = "agree" if judge_choice == annotator_choice else "disagree"
        if judge_choice == "NONE" and annotator_choice != "NONE":
            agreement = "llm_none"  # LLM 认为都不合适，标注员选了
        elif judge_choice == "ERROR" or judge_choice == "PARSE_ERROR":
            agreement = "error"

        results.append({
            "task_id": task["task_id"],
            "annotator_id": task["annotator_id"],
            "annotator_choice": annotator_choice,
            "annotator_reason": task["annotator_reason"][:100],
            "judge_choice": judge_choice,
            "judge_confidence": judge_result.get("confidence", 0),
            "judge_reasoning": judge_result.get("reasoning", "")[:100],
            "agreement": agreement,
            "job_title": task["job_title"],
            "is_validation": task["is_validation"],
        })
        time.sleep(sleep)
    return results


def generate_report(results: List[Dict], output_path: Path | None = None) -> str:
    """生成审计报告。

    Args:
        results: 审计结果列表。
        output_path: 输出文件路径。

    Returns:
        str: Markdown 报告。
    """
    total = len(results)
    agree = sum(1 for r in results if r["agreement"] == "agree")
    disagree = sum(1 for r in results if r["agreement"] == "disagree")
    llm_none = sum(1 for r in results if r["agreement"] == "llm_none")
    errors = sum(1 for r in results if r["agreement"] == "error")
    valid = total - errors

    lines = [
        "# 标注质量审计报告（DeepSeek V4 Pro 独立评判）",
        "",
        f"审计时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"审计任务数: {total}（有效: {valid}）",
        "",
        "## 一、总体一致率",
        "",
        f"| 指标 | 数值 |",
        f"|------|------|",
        f"| DeepSeek 与标注员一致 | {agree}/{valid} = **{agree*100/max(valid,1):.1f}%** |",
        f"| DeepSeek 与标注员不一致 | {disagree}/{valid} = **{disagree*100/max(valid,1):.1f}%** |",
        f"| LLM 判 NONE，标注员选了 | **{llm_none}** 条 |",
        f"| 解析错误 | {errors} |",
        "",
        "## 二、LLM 判 NONE（标注员可能误选）",
        "",
    ]

    none_cases = [r for r in results if r["agreement"] == "llm_none"]
    if none_cases:
        lines.append(f"共 {len(none_cases)} 条 LLM 认为5个候选都不匹配，但标注员仍选了一个：")
        lines.append("")
        lines.append("| Task ID | 岗位 | 标注员选 | LLM 判 | 标注员理由 |")
        lines.append("|---------|------|---------|--------|-----------|")
        for r in none_cases[:20]:
            lines.append(
                f"| {r['task_id']} | {str(r['job_title'])[:25]} "
                f"| {r['annotator_choice']} | NONE "
                f"| {r['annotator_reason'][:40]} |"
            )
    else:
        lines.append("无。")

    lines.extend([
        "",
        "## 三、不一致案例（需人工复审）",
        "",
    ])
    disagree_cases = [r for r in results if r["agreement"] == "disagree"]
    if disagree_cases:
        lines.append(f"共 {len(disagree_cases)} 条 DeepSeek 选择与标注员不同：")
        lines.append("")
        lines.append("| Task ID | 岗位 | 标注员 → | LLM → | LLM 置信度 | LLM 理由 |")
        lines.append("|---------|------|----------|--------|-----------|----------|")
        for r in disagree_cases[:30]:
            lines.append(
                f"| {r['task_id']} | {str(r['job_title'])[:20]} "
                f"| {r['annotator_choice']} | {r['judge_choice']} "
                f"| {r['judge_confidence']:.2f} | {r['judge_reasoning'][:40]} |"
            )

    # 不一致率按标注员分组
    lines.extend([
        "",
        "## 四、标注员个体差异",
        "",
    ])
    ann_disagree = Counter()
    ann_total = Counter()
    for r in results:
        aid = r["annotator_id"]
        ann_total[aid] += 1
        if r["agreement"] in ("disagree", "llm_none"):
            ann_disagree[aid] += 1

    lines.append("| 标注员 | 审计数 | 不一致数 | 不一致率 |")
    lines.append("|--------|--------|---------|---------|")
    for aid in sorted(ann_total.keys(), key=lambda x: -ann_total[x]):
        total_a = ann_total[aid]
        disagree_a = ann_disagree.get(aid, 0)
        rate = disagree_a / max(total_a, 1)
        flag = " ⚠️" if rate > 0.3 else ""
        lines.append(f"| #{aid} | {total_a} | {disagree_a} | {rate:.0%}{flag} |")

    # 建议
    lines.extend([
        "",
        "## 五、建议",
        "",
        f"1. **LLM 判 NONE 的 {llm_none} 条**：标注员可能仅看岗位名未读 JD，建议全部人工复审。",
        f"2. **不一致的 {disagree} 条**：建议人工复审不一致案例，确定是标注错误还是 LLM 错误。",
        f"3. **高不一致率标注员**：建议对其标注做专项审查或重新培训。",
        f"4. **一致率 {agree*100/max(valid,1):.0f}%** 可作为标注数据可信度的参考下限。",
        "",
        "> 注意：LLM 本身也可能出错。本审计仅提供不一致信号，最终仲裁需人工判断。",
    ])

    report = "\n".join(lines)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        logger.info("报告已保存: %s", output_path)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM 标注质量审计")
    parser.add_argument("--limit", type=int, default=100, help="审计条数")
    parser.add_argument("--validation-only", action="store_true", help="只看验证集")
    parser.add_argument("--output", type=str, default="", help="报告输出路径")
    parser.add_argument("--json-output", type=str, default="", help="详细 JSON 输出路径")
    args = parser.parse_args()

    db_path = str(PROJECT_ROOT / "output" / "recruit.duckdb")

    # 1. 加载
    tasks = load_annotated_tasks(db_path, args.validation_only, args.limit)
    logger.info("加载 %d 条已标注任务", len(tasks))
    if not tasks:
        logger.error("无数据")
        sys.exit(1)

    # 2. 审计
    auditor = AnnotationAuditor()
    logger.info("开始 DeepSeek 审计...")
    results = run_audit(tasks, auditor)

    # 3. 报告
    output_path = None
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = PROJECT_ROOT / "output" / "reports" / "annotation_audit.md"
    report = generate_report(results, output_path)
    print(report)

    # 4. JSON 详细数据
    json_out = args.json_output or str(output_path.with_suffix(".json"))
    Path(json_out).parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("详细数据: %s", json_out)


if __name__ == "__main__":
    main()
