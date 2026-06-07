"""DeepSeek V4 Pro 生成器（v2 替换 Qwen3-8B）。

功能:
- 基于检索候选 + 《职业大典》条目进行语义匹配
- 输出格式对齐 eval_annotation_quality.py（best_candidate + confidence + reasoning）
- 支持两种模式: 'rag'（检索模式）、'judge'（给定候选评判模式）
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from .config import RAGConfig

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env.local"))


# ---------------------------------------------------------------------------
# Prompt 模板
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """你是《中华人民共和国职业分类大典》（2022年版）的资深分类专家。
你的任务是根据招聘岗位的标题和描述，从检索到的候选职业细类中选择最匹配的一个。

评判原则：
1. 以实际工作内容（job_requirements）为主要判断依据，不要只看岗位名称。
2. 充分利用候选职业的层级路径（大类→中类→小类→细类）作为约束。
3. 代码结构为 X-XX-XX-XX，代表大类-中类-小类-细类，同路径前缀的候选优先考虑。
4. 如果所有候选都不合适，请选择 "NONE"。
5. 输出必须是严格的 JSON。"""

RAG_USER_TEMPLATE = """请从以下 {n} 个候选职业中，选择与招聘岗位最匹配的一个。

【招聘岗位】
岗位名称：{job_title}
岗位要求：
{job_requirements}

【候选职业】
{candidates_text}

请输出 JSON：
{{"best_candidate": "1"~"{n}" 或 "NONE", "best_code": "职业代码", "best_title": "职业名称", "confidence": 0.0-1.0, "reasoning": "理由(30字内)", "evidence": "引用的关键职业大典内容(30字内)"}}"""  # noqa: E501

JUDGE_SYSTEM_PROMPT = """你是《中华人民共和国职业分类大典》（2022年版）的资深分类专家。
你的任务是根据招聘岗位的标题和描述，从 5 个候选职业中选择最匹配的一个。

评判原则：
1. 以岗位描述中的实际工作内容为主要判断依据，不要只看岗位名称。
2. 如果岗位名称与描述不一致，以描述为准。
3. 候选职业的代码提供了职业大类信息，大类相同但细类不同时优先考虑工作内容重叠度。
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
# DeepSeekGenerator
# ---------------------------------------------------------------------------

class DeepSeekGenerator:
    """DeepSeek V4 Pro 生成器。

    支持两种模式：
    - rag: 从 RAG 检索的候选列表中选出最佳匹配
    - judge: 给定固定的 5 个候选（A~E），选出最佳匹配（对齐 eval_annotation_quality）
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY 未在 .env.local 中设置")
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    # ------------------------------------------------------------------
    # RAG 模式: 从检索候选中选择
    # ------------------------------------------------------------------

    def generate(
        self,
        query: str,
        job_title: str,
        job_requirements: str,
        candidates: List[Dict],
    ) -> Dict[str, Any]:
        """RAG 模式：从检索候选列表中选择最佳匹配职业细类。

        Args:
            query: 原始查询文本（保留用于兼容）。
            job_title: 岗位名称。
            job_requirements: 岗位要求描述。
            candidates: 检索返回的 top_k 候选列表。

        Returns:
            Dict: {"best_candidate": str, "best_code": str, "best_title": str,
                   "confidence": float, "reasoning": str, "evidence": str}
        """
        if not candidates:
            return {
                "best_candidate": "NONE",
                "best_code": "",
                "best_title": "",
                "confidence": 0.0,
                "reasoning": "无候选可供判断",
                "evidence": "",
            }

        candidates_text = self._build_candidates_text(candidates)
        user_prompt = RAG_USER_TEMPLATE.format(
            n=len(candidates),
            job_title=job_title,
            job_requirements=job_requirements[:3000],
            candidates_text=candidates_text,
        )

        raw = self._call_api(RAG_SYSTEM_PROMPT, user_prompt)
        parsed = self._parse_json(raw)

        return {
            "best_candidate": parsed.get("best_candidate", "NONE"),
            "best_code": parsed.get("best_code", ""),
            "best_title": parsed.get("best_title", ""),
            "confidence": float(parsed.get("confidence", 0)),
            "reasoning": str(parsed.get("reasoning", ""))[:200],
            "evidence": str(parsed.get("evidence", ""))[:200],
        }

    def _build_candidates_text(self, candidates: List[Dict]) -> str:
        """构建候选列表文本。

        Args:
            candidates: 检索候选列表。

        Returns:
            str: 格式化的候选文本。
        """
        parts = []
        for c in candidates:
            hier = c.get("hierarchy", {})
            hier_path = " > ".join(
                hier.get(f, "") for f in ["大类", "中类", "小类", "细类"] if hier.get(f)
            )
            parts.append(
                f"候选{c['rank']}: [{c['code']}] {c['title']}\n"
                f"  层级路径: {hier_path}\n"
                f"  职业定义: {c.get('desc', '')[:200]}\n"
                f"  工作任务: {c.get('tasks', '')[:200]}\n"
                f"  检索分数: {c['score']:.4f}"
            )
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Judge 模式: 给定固定候选，选出最佳（对齐 eval_annotation_quality）
    # ------------------------------------------------------------------

    def judge(
        self,
        job_title: str,
        job_requirements: str,
        candidates: List[Dict],
    ) -> Dict[str, Any]:
        """Judge 模式：从固定的 5 个候选（A~E）中选出最佳匹配。

        输出格式对齐 eval_annotation_quality.py 的期望:
        {"best_candidate": "A"|"B"|...|"NONE", "confidence": float, "reasoning": str}

        Args:
            job_title: 岗位名称。
            job_requirements: 岗位要求描述。
            candidates: 5 个候选，每个含 code/title/desc。

        Returns:
            Dict: {"best_candidate": str, "confidence": float, "reasoning": str}
        """
        # 确保有 5 个候选，不足补空
        padded = list(candidates)
        while len(padded) < 5:
            padded.append({"code": "", "title": "(空)", "desc": ""})

        user_prompt = JUDGE_USER_TEMPLATE.format(
            job_title=job_title,
            job_requirements=job_requirements[:3000],
            code_a=padded[0].get("code", ""),
            title_a=padded[0].get("title", ""),
            desc_a=padded[0].get("desc", "")[:300],
            code_b=padded[1].get("code", ""),
            title_b=padded[1].get("title", ""),
            desc_b=padded[1].get("desc", "")[:300],
            code_c=padded[2].get("code", ""),
            title_c=padded[2].get("title", ""),
            desc_c=padded[2].get("desc", "")[:300],
            code_d=padded[3].get("code", ""),
            title_d=padded[3].get("title", ""),
            desc_d=padded[3].get("desc", "")[:300],
            code_e=padded[4].get("code", ""),
            title_e=padded[4].get("title", ""),
            desc_e=padded[4].get("desc", "")[:300],
        )

        raw = self._call_api(JUDGE_SYSTEM_PROMPT, user_prompt)
        parsed = self._parse_json(raw)

        return {
            "best_candidate": parsed.get("best_candidate", "NONE"),
            "confidence": float(parsed.get("confidence", 0)),
            "reasoning": str(parsed.get("reasoning", ""))[:200],
        }

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        """调用 DeepSeek API。

        Args:
            system_prompt: 系统提示。
            user_prompt: 用户提示。

        Returns:
            str: 模型原始输出文本。
        """
        try:
            response = self.client.chat.completions.create(
                model=self.config.generator_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("DeepSeek API 调用失败: %s", exc)
            return ""

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        """从 LLM 输出中提取 JSON。

        Args:
            raw: LLM 原始输出。

        Returns:
            Dict: 解析后的字典，失败返回 {}。
        """
        text = (raw or "").strip()
        # 移除 markdown 代码块
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:]) if len(lines) > 1 else text
            text = text.rsplit("```", 1)[0].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # 尝试提取第一个 {...}
        import re
        m = re.search(r"\{[^{}]*\}", text)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return {}
