"""LLM 直接抽取软技能。

绕过关键词词典匹配，直接让 LLM 从岗位描述中抽取软技能并分类到大五维度。
用于补充词典匹配无法覆盖的长短语/描述性表达。

用法::

    from src.skill_extraction.soft_skill_llm_extractor import extract_soft_skills

    skills = extract_soft_skills(
        text="具备良好的沟通表达能力和较强的抗压能力",
        llm_client=client,
    )
    # [{"name": "沟通能力", "dimension": "extraversion"}, ...]
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

EXTRACT_SYSTEM_PROMPT = """你是人力资源NLP专家。从岗位描述中提取软技能（人际特质、性格品质、通用能力），忽略硬技能（工具、软件、编程语言）。

## 大五人格维度
- conscientiousness: 尽责性（责任心、执行力、细心、认真、踏实、吃苦耐劳）
- extraversion: 外向性（沟通、表达、谈判、领导、组织、开朗）
- openness: 开放性（创新、学习、思维、上进、进取、适应）
- agreeableness: 宜人性（合作、团队、服务、同理心）
- neuroticism: 情绪稳定性（抗压、情绪管理、冷静）

## 规则
1. 提取文本中明确提到的软技能（不要凭空推断）
2. 每个技能输出规范名称（2-8个字），不要输出完整短语
3. 标注正确的大五维度
4. 每个文本最多提取10个软技能
5. 只输出 JSON 数组，不要其他内容

输出格式: [{"name": "沟通能力", "dimension": "extraversion"}, ...]"""


def extract_soft_skills(
    text: str,
    llm_client: Any,
    *,
    max_skills: int = 10,
) -> List[Dict[str, str]]:
    """使用 LLM 从文本中直接抽取软技能。

    参数:
        text: 岗位描述文本。
        llm_client: LLM 客户端（需支持 complete_json 方法）。
        max_skills: 每段文本最多抽取的软技能数。

    返回:
        list[dict]: 抽取结果，每项含 ``name`` 和 ``dimension``。
    """
    if not text or not text.strip():
        return []

    prompt = EXTRACT_SYSTEM_PROMPT.replace("最多10个软技能", f"最多{max_skills}个软技能")

    try:
        resp = llm_client.complete_json(
            system_prompt=prompt,
            user_prompt=f"岗位描述:\n{text[:3000]}",
            strength="cheap",
            max_output_tokens=500,
        )
    except Exception as exc:
        logger.warning("LLM 抽取失败: %s", exc)
        return []

    if not isinstance(resp, list):
        return []

    skills: List[Dict[str, str]] = []
    for item in resp:
        if isinstance(item, dict):
            name = (item.get("name") or "").strip()
            dimension = (item.get("dimension") or "").strip()
            if name and dimension:
                skills.append({"name": name, "dimension": dimension})
    return skills
