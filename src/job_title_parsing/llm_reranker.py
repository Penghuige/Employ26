"""LLM 候选重排序模块。"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """你是《中国职业分类大典》的资深分类专家。
你的任务是根据招聘岗位的标题和描述，评估候选职业细类与该岗位的匹配程度。

评估原则：
1. 核心判断依据是岗位的实际工作内容与候选职业的任务描述是否一致。
2. 岗位名称相似但工作内容不同 → 低匹配。
3. 岗位名称不同但实际工作高度重合 → 高匹配。
4. 岗位标题中的英文缩写（如 CNC、CAD、OTC、MES、QA、QC、SEO、SEM 等）
   应作为技术关键词保留原意，不要随意翻译或忽略。
5. 如果所有候选都与岗位描述明显不符，请明确指出无匹配。
6. 你的输出必须是严格的 JSON，不要附带任何解释性文字。"""

RERANK_PROMPT_TEMPLATE = """请评估以下招聘岗位与候选职业细类的匹配度。

【招聘岗位】
岗位名称：{job_title}
岗位描述：{job_description}

【候选职业列表】
{candidates_text}

【任务要求】
1. 逐一判断每个候选职业与该岗位的匹配程度（0.0-1.0）。
2. 给出匹配或不匹配的具体理由（引用岗位描述中的关键信息）。
3. 重新排序，最相关的排在最前面。
4. 如果所有候选得分都低于 0.4，将 all_irrelevant 设为 true。

请仅输出如下 JSON（不要包含 markdown 代码块标记）：
{{
  "reranked": [
    {{
      "code": "职业代码",
      "title": "职业名称",
      "score": 0.85,
      "reason": "匹配理由"
    }}
  ],
  "all_irrelevant": false,
  "summary": "一句话总结本次匹配结果"
}}"""


def _build_candidates_text(candidates: List[Dict[str, Any]]) -> str:
    """将候选列表格式化为 LLM 可读的文本。

    Args:
        candidates: MatchPipeline 产出的候选列表。

    Returns:
        str: 编号后的候选文本。
    """
    lines: List[str] = []
    for i, cand in enumerate(candidates, 1):
        evidence = cand.get("evidence", {})
        lines.append(
            f"{i}. [{cand.get('code', '')}] {cand.get('title', '')}\n"
            f"   匹配分数: {cand.get('final_score', 0):.4f}\n"
            f"   标题命中: {evidence.get('title_hit', '')}\n"
            f"   任务命中: {evidence.get('task_hit', [])}\n"
            f"   层级命中: {evidence.get('hierarchy_hit', '')}"
        )
    return "\n".join(lines)


def _strip_json_fence(text: str) -> str:
    """移除 LLM 输出中可能包裹的 ```json ... ``` 标记。

    Args:
        text: LLM 原始输出文本。

    Returns:
        str: 清理后的 JSON 文本。
    """
    text = text.strip()
    fence_match = re.match(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return text


def _repair_truncated_json(text: str) -> str:
    """尝试修复 LLM 输出中被截断的 JSON。

    Args:
        text: 可能被截断的 JSON 文本。

    Returns:
        str: 尽力修复后的 JSON 文本。
    """
    text = text.strip()
    # 补全缺失的闭合括号
    open_braces = text.count("{") - text.count("}")
    open_brackets = text.count("[") - text.count("]")
    # 移除末尾可能的不完整字段
    last_comma = text.rfind(",")
    if last_comma > 0:
        after_comma = text[last_comma + 1:].strip()
        if not after_comma or not after_comma.endswith(("}", "]", '"')):
            text = text[:last_comma]
    text += "]" * max(0, open_brackets)
    text += "}" * max(0, open_braces)
    return text


@dataclass(frozen=True)
class RerankResult:
    """LLM 重排序后的单条匹配结果。"""

    job_id: Any
    job_title: str
    baseline_top1_code: str
    baseline_top1_title: str
    baseline_top1_score: float
    reranked_candidates: List[Dict[str, Any]]
    all_irrelevant: bool
    summary: str
    llm_call_ok: bool
    error_message: str = ""
    fell_back_to_baseline: bool = False
    fallback_reason: str = ""


class LLMReranker:
    """基于 LLM 的候选职业重排序器。

    LLM 调用统一通过 `src.model_platform.llm`，默认后端由
    `config/model_runtime.yaml` 决定。

    使用方式：
        reranker = LLMReranker()
        result = reranker.rerank(job_title, job_description, candidates)
    """

    def __init__(
        self,
        backend: str | None = None,
        vllm_config_path: str | Path | None = None,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        request_timeout: int = 90,
    ):
        """初始化 LLM 重排序器。

        Args:
            backend: 可选后端覆盖；为空时读取模型平台默认配置。
            vllm_config_path: 兼容旧调用保留；当前后端配置统一读取 `config/model_runtime.yaml`。
            system_prompt: 自定义 system prompt，为空时使用内置默认。
            temperature: LLM 采样温度（低值确保输出稳定）。
            max_tokens: 最大输出 token 数。
            request_timeout: HTTP 请求超时秒数。
        """
        self._backend = backend
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._request_timeout = request_timeout
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        self._vllm_config_path = vllm_config_path
        self._client: Any = None

    @property
    def client(self) -> Any:
        """惰性初始化统一 LLM client。"""
        if self._client is None:
            from src.model_platform.llm import create_llm_client

            self._client = create_llm_client(backend=self._backend)
        return self._client

    def is_server_available(self) -> bool:
        """检查选定的 LLM 后端是否可用。

        Returns:
            bool: 后端可用返回 True。
        """
        try:
            self.client.complete_text(
                system_prompt="你是健康检查助手。",
                user_prompt="请只回复 ok",
                max_output_tokens=8,
                temperature=0.0,
            )
            return True
        except Exception:
            return False

    def rerank(
        self,
        job_title: str,
        job_description: str,
        candidates: List[Dict[str, Any]],
        job_id: Any = None,
    ) -> RerankResult:
        """对单个岗位的 TopK 候选进行 LLM 重排序。

        Args:
            job_title: 原始岗位名称。
            job_description: 岗位描述文本。
            candidates: MatchPipeline 产出的 TopK 候选列表。
            job_id: 岗位唯一标识（可选）。

        Returns:
            RerankResult: 包含重排序后的候选及元信息。
        """
        baseline_top1 = candidates[0] if candidates else {}
        baseline_code = baseline_top1.get("code", "")
        baseline_title = baseline_top1.get("title", "")
        baseline_score = baseline_top1.get("final_score", 0.0)

        if not candidates:
            return RerankResult(
                job_id=job_id,
                job_title=job_title,
                baseline_top1_code="",
                baseline_top1_title="",
                baseline_top1_score=0.0,
                reranked_candidates=[],
                all_irrelevant=True,
                summary="无候选可供重排",
                llm_call_ok=False,
                error_message="候选列表为空",
            )

        candidates_text = _build_candidates_text(candidates)
        user_prompt = RERANK_PROMPT_TEMPLATE.format(
            job_title=job_title,
            job_description=job_description[:2000] if job_description else "（无描述）",
            candidates_text=candidates_text,
        )

        try:
            content = self.client.complete_text(
                system_prompt=self._system_prompt,
                user_prompt=user_prompt,
                strength="cheap",
                max_output_tokens=self._max_tokens,
                reasoning_effort="low",
                temperature=self._temperature,
            )

            if not content:
                return RerankResult(
                    job_id=job_id,
                    job_title=job_title,
                    baseline_top1_code=baseline_code,
                    baseline_top1_title=baseline_title,
                    baseline_top1_score=baseline_score,
                    reranked_candidates=list(candidates),
                    all_irrelevant=False,
                    summary="LLM 返回空内容，保留原始排序",
                    llm_call_ok=False,
                    error_message="LLM 返回空 content",
                )

            json_text = _strip_json_fence(content)
            parsed = json.loads(json_text)

        except json.JSONDecodeError:
            repaired = _repair_truncated_json(_strip_json_fence(content))
            try:
                parsed = json.loads(repaired)
            except json.JSONDecodeError:
                logger.warning("LLM 输出 JSON 解析失败: %s", content[:300])
                return RerankResult(
                    job_id=job_id,
                    job_title=job_title,
                    baseline_top1_code=baseline_code,
                    baseline_top1_title=baseline_title,
                    baseline_top1_score=baseline_score,
                    reranked_candidates=list(candidates),
                    all_irrelevant=False,
                    summary="LLM JSON 解析失败，保留原始排序",
                    llm_call_ok=False,
                    error_message=f"JSON 解析失败: {content[:200]}",
                )
        except Exception as exc:
            logger.warning("LLM 调用失败: %s", exc)
            return RerankResult(
                job_id=job_id,
                job_title=job_title,
                baseline_top1_code=baseline_code,
                baseline_top1_title=baseline_title,
                baseline_top1_score=baseline_score,
                reranked_candidates=list(candidates),
                all_irrelevant=False,
                summary=f"LLM 调用失败: {exc}",
                llm_call_ok=False,
                error_message=str(exc),
            )

        reranked = parsed.get("reranked", [])
        all_irrelevant = bool(parsed.get("all_irrelevant", False))
        summary = str(parsed.get("summary", ""))

        # 用原始候选信息补全 LLM 未返回的字段
        original_by_code: Dict[str, Dict[str, Any]] = {
            c.get("code", ""): c for c in candidates
        }
        enriched: List[Dict[str, Any]] = []
        for item in reranked:
            code = item.get("code", "")
            orig = original_by_code.get(code, {})
            enriched.append({
                "code": code,
                "title": item.get("title", orig.get("title", "")),
                "llm_score": item.get("score", 0.0),
                "baseline_score": orig.get("final_score", 0.0),
                "reason": item.get("reason", ""),
                "evidence": orig.get("evidence", {}),
            })

        # 如果 LLM 漏掉了某些候选，补回末尾
        returned_codes = {item.get("code", "") for item in reranked}
        for cand in candidates:
            if cand.get("code", "") not in returned_codes:
                enriched.append({
                    "code": cand.get("code", ""),
                    "title": cand.get("title", ""),
                    "llm_score": 0.0,
                    "baseline_score": cand.get("final_score", 0.0),
                    "reason": "LLM 未评估该候选，保留原始顺序",
                    "evidence": cand.get("evidence", {}),
                })

        # 阈值回退：LLM top1 分数过低时保留基线 top1
        LLM_LOW_CONFIDENCE_THRESHOLD = 0.50
        LLM_OVERCONFIDENCE_THRESHOLD = 0.85
        fell_back = False
        fallback_reason = ""
        if enriched:
            llm_top1_code = enriched[0].get("code", "")
            llm_top1_score = enriched[0].get("llm_score", 0)
            baseline_top_cand = candidates[0] if candidates else {}
            baseline_code = baseline_top_cand.get("code", "")
            code_changed = baseline_code and llm_top1_code and llm_top1_code != baseline_code

            # 保护 1：LLM 低置信度 → 回退基线
            if code_changed and llm_top1_score < LLM_LOW_CONFIDENCE_THRESHOLD:
                fell_back = True
                fallback_reason = (
                    f"LLM top1 分数 {llm_top1_score:.2f} < 低阈值 {LLM_LOW_CONFIDENCE_THRESHOLD}"
                )
            # 保护 2：LLM 高置信度但变更了基线 → 疑似幻觉，回退基线
            elif code_changed and llm_top1_score >= LLM_OVERCONFIDENCE_THRESHOLD:
                fell_back = True
                fallback_reason = (
                    f"LLM 变更了基线 top1（{baseline_code}→{llm_top1_code}）且置信度极高 "
                    f"({llm_top1_score:.2f} >= {LLM_OVERCONFIDENCE_THRESHOLD})，疑似幻觉，回退到基线"
                )

            if fell_back:
                baseline_enriched = {
                    "code": baseline_code,
                    "title": baseline_top_cand.get("title", ""),
                    "llm_score": baseline_top_cand.get("final_score", 0.0),
                    "baseline_score": baseline_top_cand.get("final_score", 0.0),
                    "reason": (
                        f"[分歧保护回退] {fallback_reason}。"
                        f"LLM 原评分: {llm_top1_score:.2f}，"
                        f"原候选: {enriched[0].get('title', 'N/A')}，"
                        f"理由: {enriched[0].get('reason', 'N/A')[:80]}"
                    ),
                    "evidence": baseline_top_cand.get("evidence", {}),
                }
                enriched = [baseline_enriched] + enriched

        return RerankResult(
            job_id=job_id,
            job_title=job_title,
            baseline_top1_code=baseline_code,
            baseline_top1_title=baseline_title,
            baseline_top1_score=baseline_score,
            reranked_candidates=enriched,
            all_irrelevant=all_irrelevant,
            summary=summary,
            llm_call_ok=True,
            fell_back_to_baseline=fell_back,
            fallback_reason=fallback_reason,
        )

    def rerank_batch(
        self,
        jobs: List[Dict[str, Any]],
        candidates_key: str = "candidates",
        title_key: str = "job_title",
        desc_key: str = "job_description",
        id_key: str = "job_id",
        show_progress: bool = True,
        sleep_between: float = 0.5,
    ) -> List[RerankResult]:
        """批量对多个岗位的候选进行 LLM 重排序。

        Args:
            jobs: 匹配结果列表，每个元素需含候选和岗位信息。
            candidates_key: 候选字段名。
            title_key: 岗位名称字段名。
            desc_key: 岗位描述字段名。
            id_key: 岗位 ID 字段名。
            show_progress: 是否显示进度。
            sleep_between: 两次 LLM 调用之间的等待秒数，避免打爆服务。

        Returns:
            List[RerankResult]: 每条岗位的重排序结果。
        """
        from tqdm.auto import tqdm

        results: List[RerankResult] = []
        iterator = tqdm(jobs, desc="LLM 重排序", unit="job") if show_progress else jobs

        for job in iterator:
            job_title = str(job.get(title_key, ""))
            job_description = str(job.get(desc_key, ""))
            job_id = job.get(id_key)
            candidates = job.get(candidates_key, [])
            if isinstance(candidates, str):
                try:
                    candidates = json.loads(candidates)
                except json.JSONDecodeError:
                    candidates = []

            result = self.rerank(
                job_title=job_title,
                job_description=job_description,
                candidates=candidates,
                job_id=job_id,
            )
            results.append(result)

            if sleep_between > 0:
                time.sleep(sleep_between)

        return results
