"""Common utilities for fully automatic LLM-based dataset labeling."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
import re
from typing import Dict, Iterable, List, Sequence, Tuple

import duckdb
import pandas as pd

from .config import SkillExtractionConfig, load_skill_extraction_config


logger = logging.getLogger(__name__)

SOURCE_COLUMNS: Sequence[str] = (
    "sample_row_id",
    "__source_table",
    "__source_row_number",
    "岗位名称",
    "岗位描述_清洗",
    "任职要求_items_text",
    "岗位职责_items_text",
    "sections_brief",
    "occupation_title",
    "occupation_code",
)

TEXT_FIELD_PRIORITY: Sequence[str] = (
    "任职要求_items_text",
    "岗位职责_items_text",
    "岗位描述_清洗",
    "text",
)


def _quote_identifier(identifier: str) -> str:
    """安全转义 DuckDB 标识符。

    参数:
        identifier: 原始列名或别名。

    返回:
        str: 可直接拼接进 DuckDB SQL 的双引号标识符。
    """
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


def safe_text(value: object) -> str:
    """将任意输入稳健地转换为普通字符串。

    参数:
        value: 任意对象，可能为 `None`、数字、字符串或缺失值。

    返回:
        str: 去除空白后的字符串；若值为空或等价于 `nan`，则返回空字符串。
    """
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def normalize_skill_key(value: object) -> str:
    """将技能名归一化为便于比较的 key。

    参数:
        value: 原始技能名或 alias。

    返回:
        str: 经过 `safe_text + casefold` 处理后的结果。
    """
    return safe_text(value).casefold()


def extract_match_text(row: Dict[str, object]) -> str:
    """按统一优先级抽取岗位文本。

    参数:
        row: 单条岗位记录。

    返回:
        str: 优先级最高且非空的文本字段。

    说明:
        当前优先级依次为：
        `任职要求_items_text` -> `岗位职责_items_text` -> `岗位描述_清洗` -> `text`
    """
    for field_name in TEXT_FIELD_PRIORITY:
        text = safe_text(row.get(field_name, ""))
        if text:
            return text
    return ""


def build_sample_id(row: Dict[str, object], fallback_index: int) -> str:
    """为自动标注样本生成稳定的 `sample_id`。

    参数:
        row: 原始岗位记录。
        fallback_index: 当源记录没有稳定主键时使用的兜底序号。

    返回:
        str: 稳定、可复现的样本编号。
    """
    for field_name in ("sample_id", "sample_row_id", "__source_row_number"):
        value = safe_text(row.get(field_name, ""))
        if value:
            return value
    return f"sample_{fallback_index:07d}"


def load_requirement_match_rows(
    config: SkillExtractionConfig | None = None,
    source_table: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """从 DuckDB 读取自动标注所需的岗位样本。

    参数:
        config: 可选的技能抽取配置对象；为空时自动加载默认配置。
        source_table: 可选的源表名；为空时使用配置中的默认表。
        limit: 可选的读取上限，通常用于调试。

    返回:
        pd.DataFrame: 包含自动标注必需字段的样本表。

    说明:
        函数会先检查实际列名，并对缺失列补 `NULL AS column`，
        这样可以提高对上游表结构波动的兼容性。
    """
    config = config or load_skill_extraction_config()
    source_table = source_table or config.requirement_match_table
    limit_clause = f"LIMIT {int(limit)}" if limit else ""
    with duckdb.connect(str(config.db_path), read_only=True) as conn:
        conn.execute(f"PRAGMA threads={config.duckdb_threads}")
        available_columns = {
            row[0]
            for row in conn.execute(f"DESCRIBE {source_table}").fetchall()
        }
        select_expressions: List[str] = []
        for column_name in SOURCE_COLUMNS:
            if column_name in available_columns:
                select_expressions.append(_quote_identifier(column_name))
            else:
                select_expressions.append(f"NULL AS {_quote_identifier(column_name)}")
        query = f"""
            SELECT
                {", ".join(select_expressions)}
            FROM {source_table}
            {limit_clause}
        """
        return conn.execute(query).df()


def prepare_labeling_frame(
    source_df: pd.DataFrame,
    max_text_chars: int = 900,
    min_text_chars: int = 20,
) -> pd.DataFrame:
    """将原始岗位样本整理成标准化标注输入表。

    参数:
        source_df: 原始 DuckDB 查询结果。
        max_text_chars: 单条样本文本允许保留的最大字符数。
        min_text_chars: 参与标注的最小文本长度阈值。

    返回:
        pd.DataFrame: 完成清洗、去重和字段对齐后的 DataFrame。
    """
    records: List[Dict[str, str]] = []
    seen_texts: set[str] = set()

    for row_index, row in enumerate(source_df.to_dict(orient="records")):
        text = extract_match_text(row)
        if len(text) < int(min_text_chars):
            continue

        normalized_key = text[: max_text_chars * 2].casefold()
        if normalized_key in seen_texts:
            continue
        seen_texts.add(normalized_key)

        records.append(
            {
                "sample_id": build_sample_id(row, row_index),
                "job_title": safe_text(row.get("岗位名称", "")),
                "occupation_title": safe_text(row.get("occupation_title", "")),
                "occupation_code": safe_text(row.get("occupation_code", "")),
                "source_table": safe_text(row.get("__source_table", "")),
                "source_row_number": safe_text(row.get("__source_row_number", "")),
                "text": text[:max_text_chars],
            }
        )

    return pd.DataFrame(records)


def stratified_sample_frame(
    frame: pd.DataFrame,
    sample_size: int,
    seed: int = 42,
) -> pd.DataFrame:
    """按职业维度做轮转采样，尽量降低类别偏斜。

    参数:
        frame: 标准化后的样本表。
        sample_size: 目标抽样数量。
        seed: 随机种子。

    返回:
        pd.DataFrame: 分层采样后的样本表。
    """
    if frame.empty or sample_size <= 0 or len(frame) <= sample_size:
        return frame.copy()

    rng = random.Random(seed)
    group_rows: List[List[Dict[str, str]]] = []
    group_key_series = frame["occupation_code"].fillna("").astype(str)
    fallback_series = frame["occupation_title"].fillna("").astype(str)

    frame = frame.copy()
    frame["_group_key"] = group_key_series.where(group_key_series != "", fallback_series)
    frame["_group_key"] = frame["_group_key"].replace("", "__ungrouped__")

    for _, group_df in frame.groupby("_group_key"):
        rows = group_df.drop(columns=["_group_key"]).to_dict(orient="records")
        rng.shuffle(rows)
        group_rows.append(rows)

    rng.shuffle(group_rows)
    sampled_records: List[Dict[str, str]] = []
    while len(sampled_records) < sample_size and any(group_rows):
        next_group_rows: List[List[Dict[str, str]]] = []
        for rows in group_rows:
            if not rows:
                continue
            sampled_records.append(rows.pop())
            if rows:
                next_group_rows.append(rows)
            if len(sampled_records) >= sample_size:
                break
        group_rows = next_group_rows

    return pd.DataFrame(sampled_records)


def build_chat_prompts(
    llm,
    prompt_pairs: Sequence[Tuple[str, str]],
) -> List[str]:
    """将 `(system, user)` prompt 对渲染成完整聊天提示词。

    参数:
        llm: 已初始化的 vLLM 模型对象。
        prompt_pairs: `(system_prompt, user_prompt)` 二元组列表。

    返回:
        List[str]: 应用 chat template 后的完整 prompt 文本列表。
    """
    tokenizer = llm.get_tokenizer()
    rendered_prompts: List[str] = []
    for system_prompt, user_prompt in prompt_pairs:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt = f"system: {system_prompt}\nuser: {user_prompt}\nassistant:"
        rendered_prompts.append(prompt)
    return rendered_prompts


def run_prompt_pairs(
    model_path: str | Path,
    prompt_pairs: Sequence[Tuple[str, str]],
    batch_size: int = 16,
    gpu_memory_utilization: float = 0.80,
    max_model_len: int = 8192,
    max_num_seqs: int = 48,
    temperature: float = 0.1,
    max_tokens: int = 1536,
    top_p: float = 0.9,
    repetition_penalty: float = 1.05,
) -> List[str]:
    """批量执行 prompt，并返回原始生成文本。

    参数:
        model_path: 本地 LLM 模型目录。
        prompt_pairs: 待执行的 prompt 对列表。
        batch_size: 每轮送入 vLLM 的 prompt 数量。
        gpu_memory_utilization: 显存占比。
        max_model_len: 最大上下文长度。
        max_num_seqs: 最大并发序列数。
        temperature: 采样温度。
        max_tokens: 最大生成 token 数。
        top_p: nucleus sampling 参数。
        repetition_penalty: 重复惩罚系数。

    返回:
        List[str]: 与输入 prompt 一一对应的原始模型输出。
    """
    from vllm import SamplingParams
    from .merge_similar_skills import init_vllm_engine

    llm = init_vllm_engine(
        model_path=str(model_path),
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )
    rendered_prompts = build_chat_prompts(llm, prompt_pairs)
    sampling_params = SamplingParams(
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        top_p=float(top_p),
        repetition_penalty=float(repetition_penalty),
    )

    outputs: List[str] = []
    total_batches = max(1, (len(rendered_prompts) + batch_size - 1) // batch_size)
    for batch_index in range(total_batches):
        start = batch_index * batch_size
        end = min(start + batch_size, len(rendered_prompts))
        logger.info(
            "Running LLM labeling batch %d/%d (%d prompts)",
            batch_index + 1,
            total_batches,
            end - start,
        )
        batch_outputs = llm.generate(rendered_prompts[start:end], sampling_params)
        for output in batch_outputs:
            outputs.append(output.outputs[0].text)
    return outputs


def extract_json_from_response(text: str) -> Dict | None:
    """从原始 LLM 输出中尽量稳定地提取 JSON。

    参数:
        text: LLM 原始输出文本。

    返回:
        Dict | None: 解析成功时返回字典；失败时返回 `None`。

    处理逻辑:
        1. 去掉 `<think>` 思考内容；
        2. 去掉 markdown 代码块；
        3. 尝试直接解析 JSON；
        4. 若失败，则截取最外层 `{...}` 再试；
        5. 修复常见尾逗号或引号问题后再解析。
    """
    if not text or not text.strip():
        return None

    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"```(?:json)?\s*\n?", "", cleaned)
    cleaned = cleaned.replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    brace_match = re.search(r"\{[\s\S]*\}", cleaned)
    if brace_match:
        candidate = brace_match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
            fixed = fixed.replace("'", '"')
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                return None
    return None


def write_jsonl(path: str | Path, rows: Iterable[Dict]) -> None:
    """将字典序列写入 UTF-8 JSONL 文件。

    参数:
        path: 输出路径。
        rows: 可迭代的字典记录。
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")
