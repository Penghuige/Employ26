# =============================================================================
# 模块：src/rag/qc_utils.py
# 功能：RAG + Qwen3 质检共享工具函数
#       供 D2_filter.py 和 D5_qwen3_auto_label.py 复用，避免重复实现
#
# 对外接口：
#   - safe_str(value) -> str
#   - build_rag_context(candidates, max_desc_len) -> str
#   - build_qc_prompt(..., has_prediction) -> str
#   - extract_json(text) -> Dict
#   - normalize_label(obj, predicted_title) -> Dict
#   - load_retriever(cfg) -> (OccupationRetriever, List[Dict])
#   - load_qwen_model(model_path, dtype_str, device_map) -> (tokenizer, model)
#   - batched_generate(tokenizer, model, prompts, ...) -> List[str]
#   - batched_generate_with_client(prompts, ...) -> List[str]
# =============================================================================

import ast
import json
import re
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.rag.config import RAGConfig
from src.rag.kb_builder import (
    build_chunks,
    load_metadata,
    load_occupation_records,
    load_saved_payload,
    save_metadata,
)
from src.rag.retriever import OccupationRetriever
from src.model_platform.llm import LLMClient, create_llm_client

# 合法错误类型枚举（与 docs/manual_label_template.txt 保持同步）
VALID_ERROR_TYPES = {
    "title_ambiguous",
    "desc_noise",
    "assistant_intern_confusion",
    "coarse_to_fine_mismatch",
    "cross_domain_confusion",
    "dictionary_gap",
    "low_confidence_borderline",
    "other",
}


# =============================================================================
# 字符串工具
# =============================================================================

def safe_str(value) -> str:
    """安全转字符串，处理 None 与 float NaN。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


# =============================================================================
# RAG 上下文构建
# =============================================================================

def build_rag_context(candidates: List[Dict], max_desc_len: int = 180) -> str:
    """将检索候选格式化为可嵌入 prompt 的上下文文本。

    优先使用新版 kb_builder 字段（task_items/sub_titles/title_main），
    回退兼容旧版字段（title/desc/tasks）。

    参数：
        candidates: 每条含 rank/score/code/title/desc/tasks 字段的候选列表
        max_desc_len: 每条候选定义/任务的最大截断长度

    返回：
        多行格式化字符串
    """
    chunks = []
    for c in candidates:
        title_display = c.get("title_main") or c.get("title", "")
        desc = str(c.get("desc", ""))[:max_desc_len]

        # 优先用拆分后的 task_items，回退到原始 tasks
        task_items = c.get("task_items", [])
        if task_items:
            tasks_text = "；".join(task_items)[:max_desc_len]
        else:
            tasks_text = str(c.get("tasks", ""))[:max_desc_len]

        # 细分工种/别名（新版字段）
        sub_titles = c.get("sub_titles", [])
        sub_str = f" 细分工种:{'/'.join(sub_titles)}" if sub_titles else ""

        chunks.append(
            f"[候选{c['rank']}] 代码:{c['code']} 名称:{title_display}{sub_str} 相似度:{c['score']:.4f}\n"
            f"定义:{desc}\n任务:{tasks_text}"
        )
    return "\n\n".join(chunks)


# =============================================================================
# Prompt 构建
# =============================================================================

def build_qc_prompt(
    job_title: str,
    job_desc: str,
    predicted_title: str,
    predicted_code: str,
    predicted_score: str,
    rag_context: str,
    has_prediction: bool = True,
) -> str:
    """构建含 RAG 知识库上下文的质检 prompt。

    参数：
        has_prediction: False 时为 tier3（无系统预测），走推荐职业分支；
                        True 时为 tier1/tier2，走质检判断分支。

    设计原则：
    1) /no_think 关闭 Qwen3 思维链，避免 <think> 污染输出
    2) is_correct=0 时明确要求 gold_title 与 predicted_title 不同
    3) 知识库候选置于末尾，利用注意力机制
    """
    if not has_prediction:
        # tier3：无系统预测，任务变为直接推荐最合适职业
        return (
            "/no_think\n"
            "你是职业分类标注员。该岗位尚无系统预测结果，请根据岗位信息和知识库候选，"
            "直接为该岗位推荐最合适的职业分类。\n"
            "只能输出一行合法 JSON，禁止输出解释、markdown、<think> 内容。\n"
            "JSON 键：is_correct,gold_title,gold_code,error_type,error_note\n"
            "规则：\n"
            "1) 无系统预测，is_correct 固定填 0。\n"
            "2) gold_title/gold_code 必须从知识库候选中选取最合适的一项。\n"
            "3) error_type 填 dictionary_gap（系统未覆盖此职业）。\n"
            "4) error_note 简要说明推荐理由。\n\n"
            f"岗位名称: {job_title}\n"
            f"岗位描述: {str(job_desc)[:400]}\n\n"
            f"知识库候选（请从中选取 gold_title/gold_code）:\n{rag_context}\n"
        )

    # tier1/tier2：有系统预测，执行质检判断
    return (
        "/no_think\n"
        "你是职业分类质检员。请基于岗位信息、当前预测和知识库候选，判断预测是否正确。\n"
        "只能输出一行合法 JSON，禁止输出解释、markdown、<think> 内容。\n"
        "JSON 键：is_correct,gold_title,gold_code,error_type,error_note\n"
        "规则：\n"
        "1) is_correct=1 预测正确，=0 预测错误。\n"
        "2) is_correct=1 时：gold_title 与 gold_code 必须从知识库候选中选取与预测最吻合的一项，"
        "error_type/error_note 置空。\n"
        "3) is_correct=0 时：gold_title/gold_code 必须从知识库候选中选取最正确的一项，"
        "且该项名称必须与 predicted_title 不同；\n"
        "   严禁凭空编造不在候选列表中的职业名称或代码。\n"
        "   error_type 从以下选一：title_ambiguous,desc_noise,assistant_intern_confusion,\n"
        "   coarse_to_fine_mismatch,cross_domain_confusion,dictionary_gap,\n"
        "   low_confidence_borderline,other\n"
        "4) 判断依据：仅凭岗位描述与候选职业定义/任务的语义匹配程度，不受预测分数高低影响。\n"
        "5) 若知识库候选中没有比当前预测更合适的职业，则判 is_correct=1。\n\n"
        f"岗位名称: {job_title}\n"
        f"岗位描述: {str(job_desc)[:400]}\n"
        f"当前预测职业: {predicted_title}\n"
        f"当前预测代码: {predicted_code}\n"
        f"当前预测分数: {predicted_score}\n\n"
        f"知识库候选（gold_title/gold_code 只能从此列表中选取）:\n{rag_context}\n"
    )


# =============================================================================
# JSON 解析与标签规范化
# =============================================================================

def extract_json(text: str) -> Dict:
    """从模型输出中鲁棒提取 JSON。

    提取策略（优先级递减）：
    1) 剥离 <think>...</think> 后直接解析
    2) 清理 markdown 标记后整段解析
    3) 从文本中抽取 {...} 块逐一尝试
    4) 全部失败返回解析失败占位
    """
    raw = safe_str(text)
    cleaned = raw

    # 步骤1：剥离思维链
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1]
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned)

    # 步骤2：清理 markdown 标记
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    # 步骤3：整段解析
    for loader in (json.loads, ast.literal_eval):
        try:
            obj = loader(cleaned)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # 步骤4：块提取
    for source in (cleaned, raw):
        for pattern in (r"\{[\s\S]*\}", r"\{[\s\S]*?\}"):
            for cand in re.findall(pattern, source):
                for loader in (json.loads, ast.literal_eval):
                    try:
                        obj = loader(cand)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        pass

    return {
        "is_correct": 0,
        "gold_title": "",
        "gold_code": "",
        "error_type": "other",
        "error_note": "qwen_output_parse_failed",
    }


def normalize_label(obj: Dict, predicted_title: str = "") -> Dict:
    """规范化模型输出标签，后处理修复 gold==predicted 矛盾。

    规则：
    - is_correct 只能是 0 或 1
    - is_correct=1 时清空错误字段
    - is_correct=0 时 error_type 必须在合法枚举内
    - is_correct=0 且 gold_title==predicted_title 时清空 gold（矛盾修复）
    """
    is_correct = 1 if safe_str(obj.get("is_correct", "0")) in {"1", "true", "True"} else 0
    gold_title = safe_str(obj.get("gold_title", ""))
    gold_code = safe_str(obj.get("gold_code", ""))
    error_type = safe_str(obj.get("error_type", ""))
    error_note = safe_str(obj.get("error_note", ""))

    if is_correct == 1:
        error_type = ""
        error_note = ""
    else:
        if error_type not in VALID_ERROR_TYPES:
            error_type = "other"
        if not error_note:
            error_note = "auto_labeled_as_incorrect"
        # 后处理：gold==predicted 为矛盾输出，清空 gold
        if predicted_title and gold_title and gold_title.strip() == predicted_title.strip():
            gold_title = ""
            gold_code = ""
            error_note = f"[post_fix] gold==predicted矛盾，已清空。原:{error_note}"

    return {
        "is_correct": is_correct,
        "gold_title": gold_title,
        "gold_code": gold_code,
        "error_type": error_type,
        "error_note": error_note,
    }


# =============================================================================
# RAG 知识库加载
# =============================================================================

def load_retriever(cfg: RAGConfig) -> Tuple[OccupationRetriever, List[Dict]]:
    """初始化 BGE 检索器，存在缓存直接加载，否则从 Excel 重建索引。

    参数：
        cfg: RAGConfig 实例（含索引路径、元数据路径等）

    返回：
        (retriever, records) 元组
    """
    import os
    retriever = OccupationRetriever(cfg)

    if os.path.exists(cfg.index_path) and os.path.exists(cfg.metadata_path):
        print("[RAG] 发现已有索引缓存，直接加载...")
        retriever.load_index()
        records = load_metadata(cfg.metadata_path)
    else:
        print("[RAG] 未发现缓存，从 Excel 重建知识库索引（仅首次需要）...")
        records = load_occupation_records(cfg)
        if not records:
            raise ValueError("[RAG] 知识库为空，请检查 Excel 文件内容与路径。")
        retriever.build_index(records)
        retriever.save_index()
        save_metadata(cfg, records)

    print(f"[RAG] 知识库加载完成，共 {len(records)} 条职业条目。")
    return retriever, records


def load_task_chunk_retriever(
    cfg: RAGConfig,
) -> Tuple[OccupationRetriever, List[Dict], List[Dict]]:
    """初始化 task-chunk 检索器。

    返回：
        retriever: 针对 task chunk 的向量检索器
        records: 职业记录级元数据
        task_chunks: 仅包含 chunk_type=task 的任务块
    """
    import os

    task_cfg = replace(cfg, index_path=cfg.task_index_path)
    retriever = OccupationRetriever(task_cfg)

    payload = load_saved_payload(cfg.metadata_path) if os.path.exists(cfg.metadata_path) else {}
    records = payload.get("records", [])
    chunks = payload.get("chunks", [])

    if not records:
        records = load_occupation_records(cfg)

    if not chunks:
        chunks = build_chunks(cfg, records)
        save_metadata(cfg, records, chunks)

    task_chunks = [chunk for chunk in chunks if chunk.get("chunk_type") == "task"]
    if not task_chunks:
        raise ValueError("[RAG] 未找到 task chunk，请检查知识库任务字段或切块配置。")

    if os.path.exists(task_cfg.index_path):
        print("[RAG] 发现已有 task-chunk 索引缓存，直接加载...")
        retriever.load_index()
    else:
        print("[RAG] 未发现 task-chunk 索引缓存，正在从 chunks 重建...")
        retriever.build_index(task_chunks)
        retriever.save_index()

    print(
        f"[RAG] task-chunk 检索器加载完成，共 {len(records)} 条职业记录，"
        f"{len(task_chunks)} 个任务块。"
    )
    return retriever, records, task_chunks


# =============================================================================
# Qwen3 模型加载
# =============================================================================

def load_qwen_model(
    model_path: str,
    dtype_str: str = "bfloat16",
    device_map: str = "auto",
):
    """加载 Qwen3 模型与 Tokenizer。

    参数：
        model_path: 本地模型路径
        dtype_str: 精度类型字符串（bfloat16/float16/float32）
        device_map: 设备分配策略（auto/cuda/cpu）

    返回：
        (tokenizer, model) 元组
    """
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(dtype_str.lower(), torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"  # decoder-only 批量生成必须左填充
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    # 清理与贪心解码冲突的默认 generation_config 参数，消除 warning
    model.generation_config.temperature = None
    model.generation_config.top_k = None
    model.generation_config.top_p = None
    model.eval()
    return tokenizer, model


# =============================================================================
# 批量推理
# =============================================================================

def batched_generate(
    tokenizer,
    model,
    prompts: List[str],
    max_new_tokens: int = 128,
    do_sample: bool = False,
    max_length: int = 4096,
) -> List[str]:
    """对一批 prompt 执行批量推理，返回生成文本列表。

    参数：
        tokenizer/model: 已加载的 Qwen3 模型
        prompts: prompt 字符串列表
        max_new_tokens: 最大生成 token 数
        do_sample: False=贪心解码，True=采样
        max_length: tokenizer 截断长度，防止 OOM

    关键实现细节：
    - enable_thinking=False 关闭 Qwen3 思维链
    - 使用 input_ids.shape[1] 精确定位生成起始，left-padding 下不会错位
    """
    rendered = []
    for p in prompts:
        msg = [{"role": "user", "content": p}]
        try:
            text = tokenizer.apply_chat_template(
                msg, tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # Qwen3 官方参数，关闭思维链
            )
        except TypeError:
            # 旧版 transformers 不支持 enable_thinking，降级处理
            text = tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True)
        rendered.append(text)

    enc = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    gen_kwargs: Dict = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    with torch.no_grad():
        out = model.generate(**enc, **gen_kwargs)

    # 精确截取生成部分（left-padding 下不能用固定偏移）
    input_len = enc["input_ids"].shape[1]
    return [
        tokenizer.decode(seq[input_len:], skip_special_tokens=True)
        for seq in out
    ]


def batched_generate_with_client(
    prompts: List[str],
    *,
    client: LLMClient | None = None,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> List[str]:
    """通过统一 LLM client 批量生成文本。

    该函数是 BGE/RAG 质检脚本的新主路径。默认使用 WSL vLLM HTTP
    服务，不再在 Windows 侧直接加载 Qwen 权重。
    """
    llm = client or create_llm_client()
    return llm.batch_complete_text(
        [("你是职业分类质检员。只输出用户要求的 JSON。", prompt) for prompt in prompts],
        max_output_tokens=max_new_tokens,
        temperature=temperature,
    )
