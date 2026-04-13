# -*- coding: utf-8 -*-
"""
Qwen3 本地推理模块

使用 transformers 库直接加载本地 Qwen3-8B 模型进行推理。
模型路径：D:\\model\\Qwen3-8B

RTX 4090 24G 推荐配置：batch_size=32, max_new_tokens=512
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MODEL_PATH = r"D:\model\Qwen3-8B"

# RTX 4090 24G 推荐批大小
DEFAULT_BATCH_SIZE = 32


class Qwen3Extractor:
    """
    本地 Qwen3-8B 推理器。
    懒加载模型，首次调用时才初始化，避免导入时占用显存。
    支持真正的 batch 推理（left-padding），充分利用 RTX 4090 24G。
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        device: str = "auto",
        max_new_tokens: int = 512,
        thinking: bool = False,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """
        Args:
            model_path     : 本地模型路径
            device         : 'auto' / 'cuda' / 'cpu'
            max_new_tokens : 最大生成长度
            thinking       : 是否开启 Qwen3 思维链（影响速度，但提升复杂文本准确率）
            batch_size     : 批推理大小，RTX 4090 24G 推荐 32
        """
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.thinking = thinking
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------

    def _load_model(self):
        """懒加载模型和分词器，并将 tokenizer 设为 left-padding（batch 推理必须）"""
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "请先安装依赖：pip install transformers torch accelerate"
            ) from e

        logger.info(f"正在加载模型：{self.model_path}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        # batch 推理必须使用 left-padding
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map=self.device,
            trust_remote_code=True,
        )
        self._model.eval()
        logger.info("模型加载完成")

    # ------------------------------------------------------------------
    # JSON 解析
    # ------------------------------------------------------------------

    def _parse_json_output(self, raw: str) -> Optional[dict]:
        """从模型原始输出中提取 JSON 对象"""
        # 去除 Qwen3 思维链 <think>...</think> 部分
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{[\s\S]+\}", raw)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        logger.warning(f"JSON 解析失败，原始输出：{raw[:200]}")
        return None

    # ------------------------------------------------------------------
    # 单条推理
    # ------------------------------------------------------------------

    def extract(self, jd_text: str, mode: str = "few_shot") -> Optional[dict]:
        """
        对单条岗位描述执行结构化抽取。

        Args:
            jd_text : 岗位描述文本
            mode    : 'zero_shot' 或 'few_shot'

        Returns:
            dict 包含 skills/tools/certs/benefits/duties/headcount/job_type
            解析失败时返回 None
        """
        results = self._extract_batch_raw([jd_text], mode=mode)
        return results[0]

    # ------------------------------------------------------------------
    # 核心：真正的 batch 推理
    # ------------------------------------------------------------------

    def _extract_batch_raw(
        self,
        jd_texts: list[str],
        mode: str = "few_shot",
    ) -> list[Optional[dict]]:
        """
        对一个 mini-batch 的文本执行 batch 推理。
        使用 left-padding 对齐，一次 forward 处理整批。

        Args:
            jd_texts : 一个 batch 内的岗位描述列表
            mode     : 'zero_shot' 或 'few_shot'

        Returns:
            与 jd_texts 等长的结果列表
        """
        import torch
        from prompt_builder import build_prompt

        self._load_model()

        # 构建每条样本的 prompt 文本
        texts = []
        for jd in jd_texts:
            messages = build_prompt(jd, mode=mode, thinking=self.thinking)
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            texts.append(text)

        # Batch tokenize（left-padding）
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(self._model.device)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        # 逐条解码（只取新生成的 token）
        results = []
        for i, seq in enumerate(outputs):
            new_tokens = seq[input_len:]
            raw = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
            results.append(self._parse_json_output(raw))
        return results

    # ------------------------------------------------------------------
    # 批量推理主接口（分 mini-batch，带进度条和计时）
    # ------------------------------------------------------------------

    def extract_batch(
        self,
        jd_list: list[str],
        mode: str = "few_shot",
        show_progress: bool = True,
        batch_size: Optional[int] = None,
    ) -> list[Optional[dict]]:
        """
        批量抽取，自动分 mini-batch，充分利用 RTX 4090 显存。
        显示双层进度条：外层按 batch 计，内层按 entry 计。
        同时打印每个 batch 耗时和每条平均耗时。

        Args:
            jd_list      : 岗位描述列表
            mode         : 'zero_shot' 或 'few_shot'
            show_progress: 是否显示 tqdm 进度条
            batch_size   : 覆盖实例默认 batch_size

        Returns:
            与 jd_list 等长的结果列表
        """
        try:
            from tqdm import tqdm
            _tqdm_available = True
        except ImportError:
            _tqdm_available = False
            if show_progress:
                logger.warning("tqdm 未安装，进度条不可用。pip install tqdm")

        bs = batch_size if batch_size is not None else self.batch_size
        total = len(jd_list)
        results: list[Optional[dict]] = [None] * total

        # 将列表切分为 mini-batches
        batches = [
            (start, jd_list[start: start + bs])
            for start in range(0, total, bs)
        ]
        n_batches = len(batches)

        logger.info(
            f"开始批量推理：共 {total} 条，batch_size={bs}，"
            f"共 {n_batches} 个 batch"
        )

        # 外层进度条：按 batch
        batch_iter = (
            tqdm(batches, desc="Batch进度", unit="batch", position=0)
            if show_progress and _tqdm_available
            else batches
        )
        # 内层进度条：按 entry（嵌套，leave=False 避免残留）
        entry_bar = (
            tqdm(total=total, desc="Entry进度", unit="条", position=1, leave=True)
            if show_progress and _tqdm_available
            else None
        )

        total_wall_start = time.perf_counter()

        for batch_idx, (start, batch_jds) in enumerate(batch_iter):
            batch_start = time.perf_counter()

            batch_results = self._extract_batch_raw(batch_jds, mode=mode)

            batch_elapsed = time.perf_counter() - batch_start
            per_entry_ms = batch_elapsed / len(batch_jds) * 1000

            # 写回结果
            for j, res in enumerate(batch_results):
                results[start + j] = res

            # 更新内层进度条
            if entry_bar is not None:
                entry_bar.update(len(batch_jds))
                entry_bar.set_postfix(
                    batch=f"{batch_idx + 1}/{n_batches}",
                    batch_s=f"{batch_elapsed:.1f}s",
                    per_entry=f"{per_entry_ms:.0f}ms",
                )

            # 更新外层进度条附加信息
            if show_progress and _tqdm_available and hasattr(batch_iter, "set_postfix"):
                batch_iter.set_postfix(
                    batch_s=f"{batch_elapsed:.1f}s",
                    per_entry=f"{per_entry_ms:.0f}ms",
                    done=f"{start + len(batch_jds)}/{total}",
                )

            logger.info(
                f"[Batch {batch_idx + 1:3d}/{n_batches}] "
                f"{len(batch_jds)} 条 | "
                f"耗时 {batch_elapsed:.2f}s | "
                f"均 {per_entry_ms:.0f}ms/条 | "
                f"已完成 {start + len(batch_jds)}/{total}"
            )

        if entry_bar is not None:
            entry_bar.close()

        total_elapsed = time.perf_counter() - total_wall_start
        avg_ms = total_elapsed / total * 1000 if total else 0
        logger.info(
            f"批量推理完成：{total} 条 | "
            f"总耗时 {total_elapsed:.1f}s | "
            f"全程均 {avg_ms:.0f}ms/条"
        )
        return results
