import ast
import json
import re
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import RAGConfig


class QwenGenerator:
    """Qwen3-8B 生成器。

    设计目标：
    - 严格基于检索候选回答；
    - 尽量输出可解析 JSON，避免自由文本导致流程中断。
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.generator_model_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            config.generator_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # 清理与贪心解码冲突的默认参数，避免控制台 warning。
        self.model.generation_config.temperature = None
        self.model.generation_config.top_k = None
        self.model.generation_config.top_p = None

        self.model.eval()

    @staticmethod
    def _build_context(candidates: List[Dict]) -> str:
        """把检索候选拼装成上下文。

        为了降低 prompt 噪声和长度，定义/任务做截断。
        """
        chunks = []
        for c in candidates:
            desc = str(c.get("desc", ""))[:220]
            tasks = str(c.get("tasks", ""))[:280]
            chunks.append(
                f"[候选{c['rank']}] 代码:{c['code']} 名称:{c['title']}\n"
                f"定义:{desc}\n"
                f"任务:{tasks}\n"
                f"检索分数:{c['score']:.4f}"
            )
        return "\n\n".join(chunks)

    def _generate_text(self, prompt: str, max_new_tokens: int = None) -> str:
        """执行一次模型生成并返回纯文本。"""
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

        enc = self.tokenizer(rendered, return_tensors="pt")
        enc = {k: v.to(self.model.device) for k, v in enc.items()}

        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.config.do_sample:
            gen_kwargs["temperature"] = self.config.temperature

        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)

        input_len = int(enc["attention_mask"].sum().item())
        return self.tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()

    @staticmethod
    def _extract_json(raw_text: str) -> Dict:
        """从模型输出中尽可能鲁棒地提取 JSON。"""
        text = str(raw_text or "").strip()
        text = text.replace("```json", "").replace("```", "")
        text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()

        # 1) 优先尝试整段解析
        for loader in (json.loads, ast.literal_eval):
            try:
                obj = loader(text)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

        # 2) 回退：抽取文本中第一个 {...}
        candidates = re.findall(r"\{[\s\S]*\}", text)
        for cand in candidates:
            for loader in (json.loads, ast.literal_eval):
                try:
                    obj = loader(cand)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    pass

        return {}

    def generate(self, query: str, candidates: List[Dict]) -> Dict:
        """执行 RAG 生成，返回结构化结果。"""
        context = self._build_context(candidates)

        # /no_think 用于尽量关闭推理过程外显，减少输出被“思考文本”占满。
        prompt = (
            "/no_think\n"
            "你是职业分类助手。请只基于给定候选进行判断，不允许使用外部知识。\n"
            "你必须仅输出一行 JSON，不允许输出任何解释、markdown、或<think>内容。\n"
            "JSON 键必须为：final_code,final_title,confidence,reason,evidence_rank。\n"
            "若证据不足，请输出 final_code='UNCERTAIN'、final_title='不确定'。\n\n"
            f"用户输入:\n{query}\n\n"
            f"候选知识:\n{context}\n"
        )

        text = self._generate_text(prompt)
        data = self._extract_json(text)

        # 若首轮依然失败，进行一次“JSON 修复”二次生成。
        if not data:
            repair_prompt = (
                "/no_think\n"
                "请把下面文本改写为一行合法 JSON，仅保留键："
                "final_code,final_title,confidence,reason,evidence_rank。\n"
                "如果文本中没有明确结论，请输出 final_code='UNCERTAIN'、final_title='不确定'。\n\n"
                f"原始文本:\n{text[:1200]}"
            )
            repaired = self._generate_text(repair_prompt, max_new_tokens=160)
            data = self._extract_json(repaired)

        if isinstance(data, dict) and data:
            # 补齐缺失键，保持输出结构稳定。
            data.setdefault("final_code", "UNCERTAIN")
            data.setdefault("final_title", "不确定")
            data.setdefault("confidence", 0.0)
            data.setdefault("reason", "")
            data.setdefault("evidence_rank", "")
            return data

        return {
            "final_code": "UNCERTAIN",
            "final_title": "不确定",
            "confidence": 0.0,
            "reason": f"模型输出未能解析为 JSON: {text[:400]}",
            "evidence_rank": "",
        }
