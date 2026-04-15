"""
本地 Qwen3-8B 职业技能词典工作流。

职责：
1. 读取 `occupation_skill_pipeline.py` 生成的 prompt；
2. 调用本地模型 `D:\\model\\Qwen3-8B` 批量生成技能词典 JSON；
3. 把 JSON 落盘到 `output/skill_extraction/llm_outputs`；
4. 自动导入词典并驱动后续迭代。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
import shutil
from typing import Dict, Iterable, List, Sequence

import pandas as pd

from ..config import load_skill_extraction_config
from .init_llm_output_layout import initialize_layout
from .occupation_skill_pipeline import OccupationSkillPipeline


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


DEFAULT_MODEL_PATH = Path(r"D:\model\Qwen3-8B")


def _safe_text(value: object) -> str:
    """安全转字符串。"""
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _extract_json_from_fenced_blocks(text: str) -> List[str]:
    """从 markdown fenced code block 中提取 JSON。"""
    pattern = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
    return [match.group(1).strip() for match in pattern.finditer(text) if match.group(1).strip()]


def _extract_balanced_json_fragments(text: str) -> List[str]:
    """从普通文本中提取平衡 JSON 对象。"""
    fragments: List[str] = []
    stack: List[str] = []
    start_index: int | None = None
    in_string = False
    escape = False

    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char in "{[":
            if not stack:
                start_index = index
            stack.append(char)
            continue

        if char in "}]":
            if not stack:
                continue
            expected = "{" if char == "}" else "["
            if stack[-1] != expected:
                stack.clear()
                start_index = None
                continue
            stack.pop()
            if not stack and start_index is not None:
                fragment = text[start_index : index + 1].strip()
                if fragment:
                    fragments.append(fragment)
                start_index = None
    return fragments


def _parse_skill_payload(text: str) -> Dict | None:
    """从模型输出中解析技能词典 payload。"""
    candidates: List[str] = []
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)
    candidates.extend(_extract_json_from_fenced_blocks(text))
    candidates.extend(_extract_balanced_json_fragments(text))

    seen = set()
    for candidate in candidates:
        normalized = candidate.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        try:
            payload = json.loads(normalized)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and _safe_text(payload.get("detail_path", "")):
            if isinstance(payload.get("skills"), list) or isinstance(payload.get("missing_skills"), list):
                return payload
    return None


def _estimate_chars(text: str) -> int:
    """估算 prompt 长度。"""
    return len(text or "")


def _iter_prompt_files(prompt_dir: Path) -> List[Path]:
    """列出 prompt 文件。"""
    if not prompt_dir.exists():
        return []
    return sorted([path for path in prompt_dir.glob("*.md") if path.is_file()])


def _archive_json_files(json_dir: Path, imported_dir: Path) -> int:
    """把已导入 JSON 归档到 imported 目录。"""
    imported_dir.mkdir(parents=True, exist_ok=True)
    moved_count = 0
    for json_path in sorted(json_dir.glob("*.json")):
        target_path = imported_dir / json_path.name
        if target_path.exists():
            target_path.unlink()
        shutil.move(str(json_path), str(target_path))
        moved_count += 1
    return moved_count


@dataclass(frozen=True)
class PromptTask:
    """单个 prompt 生成任务。"""

    prompt_path: Path
    prompt_text: str
    prompt_key: str
    raw_output_path: Path
    json_output_path: Path
    error_output_path: Path
    char_length: int


class LocalQwenSkillGenerator:
    """本地 Qwen3-8B 批量技能词典生成器。"""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        device_map: str = "auto",
        max_input_tokens: int = 6144,
        max_new_tokens: int = 1024,
        base_batch_size: int = 4,
        allow_cpu: bool = False,
    ):
        self.model_path = Path(model_path)
        self.device_map = device_map
        self.max_input_tokens = int(max_input_tokens)
        self.max_new_tokens = int(max_new_tokens)
        self.base_batch_size = int(base_batch_size)
        self.allow_cpu = bool(allow_cpu)
        self._tokenizer = None
        self._model = None

    def _load_model(self) -> None:
        """懒加载模型。"""
        if self._model is not None and self._tokenizer is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(f"模型路径不存在: {self.model_path}")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("请先安装依赖：pip install transformers torch accelerate") from exc

        if not torch.cuda.is_available() and not self.allow_cpu:
            raise RuntimeError(
                "当前环境未检测到 CUDA，Qwen3-8B 在 CPU 上会非常慢。"
                "如确认要在 CPU 上运行，请加参数 `--allow-cpu`。"
            )

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        logger.info("加载 Qwen 模型: %s", self.model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), trust_remote_code=True)
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype=dtype,
            device_map=self.device_map,
            trust_remote_code=True,
        )
        self._model.eval()
        logger.info("Qwen 模型加载完成")

    def generate_directory(
        self,
        prompt_dir: Path,
        raw_dir: Path,
        json_dir: Path,
        error_dir: Path,
        overwrite: bool = False,
    ) -> Dict:
        """对一个目录下的 prompt 批量生成结果。"""
        prompt_files = _iter_prompt_files(prompt_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)
        error_dir.mkdir(parents=True, exist_ok=True)

        tasks: List[PromptTask] = []
        for prompt_path in prompt_files:
            prompt_text = prompt_path.read_text(encoding="utf-8")
            prompt_key = prompt_path.stem
            raw_output_path = raw_dir / f"{prompt_key}__qwen3_8b.txt"
            json_output_path = json_dir / f"{prompt_key}__qwen3_8b.json"
            error_output_path = error_dir / f"{prompt_key}__qwen3_8b.error.txt"

            if json_output_path.exists() and not overwrite:
                continue

            tasks.append(
                PromptTask(
                    prompt_path=prompt_path,
                    prompt_text=prompt_text,
                    prompt_key=prompt_key,
                    raw_output_path=raw_output_path,
                    json_output_path=json_output_path,
                    error_output_path=error_output_path,
                    char_length=_estimate_chars(prompt_text),
                )
            )

        if not tasks:
            return {
                "prompt_dir": str(prompt_dir),
                "discovered_prompts": len(prompt_files),
                "pending_prompts": 0,
                "generated_json_count": 0,
                "failed_count": 0,
                "manifest_path": "",
            }

        self._load_model()
        tasks = sorted(tasks, key=lambda item: item.char_length)

        generated_json_count = 0
        failed_count = 0
        manifest_rows: List[Dict] = []
        batch_index = 0
        start = 0
        while start < len(tasks):
            batch_size = self._pick_batch_size(tasks[start].char_length)
            batch = tasks[start : start + batch_size]
            batch_index += 1
            logger.info(
                "生成 batch %s: size=%s, chars=%s~%s",
                batch_index,
                len(batch),
                batch[0].char_length,
                batch[-1].char_length,
            )
            outputs = self._generate_batch([task.prompt_text for task in batch])
            for task, raw_output in zip(batch, outputs):
                task.raw_output_path.write_text(raw_output, encoding="utf-8")
                payload = _parse_skill_payload(raw_output)
                if payload is None:
                    task.error_output_path.write_text(raw_output, encoding="utf-8")
                    failed_count += 1
                    manifest_rows.append(
                        {
                            "prompt_file": str(task.prompt_path),
                            "prompt_key": task.prompt_key,
                            "char_length": task.char_length,
                            "status": "failed",
                            "json_output_path": "",
                            "raw_output_path": str(task.raw_output_path),
                        }
                    )
                    continue

                task.json_output_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                if task.error_output_path.exists():
                    task.error_output_path.unlink()
                generated_json_count += 1
                manifest_rows.append(
                    {
                        "prompt_file": str(task.prompt_path),
                        "prompt_key": task.prompt_key,
                        "char_length": task.char_length,
                        "status": "ok",
                        "json_output_path": str(task.json_output_path),
                        "raw_output_path": str(task.raw_output_path),
                    }
                )
            start += len(batch)

        manifest_path = json_dir.parent / "generation_manifest.csv"
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False, encoding="utf-8-sig")
        return {
            "prompt_dir": str(prompt_dir),
            "discovered_prompts": len(prompt_files),
            "pending_prompts": len(tasks),
            "generated_json_count": generated_json_count,
            "failed_count": failed_count,
            "manifest_path": str(manifest_path),
        }

    def _pick_batch_size(self, char_length: int) -> int:
        """按 prompt 长度动态调整 batch size，减少 padding 和 OOM 风险。"""
        if char_length >= 12000:
            return 1
        if char_length >= 8000:
            return min(self.base_batch_size, 2)
        if char_length >= 5000:
            return min(self.base_batch_size, 3)
        return max(1, self.base_batch_size)

    def _render_chat_texts(self, prompts: Sequence[str]) -> List[str]:
        """将 prompt 转成 Qwen chat template。"""
        rendered: List[str] = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            try:
                text = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            rendered.append(text)
        return rendered

    def _generate_batch(self, prompts: Sequence[str]) -> List[str]:
        """对一个 batch 的 prompt 执行推理。"""
        import torch

        rendered = self._render_chat_texts(prompts)
        enc = self._tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
        )
        enc = {key: value.to(self._model.device) for key, value in enc.items()}
        input_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self._model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        return [
            self._tokenizer.decode(sequence[input_len:], skip_special_tokens=True)
            for sequence in outputs
        ]


class QwenSkillWorkflow:
    """Qwen + 技能词典自动工作流。"""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config = load_skill_extraction_config()
        self.pipeline = OccupationSkillPipeline(self.config)
        self.generator = LocalQwenSkillGenerator(
            model_path=Path(args.model_path),
            device_map=args.device_map,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            base_batch_size=args.batch_size,
            allow_cpu=args.allow_cpu,
        )
        self.llm_output_root = self.config.output_dir / "llm_outputs"

    def run(self) -> None:
        """执行完整工作流。"""
        initialize_layout(self.llm_output_root, force=False)

        if self.args.prepare_first:
            self.pipeline.prepare(
                train_size=self.args.train_size,
                validation_batch_size=self.args.validation_batch_size,
                seed=self.args.seed,
                limit_job_rows=self.args.limit_job_rows,
                limit_categories=self.args.limit_categories,
                match_workers=self.args.match_workers,
                match_chunk_size=self.args.match_chunk_size,
                parse_workers=self.args.parse_workers,
                show_progress=self.args.progress,
            )

        self._run_train_stage()

        for _ in range(self.args.max_rounds):
            round_no = self._run_validation_round()
            if round_no is None:
                logger.info("没有新的补词轮次，工作流结束")
                break

    def generate_only(self) -> None:
        """只跑某一阶段的生成，不导入、不迭代。"""
        initialize_layout(self.llm_output_root, force=False)
        stage = self.args.stage
        round_name = self._normalize_round_name(stage=stage, round_name=self.args.round)
        prompt_dir, raw_dir, json_dir, error_dir, imported_dir = self._stage_dirs(stage=stage, round_name=round_name)

        if imported_dir.exists() and any(imported_dir.glob("*.json")) and not self.args.force:
            logger.info("检测到该阶段已有已导入结果，若需重新生成请加 --force")
            return

        summary = self.generator.generate_directory(
            prompt_dir=prompt_dir,
            raw_dir=raw_dir,
            json_dir=json_dir,
            error_dir=error_dir,
            overwrite=self.args.force,
        )
        logger.info("生成完成: %s", json.dumps(summary, ensure_ascii=False))

    def import_only(self) -> None:
        """只导入某一阶段的 JSON。"""
        stage = self.args.stage
        round_name = self._normalize_round_name(stage=stage, round_name=self.args.round)
        _, _, json_dir, _, imported_dir = self._stage_dirs(stage=stage, round_name=round_name)
        if not json_dir.exists():
            logger.info("待导入目录不存在: %s", json_dir)
            return
        self.pipeline.import_llm_results(json_dir, recursive=False, dry_run=self.args.dry_run)
        if not self.args.dry_run:
            moved = _archive_json_files(json_dir=json_dir, imported_dir=imported_dir)
            logger.info("已归档导入 JSON: %s", moved)

    def _run_train_stage(self) -> None:
        """执行训练阶段：生成 -> 导入。"""
        round_name = "round_00"
        prompt_dir, raw_dir, json_dir, error_dir, imported_dir = self._stage_dirs(stage="train", round_name=round_name)
        if imported_dir.exists() and any(imported_dir.glob("*.json")) and not self.args.force_train:
            logger.info("训练阶段已有已导入结果，跳过 train 生成。若需重跑请加 --force-train")
            return

        summary = self.generator.generate_directory(
            prompt_dir=prompt_dir,
            raw_dir=raw_dir,
            json_dir=json_dir,
            error_dir=error_dir,
            overwrite=self.args.force_train,
        )
        logger.info("训练阶段生成完成: %s", json.dumps(summary, ensure_ascii=False))

        if summary["generated_json_count"] == 0 and not any(json_dir.glob("*.json")):
            logger.info("训练阶段没有可导入 JSON")
            return

        self.pipeline.import_llm_results(json_dir, recursive=False, dry_run=False)
        moved = _archive_json_files(json_dir=json_dir, imported_dir=imported_dir)
        logger.info("训练阶段已归档 JSON: %s", moved)

    def _run_validation_round(self) -> int | None:
        """执行一轮验证，如果有补词 prompt，则生成并导入。"""
        self.pipeline.iterate(
            validation_batch_size=self.args.validation_batch_size,
            coverage_threshold=self.args.coverage_threshold,
            limit_categories=self.args.limit_categories,
            parse_workers=self.args.parse_workers,
        )

        state = self.pipeline._load_state(validation_batch_size=self.args.validation_batch_size)
        round_no = int(state.get("global_round", 0))
        if round_no <= 0:
            return None

        round_name = f"round_{round_no:02d}"
        prompt_dir, raw_dir, json_dir, error_dir, imported_dir = self._stage_dirs(stage="supplement", round_name=round_name)
        prompt_files = _iter_prompt_files(prompt_dir)
        if not prompt_files:
            return None

        if imported_dir.exists() and any(imported_dir.glob("*.json")) and not self.args.force:
            logger.info("补词阶段 %s 已有已导入结果，跳过生成", round_name)
            return round_no

        summary = self.generator.generate_directory(
            prompt_dir=prompt_dir,
            raw_dir=raw_dir,
            json_dir=json_dir,
            error_dir=error_dir,
            overwrite=self.args.force,
        )
        logger.info("补词阶段 %s 生成完成: %s", round_name, json.dumps(summary, ensure_ascii=False))

        if summary["generated_json_count"] == 0 and not any(json_dir.glob("*.json")):
            logger.info("补词阶段 %s 没有可导入 JSON", round_name)
            return round_no

        self.pipeline.import_llm_results(json_dir, recursive=False, dry_run=False)
        moved = _archive_json_files(json_dir=json_dir, imported_dir=imported_dir)
        logger.info("补词阶段 %s 已归档 JSON: %s", round_name, moved)
        return round_no

    def _stage_dirs(self, stage: str, round_name: str) -> tuple[Path, Path, Path, Path, Path]:
        """获取一个阶段的 prompt/raw/json/error/imported 目录。"""
        if stage == "train":
            prompt_dir = self.config.prompt_train_dir
        elif stage == "supplement":
            prompt_dir = self.config.prompt_supplement_dir / round_name
        else:
            raise ValueError(f"不支持的 stage: {stage}")

        stage_root = self.llm_output_root / stage / round_name
        raw_dir = stage_root / "raw"
        json_dir = stage_root / "json"
        error_dir = stage_root / "errors"
        imported_dir = stage_root / "imported"
        for target in [raw_dir, json_dir, error_dir, imported_dir]:
            target.mkdir(parents=True, exist_ok=True)
        return prompt_dir, raw_dir, json_dir, error_dir, imported_dir

    @staticmethod
    def _normalize_round_name(stage: str, round_name: str | None) -> str:
        """标准化轮次名称。"""
        if stage == "train":
            return "round_00"
        if not round_name:
            raise ValueError("supplement 阶段必须指定 --round，例如 round_01")
        return str(round_name)


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""
    parser = argparse.ArgumentParser(description="本地 Qwen3-8B 职业技能词典工作流")
    subparsers = parser.add_subparsers(dest="command")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="本地 Qwen 模型路径")
    common.add_argument("--device-map", default="auto", help="transformers device_map")
    common.add_argument("--batch-size", type=int, default=4, help="基础 batch size")
    common.add_argument("--max-input-tokens", type=int, default=6144, help="输入最大 token 长度")
    common.add_argument("--max-new-tokens", type=int, default=1024, help="输出最大 token 长度")
    common.add_argument("--allow-cpu", action="store_true", help="允许在 CPU 上运行")
    common.add_argument("--force", action="store_true", help="覆盖已有生成结果")

    generate = subparsers.add_parser("generate", parents=[common], help="只生成某一阶段的模型输出")
    generate.add_argument("--stage", choices=["train", "supplement"], required=True, help="生成阶段")
    generate.add_argument("--round", default=None, help="补词阶段轮次，例如 round_01")

    import_cmd = subparsers.add_parser("import", help="只导入某一阶段已生成的 JSON")
    import_cmd.add_argument("--stage", choices=["train", "supplement"], required=True, help="导入阶段")
    import_cmd.add_argument("--round", default=None, help="补词阶段轮次，例如 round_01")
    import_cmd.add_argument("--dry-run", action="store_true", help="只预演导入，不写回词典")

    run = subparsers.add_parser("run", parents=[common], help="完整执行 prepare/train/import/iterate")
    run.add_argument("--prepare-first", action="store_true", help="先执行 prepare")
    run.add_argument("--train-size", type=int, default=100, help="每个细类训练样本数")
    run.add_argument("--validation-batch-size", type=int, default=10, help="每轮每个细类验证样本数")
    run.add_argument("--coverage-threshold", type=float, default=0.95, help="覆盖率阈值")
    run.add_argument("--max-rounds", type=int, default=10, help="最多执行多少轮 iterate")
    run.add_argument("--seed", type=int, default=42, help="prepare 随机种子")
    run.add_argument("--limit-job-rows", type=int, default=None, help="调试用，限制每张招聘表读取行数")
    run.add_argument("--limit-categories", type=int, default=None, help="调试用，限制细类数量")
    run.add_argument("--match-workers", type=int, default=4, help="保留参数，兼容 prepare")
    run.add_argument("--match-chunk-size", type=int, default=256, help="保留参数，兼容 prepare")
    run.add_argument("--parse-workers", type=int, default=1, help="岗位描述切分并发数")
    run.add_argument("--progress", action="store_true", help="保留参数，兼容 prepare")
    run.add_argument("--force-train", action="store_true", help="即使训练阶段已导入，也重新跑 train")

    init = subparsers.add_parser("init-layout", help="初始化 llm_outputs 目录模板")
    init.add_argument("--target", default=r"output\skill_extraction\llm_outputs", help="输出目录")

    return parser


def main() -> None:
    """CLI 入口。"""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "init-layout":
        initialize_layout(Path(args.target), force=False)
        logger.info("已初始化 LLM 输出目录: %s", args.target)
        return

    if args.command in {"generate", "import", "run"}:
        workflow = QwenSkillWorkflow(args)
        if args.command == "generate":
            workflow.generate_only()
            return
        if args.command == "import":
            workflow.import_only()
            return
        if args.command == "run":
            workflow.run()
            return

    parser.print_help()


if __name__ == "__main__":
    main()
