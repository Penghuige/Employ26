"""v2 平面化词典流水线入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_skill_extraction_config
from .occupation_skill_pipeline import (
    DEFAULT_COVERAGE_THRESHOLD,
    DEFAULT_MODEL_PATH,
    FlatSkillPipeline,
)

RECOMMENDED_LOCAL_MODELS = [
    "models/hf/Qwen2.5-14B-Instruct",
    "models/hf/DeepSeek-R1-Distill-Qwen-14B",
    "models/hf/Qwen2.5-7B-Instruct",
]


def build_parser() -> argparse.ArgumentParser:
    """构建 v2 流水线命令行参数。"""
    parser = argparse.ArgumentParser(
        description="v2 平面化硬技能词典流水线",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Qwen3-8B 模型路径（默认从 config/database.yaml 的 LLM_model_path 读取）",
    )
    parser.add_argument("--train-size", type=int, default=100, help="每个职业中类的训练样本数")
    parser.add_argument("--validation-size", type=int, default=10, help="每个职业中类的验证样本数")
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=DEFAULT_COVERAGE_THRESHOLD,
        help=f"覆盖率阈值 (默认: {DEFAULT_COVERAGE_THRESHOLD})",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--limit-job-rows", type=int, default=None, help="调试用，限制每张招聘表读取行数")
    parser.add_argument("--limit-categories", type=int, default=None, help="调试用，限制处理的职业中类数量")
    parser.add_argument("--parse-workers", type=int, default=1, help="岗位描述解析并发数")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80, help="GPU 显存利用率")
    parser.add_argument("--max-model-len", type=int, default=8192, help="vLLM 最大序列长度")
    parser.add_argument("--max-num-seqs", type=int, default=48, help="vLLM 最大并发序列数")
    parser.add_argument("--output", type=str, default=None, help="输出词典路径")
    parser.add_argument("--print-model-choice", action="store_true", help="打印当前自动选择的本地模型并退出")
    return parser


def _resolve_model_path(config, cli_model: str | None) -> str:
    if cli_model:
        return cli_model
    return str(config.llm_model_path)


def main() -> None:
    """执行 v2 平面化流水线。"""
    parser = build_parser()
    args = parser.parse_args()
    config = load_skill_extraction_config()
    resolved_model = _resolve_model_path(config, args.model)
    if args.print_model_choice:
        print(resolved_model)
        return
    pipeline = FlatSkillPipeline(
        config=config,
        model_path=resolved_model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
    )
    pipeline.run(
        train_size=args.train_size,
        validation_size=args.validation_size,
        coverage_threshold=args.coverage_threshold,
        seed=args.seed,
        limit_job_rows=args.limit_job_rows,
        limit_categories=args.limit_categories,
        parse_workers=args.parse_workers,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
