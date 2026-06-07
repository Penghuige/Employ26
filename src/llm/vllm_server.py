"""CLI for managing the local vLLM OpenAI-compatible server."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.vllm_utils import (  # noqa: E402
    DEFAULT_VLLM_CONFIG_PATH,
    build_subprocess_env,
    build_vllm_command,
    check_environment,
    load_vllm_config,
    print_api_info,
    wait_until_ready,
)


def serve(config_path: str | None = None, skip_check: bool = False) -> None:
    """Start vLLM inside WSL and block until the server exits."""
    config = load_vllm_config(config_path)
    print_api_info(config)
    if not skip_check:
        check_environment(config)
    print("正在 WSL 中启动 vLLM。调用 API 时请保持该终端窗口运行。")
    subprocess.run(build_vllm_command(config), check=True, env=build_subprocess_env())


def build_parser() -> argparse.ArgumentParser:
    """Build the server CLI parser."""
    parser = argparse.ArgumentParser(
        description="管理本地 vLLM OpenAI 兼容 API 服务。",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_VLLM_CONFIG_PATH),
        help="vLLM TOML 配置文件路径",
    )
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser("serve", help="在 WSL 中启动 vLLM 服务")
    serve_parser.add_argument("--skip-check", action="store_true", help="跳过环境检查，直接启动")

    subparsers.add_parser("check", help="检查 WSL、模型路径、vLLM 和 GPU")
    subparsers.add_parser("info", help="打印 API、模型路径和启动命令")

    wait_parser = subparsers.add_parser("wait", help="等待 API 服务就绪")
    wait_parser.add_argument("--timeout", type=int, default=None, help="等待秒数")

    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and run the requested server action."""
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "serve"
    config = load_vllm_config(args.config)

    if command == "serve":
        serve(config_path=args.config, skip_check=getattr(args, "skip_check", False))
        return

    if command == "check":
        check_environment(config)
        return

    if command == "info":
        print_api_info(config)
        print("服务启动命令:")
        print(subprocess.list2cmdline(build_vllm_command(config)))
        return

    if command == "wait":
        wait_until_ready(config, timeout_seconds=args.timeout)
        return

    parser.error(f"未知命令: {command}")


if __name__ == "__main__":
    main()
