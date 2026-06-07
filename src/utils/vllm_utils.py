"""vLLM server/session shared utilities."""

from __future__ import annotations

import os
import shlex
import subprocess
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from src.utils.utils import PROJECT_ROOT, safe_print


DEFAULT_VLLM_CONFIG_PATH = PROJECT_ROOT / "config" / "vllm.toml"
PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


@dataclass(frozen=True)
class VLLMConfig:
    """Local vLLM service configuration."""

    host: str = "127.0.0.1"
    port: int = 8100
    distro: str = "Ubuntu-24.04"
    model_name: str = "Qwen3.6-27B-int4-AutoRound"
    windows_model_dir: str = ""
    wsl_model_dir: str = "/home/lin/models/Qwen3.6-27B-int4-AutoRound"
    wsl_vllm_env: str = "/home/lin/llm-env"
    wsl_vllm_bin: str = "/home/lin/llm-env/bin/vllm"
    startup_timeout_seconds: int = 1800
    request_timeout_seconds: int = 600
    dtype: str = "half"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.92
    max_num_seqs: int = 4
    enable_prefix_caching: bool = True
    safetensors_load_strategy: str = "prefetch"
    trust_remote_code: bool = True
    enable_reasoning: bool = False
    reasoning_parser: str = "qwen3"
    enable_auto_tool_choice: bool = True
    tool_call_parser: str = "qwen3_xml"
    extra_args: list[str] = field(default_factory=list)
    temperature: float = 0.2
    max_tokens: int = 1024
    enable_thinking: bool = False
    system_prompt: str = "你是一个中文助手。回答要直接、清晰，并尽量保持上下文连续。"

    @property
    def api_base(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def chat_completions_url(self) -> str:
        return f"{self.api_base}/chat/completions"


def load_vllm_config(config_path: str | Path | None = None) -> VLLMConfig:
    """Load vLLM configuration from TOML."""
    path = Path(config_path) if config_path else DEFAULT_VLLM_CONFIG_PATH
    if not path.exists():
        return VLLMConfig()

    with path.open("rb") as file:
        data = tomllib.load(file)

    server = data.get("server", {})
    wsl = data.get("wsl", {})
    model = data.get("model", {})
    vllm = data.get("vllm", {})
    chat = data.get("chat", {})
    vllm_env = str(wsl.get("vllm_env", VLLMConfig.wsl_vllm_env))
    vllm_bin = str(wsl.get("vllm_bin") or f"{vllm_env.rstrip('/')}/bin/vllm")

    return VLLMConfig(
        host=str(server.get("host", VLLMConfig.host)),
        port=int(server.get("port", VLLMConfig.port)),
        distro=str(wsl.get("distro", VLLMConfig.distro)),
        model_name=str(model.get("name", VLLMConfig.model_name)),
        windows_model_dir=str(model.get("windows_dir", VLLMConfig.windows_model_dir) or ""),
        wsl_model_dir=str(model.get("wsl_dir", VLLMConfig.wsl_model_dir)),
        wsl_vllm_env=vllm_env,
        wsl_vllm_bin=vllm_bin,
        startup_timeout_seconds=int(
            server.get("startup_timeout_seconds", VLLMConfig.startup_timeout_seconds)
        ),
        request_timeout_seconds=int(
            server.get("request_timeout_seconds", VLLMConfig.request_timeout_seconds)
        ),
        dtype=str(vllm.get("dtype", VLLMConfig.dtype)),
        max_model_len=int(vllm.get("max_model_len", VLLMConfig.max_model_len)),
        gpu_memory_utilization=float(
            vllm.get("gpu_memory_utilization", VLLMConfig.gpu_memory_utilization)
        ),
        max_num_seqs=int(vllm.get("max_num_seqs", VLLMConfig.max_num_seqs)),
        enable_prefix_caching=bool(
            vllm.get("enable_prefix_caching", VLLMConfig.enable_prefix_caching)
        ),
        safetensors_load_strategy=str(
            vllm.get("safetensors_load_strategy", VLLMConfig.safetensors_load_strategy)
        ),
        trust_remote_code=bool(vllm.get("trust_remote_code", VLLMConfig.trust_remote_code)),
        enable_reasoning=bool(vllm.get("enable_reasoning", VLLMConfig.enable_reasoning)),
        reasoning_parser=str(vllm.get("reasoning_parser", VLLMConfig.reasoning_parser)),
        enable_auto_tool_choice=bool(
            vllm.get("enable_auto_tool_choice", VLLMConfig.enable_auto_tool_choice)
        ),
        tool_call_parser=str(vllm.get("tool_call_parser", VLLMConfig.tool_call_parser)),
        extra_args=[str(item) for item in vllm.get("extra_args", [])],
        temperature=float(chat.get("temperature", VLLMConfig.temperature)),
        max_tokens=int(chat.get("max_tokens", VLLMConfig.max_tokens)),
        enable_thinking=bool(chat.get("enable_thinking", VLLMConfig.enable_thinking)),
        system_prompt=str(chat.get("system_prompt", VLLMConfig.system_prompt)),
    )


def create_http_session() -> requests.Session:
    """Create a requests session that ignores proxy env vars."""
    session = requests.Session()
    session.trust_env = False
    return session


def build_subprocess_env() -> dict[str, str]:
    """Create a Windows subprocess environment suitable for calling WSL."""
    env = os.environ.copy()
    env["SystemRoot"] = env.get("SystemRoot") or r"C:\Windows"
    env["windir"] = env.get("windir") or r"C:\Windows"
    for key in PROXY_ENV_KEYS:
        env.pop(key, None)
    env["NO_PROXY"] = "localhost,127.0.0.1,::1"
    env["no_proxy"] = "localhost,127.0.0.1,::1"
    return env


def _is_gguf(config: VLLMConfig) -> bool:
    """Detect whether the model path points to a GGUF file."""
    return config.wsl_model_dir.lower().endswith(".gguf")


def build_llama_command(config: VLLMConfig) -> list[str]:
    """Build the WSL command that starts llama.cpp OpenAI-compatible server for GGUF models."""
    python_bin = f"{config.wsl_vllm_env.rstrip('/')}/bin/python"

    llama_args = [
        python_bin,
        "-m", "llama_cpp.server",
        "--model", config.wsl_model_dir,
        "--host", "0.0.0.0",
        "--port", str(config.port),
        "--n_ctx", str(config.max_model_len),
        "--n_gpu_layers", "-1",
        "--chat_template_kwargs", '{"enable_thinking": false}',
    ]

    serve_command = " ".join(shlex.quote(str(arg)) for arg in llama_args)
    shell_command = (
        "unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy; "
        f"exec {serve_command}"
    )
    return ["wsl", "-d", config.distro, "--", "bash", "-lc", shell_command]


def build_vllm_command(config: VLLMConfig) -> list[str]:
    """Build the Windows subprocess command that starts the model server inside WSL.

    Automatically detects GGUF files and routes to llama.cpp server.
    Falls back to vLLM serve for HuggingFace-format models.
    """
    if _is_gguf(config):
        return build_llama_command(config)

    vllm_args = [
        config.wsl_vllm_bin,
        "serve",
        config.wsl_model_dir,
        "--served-model-name",
        config.model_name,
        "--dtype",
        config.dtype,
        "--max-model-len",
        str(config.max_model_len),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
        "--max-num-seqs",
        str(config.max_num_seqs),
        "--safetensors-load-strategy",
        config.safetensors_load_strategy,
        "--host",
        "0.0.0.0",
        "--port",
        str(config.port),
    ]
    if config.enable_prefix_caching:
        vllm_args.append("--enable-prefix-caching")
    if config.trust_remote_code:
        vllm_args.append("--trust-remote-code")
    if config.enable_reasoning:
        if config.reasoning_parser:
            vllm_args.extend(["--reasoning-parser", config.reasoning_parser])
        if config.enable_auto_tool_choice:
            vllm_args.append("--enable-auto-tool-choice")
        if config.tool_call_parser:
            vllm_args.extend(["--tool-call-parser", config.tool_call_parser])
    vllm_args.extend(config.extra_args)

    serve_command = " ".join(shlex.quote(arg) for arg in vllm_args)
    shell_command = (
        "unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy; "
        f"exec {serve_command}"
    )
    return ["wsl", "-d", config.distro, "--", "bash", "-lc", shell_command]


def run_wsl(
    command: str,
    config: VLLMConfig,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    """Run one bash command in the configured WSL distro."""
    return subprocess.run(
        ["wsl", "-d", config.distro, "--", "bash", "-lc", command],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=build_subprocess_env(),
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def check_environment(config: VLLMConfig) -> None:
    """Check model path, runtime executable, and GPU availability."""
    if config.windows_model_dir and not os.path.isdir(config.windows_model_dir):
        raise FileNotFoundError(f"Windows model directory does not exist: {config.windows_model_dir}")

    model_path = shlex.quote(config.wsl_model_dir)

    if _is_gguf(config):
        # GGUF: check file exists, llama-cpp-python is installed, and GPU is available
        python_bin = f"{config.wsl_vllm_env.rstrip('/')}/bin/python"
        checks = run_wsl(
            f"set -e; test -f {model_path}; "
            f"{python_bin} -c \"import llama_cpp; print('llama-cpp-python ' + llama_cpp.__version__)\"; "
            "echo 'Backend: llama.cpp (GGUF)'",
            config=config,
            timeout=120,
        )
    else:
        # HuggingFace model: check directory, vLLM binary, and GPU
        checks = run_wsl(
            "set -e; "
            f"test -e {model_path}; "
            f"test -x {shlex.quote(config.wsl_vllm_bin)}; "
            f"{shlex.quote(config.wsl_vllm_bin)} --version; "
            "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader; "
            "echo 'Backend: vLLM'",
            config=config,
            timeout=120,
        )
    safe_print(checks.stdout.strip())


def wait_until_ready(
    config: VLLMConfig,
    timeout_seconds: int | None = None,
    session: requests.Session | None = None,
) -> None:
    """Poll `/v1/models` until the local vLLM API responds."""
    timeout = timeout_seconds or config.startup_timeout_seconds
    http = session or create_http_session()
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = http.get(f"{config.api_base}/models", timeout=5)
            if response.ok:
                print(f"vLLM API 已就绪: {config.api_base}")
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"vLLM API 在 {timeout} 秒内未就绪: {config.api_base}")


def check_server(config: VLLMConfig, session: requests.Session | None = None) -> None:
    """Raise a clear error if the vLLM OpenAI-compatible API is unavailable."""
    http = session or create_http_session()
    try:
        response = http.get(f"{config.api_base}/models", timeout=10)
    except requests.RequestException as exc:
        raise RuntimeError(
            f"无法连接到 {config.api_base}。请先启动模型服务:\n"
            f"  .\\.conda\\python.exe -m src.llm.vllm_server serve\n"
            f"原始错误: {exc}"
        ) from exc

    if not response.ok:
        raise RuntimeError(
            f"服务可连接，但 /v1/models 返回 HTTP {response.status_code}。\n"
            f"响应内容:\n{response.text[:2000]}"
        )


def chat_completion(
    *,
    config: VLLMConfig,
    messages: list[dict[str, Any]],
    session: requests.Session | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    enable_thinking: bool | None = None,
) -> dict[str, Any]:
    """Call `/v1/chat/completions` and return the decoded JSON response."""
    http = session or create_http_session()
    resolved_thinking = config.enable_thinking if enable_thinking is None else enable_thinking
    payload: dict[str, Any] = {
        "model": config.model_name,
        "messages": messages,
        "temperature": config.temperature if temperature is None else temperature,
        "max_tokens": config.max_tokens if max_tokens is None else max_tokens,
    }
    # llama.cpp expects enable_thinking as a top-level key;
    # vLLM expects it nested inside chat_template_kwargs
    if _is_gguf(config):
        payload["enable_thinking"] = resolved_thinking
    else:
        payload["chat_template_kwargs"] = {"enable_thinking": resolved_thinking}
    response = http.post(
        config.chat_completions_url,
        json=payload,
        timeout=config.request_timeout_seconds,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"聊天 API 返回 HTTP {response.status_code}。\n"
            "如果是 502，常见原因是 vLLM 加载或运行模型失败。\n"
            f"响应内容:\n{response.text[:4000]}"
        ) from exc
    return response.json()


def extract_message_parts(response_data: dict[str, Any]) -> tuple[str, str, str | None]:
    """Return content, reasoning, and finish_reason from a chat response."""
    choice = response_data["choices"][0]
    message = choice.get("message", {})
    content = message.get("content") or ""
    reasoning = message.get("reasoning") or ""
    return content, reasoning, choice.get("finish_reason")


def print_api_info(config: VLLMConfig) -> None:
    """Print important server and model paths."""
    backend = "llama.cpp (GGUF)" if _is_gguf(config) else "vLLM"
    print(f"推理后端: {backend}")
    print(f"API 基础地址: {config.api_base}")
    print(f"聊天补全地址: {config.chat_completions_url}")
    print(f"模型名称: {config.model_name}")
    print(f"Windows 模型目录: {config.windows_model_dir or '未配置'}")
    print(f"WSL 模型目录: {config.wsl_model_dir}")
    print(f"WSL 发行版: {config.distro}")
    print(f"WSL vLLM 路径: {config.wsl_vllm_bin}")
