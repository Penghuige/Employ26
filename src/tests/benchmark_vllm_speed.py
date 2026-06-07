"""测试本地 vLLM OpenAI 兼容接口的 token 处理速度。

该脚本会向 `/v1/chat/completions` 发送固定测试请求，并根据接口返回的
`usage.prompt_tokens`、`usage.completion_tokens` 和本地计时计算速度。

常用命令：
    .\\.conda\\python.exe src/test/benchmark_vllm_speed.py
    .\\.conda\\python.exe src/test/benchmark_vllm_speed.py --parallel 4 --rounds 8
"""

import argparse
import concurrent.futures
import statistics
import time
from dataclasses import dataclass

import requests


API_URL = "http://127.0.0.1:8100/v1/chat/completions"
MODEL_NAME = "Qwen3.6-27B-int4-AutoRound"
REQUEST_TIMEOUT_SECONDS = 600


@dataclass(frozen=True)
class SpeedResult:
    """保存单次请求的耗时和 token 统计。"""

    elapsed_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @property
    def completion_tokens_per_second(self) -> float:
        """返回输出 token 速度。"""
        return self.completion_tokens / self.elapsed_seconds

    @property
    def total_tokens_per_second(self) -> float:
        """返回总 token 速度，包含 prompt token 和 completion token。"""
        return self.total_tokens / self.elapsed_seconds


def build_payload(max_tokens: int) -> dict:
    """构造用于测速的请求体。

    为了测量生成速度，请求会要求模型输出较长的编号列表，并关闭 thinking。

    Args:
        max_tokens: 允许模型最多生成的 token 数。

    Returns:
        可直接传给 requests.post(json=...) 的字典。
    """
    return {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "你是一个严格执行格式要求的中文助手。不要解释，只输出正文。",
            },
            {
                "role": "user",
                "content": (
                    "请输出 120 条不同的软件开发硬技能，每条一行，格式为："
                    "序号. 技能名 - 8到12字中文说明。"
                ),
            },
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def run_once(session: requests.Session, max_tokens: int) -> SpeedResult:
    """执行一次测速请求并返回速度统计。

    Args:
        session: 复用连接的 requests 会话。
        max_tokens: 单次请求允许生成的最大 token 数。

    Returns:
        SpeedResult，包含耗时、输入 token、输出 token 和总 token。

    Raises:
        requests.HTTPError: 当接口返回非 2xx 状态码时抛出。
    """
    start = time.perf_counter()
    response = session.post(API_URL, json=build_payload(max_tokens), timeout=REQUEST_TIMEOUT_SECONDS)
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    usage = response.json()["usage"]
    return SpeedResult(
        elapsed_seconds=elapsed,
        prompt_tokens=usage.get("prompt_tokens") or 0,
        completion_tokens=usage.get("completion_tokens") or 0,
        total_tokens=usage.get("total_tokens") or 0,
    )


def print_summary(results: list[SpeedResult], wall_seconds: float) -> None:
    """打印测速汇总。

    Args:
        results: 多次请求的测速结果。
        wall_seconds: 整个测试批次的墙钟耗时。
    """
    completion_tokens = sum(item.completion_tokens for item in results)
    total_tokens = sum(item.total_tokens for item in results)
    per_request_output_speeds = [item.completion_tokens_per_second for item in results]

    print(f"请求数: {len(results)}")
    print(f"墙钟耗时: {wall_seconds:.2f} 秒")
    print(f"输出 token 总数: {completion_tokens}")
    print(f"总 token 数: {total_tokens}")
    print(f"整体输出吞吐: {completion_tokens / wall_seconds:.2f} token/s")
    print(f"整体总吞吐: {total_tokens / wall_seconds:.2f} token/s")
    print(f"单请求输出速度均值: {statistics.mean(per_request_output_speeds):.2f} token/s")
    print(f"单请求输出速度中位数: {statistics.median(per_request_output_speeds):.2f} token/s")


def main() -> None:
    """解析命令行参数并执行测速。"""
    parser = argparse.ArgumentParser(description="Benchmark local vLLM token speed.")
    parser.add_argument("--rounds", type=int, default=3, help="总请求次数")
    parser.add_argument("--parallel", type=int, default=1, help="并发请求数")
    parser.add_argument("--max-tokens", type=int, default=512, help="每个请求的最大输出 token 数")
    args = parser.parse_args()

    started = time.perf_counter()
    if args.parallel <= 1:
        session = requests.Session()
        session.trust_env = False
        results = [run_once(session, args.max_tokens) for _ in range(args.rounds)]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = []
            for _ in range(args.rounds):
                session = requests.Session()
                session.trust_env = False
                futures.append(executor.submit(run_once, session, args.max_tokens))
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

    print_summary(results, time.perf_counter() - started)


if __name__ == "__main__":
    main()
