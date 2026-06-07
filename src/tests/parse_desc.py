"""
最小化验证本地 Qwen3-8B 是否能通过 vLLM 正常启动并生成文本。

为什么之前会报错：
- 你的脚本把 `gpu_memory_utilization` 固定成了 `0.90`
- 24GB 显卡在启动时希望预留约 `21.59 GiB`
- 但实际空闲显存只有 `19.61 GiB`
- 所以 vLLM 在初始化阶段直接拒绝启动

这版脚本做了两点修复：
1. 默认把显存占用目标调低到更稳的 `0.80`
2. 允许用环境变量覆盖，方便你在不同机器上试参数

可选环境变量：
- `QWEN_MODEL_PATH`：模型目录，默认 `D:\\model\\Qwen3-8B`
- `VLLM_GPU_MEMORY_UTILIZATION`：显存占用比例，默认 `0.80`
- `QWEN_MAX_MODEL_LEN`：上下文长度，默认 `4096`
- `CUDA_VISIBLE_DEVICES`：默认仍使用 `0`
"""

from __future__ import annotations

import os

from vllm import LLM, SamplingParams


os.environ["VLLM_HOST_IP"] = "127.0.0.1"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", r"D:\model\Qwen3-8B")
GPU_MEMORY_UTILIZATION = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.80"))
MAX_MODEL_LEN = int(os.environ.get("QWEN_MAX_MODEL_LEN", "4096"))


def build_llm() -> LLM:
    """
    构建 vLLM 实例。

    说明：
    - `trust_remote_code` 对当前这条加载路径没有实际作用，因此不再传入；
    - 这里只保留最小必要参数，减少测试阶段的干扰因素；
    - `gpu_memory_utilization` 降低后，通常就能绕开“启动时预留显存过高”的报错。
    """
    return LLM(
        model=MODEL_PATH,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enforce_eager=True,
        disable_log_stats=True,
    )


def main() -> None:
    print(f"model={MODEL_PATH}")
    print(f"gpu_memory_utilization={GPU_MEMORY_UTILIZATION}")
    print(f"max_model_len={MAX_MODEL_LEN}")

    try:
        llm = build_llm()
    except ValueError as exc:
        message = str(exc)
        if "Free memory on device" in message and "gpu memory utilization" in message.lower():
            raise SystemExit(
                "\n".join(
                    [
                        "vLLM 启动失败：当前空闲显存小于目标预留显存。",
                        f"当前脚本使用的 gpu_memory_utilization={GPU_MEMORY_UTILIZATION:.2f}",
                        "可以继续尝试：",
                        "1. 先关闭其他占用显存的程序",
                        "2. 临时调低环境变量 VLLM_GPU_MEMORY_UTILIZATION，例如 0.75",
                        "3. 如仍不足，再把 QWEN_MAX_MODEL_LEN 调低到 3072 或 2048",
                    ]
                )
            ) from exc
        raise

    params = SamplingParams(max_tokens=256, temperature=0.7)
    output = llm.generate(["Hello!"], sampling_params=params)
    print(output[0].outputs[0].text)


if __name__ == "__main__":
    main()
