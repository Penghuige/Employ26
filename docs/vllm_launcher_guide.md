# vLLM Windows 智能体对话服务使用指南

## 概述

`vllm_launcher.py` 是一个 Windows 兼容的 vLLM 服务启动器。由于 vLLM 原生 API 服务器依赖 Linux 的 `fork` 和 ZMQ IPC，无法在 Windows 上运行，因此本启动器封装了一个轻量级 **FastAPI** 服务器，提供 **OpenAI 兼容** 的 `/v1/chat/completions` 接口，支持：

- ✅ 多轮对话（chat completions）
- ✅ SSE 流式输出（streaming）
- ✅ **函数调用 / 工具调用（Function Calling / Tool Use）——即"智能体对话"**
- ✅ Embedding 推理（`--task embed`）
- ✅ 持续批处理（continuous batching）
- ✅ 交互式模型选择器

---

## 快速启动

### 1. 环境准备

```bash
# 确保已安装 vLLM（Windows 兼容版本）和 FastAPI
pip install vllm fastapi uvicorn
```

### 2. 启动服务

```bash
# 交互式选择模型（自动扫描常见路径）
python -m src.llm.vllm_launcher --port 8100

# 直接指定模型
python -m src.llm.vllm_launcher --model models/hf/Qwen2.5-14B-Instruct --port 8100

# 使用 GGUF 量化模型
python -m src.llm.vllm_launcher --model models/gguf/qwen2.5-7b-q4.gguf --port 8100
```

### 3. 验证服务

```bash
curl http://127.0.0.1:8100/health
# → {"status":"ok"}

curl http://127.0.0.1:8100/v1/models
# → {"object":"list","data":[{"id":"Qwen2.5-14B-Instruct",...}]}
```

---

## 核心命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | 交互式选择 | 模型路径或 HuggingFace ID |
| `--models-dir` | — | 额外的模型扫描目录 |
| `--port` | `8100` | 服务端口 |
| `--host` | `127.0.0.1` | 绑定地址 |
| `--max-model-len` | 自动检测（上限8192） | 上下文长度上限 |
| `--gpu-memory-utilization` | `0.8` | GPU 显存利用率（0~1） |
| `--max-num-seqs` | `8` | 最大并发序列数 |
| `--tensor-parallel-size` | `1` | 张量并行数（多GPU） |
| `--gpu-id` | — | 指定 GPU 设备索引 |
| `--enable-prefix-caching` | `false` | 启用前缀缓存（复用 KV cache） |
| `--cpu-offload-gb` | `0` | 卸载到 CPU 的权重 GiB 数 |
| `--task` | `generate` | 任务类型：`generate`（对话）或 `embed`（嵌入） |
| `--trust-remote-code` | `false` | 允许自定义模型代码 |
| `--enforce-eager` | `false` | 禁用 CUDA graph（仅调试用） |

### 显存不足时的调参建议

如果启动报 `KV cache` 相关错误，按以下顺序收紧参数：

```bash
# 第1步：降低上下文长度
python -m src.llm.vllm_launcher --max-model-len 4096

# 第2步：降低显存利用率
python -m src.llm.vllm_launcher --gpu-memory-utilization 0.70

# 第3步：减少并发序列
python -m src.llm.vllm_launcher --max-num-seqs 4

# 第4步：卸载部分权重到 CPU
python -m src.llm.vllm_launcher --cpu-offload-gb 4
```

---

## 智能体对话（Agent / Tool Calling）

这是本服务的核心能力——让模型能够**自主决定调用外部工具**来完成复杂任务。

### 工作原理

服务启动后，提供标准的 OpenAI `/v1/chat/completions` 端点。当请求中传入 `tools` 参数时，服务会：

1. **检测模型是否原生支持工具调用模板**（如 Qwen2.5、Llama 3.x）
2. 若原生支持 → 将 tools 注入 chat template
3. 若不支持 → 将工具描述注入 system prompt 作为后备方案
4. 解析模型输出的工具调用，返回 OpenAI 兼容的 `tool_calls` 格式

### 支持的输出格式

以下三种 LLM 工具调用格式均可被解析：

```
格式1（Qwen2.5 / Hermes）:
<tool_call>{"name": "get_weather", "arguments": {"city": "广州"}}</tool_call>

格式2（通用 JSON）:
{"tool": "get_weather", "arguments": {"city": "广州"}}

格式3（直接 JSON）:
{"name": "get_weather", "arguments": {"city": "广州"}}
```

### Python 调用示例

```python
import requests

BASE = "http://127.0.0.1:8100/v1/chat/completions"

# === 1. 定义工具（OpenAI Function Calling 格式） ===
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如：广州、北京"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "在招聘数据库中搜索岗位信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "搜索关键词，如岗位名称"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回结果数量上限",
                        "default": 10
                    }
                },
                "required": ["keyword"]
            }
        }
    }
]

# === 2. 发送对话请求（含工具） ===
messages = [
    {"role": "system", "content": "你是一个招聘数据分析助手。"},
    {"role": "user", "content": "广州今天的天气如何？如果适合外出，帮我查一下广州最近有哪些数据分析岗位。"}
]

resp = requests.post(BASE, json={
    "messages": messages,
    "tools": tools,
    "tool_choice": "auto",       # auto: 模型自动决定是否调用工具
    "temperature": 0.7,
    "max_tokens": 1024,
})
result = resp.json()

# === 3. 处理工具调用响应 ===
choice = result["choices"][0]
msg = choice["message"]

if choice["finish_reason"] == "tool_calls":
    # 模型想要调用工具
    for tc in msg["tool_calls"]:
        func_name = tc["function"]["name"]
        func_args = json.loads(tc["function"]["arguments"])
        print(f"模型请求调用: {func_name}({func_args})")

        # 执行实际的工具函数
        if func_name == "get_weather":
            tool_result = call_weather_api(func_args["city"])
        elif func_name == "search_database":
            tool_result = query_duckdb(func_args["keyword"], func_args.get("limit", 10))

        # 将工具结果追加到消息历史
        messages.append(msg)  # assistant 的 tool_calls 消息
        messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": json.dumps(tool_result, ensure_ascii=False)
        })

    # 继续对话，让模型基于工具结果生成最终回复
    resp2 = requests.post(BASE, json={
        "messages": messages,
        "tools": tools,
        "temperature": 0.7,
        "max_tokens": 1024,
    })
    final = resp2.json()
    print(final["choices"][0]["message"]["content"])

elif choice["finish_reason"] == "stop":
    # 模型直接回复文本
    print(msg["content"])
```

### 流式输出（SSE Streaming）

```python
import requests
import json

resp = requests.post(
    "http://127.0.0.1:8100/v1/chat/completions",
    json={
        "messages": [{"role": "user", "content": "介绍一下广东省的IT行业就业趋势"}],
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 2048,
    },
    stream=True,
)

for line in resp.iter_lines():
    if line:
        line = line.decode("utf-8")
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            content = chunk["choices"][0]["delta"].get("content", "")
            if content:
                print(content, end="", flush=True)
```

---

## 与 OpenAI SDK 集成

由于服务是 OpenAI 兼容的，可以直接使用 `openai` Python SDK：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8100/v1",
    api_key="not-needed",  # 本地服务不需要真实的 API key
)

# 普通对话
response = client.chat.completions.create(
    model="Qwen2.5-14B-Instruct",
    messages=[
        {"role": "user", "content": "你好，请介绍一下你自己"}
    ],
    temperature=0.7,
    max_tokens=512,
)
print(response.choices[0].message.content)

# 智能体对话（带工具）
response = client.chat.completions.create(
    model="Qwen2.5-14B-Instruct",
    messages=[
        {"role": "user", "content": "帮我查询广州的数据分析师岗位"}
    ],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "search_jobs",
                "description": "搜索招聘岗位",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市"},
                        "title": {"type": "string", "description": "岗位名称"},
                    },
                    "required": ["city", "title"],
                },
            },
        }
    ],
    tool_choice="auto",
)

msg = response.choices[0].message
if msg.tool_calls:
    for tc in msg.tool_calls:
        print(f"→ 调用工具: {tc.function.name}({tc.function.arguments})")
else:
    print(msg.content)
```

---

## Embedding 模式

用于文本向量化（如 BGE 系列模型）：

```bash
python -m src.llm.vllm_launcher --model models/hf/bge-large-zh-v1.5 --task embed --port 8101
```

```python
resp = requests.post("http://127.0.0.1:8101/v1/embeddings", json={
    "input": ["数据分析师岗位要求", "Python开发工程师"],
})
embeddings = [d["embedding"] for d in resp.json()["data"]]
```

---

## 完整的多轮智能体对话示例

以下是一个完整的"招聘岗位匹配助手"智能体——模型能够查询数据库、分析薪资、推荐岗位：

```python
import json
import requests

BASE = "http://127.0.0.1:8100/v1/chat/completions"

# 定义工具集
tools = [
    {
        "type": "function",
        "function": {
            "name": "query_jobs_by_title",
            "description": "根据岗位名称查询招聘数据",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "岗位名称关键词"},
                    "city": {"type": "string", "description": "城市，可选"},
                    "limit": {"type": "integer", "description": "返回条数，默认10"},
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_salary_stats",
            "description": "获取某岗位的薪资统计数据",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_title": {"type": "string", "description": "岗位名称"},
                    "city": {"type": "string", "description": "城市"},
                },
                "required": ["job_title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_industry_trend",
            "description": "获取某行业的招聘趋势数据",
            "parameters": {
                "type": "object",
                "properties": {
                    "industry": {"type": "string", "description": "行业名称"},
                },
                "required": ["industry"],
            },
        },
    },
]

# --- 工具实现（示意） ---
def query_jobs_by_title(title, city=None, limit=10):
    """实际项目中查询 DuckDB"""
    return {"jobs": [{"title": "数据分析师", "company": "XX科技", "salary": "15-25K"}], "total": 120}

def get_salary_stats(job_title, city=None):
    return {"avg": 18500, "median": 17000, "p25": 12000, "p75": 25000}

def get_industry_trend(industry):
    return {"trend": "up", "growth": "+15%", "top_cities": ["深圳", "广州", "杭州"]}

TOOL_MAP = {
    "query_jobs_by_title": query_jobs_by_title,
    "get_salary_stats": get_salary_stats,
    "get_industry_trend": get_industry_trend,
}

# --- 对话循环 ---
def agent_chat(user_input: str):
    messages = [
        {"role": "system", "content": "你是一个招聘市场分析助手。你可以查询岗位数据、薪资统计和行业趋势。优先使用工具获取数据，再做分析。"},
        {"role": "user", "content": user_input},
    ]

    max_rounds = 5  # 防止无限循环
    for _ in range(max_rounds):
        resp = requests.post(BASE, json={
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.7,
            "max_tokens": 1024,
        }).json()

        choice = resp["choices"][0]
        msg = choice["message"]

        if choice["finish_reason"] == "stop":
            return msg["content"]

        if choice["finish_reason"] == "tool_calls":
            messages.append(msg)
            for tc in msg["tool_calls"]:
                func = TOOL_MAP.get(tc["function"]["name"])
                if func:
                    args = json.loads(tc["function"]["arguments"])
                    result = func(**args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result, ensure_ascii=False),
                    })

    return "对话超过最大轮次，已终止。"

# 测试
print(agent_chat("深圳的数据分析师岗位薪资如何？行业前景怎么样？"))
```

---

## 模型选择与配置持久化

启动时不指定 `--model`，服务会进入**交互式模型选择器**：

```
============================================================
  vLLM Windows - Model Selection
============================================================

Scanning for models...

Found 5 model(s):

    #  Model                                    Type        Size  Path
  ---  ----                                    ----        ----  ----
  [ 1] Qwen2.5-14B-Instruct                    qwen2       28.2G  D:\models\hf\Qwen2.5-14B-Instruct
  [ 2] Qwen3-8B                                qwen3       16.3G  D:\models\hf\Qwen3-8B
  [ 3] bge-large-zh-v1.5                       bert         0.6G  D:\models\hf\bge-large-zh-v1.5
  [ 4] qwen2.5-7b-instruct-q4_k_m              gguf         4.7G  D:\models\gguf\qwen2.5-7b-instruct-q4_k_m.gguf  *
  [ 5] deepseek-coder-6.7b-instruct            deepseek    13.4G  C:\Users\...\hub\models--deepseek-ai--DeepSeek-Coder-V2

  [ 0] Enter path manually

  * = last used (press Enter to reuse)

Select model number [4]:
```

- 自动扫描 HF 缓存、常见模型目录（C:/D:/E:/下的 models/LLM/llm/huggingface）
- 支持 HF 模型目录和 GGUF 量化文件
- 选择结果保存到 `.vllm_config.json`，下次启动按 Enter 即可复用

---

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace 缓存目录 |
| `CUDA_VISIBLE_DEVICES` | 全部 GPU | 等价于 `--gpu-id` |
| `VLLM_SAFE_MAX_MODEL_LEN` | `8192` | 安全上下文长度上限（防止显存溢出） |

---

## 注意事项

1. **structured_outputs（JSON mode）在 Windows V1 引擎上不可用**——`response_format` 参数会被忽略，调用方需自行做 JSON 解析/修复
2. **不支持 `uvloop`**——启动时自动 stub 为 `asyncio.run`
3. **工具调用准确度取决于模型**——Qwen2.5、Llama 3.x 原生支持工具模板，效果最好；其他模型使用 system prompt 注入方案，可能不稳定
4. **首次启动需加载模型到 GPU 显存**，14B 模型约需 28GB，7B 量化约需 5-8GB
