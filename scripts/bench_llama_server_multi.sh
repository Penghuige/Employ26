#!/usr/bin/env bash
set -euo pipefail

MODEL="/home/lin/models/Qwen3.6-27B-Uncensored-Balanced/Qwen3.6-27B-Uncensored-HauhauCS-Balanced-Q4_K_P.gguf"
SERVER="/home/lin/llama.cpp/build/bin/llama-server"
PORT="$1"
LOG="$2"
REQUESTS="$3"
MAX_TOKENS="$4"
shift 4

rm -f "$LOG"
"$SERVER" \
  -m "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  -c 4096 \
  -ngl 999 \
  -fa on \
  -b 2048 \
  -ub 512 \
  --alias bench \
  "$@" >"$LOG" 2>&1 &

PID=$!

cleanup() {
  kill "$PID" 2>/dev/null || true
  wait "$PID" 2>/dev/null || true
}
trap cleanup EXIT

for i in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    break
  fi
  sleep 1
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "server died" >&2
    tail -80 "$LOG" >&2 || true
    exit 1
  fi
done

python3 - <<PY
import json
import time
import urllib.request

port = $PORT
requests_count = $REQUESTS
max_tokens = $MAX_TOKENS
results = []
for idx in range(requests_count):
    payload = {
        "model": "bench",
        "messages": [
            {
                "role": "system",
                "content": "/no_think 你只输出中文词语列表，不解释。",
            },
            {
                "role": "user",
                "content": f"第{idx + 1}轮：请输出80个不同的中文双字词，用逗号分隔。",
            },
        ],
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        data = json.loads(response.read().decode("utf-8"))
    elapsed = time.perf_counter() - started
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens") or 0
    results.append(
        {
            "round": idx + 1,
            "elapsed_seconds": round(elapsed, 3),
            "completion_tokens": completion_tokens,
            "completion_tps": round(completion_tokens / elapsed, 2) if elapsed else None,
            "usage": usage,
            "finish_reason": data["choices"][0].get("finish_reason"),
        }
    )
print(json.dumps(results, ensure_ascii=False, indent=2))
PY

echo "---- prompt cache update lines ----"
grep -E "prompt cache update took|eval time|prompt eval time|done request" "$LOG" | tail -80 || true
