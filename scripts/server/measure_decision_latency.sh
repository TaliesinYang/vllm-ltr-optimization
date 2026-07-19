#!/usr/bin/env bash
set -euo pipefail

LTR_ROOT="${LTR_ROOT:-/hy-tmp/ltr}"
VENV="${VENV:-$LTR_ROOT/venv}"
DECISION_ENDPOINT="${DECISION_ENDPOINT:-http://127.0.0.1:9200}"
OUTPUT="${OUTPUT:-$LTR_ROOT/manifest.decision-latency.json}"

curl -fsS "$DECISION_ENDPOINT/healthz" >/dev/null
"$VENV/bin/python" - "$DECISION_ENDPOINT" "$OUTPUT" <<'PY'
import concurrent.futures, json, math, statistics, sys, time, urllib.request, uuid
from pathlib import Path

endpoint, output = sys.argv[1], Path(sys.argv[2])
concurrency, warmups, measured = 8, 20, 200

def one():
    request_id = f"latency-{uuid.uuid4().hex}"
    payload = {
        "schema_version": "1.0", "request_id": request_id,
        "decision_id": f"dec-{request_id}", "model_id": "qwen3.5-9b",
        "request_age_ms": 0,
        "messages": [{"role": "user", "content": "Measure warm decision latency."}],
        "tool_schema_text": "[]",
        "generation_controls": {"temperature": 0.0, "top_p": 1.0, "seed": 42, "max_tokens": 4096},
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(endpoint + "/v1/decision", data=body, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"decision returned {response.status}")
        json.load(response)
    return (time.perf_counter() - started) * 1000.0

with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
    list(pool.map(lambda _: one(), range(warmups)))
    samples = list(pool.map(lambda _: one(), range(measured)))
ordered = sorted(samples)
p99 = ordered[math.ceil(0.99 * len(ordered)) - 1]
timeout_ms = max(2000, math.ceil(1.25 * p99))
payload = {
    "schema_version": "decision-latency-v1", "endpoint": endpoint,
    "concurrency": concurrency, "warmup_count": warmups,
    "sample_count": measured, "samples_ms": samples,
    "concurrent_p99_ms": p99, "mean_ms": statistics.fmean(samples),
    "timeout_formula": "max(2000, ceil(1.25 * concurrent_p99_ms))",
    "timeout_ms": timeout_ms,
}
output.parent.mkdir(parents=True, exist_ok=True)
temporary = output.with_suffix(output.suffix + ".partial")
temporary.write_text(json.dumps(payload, indent=2) + "\n")
temporary.replace(output)
print(json.dumps({"p99_ms": p99, "timeout_ms": timeout_ms}))
PY
