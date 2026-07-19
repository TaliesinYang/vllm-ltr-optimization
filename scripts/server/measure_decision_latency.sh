#!/usr/bin/env bash
set -euo pipefail

LTR_ROOT="${LTR_ROOT:-/hy-tmp/ltr}"
VENV="${VENV:-$LTR_ROOT/venv}"
REPO_ROOT="${REPO_ROOT:-$LTR_ROOT/repo}"
ARTIFACTS="${ARTIFACTS:-$LTR_ROOT/artifacts/current}"
DECISION_ENDPOINT="${DECISION_ENDPOINT:-http://127.0.0.1:9200}"
OUTPUT="${OUTPUT:-$LTR_ROOT/manifest.decision-latency.json}"
MIXED_WORKLOAD="${MIXED_WORKLOAD:-$ARTIFACTS/mixed.v2.jsonl}"

curl -fsS "$DECISION_ENDPOINT/healthz" >/dev/null
PYTHONPATH="$REPO_ROOT" "$VENV/bin/python" - "$DECISION_ENDPOINT" "$OUTPUT" "$MIXED_WORKLOAD" <<'PY'
import concurrent.futures, json, math, statistics, sys, time, urllib.request, uuid
from pathlib import Path
from ltr_training.tier2 import build_request

endpoint, output, workload = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
# Representative REAL tool request (long schema) — the greeting stub under-
# measured latency ~2x and set the gateway timeout too low.
_row = next(json.loads(l) for l in open(workload) if l.strip())
_req = build_request({"prompt": _row["prompt"], "tool_schema": _row["tool_schema"], "history": _row.get("history", [])}, model="qwen3.5-9b")
_msgs = [{"role": m["role"], "content": m.get("content", "")} for m in _req["messages"]]
_tools = _req.get("tools")
_schema_text = _row["tool_schema"]
concurrency, warmups, measured = 8, 20, 200

def one():
    request_id = f"latency-{uuid.uuid4().hex}"
    payload = {
        "schema_version": "1.0", "request_id": request_id,
        "decision_id": f"dec-{request_id}", "model_id": "qwen3.5-9b",
        "request_age_ms": 0,
        "messages": _msgs,
        "tool_schema_text": _schema_text,
        "workflow_id": request_id, "step_id": "0", "conversation_id": request_id,
        "previous_tool_gap_ms": 0,
        "generation_controls": {"temperature": 0.0, "top_p": 1.0, "seed": 42, "max_tokens": 4096},
    }
    if _tools:
        payload["tools"] = _tools
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
timeout_ms = max(6000, math.ceil(2.0 * p99))
payload = {
    "schema_version": "decision-latency-v1", "endpoint": endpoint,
    "concurrency": concurrency, "warmup_count": warmups,
    "sample_count": measured, "samples_ms": samples,
    "concurrent_p99_ms": p99, "mean_ms": statistics.fmean(samples),
    "timeout_formula": "max(6000, ceil(2.0 * concurrent_p99_ms))",
    "timeout_ms": timeout_ms,
}
output.parent.mkdir(parents=True, exist_ok=True)
temporary = output.with_suffix(output.suffix + ".partial")
temporary.write_text(json.dumps(payload, indent=2) + "\n")
temporary.replace(output)
print(json.dumps({"p99_ms": p99, "timeout_ms": timeout_ms}))
PY
