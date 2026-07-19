#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STATE_DIR="${LTR_SMOKE_STATE_DIR:-/tmp/ltr-smoke}"
EVIDENCE_DIR="${LTR_SMOKE_EVIDENCE_DIR:-$REPO_ROOT/runs/smoke-local-chain}"
PYTHON="${LTR_SMOKE_PYTHON:-$REPO_ROOT/.worktrees/final-training-artifacts/.venv/bin/python}"
GATEWAY_BIN="${LTR_SMOKE_GATEWAY_BIN:-/Users/alex/develop/VeloxMesh/bin/gateway}"
CHECKPOINT="${LTR_SMOKE_CHECKPOINT:-$REPO_ROOT/checkpoints_best_predictor}"
WORKLOAD="${LTR_SMOKE_WORKLOAD:-$REPO_ROOT/runs/workloads-v2/workload-id.v2.jsonl}"

FAKE_PORT="${LTR_SMOKE_FAKE_PORT:-8000}"
DECISION_PORT="${LTR_SMOKE_DECISION_PORT:-9200}"
GATEWAY_PORT="${LTR_SMOKE_GATEWAY_PORT:-9100}"
API_KEY="${LTR_SMOKE_API_KEY:-vx-dev}"
MODEL="${LTR_SMOKE_MODEL:-qwen3.5-9b}"

FAKE_PIDFILE="$STATE_DIR/fake-vllm.pid"
DECISION_PIDFILE="$STATE_DIR/decision.pid"
GATEWAY_PIDFILE="$STATE_DIR/gateway.pid"
PIDFILES=("$GATEWAY_PIDFILE" "$DECISION_PIDFILE" "$FAKE_PIDFILE")

CAPTURE="$STATE_DIR/capture.jsonl"
SUMMARY="$STATE_DIR/summary.json"
QUANTILE_MANIFEST="$STATE_DIR/rank_quantiles.smoke-fixture.json"
FAKE_LOG="$STATE_DIR/fake-vllm.log"
DECISION_LOG="$STATE_DIR/decision.log"
GATEWAY_LOG="$STATE_DIR/gateway.log"

cleanup() {
  set +e
  for pidfile in "${PIDFILES[@]}"; do
    if [[ ! -f "$pidfile" ]]; then
      continue
    fi
    pid="$(<"$pidfile")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
  done
}
trap cleanup EXIT INT TERM

require_file() {
  local path="$1"
  [[ -e "$path" ]] || { echo "required path missing: $path" >&2; exit 1; }
}

ensure_no_live_pid() {
  local pidfile="$1"
  if [[ -f "$pidfile" ]]; then
    local pid
    pid="$(<"$pidfile")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      echo "refusing to overwrite live smoke pid $pid from $pidfile" >&2
      exit 1
    fi
    rm -f "$pidfile"
  fi
}

wait_for_http() {
  local name="$1"
  local url="$2"
  local pidfile="$3"
  local logfile="$4"
  local attempts="$5"
  local pid
  pid="$(<"$pidfile")"
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    if curl -fsS "$url" >/dev/null; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$name exited before becoming ready" >&2
      tail -100 "$logfile" >&2 || true
      return 1
    fi
    sleep 1
  done
  echo "$name health check timed out: $url" >&2
  tail -100 "$logfile" >&2 || true
  return 1
}

require_file "$PYTHON"
require_file "$GATEWAY_BIN"
require_file "$CHECKPOINT"
require_file "$WORKLOAD"
mkdir -p "$STATE_DIR"
for pidfile in "${PIDFILES[@]}"; do
  ensure_no_live_pid "$pidfile"
done
rm -f \
  "$CAPTURE" "$SUMMARY" "$QUANTILE_MANIFEST" \
  "$FAKE_LOG" "$DECISION_LOG" "$GATEWAY_LOG"

nohup "$PYTHON" "$REPO_ROOT/scripts/fake_vllm_server.py" \
  --host 127.0.0.1 --port "$FAKE_PORT" --capture "$CAPTURE" \
  >"$FAKE_LOG" 2>&1 &
printf '%s\n' "$!" >"$FAKE_PIDFILE"
wait_for_http \
  "fake vLLM" "http://127.0.0.1:$FAKE_PORT/healthz" \
  "$FAKE_PIDFILE" "$FAKE_LOG" 30

PYTHONPATH="$REPO_ROOT" "$PYTHON" - "$QUANTILE_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

from scheduler_benchmark.rank_quantiles import (
    APPROXIMATION_NOTICE,
    MAPPING_VERSION,
)

path = Path(sys.argv[1])
manifest = {
    "mapping_version": MAPPING_VERSION,
    "model_version": "smoke-fixture",
    "approximation_notice": APPROXIMATION_NOTICE,
    "sample_count": 6000,
    "percentiles": {str(p): float(10 + 5 * p) for p in range(10, 100)},
    "global_quantiles": {"50": 260.0, "70": 360.0, "90": 460.0},
    "fixture_only": True,
    "fixture_semantics": (
        "Synthetic p10..p99 values and contractual sample_count=6000; "
        "valid only for local chain integration smoke."
    ),
}
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

PYTHONPATH="$REPO_ROOT" nohup "$PYTHON" \
  "$REPO_ROOT/scripts/run_decision_service.py" \
  --host 127.0.0.1 \
  --port "$DECISION_PORT" \
  --predictor bert \
  --checkpoint "$CHECKPOINT" \
  --quantile-manifest "$QUANTILE_MANIFEST" \
  --max-concurrency 8 \
  >"$DECISION_LOG" 2>&1 &
printf '%s\n' "$!" >"$DECISION_PIDFILE"
wait_for_http \
  "decision service" "http://127.0.0.1:$DECISION_PORT/healthz" \
  "$DECISION_PIDFILE" "$DECISION_LOG" 180

GATEWAY_DATA_ADDR="127.0.0.1:$GATEWAY_PORT" \
DEV_API_KEY="$API_KEY" \
DEFAULT_PROVIDER="openai-primary" \
OPENAI_PRIMARY_BASE_URL="http://127.0.0.1:$FAKE_PORT/v1" \
OPENAI_PRIMARY_API_KEY="unused" \
OPENAI_PRIMARY_MODELS="$MODEL" \
OPENAI_PRIMARY_DEFAULT_MODEL="$MODEL" \
LTR_DECISION_ENDPOINT="http://127.0.0.1:$DECISION_PORT" \
LTR_DECISION_TIMEOUT_MS="2000" \
  nohup "$GATEWAY_BIN" >"$GATEWAY_LOG" 2>&1 &
printf '%s\n' "$!" >"$GATEWAY_PIDFILE"
wait_for_http \
  "gateway" "http://127.0.0.1:$GATEWAY_PORT/healthz" \
  "$GATEWAY_PIDFILE" "$GATEWAY_LOG" 60

PYTHONPATH="$REPO_ROOT" "$PYTHON" "$REPO_ROOT/scripts/smoke_chain_driver.py" \
  --endpoint "http://127.0.0.1:$GATEWAY_PORT/v1/chat/completions" \
  --api-key "$API_KEY" \
  --workload "$WORKLOAD" \
  --model "$MODEL" \
  --count 20 \
  >"$SUMMARY"

"$PYTHON" - "$SUMMARY" "$CAPTURE" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
capture_path = Path(sys.argv[2])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
assert summary["requested"] == 20, summary
assert summary["completed"] == 20, summary
assert summary["error_count"] == 0, summary
assert summary["errors"] == [], summary
assert summary["usage_tokens_nonzero_count"] == 20, summary

records = [
    json.loads(line)
    for line in capture_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
assert len(records) >= 20, f"expected at least 20 captures, got {len(records)}"
reliable_with_estimate = 0
for index, record in enumerate(records, start=1):
    validation = record.get("validation")
    assert isinstance(validation, dict) and validation.get("ok") is True, (
        index,
        validation,
    )
    body = record.get("body")
    assert isinstance(body, dict), (index, body)
    chat_kwargs = body.get("chat_template_kwargs")
    assert isinstance(chat_kwargs, dict), (index, chat_kwargs)
    assert chat_kwargs.get("enable_thinking") is False, (index, chat_kwargs)
    xargs = body.get("vllm_xargs")
    assert isinstance(xargs, dict), (index, xargs)
    assert "ltr_tool_schema" not in xargs, (index, xargs)
    flag = xargs.get("prediction_reliable")
    assert type(flag) is int and flag in (0, 1), (index, flag, type(flag))
    if flag == 1:
        estimate = xargs.get("workflow_estimated_tokens")
        assert type(estimate) is int and estimate >= 1, (index, estimate)
        reliable_with_estimate += 1

assert reliable_with_estimate >= 1, "no reliable capture carried a token estimate"
print(
    json.dumps(
        {
            "captures": len(records),
            "reliable_with_estimate": reliable_with_estimate,
            "summary": summary,
        },
        sort_keys=True,
    )
)
PY

mkdir -p "$EVIDENCE_DIR"
cp \
  "$CAPTURE" "$SUMMARY" "$QUANTILE_MANIFEST" \
  "$FAKE_LOG" "$DECISION_LOG" "$GATEWAY_LOG" \
  "$EVIDENCE_DIR/"
echo "SMOKE_OK"
