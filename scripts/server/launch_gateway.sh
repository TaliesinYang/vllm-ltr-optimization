#!/usr/bin/env bash
set -euo pipefail

LTR_ROOT="${LTR_ROOT:-/hy-tmp/ltr}"
GATEWAY_BIN="${GATEWAY_BIN:-$LTR_ROOT/bin/gateway}"
LATENCY_MANIFEST="${LATENCY_MANIFEST:-$LTR_ROOT/manifest.decision-latency.json}"
GATEWAY_PORT="${GATEWAY_PORT:-9100}"
VLLM_PORT="${VLLM_PORT:-8000}"
DECISION_PORT="${DECISION_PORT:-9200}"
RUN_DIR="$LTR_ROOT/runs/services"
PIDFILE="$RUN_DIR/gateway.pid"
STARTFILE="$PIDFILE.starttime"
LOGFILE="$RUN_DIR/gateway.log"
mkdir -p "$RUN_DIR"

[[ -x "$GATEWAY_BIN" ]] || { echo "gateway binary missing: $GATEWAY_BIN" >&2; exit 1; }
timeout_ms="$(python3 - "$LATENCY_MANIFEST" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))["timeout_ms"]
if not isinstance(value, int) or value < 2000:
    raise SystemExit("decision timeout manifest must contain timeout_ms >= 2000")
print(value)
PY
)"
if [[ -f "$PIDFILE" ]] && kill -0 "$(<"$PIDFILE")" 2>/dev/null; then
  echo "gateway is already running" >&2
  exit 1
fi
# shellcheck disable=SC2329 # invoked by trap while this launcher waits for health
cleanup() {
  if [[ -f "$PIDFILE" && -f "$STARTFILE" ]]; then
    saved_pid="$(<"$PIDFILE")"
    saved_start="$(<"$STARTFILE")"
    if [[ "$saved_pid" =~ ^[0-9]+$ && "$saved_start" =~ ^[0-9]+$ && -r "/proc/$saved_pid/stat" ]]; then
      if current_start="$(awk '{print $22}' "/proc/$saved_pid/stat" 2>/dev/null)"; then
        [[ "$current_start" == "$saved_start" ]] && kill "$saved_pid" 2>/dev/null || true
      fi
    fi
  fi
  rm -f "$PIDFILE" "$STARTFILE"
}
trap cleanup INT TERM

GATEWAY_DATA_ADDR="127.0.0.1:$GATEWAY_PORT" \
DEV_API_KEY="${DEV_API_KEY:-vx-dev}" \
DEFAULT_PROVIDER="openai-primary" \
OPENAI_PRIMARY_BASE_URL="http://127.0.0.1:$VLLM_PORT/v1" \
OPENAI_PRIMARY_API_KEY="${OPENAI_PRIMARY_API_KEY:-unused}" \
OPENAI_PRIMARY_MODELS="qwen3.5-9b" \
OPENAI_PRIMARY_DEFAULT_MODEL="qwen3.5-9b" \
LTR_DECISION_ENDPOINT="${LTR_DECISION_ENDPOINT-http://127.0.0.1:$DECISION_PORT}" \
LTR_DECISION_TIMEOUT_MS="$timeout_ms" \
  nohup "$GATEWAY_BIN" >"$LOGFILE" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$PIDFILE"
[[ -r "/proc/$pid/stat" ]] || { rm -f "$PIDFILE" "$STARTFILE"; tail -100 "$LOGFILE" >&2; exit 1; }
if ! awk '{print $22}' "/proc/$pid/stat" >"$STARTFILE"; then
  rm -f "$PIDFILE" "$STARTFILE"
  tail -100 "$LOGFILE" >&2
  exit 1
fi

for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:$GATEWAY_PORT/healthz" >/dev/null; then
    trap - INT TERM
    echo "gateway ready pid=$pid timeout_ms=$timeout_ms"
    exit 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    cleanup
    tail -100 "$LOGFILE" >&2
    exit 1
  fi
  sleep 2
done
echo "gateway health check timed out" >&2
cleanup
exit 1
