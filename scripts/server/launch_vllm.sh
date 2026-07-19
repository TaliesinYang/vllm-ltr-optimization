#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 2 ]] || { echo "usage: $0 <scheduler-FQCN> <run-tag>" >&2; exit 2; }
SCHEDULER_CLS="$1"
RUN_TAG="$2"
[[ "$RUN_TAG" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid run tag" >&2; exit 2; }

LTR_ROOT="${LTR_ROOT:-/hy-tmp/ltr}"
REPO_ROOT="${REPO_ROOT:-$LTR_ROOT/repo}"
VENV="${VENV:-$LTR_ROOT/venv}"
MODEL_DIR="${MODEL_DIR:-/hy-tmp/models/qwen3.5-9b}"
PORT="${VLLM_PORT:-8000}"
RUN_DIR="$LTR_ROOT/runs/$RUN_TAG"
PIDFILE="$RUN_DIR/vllm.pid"
STARTFILE="$PIDFILE.starttime"
LOGFILE="$RUN_DIR/vllm.log"
mkdir -p "$RUN_DIR"

if [[ -f "$PIDFILE" ]] && kill -0 "$(<"$PIDFILE")" 2>/dev/null; then
  echo "vLLM is already running for $RUN_TAG" >&2
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

PYTHONPATH="$REPO_ROOT" LTR_PREDICTOR=gateway LTR_ORDER_LOG="$RUN_DIR/order.jsonl" \
  nohup "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" --served-model-name qwen3.5-9b --dtype bfloat16 \
  --port "$PORT" --scheduler-cls "$SCHEDULER_CLS" \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --max-model-len 8192 >"$LOGFILE" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$PIDFILE"
[[ -r "/proc/$pid/stat" ]] || { rm -f "$PIDFILE" "$STARTFILE"; tail -100 "$LOGFILE" >&2; exit 1; }
if ! awk '{print $22}' "/proc/$pid/stat" >"$STARTFILE"; then
  rm -f "$PIDFILE" "$STARTFILE"
  tail -100 "$LOGFILE" >&2
  exit 1
fi

for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
    trap - INT TERM
    echo "vLLM ready pid=$pid scheduler=$SCHEDULER_CLS"
    exit 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    cleanup
    tail -100 "$LOGFILE" >&2
    exit 1
  fi
  sleep 2
done
echo "vLLM health check timed out" >&2
cleanup
exit 1
