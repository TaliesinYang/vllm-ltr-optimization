#!/usr/bin/env bash
set -euo pipefail

LTR_ROOT="${LTR_ROOT:-/hy-tmp/ltr}"
REPO_ROOT="${REPO_ROOT:-$LTR_ROOT/repo}"
VENV="${VENV:-$LTR_ROOT/venv}"
ARTIFACTS="${ARTIFACTS:-$LTR_ROOT/artifacts/current}"
PORT="${DECISION_PORT:-9200}"
RUN_DIR="$LTR_ROOT/runs/services"
PIDFILE="$RUN_DIR/decision.pid"
STARTFILE="$PIDFILE.starttime"
LOGFILE="$RUN_DIR/decision.log"
mkdir -p "$RUN_DIR"

if [[ -f "$PIDFILE" ]] && kill -0 "$(<"$PIDFILE")" 2>/dev/null; then
  echo "decision service is already running" >&2
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

PYTHONPATH="$REPO_ROOT" nohup "$VENV/bin/python" "$REPO_ROOT/scripts/run_decision_service.py" \
  --host 127.0.0.1 --port "$PORT" --predictor bert \
  --checkpoint "$ARTIFACTS/checkpoints_best_predictor" \
  --quantile-manifest "$ARTIFACTS/rank_quantiles.json" \
  --max-concurrency 8 >"$LOGFILE" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$PIDFILE"
[[ -r "/proc/$pid/stat" ]] || { rm -f "$PIDFILE" "$STARTFILE"; tail -100 "$LOGFILE" >&2; exit 1; }
if ! awk '{print $22}' "/proc/$pid/stat" >"$STARTFILE"; then
  rm -f "$PIDFILE" "$STARTFILE"
  tail -100 "$LOGFILE" >&2
  exit 1
fi

for _ in $(seq 1 90); do
  if curl -fsS "http://127.0.0.1:$PORT/healthz" >/dev/null; then
    trap - INT TERM
    echo "decision service ready pid=$pid"
    exit 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    cleanup
    tail -100 "$LOGFILE" >&2
    exit 1
  fi
  sleep 2
done
echo "decision service health check timed out" >&2
cleanup
exit 1
