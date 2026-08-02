#!/usr/bin/env bash
# Phase 1 driver: for each (cache-mode, policy, restart) arm, start a fresh vLLM
# server, replay the pairs, tear it down. A fresh server per arm is the only way
# to guarantee the cache state we claim -- never reuse one across arms.
#
#   bash run_phase1.sh <trace> <model> <tp> <max_model_len> <outdir> [restarts]
set -uo pipefail

TRACE=${1:?trace path}
MODEL=${2:?model id}
TP=${3:-1}
MAXLEN=${4:-32768}
OUTDIR=${5:?output dir}
RESTARTS=${6:-3}

VENV=${VENV:-$HOME/.venv126}/bin
PORT=8000

# vLLM refuses V1 below compute capability 8.0. The first rig attempted was
# Volta (sm_70) and could only have run V0, hence the default below; the
# committed results were NOT produced that way. They come from an RTX 4090
# Laptop (sm_89) with VLLM_USE_V1=1 exported by the caller. Always pass
# VLLM_USE_V1 explicitly and record it — do not rely on this default.
export VLLM_USE_V1=${VLLM_USE_V1:-0}
BASE="http://127.0.0.1:$PORT"
POLICIES=("Original" "Stable Full" "Shuffled Full")
HERE=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$OUTDIR"

start_server() {          # $1 = on|off
    local cache_flag="--enable-prefix-caching"
    [ "$1" = "off" ] && cache_flag="--no-enable-prefix-caching"
    echo "  starting server (prefix-cache=$1, tp=$TP, len=$MAXLEN)"
    "$VENV/vllm" serve "$MODEL" \
        --port "$PORT" \
        --dtype float16 \
        --tensor-parallel-size "$TP" \
        --max-model-len "$MAXLEN" \
        --gpu-memory-utilization 0.90 \
        --enable-auto-tool-choice --tool-call-parser hermes \
        $cache_flag \
        > "$OUTDIR/server_$1_$2.log" 2>&1 &
    SERVER_PID=$!
}

wait_ready() {
    for _ in $(seq 1 180); do
        if curl -sf "$BASE/health" >/dev/null 2>&1; then return 0; fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "  !! server died during startup"; return 1
        fi
        sleep 5
    done
    echo "  !! server not ready after 15 min"; return 1
}

stop_server() {
    kill "$SERVER_PID" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
    sleep 5
}

for cache in on off; do
  for run in $(seq 1 "$RESTARTS"); do
    for pol in "${POLICIES[@]}"; do
      echo "=== cache=$cache run=$run policy=$pol ==="
      start_server "$cache" "${run}_$(echo "$pol" | tr ' ' '-')"
      if wait_ready; then
          "$VENV/python" "$HERE/replay.py" \
              --base "$BASE" --trace "$TRACE" --model "$MODEL" \
              --policy "$pol" --cache-mode "$cache" --run-id "$run" \
              --out "$OUTDIR/phase1_measurements.csv"
      else
          echo "  SKIPPED (server failed)"
      fi
      stop_server
    done
  done
done

echo "done -> $OUTDIR/phase1_measurements.csv"
