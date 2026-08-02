#!/usr/bin/env bash
# Phase 2 driver: Original Full vs Frozen Thin, cold start and KV footprint.
#
# Cold start is the whole point, so every (policy, cache-mode) arm gets a fresh
# server -- a reused server would have the session heads already cached and the
# measurement would be meaningless.
#
#   bash run_phase2.sh <trace> <dataset-label> <model> <tp> <max_model_len> <outdir> [restarts]
set -uo pipefail

TRACE=${1:?trace path}
DATASET=${2:?dataset label}
MODEL=${3:?model id}
TP=${4:-1}
MAXLEN=${5:-32768}
OUTDIR=${6:?output dir}
RESTARTS=${7:-2}

VENV=${VENV:-$HOME/.venv126}/bin
PORT=8000
BASE="http://127.0.0.1:$PORT"
HERE=$(cd "$(dirname "$0")" && pwd)
export VLLM_USE_V1=${VLLM_USE_V1:-0}      # Volta: V1 unavailable, recorded not hidden

mkdir -p "$OUTDIR"

start_server() {           # $1 = on|off  $2 = tag
    local flag="--enable-prefix-caching"
    [ "$1" = "off" ] && flag="--no-enable-prefix-caching"
    "$VENV/vllm" serve "$MODEL" \
        --port "$PORT" --dtype float16 --tensor-parallel-size "$TP" \
        --max-model-len "$MAXLEN" --gpu-memory-utilization 0.90 \
        --enable-auto-tool-choice --tool-call-parser hermes \
        $flag > "$OUTDIR/server_${DATASET}_$1_$2.log" 2>&1 &
    SERVER_PID=$!
}

wait_ready() {
    for _ in $(seq 1 240); do
        curl -sf "$BASE/health" >/dev/null 2>&1 && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "  !! server died"; return 1; }
        sleep 5
    done
    echo "  !! server not ready after 20 min"; return 1
}

stop_server() {
    kill "$SERVER_PID" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; sleep 5
}

for cache in on off; do
  for run in $(seq 1 "$RESTARTS"); do
    for pol in "Original" "Frozen Thin"; do
      tag="${run}_$(echo "$pol" | tr ' ' '-')"
      echo "=== $DATASET cache=$cache run=$run policy=$pol ==="
      start_server "$cache" "$tag"
      if wait_ready; then
          # capture what the server decided it could hold, once per arm
          grep -iE "GPU KV cache size|Maximum concurrency|# GPU blocks" \
               "$OUTDIR/server_${DATASET}_${cache}_${tag}.log" | tail -3 \
               >> "$OUTDIR/kv_capacity_${DATASET}.txt"
          "$VENV/python" "$HERE/coldstart.py" \
              --base "$BASE" --trace "$TRACE" --model "$MODEL" \
              --dataset "$DATASET" --policy "$pol" --cache-mode "$cache" \
              --run-id "$run" --out "$OUTDIR/phase2_coldstart.csv"
      else
          echo "  SKIPPED"
      fi
      stop_server
    done
  done
done

echo "done -> $OUTDIR/phase2_coldstart.csv"
