#!/usr/bin/env bash
# Cross-session layout probe: fresh server per (layout, run) so head 1 is truly cold.
set -uo pipefail
VENV="$HOME/vllm-schema-exp/.venv092/bin"
MODEL=Qwen/Qwen3-8B-AWQ
TRACE="$HOME/vllm-schema-exp/probes/agent-traces-2026-07-26/agent_trace_vanilla.jsonl.gz"
OUT="$HOME/vllm-schema-exp/results/phase1b_layout"
HERE="$HOME/vllm-schema-exp/probes/schema-gpu-validation-2026-08/phase1b"
mkdir -p "$OUT"
export VLLM_USE_V1=1

for run in 1 2 3; do
  for layout in as-is hoisted; do
    echo "=== layout=$layout run=$run ==="
    "$VENV/vllm" serve "$MODEL" --port 8000 --max-model-len 32768 \
      --gpu-memory-utilization 0.90 --enable-prefix-caching \
      > "$OUT/server_${layout}_$run.log" 2>&1 &
    PID=$!
    ok=0
    for _ in $(seq 1 90); do
      curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
      kill -0 $PID 2>/dev/null || break
      sleep 5
    done
    if [ $ok = 1 ]; then
      "$VENV/python" "$HERE/layout.py" --base http://127.0.0.1:8000 \
        --trace "$TRACE" --model "$MODEL" --layout "$layout" --run-id "$run" \
        --out "$OUT/layout_measurements.csv"
    else
      echo "  server failed to start"
    fi
    kill $PID 2>/dev/null; for _ in $(seq 1 60); do kill -0 $PID 2>/dev/null || break; sleep 1; done
    kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 5
  done
done
echo LAYOUT_SWEEP_DONE
