#!/bin/bash
# Standalone gateway-overhead paired replay (FCFS direct-vLLM vs VeloxMesh gateway).
# Mirrors run_matrix.sh's overhead block; reuses the completed run dir.
set -uo pipefail
LTR=/hy-tmp/ltr; REPO=$LTR/repo; VENV=$LTR/venv
ART=$LTR/artifacts/current
RUN=$LTR/runs/rental-20260719T231309Z
STOCK=scheduler_benchmark.vllm_scheduler.StockFCFSShim
DIRECT_ENDPOINT=http://127.0.0.1:8000/v1/chat/completions
ENDPOINT=http://127.0.0.1:9100/v1/chat/completions
MODEL=qwen3.5-9b
MIXED_WL=$ART/mixed.v2.jsonl
CAP=$(python3 -c "import json;print(json.load(open('$LTR/runs/calibration/capacity.json'))['capacity_rps'])")
OUT=$RUN/gateway-overhead.json

wait_gpu_free () {
  for i in $(seq 1 40); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d " ")
    [ "${used:-99999}" -lt 5000 ] 2>/dev/null && return 0
    sleep 5
  done
  return 1
}

echo ">> overhead: tearing down OOD vLLM $(date)"
pkill -9 -f "vllm.entrypoint" 2>/dev/null
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
pkill -9 -f "multiprocessing.resource_tracker" 2>/dev/null
sleep 3
wait_gpu_free || echo "WARN: GPU not free before overhead"

tag="overhead-$(date +%s)"
echo ">> overhead: launching StockFCFSShim vLLM"
RUN_TAG="$tag" bash "$REPO/scripts/server/launch_vllm.sh" "$STOCK" "$tag" \
  || { echo "NO-GO: overhead vLLM failed"; exit 1; }

echo ">> overhead: running paired replay (direct vs gateway)"
"$VENV/bin/python" "$REPO/scripts/run_gateway_overhead.py" \
  --direct-endpoint "$DIRECT_ENDPOINT" --gateway-endpoint "$ENDPOINT" \
  --model "$MODEL" --workload "$MIXED_WL" --capacity-rps "$CAP" \
  --scheduler-cls "$STOCK" --output "$OUT" \
  --saturation 0.9 --api-key vx-dev
rc=$?
echo ">> overhead runner rc=$rc"
pkill -9 -f "vllm.entrypoint" 2>/dev/null
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
[ -s "$OUT" ] || { echo "NO-GO: no overhead output"; exit 1; }
echo "OVERHEAD_DONE rc=$rc $(date)"
