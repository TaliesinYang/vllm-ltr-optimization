#!/bin/bash
# Standalone OOD matrix + gateway-overhead runner. Reuses the completed-mixed
# run dir; bypasses the strict resume-identity gate (mixed already done + saved).
set -uo pipefail
LTR=/hy-tmp/ltr; REPO=$LTR/repo; VENV=$LTR/venv
RUN=$LTR/runs/rental-20260719T231309Z
ART=$LTR/artifacts/current
ENDPOINT=http://127.0.0.1:9100/v1/chat/completions
MODEL=qwen3.5-9b
OOD_WL=$ART/ood.v2.jsonl
CAP=$(python3 -c "import json;print(json.load(open('$LTR/runs/calibration/capacity.json'))['capacity_rps'])")
mkdir -p "$RUN/matrix-ood"
export PATH=$LTR/go/bin:$PATH PYTHONPATH=$REPO
export HF_HUB_DISABLE_XET=1

OOD=(StockFCFSShim PureLTRScheduler TailSafeScheduler GatedHybridScheduler)
echo "OOD run start $(date) capacity=$CAP" > $LTR/ood.log

wait_gpu_free () {
  # poll until GPU memory drops below 5 GiB (previous vLLM fully released)
  for i in $(seq 1 40); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d " ")
    [ "${used:-99999}" -lt 5000 ] 2>/dev/null && return 0
    sleep 5
  done
  return 1
}
launch_vllm () {
  local cls=$1 tag=$2
  # vLLM spawns a VLLM::EngineCore subprocess whose cmdline does NOT contain
  # "vllm.entrypoint" — it is the real GPU-memory holder. Must kill it too, or
  # the next policy hits "Free memory ... less than desired" at startup.
  pkill -9 -f "vllm.entrypoint" 2>/dev/null
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null
  pkill -9 -f "multiprocessing.resource_tracker" 2>/dev/null
  sleep 3
  wait_gpu_free || echo "WARN: GPU still not free before $cls" >> /hy-tmp/ltr/ood.log
  RUN_TAG="$tag" bash "$REPO/scripts/server/launch_vllm.sh" "scheduler_benchmark.vllm_scheduler.$cls" "$tag"
}
ood_complete () {
  # returns 0 if $1 json already has all repeats' runs recorded (skip re-run)
  local jf=$1
  [ -s "$jf" ] || return 1
  python3 - "$jf" << 'PYEOF'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    reps=d.get("repeats")
    scs=d.get("scenarios") or []
    ok = bool(scs) and all(len(s.get("runs",[]))>=reps for s in scs)
    # every run must have terminal status
    ok = ok and all(r.get("status") for s in scs for r in s.get("runs",[]))
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
PYEOF
}
for cls in "${OOD[@]}"; do
  if ood_complete "$RUN/matrix-ood/$cls.json"; then
    echo ">> $cls already complete — skip (no vLLM restart)" >> $LTR/ood.log
    continue
  fi
  tag="ood-$cls-$(date +%s)"
  echo ">> $cls launching vLLM" >> $LTR/ood.log
  launch_vllm "$cls" "$tag" || { echo "NO-GO: vLLM failed for $cls" >> $LTR/ood.log; exit 1; }
  echo ">> $cls running OOD 120x3" >> $LTR/ood.log
  $VENV/bin/python -m scheduler_benchmark.runner \
    --endpoint $ENDPOINT --model $MODEL --workload $OOD_WL \
    --capacity-rps $CAP --scheduler-cls scheduler_benchmark.vllm_scheduler.$cls \
    --output $RUN/matrix-ood/$cls.json --api-key vx-dev --scenario steady --load 90 \
    --profile ood --repeats 3 --resume >> $LTR/ood.log 2>&1
  rc=$?
  echo ">> $cls runner rc=$rc" >> $LTR/ood.log
  [ -s "$RUN/matrix-ood/$cls.json" ] || { echo "NO-GO: no output for $cls" >> $LTR/ood.log; exit 1; }
done
pkill -9 -f "vllm.entrypoint" 2>/dev/null
echo "OOD_ALL_DONE $(date)" >> $LTR/ood.log
