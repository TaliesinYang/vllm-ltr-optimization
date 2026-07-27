#!/usr/bin/env bash
# Block 2: six-arm decomposition of the gateway + decision-service overhead.
#
# Derived from run_overhead_only.sh, which measured only two arms (direct vs
# gateway) and so could only report a single lumped number. Six arms attribute
# that number to its parts:
#
#   D0  direct to vLLM, no gateway            transport floor
#   G0  gateway, NO decision endpoint         + gateway proxying
#   G1  gateway + stub decision service       + decision round-trip
#   G2  gateway + CPU BERT, gate DISABLED     + model cost on every request
#   G3  gateway + CPU BERT, gate-first        + gating (this is what saves)
#   G4  gateway + GPU batched, gate-first     + GPU batching
#
# Each successive arm adds exactly one component, so consecutive differences
# are the per-component costs.
#
# Ordering is ABBA: D0 G0 G1 G2 G3 G4 | G4 G3 G2 G1 G0 D0. A monotone drift over
# the session cancels in the mean of each arm's two halves instead of being
# charged to whichever arm ran late.
set -euo pipefail

LTR_ROOT="${LTR_ROOT:-/hy-tmp/ltr}"
REPO_ROOT="${REPO_ROOT:-$LTR_ROOT/repo}"
GATEWAY_REPO="${GATEWAY_REPO:-$LTR_ROOT/VeloxMesh}"
VENV="${VENV:-$LTR_ROOT/venv}"
ARTIFACTS="${ARTIFACTS:-$LTR_ROOT/artifacts/current}"
RUN_TAG="${RUN_TAG:-block2-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="$LTR_ROOT/runs/$RUN_TAG"
MIXED_WORKLOAD="${MIXED_WORKLOAD:-$ARTIFACTS/block1.jsonl}"
CAPACITY_MANIFEST="${CAPACITY_MANIFEST:-$LTR_ROOT/runs/calibration/capacity.json}"
ENDPOINT="${ENDPOINT:-http://127.0.0.1:9100/v1/chat/completions}"
DIRECT_ENDPOINT="${DIRECT_ENDPOINT:-http://127.0.0.1:8000/v1/chat/completions}"
DECISION_HEALTH="${DECISION_HEALTH:-http://127.0.0.1:9200/healthz}"
GATEWAY_HEALTH="${GATEWAY_HEALTH:-http://127.0.0.1:9100/healthz}"
VLLM_METRICS="${VLLM_METRICS:-http://127.0.0.1:8000/metrics}"
MODEL="${MODEL:-qwen3.5-9b}"
STOCK="scheduler_benchmark.vllm_scheduler.StockFCFSShim"
CHECKPOINT="${CHECKPOINT:-$LTR_ROOT/checkpoints_best_predictor}"
QUANTILE_MANIFEST="${QUANTILE_MANIFEST:-$ARTIFACTS/rank_quantiles.json}"
# Rule C tops out at 0.6233 and the comparison is `confidence < threshold`, so
# 0.5 is the only value that keeps S3 (0.5787) AND S4 (0.6233) trusted.
RELIABILITY_THRESHOLD="${RELIABILITY_THRESHOLD:-0.5}"
DRAIN_TIMEOUT_S="${DRAIN_TIMEOUT_S:-120}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-20}"

DECISION_PID=""
GATEWAY_PID=""

mkdir -p "$RUN_ROOT/arms" "$RUN_ROOT/logs" "$RUN_ROOT/manifests"

# shellcheck source=scripts/server/_matrix_common.sh
source "$REPO_ROOT/scripts/server/_matrix_common.sh"

for path in "$MIXED_WORKLOAD" "$QUANTILE_MANIFEST" "$CAPACITY_MANIFEST"; do
  [[ -f "$path" ]] || { echo "required Block-2 input missing: $path" >&2; exit 1; }
done

stop_decision() {
  [[ -n "$DECISION_PID" ]] || return 0
  kill "$DECISION_PID" 2>/dev/null || true
  wait "$DECISION_PID" 2>/dev/null || true
  DECISION_PID=""
}

stop_gateway() {
  [[ -n "$GATEWAY_PID" ]] || return 0
  kill "$GATEWAY_PID" 2>/dev/null || true
  wait "$GATEWAY_PID" 2>/dev/null || true
  GATEWAY_PID=""
}

block2_cleanup() {
  stop_gateway
  stop_decision
}
trap block2_cleanup EXIT

wait_healthy() {
  local url="$1" label="$2" deadline=$((SECONDS + 60))
  while ((SECONDS < deadline)); do
    curl -fsS "$url" >/dev/null 2>&1 && return 0
    sleep 1
  done
  echo "NO-GO: $label never became healthy at $url" >&2
  return 1
}

# Drain: do not start a measured arm while the previous one is still finishing.
# A busy engine would charge the previous arm's queue to this arm.
drain() {
  local deadline=$((SECONDS + DRAIN_TIMEOUT_S))
  local running
  while ((SECONDS < deadline)); do
    running="$(curl -fsS "$VLLM_METRICS" 2>/dev/null \
      | awk '/^vllm:num_requests_running/ {print $2; exit}')"
    if [[ -z "$running" ]]; then
      sleep 2
      return 0  # no metrics endpoint; a fixed settle is the honest fallback
    fi
    if awk -v value="$running" 'BEGIN { exit !(value + 0 == 0) }'; then
      sleep 2
      return 0
    fi
    sleep 2
  done
  echo "WARN: engine still busy after ${DRAIN_TIMEOUT_S}s drain; continuing" >&2
  return 0
}

# run_gateway_overhead.py has no warm-up flag, so warm-up is explicit: fire a
# handful of real requests at the arm's endpoint and discard them, so the first
# measured request does not pay for cold caches and lazy imports.
WARMUP_BODY="$RUN_ROOT/warmup-body.json"
build_warmup_body() {
  PYTHONPATH="$REPO_ROOT" "$VENV/bin/python" - \
    "$MIXED_WORKLOAD" "$WARMUP_BODY" "$MODEL" <<'PY'
import json, sys
from scheduler_benchmark.runner import WorkloadRequest, make_chat_payload
workload, out, model = sys.argv[1], sys.argv[2], sys.argv[3]
row = next(json.loads(line) for line in open(workload) if line.strip())
request = WorkloadRequest(
    request_id=str(row["request_id"]), prompt=str(row["prompt"]),
    baseline_service_ms=float(row.get("baseline_service_ms", 0.0)),
    max_tokens=int(row.get("max_tokens", 4096)), kind=str(row.get("kind", "tool")),
    category=str(row.get("category", "")), tool_schema=str(row.get("tool_schema", "")),
    history=[list(item) for item in row.get("history", [])],
)
payload = make_chat_payload(request, model=model)
payload["stream"] = False
payload.pop("stream_options", None)
open(out, "w").write(json.dumps(payload))
PY
}

warmup() {
  local target="$1" index
  for ((index = 0; index < WARMUP_REQUESTS; index++)); do
    curl -fsS -H 'Authorization: Bearer vx-dev' -H 'Content-Type: application/json' \
      -d @"$WARMUP_BODY" "$target" >/dev/null 2>&1 || true
  done
}

start_decision() {
  local mode="$1"
  local log="$RUN_ROOT/logs/decision-$mode.log"
  local args=(
    "$VENV/bin/python" "$REPO_ROOT/scripts/run_decision_service.py"
    --quantile-manifest "$QUANTILE_MANIFEST"
    --reliability-threshold "$RELIABILITY_THRESHOLD"
  )
  export LTR_DECISION_TORCH_THREADS=2   # 7/19 trap: thread oversubscription
  case "$mode" in
    stub) args+=(--predictor stub) ;;
    cpu_nogate)
      args+=(--predictor bert --checkpoint "$CHECKPOINT" --device cpu --no-gate) ;;
    cpu_gated)
      args+=(--predictor bert --checkpoint "$CHECKPOINT" --device cpu) ;;
    gpu_gated)
      args+=(--predictor bert --checkpoint "$CHECKPOINT" --device cuda
             --batch-max 8 --batch-window-ms 3) ;;
    *) echo "unknown decision mode: $mode" >&2; return 1 ;;
  esac
  "${args[@]}" >"$log" 2>&1 &
  DECISION_PID=$!
  wait_healthy "$DECISION_HEALTH" "decision service ($mode)"
}

start_gateway() {
  local with_decision="$1"
  local log="$RUN_ROOT/logs/gateway-$with_decision.log"
  if [[ "$with_decision" == "yes" ]]; then
    LTR_DECISION_ENDPOINT="http://127.0.0.1:9200" \
      "$REPO_ROOT/scripts/server/launch_gateway.sh" >"$log" 2>&1 &
  else
    # G0 isolates pure gateway proxying: the gateway must not consult any
    # decision service, so the endpoint is unset rather than pointed at a stub.
    LTR_DECISION_ENDPOINT="" \
      "$REPO_ROOT/scripts/server/launch_gateway.sh" >"$log" 2>&1 &
  fi
  GATEWAY_PID=$!
  wait_healthy "$GATEWAY_HEALTH" "gateway (decision=$with_decision)"
}

# One measured arm. Each arm brings up exactly the components it needs, warms
# up, drains, then measures.
run_arm() {
  local arm="$1" half="$2"
  local output="$RUN_ROOT/arms/${arm}-${half}.json"
  local target="$ENDPOINT"
  echo ">> arm $arm ($half half)"

  stop_gateway
  stop_decision

  case "$arm" in
    D0) target="$DIRECT_ENDPOINT" ;;
    G0) start_gateway no ;;
    G1) start_decision stub;       start_gateway yes ;;
    G2) start_decision cpu_nogate; start_gateway yes ;;
    G3) start_decision cpu_gated;  start_gateway yes ;;
    G4) start_decision gpu_gated;  start_gateway yes ;;
    *) echo "unknown arm: $arm" >&2; return 1 ;;
  esac

  warmup "$target"
  drain
  "$VENV/bin/python" "$REPO_ROOT/scripts/run_gateway_overhead.py" \
    --direct-endpoint "$DIRECT_ENDPOINT" --gateway-endpoint "$target" \
    --model "$MODEL" --workload "$MIXED_WORKLOAD" \
    --capacity-rps "$capacity_rps" --scheduler-cls "$STOCK" \
    --output "$output" --saturation 0.9 --api-key vx-dev \
    2>&1 | tee "$RUN_ROOT/logs/arm-${arm}-${half}.log"

  stop_gateway
  stop_decision
  drain
}

# vLLM stays up for the whole block: it is the constant every arm shares.
stop_vllm
block_tag="$(new_attempt_tag "$RUN_TAG-block2")"
prepare_vllm_evidence "$block_tag" unique
write_manifest stock_fcfs "$STOCK" block2_overhead "$block_tag" \
  "$RUN_ROOT/manifests/block2-overhead.json" \
  "$MIXED_WORKLOAD" "$capacity_rps" "$MODEL" "$VLLM_VERSION"
ACTIVE_ATTEMPT_TAG="$block_tag"
ACTIVE_ATTEMPT_MANIFEST="$RUN_ROOT/manifests/block2-overhead.json"
ACTIVE_ATTEMPT_SCHEDULER="$STOCK"
"$REPO_ROOT/scripts/server/launch_vllm.sh" "$STOCK" "$block_tag"
build_warmup_body

FORWARD=(D0 G0 G1 G2 G3 G4)
REVERSE=(G4 G3 G2 G1 G0 D0)

for arm in "${FORWARD[@]}"; do
  run_arm "$arm" first
done
for arm in "${REVERSE[@]}"; do
  run_arm "$arm" second
done

stop_vllm
collect_vllm_evidence "$block_tag" unique
mark_attempt_status "$RUN_ROOT/manifests/block2-overhead.json" "$block_tag" complete
ACTIVE_ATTEMPT_TAG=""; ACTIVE_ATTEMPT_MANIFEST=""; ACTIVE_ATTEMPT_SCHEDULER=""

python3 - "$RUN_ROOT" "$RUN_TAG" <<'PY'
import json, pathlib, sys
run_root, run_tag = pathlib.Path(sys.argv[1]), sys.argv[2]
arms = {}
for path in sorted((run_root / "arms").glob("*.json")):
    arm, half = path.stem.rsplit("-", 1)
    try:
        arms.setdefault(arm, {})[half] = json.loads(path.read_text())
    except json.JSONDecodeError:
        arms.setdefault(arm, {})[half] = {"error": "unparseable output"}
summary = {
    "schema_version": "block2-overhead-v1",
    "run_tag": run_tag,
    "ordering": "ABBA: D0 G0 G1 G2 G3 G4 | G4 G3 G2 G1 G0 D0",
    "arm_meaning": {
        "D0": "direct to vLLM, no gateway",
        "G0": "gateway without a decision endpoint",
        "G1": "gateway + stub decision service",
        "G2": "gateway + CPU BERT, gate disabled",
        "G3": "gateway + CPU BERT, gate-first",
        "G4": "gateway + GPU batched, gate-first",
    },
    "note": "consecutive arm differences are per-component costs; each half is "
            "reported separately so drift is visible rather than averaged away",
    "arms_present": {arm: sorted(halves) for arm, halves in sorted(arms.items())},
}
(run_root / "block2-summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo ">> Block 2 complete: $RUN_ROOT"
