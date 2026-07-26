#!/usr/bin/env bash
# Block 1: mixed-workload policy matrix on the trace-calibrated Block-1 workload.
#
# Derived from run_matrix.sh's mixed phase. Differences, all deliberate:
#   - six policies, including PolicyFCFS (the algorithmic FCFS control) and
#     GatedRuleCScheduler (slot-preserving gating)
#   - five repeats split into TWO LAUNCH ROUNDS with independently shuffled
#     policy order, so a systematic drift over the session (thermal, cache,
#     background load) cannot masquerade as a policy effect
#   - FCFS sentinels before, between and after the rounds: if the machine drifts,
#     the sentinels move and we can say so instead of guessing
#   - no OOD phase and no overhead phase; Block 2 owns those
#
# The shuffle is seeded from RUN_TAG, so a resumed run reproduces the same order.
set -euo pipefail

LTR_ROOT="${LTR_ROOT:-/hy-tmp/ltr}"
REPO_ROOT="${REPO_ROOT:-$LTR_ROOT/repo}"
GATEWAY_REPO="${GATEWAY_REPO:-$LTR_ROOT/VeloxMesh}"
VENV="${VENV:-$LTR_ROOT/venv}"
ARTIFACTS="${ARTIFACTS:-$LTR_ROOT/artifacts/current}"
RUN_TAG="${RUN_TAG:-block1-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="$LTR_ROOT/runs/$RUN_TAG"
MIXED_WORKLOAD="${MIXED_WORKLOAD:-$ARTIFACTS/block1.jsonl}"
CAPACITY_MANIFEST="${CAPACITY_MANIFEST:-$LTR_ROOT/runs/calibration/capacity.json}"
ENDPOINT="${ENDPOINT:-http://127.0.0.1:9100/v1/chat/completions}"
MODEL="${MODEL:-qwen3.5-9b}"
STOCK="vllm.v1.core.sched.scheduler.Scheduler"

ROUND_A_REPEATS="${ROUND_A_REPEATS:-3}"
ROUND_B_REPEATS="${ROUND_B_REPEATS:-2}"
SENTINEL_ROWS="${SENTINEL_ROWS:-120}"
SENTINEL_REPEATS="${SENTINEL_REPEATS:-1}"
SENTINEL_CLASS="scheduler_benchmark.vllm_scheduler.PolicyFCFS"

BLOCK1_CLASSES=(
  "$STOCK"
  scheduler_benchmark.vllm_scheduler.StockFCFSShim
  scheduler_benchmark.vllm_scheduler.PolicyFCFS
  scheduler_benchmark.vllm_scheduler.PromptLengthSJFScheduler
  scheduler_benchmark.vllm_scheduler.PureLTRScheduler
  scheduler_benchmark.vllm_scheduler.GatedRuleCScheduler
)

for value in "$ROUND_A_REPEATS" "$ROUND_B_REPEATS" "$SENTINEL_REPEATS" "$SENTINEL_ROWS"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "repeat/row counts must be positive integers" >&2
    exit 1
  }
done

mkdir -p "$RUN_ROOT/matrix" "$RUN_ROOT/matrix-round-b" "$RUN_ROOT/sentinels" \
  "$RUN_ROOT/manifests"

# shellcheck source=scripts/server/_matrix_common.sh
source "$REPO_ROOT/scripts/server/_matrix_common.sh"

for path in "$MIXED_WORKLOAD" "$ARTIFACTS/rank_quantiles.json" "$CAPACITY_MANIFEST"; do
  [[ -f "$path" ]] || { echo "required Block-1 input missing: $path" >&2; exit 1; }
done

curl -fsS http://127.0.0.1:9200/healthz >/dev/null
curl -fsS http://127.0.0.1:9100/healthz >/dev/null

# Deterministic per-round policy order. Seeded by RUN_TAG so a resume replays the
# same sequence; independent per round so the two rounds are not correlated.
shuffled_policies() {
  local round="$1"
  printf '%s\n' "${BLOCK1_CLASSES[@]}" | python3 -c '
import hashlib, random, sys
rows = [line.strip() for line in sys.stdin if line.strip()]
seed = int(hashlib.sha256(f"{sys.argv[1]}::{sys.argv[2]}".encode()).hexdigest()[:16], 16)
random.Random(seed).shuffle(rows)
print("\n".join(rows))
' "$RUN_TAG" "$round"
}

SENTINEL_WORKLOAD="$RUN_ROOT/sentinel-workload.jsonl"
head -n "$SENTINEL_ROWS" "$MIXED_WORKLOAD" >"$SENTINEL_WORKLOAD"
[[ -s "$SENTINEL_WORKLOAD" ]] || { echo "sentinel workload is empty" >&2; exit 1; }

# A sentinel is one short PolicyFCFS run. Its job is to detect machine drift
# ACROSS the session, so it must use the same code path as a measured policy.
run_sentinel() {
  local label="$1"
  local output_dir="$RUN_ROOT/sentinels/$label"
  mkdir -p "$output_dir"
  echo ">> sentinel $label: $SENTINEL_ROWS rows on PolicyFCFS"
  RUN_SENTINEL_LABEL="$label" run_policy \
    "$SENTINEL_CLASS" "sentinel-$label" "$SENTINEL_WORKLOAD" \
    "$SENTINEL_REPEATS" "$output_dir"
}

run_round() {
  local round="$1" repeats="$2" output_dir="$3"
  local scheduler
  mkdir -p "$output_dir"
  echo ">> Block-1 round $round: $repeats repeats, order:"
  shuffled_policies "$round" | sed 's/^/     /'
  while IFS= read -r scheduler; do
    [[ -n "$scheduler" ]] || continue
    run_policy "$scheduler" "mixed-round-$round" "$MIXED_WORKLOAD" \
      "$repeats" "$output_dir"
  done < <(shuffled_policies "$round")
}

run_sentinel 1
run_round a "$ROUND_A_REPEATS" "$RUN_ROOT/matrix"
run_sentinel 2
run_round b "$ROUND_B_REPEATS" "$RUN_ROOT/matrix-round-b"
run_sentinel 3

# Parity is a RECORDED finding, not a gate: a marginal tail delta must not
# discard the block. Both FCFS arms are present, so record their deltas.
parity_rc=0
if [[ -f "$RUN_ROOT/matrix/stock_fcfs.json" && -f "$RUN_ROOT/matrix/StockFCFSShim.json" ]]; then
  "$VENV/bin/python" "$REPO_ROOT/scripts/check_fcfs_parity.py" \
    --stock "$RUN_ROOT/matrix/stock_fcfs.json" \
    --shim "$RUN_ROOT/matrix/StockFCFSShim.json" \
    --output "$RUN_ROOT/matrix/parity.json" || parity_rc=$?
  if [[ "$parity_rc" != 0 ]]; then
    echo "WARN: stock-vs-shim parity outside tolerance (see matrix/parity.json); recorded, continuing" >&2
  fi
fi

python3 - "$RUN_ROOT" "$RUN_TAG" "$MIXED_WORKLOAD" <<'PY'
import json, pathlib, sys
run_root, run_tag, workload = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
summary = {
    "schema_version": "block1-matrix-v1",
    "run_tag": run_tag,
    "workload": workload,
    "rounds": {},
    "sentinels": {},
}
for name, folder in (("a", "matrix"), ("b", "matrix-round-b")):
    outputs = sorted(p.name for p in (run_root / folder).glob("*.json"))
    summary["rounds"][name] = {"dir": folder, "outputs": outputs}
for path in sorted((run_root / "sentinels").glob("*/*.json")):
    summary["sentinels"][path.parent.name] = str(path.relative_to(run_root))
(run_root / "block1-summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo ">> Block 1 complete: $RUN_ROOT"
