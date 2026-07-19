#!/usr/bin/env bash
set -euo pipefail

LTR_ROOT="${LTR_ROOT:-/hy-tmp/ltr}"
REPO_ROOT="${REPO_ROOT:-$LTR_ROOT/repo}"
VENV="${VENV:-$LTR_ROOT/venv}"
WORKLOAD="${WORKLOAD:-$LTR_ROOT/artifacts/current/mixed.v2.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$LTR_ROOT/runs/calibration}"
ENDPOINT="${ENDPOINT:-http://127.0.0.1:9100/v1/chat/completions}"
MODEL="${MODEL:-qwen3.5-9b}"
SCHEDULER_CLS="${SCHEDULER_CLS:-vllm.v1.core.sched.scheduler.Scheduler}"
GRID=(0.3 0.45 0.68 1.0 1.5 2.2)
SUBSET="$OUTPUT_DIR/workload-120.jsonl"
mkdir -p "$OUTPUT_DIR"
CAPACITY_OUTPUT="$OUTPUT_DIR/capacity.json"
[[ "$CAPACITY_OUTPUT" == "$OUTPUT_DIR/capacity.json" ]] || { echo "refusing unsafe capacity cleanup target" >&2; exit 1; }
rm -f "$CAPACITY_OUTPUT" "$CAPACITY_OUTPUT.partial"

[[ -f "$WORKLOAD" ]] || { echo "workload missing: $WORKLOAD" >&2; exit 1; }
python3 - "$WORKLOAD" "$SUBSET" <<'PY'
import sys
from pathlib import Path
rows = [line for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
if len(rows) < 120:
    raise SystemExit(f"calibration requires at least 120 requests, got {len(rows)}")
Path(sys.argv[2]).write_text("\n".join(rows[:120]) + "\n")
PY

for rps in "${GRID[@]}"; do
  safe_rps="${rps//./_}"
  grid_output="$OUTPUT_DIR/grid-$safe_rps.json"
  grid_runs="$OUTPUT_DIR/grid-$safe_rps.runs"
  [[ "$grid_output" == "$OUTPUT_DIR"/grid-*.json && "$grid_runs" == "$OUTPUT_DIR"/grid-*.runs ]] || {
    echo "refusing unsafe calibration cleanup target" >&2
    exit 1
  }
  rm -f "$grid_output"
  rm -rf "$grid_runs"
  # runner returns 2 when a subrun is invalid (over-saturated) — that is a
  # DATA POINT for calibration, not a fatal error. Capture it, keep going.
  runner_rc=0
  "$VENV/bin/python" -m scheduler_benchmark.runner \
    --endpoint "$ENDPOINT" --model "$MODEL" --workload "$SUBSET" \
    --capacity-rps "$rps" --scheduler-cls "$SCHEDULER_CLS" \
    --output "$grid_output" --api-key vx-dev \
    --scenario steady --load 90 --profile mixed --repeats 1 || runner_rc=$?
  if [[ "$runner_rc" != 0 && "$runner_rc" != 2 ]]; then
    echo "NO-GO: calibration runner crashed at rps=$rps (rc=$runner_rc)" >&2
    exit 1
  fi
  [[ -s "$grid_output" ]] || { echo "NO-GO: no grid output at rps=$rps" >&2; exit 1; }
done

python3 - "$OUTPUT_DIR" "${GRID[@]}" <<'PY'
import json, sys
from pathlib import Path
root, grid = Path(sys.argv[1]), [float(value) for value in sys.argv[2:]]
rows, first_failed = [], None
for index, offered in enumerate(grid):
    result = json.loads((root / f"grid-{str(offered).replace('.', '_')}.json").read_text())
    scenario = result["scenarios"][0]
    metrics = scenario["runs"][0]["metrics"]
    achieved = float(metrics["throughput_rps"])
    ratio = achieved / (0.9 * offered)
    rows.append({"capacity_rps": offered, "achieved_rps": achieved, "achievement_ratio": ratio})
    if ratio < 0.95 and first_failed is None:
        first_failed = index
if first_failed == 0:
    raise SystemExit("calibration failed: first grid point is already saturated; lower the grid floor")
if first_failed is None:
    raise SystemExit("calibration failed: grid ceiling is too low; raise it")
payload = {
    "schema_version": "saturation-calibration-v1", "request_count": 120,
    "load_pct": 90, "threshold": 0.95, "grid": rows,
    "first_saturated_rps": grid[first_failed],
    "capacity_rps": grid[first_failed - 1],
}
output = root / "capacity.json"
temporary = root / "capacity.json.partial"
temporary.write_text(json.dumps(payload, indent=2) + "\n")
temporary.replace(output)
print(json.dumps(payload))
PY
