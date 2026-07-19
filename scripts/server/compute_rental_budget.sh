#!/usr/bin/env bash
set -euo pipefail

LTR_ROOT="${LTR_ROOT:-/hy-tmp/ltr}"
ARTIFACTS="${ARTIFACTS:-$LTR_ROOT/artifacts/current}"
MIXED_WORKLOAD="${MIXED_WORKLOAD:-$ARTIFACTS/mixed.v2.jsonl}"
OOD_WORKLOAD="${OOD_WORKLOAD:-$ARTIFACTS/ood.v2.jsonl}"
OUTPUT="${OUTPUT:-$LTR_ROOT/rental-budget.json}"
CAPACITY_RPS="${CAPACITY_RPS:-0.3}"
MIXED_REQUESTS="${MIXED_REQUESTS:-}"
OOD_REQUESTS="${OOD_REQUESTS:-}"

count_rows() {
  local path="$1" explicit="$2"
  if [[ -n "$explicit" ]]; then
    printf '%s\n' "$explicit"
  elif [[ -f "$path" ]]; then
    awk 'NF {count++} END {print count+0}' "$path"
  else
    echo "set request count or provide workload: $path" >&2
    return 1
  fi
}
mixed_count="$(count_rows "$MIXED_WORKLOAD" "$MIXED_REQUESTS")"
ood_count="$(count_rows "$OOD_WORKLOAD" "$OOD_REQUESTS")"

python3 - "$OUTPUT" "$CAPACITY_RPS" "$mixed_count" "$ood_count" <<'PY'
import json, math, os, sys
from pathlib import Path

output = Path(sys.argv[1]); capacity = float(sys.argv[2])
mixed_n, ood_n = int(sys.argv[3]), int(sys.argv[4])
if capacity <= 0 or mixed_n <= 0 or ood_n <= 0:
    raise SystemExit("capacity and request counts must be positive")
env_minutes = lambda name, default: float(os.environ.get(name, default)) * 60.0
steady = lambda requests: requests / (0.9 * capacity)
grid = [0.3, 0.45, 0.68, 1.0, 1.5, 2.2]
stages = [
    ("environment_and_model_download", env_minutes("SETUP_MINUTES", 35)),
    ("restore_and_verify", env_minutes("RESTORE_MINUTES", 20)),
    ("repair_three_replay_errors", env_minutes("REPLAY_MINUTES", 15)),
    ("build_rank_quantiles", env_minutes("QUANTILE_MINUTES", 8)),
    ("measure_decision_latency_20_warm_plus_200", env_minutes("LATENCY_MINUTES", 8)),
    ("protocol_seam_and_two_request_e2e", env_minutes("PREFLIGHT_MINUTES", 10)),
    ("saturation_calibration_6x120", sum(120 / (0.9 * rps) for rps in grid)),
    ("mixed_matrix_7x3", 7 * 3 * steady(mixed_n)),
    ("ood_matrix_4x3", 4 * 3 * steady(ood_n)),
    ("gateway_overhead_paired_replay", 2 * steady(mixed_n)),
    ("model_restarts_13x3min", 13 * 3 * 60.0),
    ("results_upload", 10 * 60.0),
]
total = sum(seconds for _, seconds in stages)
limit = 5.25 * 3600
payload = {
    "schema_version": "rental-budget-v1",
    "inputs": {"capacity_rps": capacity, "mixed_requests": mixed_n, "ood_requests": ood_n},
    "arrival_formula": "N / (0.9 * capacity_rps)",
    "stages": [{"name": name, "seconds": seconds, "minutes": seconds / 60} for name, seconds in stages],
    "total_seconds": total, "total_hours": total / 3600,
    "gate_hours": 5.25, "retry_reserve_minutes": 45,
    "passed": total <= limit,
    "trim_order": [
        "reduce OOD repeats from 3 to 2",
        "drop mixed non-core policies while retaining fcfs, pure_ltr, tail_safe, gated_hybrid",
    ],
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps({"total_hours": payload["total_hours"], "passed": payload["passed"]}))
if total > limit:
    raise SystemExit("NO-GO: rental estimate exceeds 5.25 hours; apply trim_order")
PY
