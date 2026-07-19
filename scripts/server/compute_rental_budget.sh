#!/usr/bin/env bash
set -euo pipefail

LTR_ROOT="${LTR_ROOT:-/hy-tmp/ltr}"
ARTIFACTS="${ARTIFACTS:-$LTR_ROOT/artifacts/current}"
MIXED_WORKLOAD="${MIXED_WORKLOAD:-$ARTIFACTS/mixed.v2.jsonl}"
OOD_WORKLOAD="${OOD_WORKLOAD:-$ARTIFACTS/ood.v2.jsonl}"
OUTPUT="${OUTPUT:-$LTR_ROOT/rental-budget.json}"
# Pre-rental worst-case capacity floor: 0.75 rps, evidence-backed —
# our own 3090 (24G) tier2 replay sustained 202 tok/s at 8-way concurrency
# (tier2-throughput-final-8-report.json); mean output ~130 tok -> ~1.55 req/s
# service rate, halved for safety. Rental target (48G) is strictly faster.
CAPACITY_RPS="${CAPACITY_RPS:-0.75}"
MIXED_REQUESTS="${MIXED_REQUESTS:-}"
OOD_REQUESTS="${OOD_REQUESTS:-}"
MIXED_REPEATS="${MIXED_REPEATS:-3}"
OOD_REPEATS="${OOD_REPEATS:-3}"
MIXED_POLICY_COUNT="${MIXED_POLICY_COUNT:-7}"
OOD_POLICY_COUNT="${OOD_POLICY_COUNT:-4}"

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

python3 - "$OUTPUT" "$CAPACITY_RPS" "$mixed_count" "$ood_count" \
  "$MIXED_REPEATS" "$OOD_REPEATS" "$MIXED_POLICY_COUNT" "$OOD_POLICY_COUNT" <<'PY'
import json, math, os, sys
from pathlib import Path

output = Path(sys.argv[1]); capacity = float(sys.argv[2])
mixed_n, ood_n = int(sys.argv[3]), int(sys.argv[4])
mixed_repeats, ood_repeats = int(sys.argv[5]), int(sys.argv[6])
mixed_policies, ood_policies = int(sys.argv[7]), int(sys.argv[8])
if any(value <= 0 for value in (
    capacity, mixed_n, ood_n, mixed_repeats, ood_repeats,
    mixed_policies, ood_policies,
)):
    raise SystemExit("capacity, request counts, repeats, and policy counts must be positive")
env_minutes = lambda name, default: float(os.environ.get(name, default)) * 60.0
steady = lambda requests: requests / (0.9 * capacity)
grid = [0.3, 0.45, 0.68, 1.0, 1.5, 2.2]
restart_count = mixed_policies + ood_policies + 2  # preflight + overhead
stages = [
    ("environment_and_model_download", env_minutes("SETUP_MINUTES", 35)),
    ("restore_and_verify", env_minutes("RESTORE_MINUTES", 20)),
    ("repair_three_replay_errors", env_minutes("REPLAY_MINUTES", 15)),
    ("build_rank_quantiles", env_minutes("QUANTILE_MINUTES", 8)),
    ("ood_labeling_800_direct_vllm", env_minutes("OOD_LABELING_MINUTES", 25)),
    ("measure_decision_latency_20_warm_plus_200", env_minutes("LATENCY_MINUTES", 8)),
    ("protocol_seam_and_two_request_e2e", env_minutes("PREFLIGHT_MINUTES", 10)),
    ("saturation_calibration_6x120", sum(120 / (0.9 * rps) for rps in grid)),
    (f"mixed_matrix_{mixed_policies}x{mixed_repeats}", mixed_policies * mixed_repeats * steady(mixed_n)),
    (f"ood_matrix_{ood_policies}x{ood_repeats}", ood_policies * ood_repeats * steady(ood_n)),
    ("gateway_overhead_paired_replay", 2 * steady(mixed_n)),
    (f"model_restarts_{restart_count}x3min", restart_count * 3 * 60.0),
    ("results_upload", 10 * 60.0),
]
total = sum(seconds for _, seconds in stages)
limit = 7.25 * 3600
payload = {
    "schema_version": "rental-budget-v2",
    "inputs": {
        "capacity_rps": capacity,
        "mixed_requests": mixed_n,
        "ood_requests": ood_n,
        "mixed_repeats": mixed_repeats,
        "ood_repeats": ood_repeats,
        "mixed_policy_count": mixed_policies,
        "ood_policy_count": ood_policies,
    },
    "arrival_formula": "N / (0.9 * capacity_rps)",
    "stages": [{"name": name, "seconds": seconds, "minutes": seconds / 60} for name, seconds in stages],
    "total_seconds": total, "total_hours": total / 3600,
    "gate_hours": 7.25, "retry_reserve_minutes": 45,
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
    raise SystemExit("NO-GO: rental estimate exceeds 7.25 hours; apply trim_order")
PY
