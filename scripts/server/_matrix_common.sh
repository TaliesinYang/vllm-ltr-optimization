#!/usr/bin/env bash
# Shared matrix-driver machinery: process lifecycle, evidence capture, manifests
# and the per-policy launch/measure/verify cycle.
#
# Extracted VERBATIM from run_matrix.sh so the Block-1 and Block-2 drivers reuse
# the same safety behaviour instead of reimplementing it. run_matrix.sh itself is
# deliberately left untouched while the rental box is live; folding it onto this
# library is a post-rental cleanup, not a rental-day change.
#
# Sourcing contract - the caller MUST define these before sourcing:
#   LTR_ROOT REPO_ROOT GATEWAY_REPO VENV ARTIFACTS RUN_TAG RUN_ROOT
#   ENDPOINT MODEL STOCK CAPACITY_MANIFEST
# and MUST have created "$RUN_ROOT" and its subdirectories.
#
# Sourcing has SIDE EFFECTS, in this order:
#   1. installs the cleanup trap on EXIT/INT/TERM
#   2. asserts vLLM 0.24.x and sets VLLM_VERSION
#   3. runs the protocol-seam test, aborting if it was skipped
#   4. sets capacity_rps from CAPACITY_MANIFEST
# Source once, after the globals are defined and before any run_policy call.

ACTIVE_ATTEMPT_TAG="${ACTIVE_ATTEMPT_TAG:-}"
ACTIVE_ATTEMPT_MANIFEST="${ACTIVE_ATTEMPT_MANIFEST:-}"
ACTIVE_ATTEMPT_SCHEDULER="${ACTIVE_ATTEMPT_SCHEDULER:-}"
CLEANUP_RUNNING="${CLEANUP_RUNNING:-0}"

safe_stop_pidfile() {
  local pidfile="$1" signature="$2" startfile="$1.starttime"
  local pid saved_start current_start cmdline
  [[ -f "$pidfile" ]] || return 0
  pid="$(<"$pidfile")"
  if [[ ! "$pid" =~ ^[0-9]+$ || ! -f "$startfile" ]]; then
    rm -f "$pidfile" "$startfile"
    return 0
  fi
  saved_start="$(<"$startfile")"
  if [[ ! "$saved_start" =~ ^[0-9]+$ || ! -r "/proc/$pid/stat" || ! -r "/proc/$pid/cmdline" ]]; then
    rm -f "$pidfile" "$startfile"
    return 0
  fi
  if ! current_start="$(awk '{print $22}' "/proc/$pid/stat" 2>/dev/null)" || \
      ! cmdline="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null)"; then
    rm -f "$pidfile" "$startfile"
    return 0
  fi
  if [[ "$current_start" == "$saved_start" && "$cmdline" == *"$signature"* ]]; then
    kill "$pid" 2>/dev/null || true
  else
    echo "discarding stale or unexpected pidfile without signaling: $pidfile" >&2
  fi
  rm -f "$pidfile" "$startfile"
}

cleanup() {
  local pidfile
  [[ "$CLEANUP_RUNNING" == 0 ]] || return 0
  CLEANUP_RUNNING=1
  set +e
  for pidfile in "$LTR_ROOT"/runs/*/vllm.pid; do
    [[ -f "$pidfile" ]] || continue
    safe_stop_pidfile "$pidfile" "vllm.entrypoints.openai.api_server"
  done
  if [[ -n "$ACTIVE_ATTEMPT_TAG" ]]; then
    echo "cleanup: archiving failed attempt $ACTIVE_ATTEMPT_TAG scheduler=$ACTIVE_ATTEMPT_SCHEDULER" >&2
    active_evidence="$RUN_ROOT/vllm-evidence/$ACTIVE_ATTEMPT_TAG"
    if [[ ! -e "$active_evidence" ]]; then
      collect_vllm_evidence "$ACTIVE_ATTEMPT_TAG" unique
    else
      mkdir -p "$active_evidence"
      printf '%s\n' "$ACTIVE_ATTEMPT_TAG" >"$active_evidence/run-tag.txt"
      active_source="$LTR_ROOT/runs/$ACTIVE_ATTEMPT_TAG"
      for active_filename in order.jsonl vllm.log; do
        if [[ ! -f "$active_evidence/$active_filename" && -f "$active_source/$active_filename" ]]; then
          cp "$active_source/$active_filename" "$active_evidence/$active_filename"
        fi
      done
    fi
    if [[ -n "$ACTIVE_ATTEMPT_MANIFEST" && -f "$ACTIVE_ATTEMPT_MANIFEST" ]]; then
      mark_attempt_status "$ACTIVE_ATTEMPT_MANIFEST" "$ACTIVE_ATTEMPT_TAG" failed
    fi
    ACTIVE_ATTEMPT_TAG=""
    ACTIVE_ATTEMPT_MANIFEST=""
    ACTIVE_ATTEMPT_SCHEDULER=""
  fi
  # Decision + gateway are PERSISTENT services owned by the run wrapper, not by
  # a single matrix attempt — do NOT kill them here, or a preflight retry finds
  # them dead and fail-opens (all predictions unreliable). Leave them running.
  CLEANUP_RUNNING=0
  set -e
  return 0
}
trap cleanup EXIT INT TERM

VLLM_VERSION="$("$VENV/bin/python" -c 'import vllm; print(vllm.__version__)')"
[[ "$VLLM_VERSION" == 0.24.* ]] || { echo "vLLM 0.24.x required, got $VLLM_VERSION" >&2; exit 1; }
protocol_output="$("$VENV/bin/python" -m pytest "$REPO_ROOT/tests/test_vllm_protocol_seam.py" -q -rs 2>&1)"
printf '%s\n' "$protocol_output"
if grep -Eiq '(^|[^0-9])[1-9][0-9]* skipped|SKIPPED' <<<"$protocol_output"; then
  echo "NO-GO: protocol seam was skipped" >&2
  exit 1
fi

capacity_rps="$(python3 - "$CAPACITY_MANIFEST" <<'PY'
import json, sys
value = float(json.load(open(sys.argv[1]))["capacity_rps"])
if value <= 0: raise SystemExit("capacity_rps must be positive")
print(value)
PY
)"

stop_vllm() {
  local pidfile used
  for pidfile in "$LTR_ROOT"/runs/*/vllm.pid; do
    [[ -f "$pidfile" ]] || continue
    safe_stop_pidfile "$pidfile" "vllm.entrypoints.openai.api_server"
  done
  local http_down=0
  for _ in $(seq 1 60); do
    if ! curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
      http_down=1
      break
    fi
    sleep 1
  done
  if [[ "$http_down" != 1 ]]; then
    echo "vLLM did not stop" >&2
    return 1
  fi
  # vLLM spawns a VLLM::EngineCore subprocess whose cmdline does NOT contain
  # "vllm.entrypoints.openai.api_server", so the pidfile-signature stop above
  # never reaps it. It keeps the GPU allocation, so the NEXT scheduler's launch
  # hits "Free memory on device cuda:0 (…) is less than desired GPU memory
  # utilization". Reap it explicitly, then wait for the driver to reclaim the
  # memory before returning (launch_vllm.sh would otherwise fail at startup).
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
  pkill -9 -f "multiprocessing.resource_tracker" 2>/dev/null || true
  for _ in $(seq 1 40); do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
    if [[ "${used:-99999}" -lt 5000 ]] 2>/dev/null; then
      return 0
    fi
    sleep 5
  done
  echo "warning: GPU memory not reclaimed after vLLM stop (used=${used:-unknown} MiB)" >&2
  return 0
}

policy_name() {
  case "$1" in
    "$STOCK") printf '%s\n' stock_fcfs ;;
    *) local short="${1##*.}"; printf '%s\n' "$short" ;;
  esac
}

prepare_vllm_evidence() {
  local tag="$1" mode="${2:-current}" source_dir="$LTR_ROOT/runs/$1"
  [[ "$tag" == "$RUN_TAG"-* ]] || { echo "refusing unrelated vLLM evidence tag: $tag" >&2; return 1; }
  if [[ "$mode" == unique && -e "$source_dir" ]]; then
    echo "refusing to overwrite existing vLLM attempt: $source_dir" >&2
    return 1
  fi
  mkdir -p "$source_dir"
  : >"$source_dir/order.jsonl"
  rm -f "$source_dir/vllm.log"
}

collect_vllm_evidence() {
  local tag="$1" mode="${2:-current}" source_dir="$LTR_ROOT/runs/$1" target_dir="$RUN_ROOT/vllm-evidence/$1"
  [[ "$tag" == "$RUN_TAG"-* ]] || { echo "refusing unrelated vLLM evidence tag: $tag" >&2; return 1; }
  if [[ "$mode" == unique && -e "$target_dir" ]]; then
    echo "refusing to overwrite archived vLLM attempt: $target_dir" >&2
    return 1
  fi
  mkdir -p "$target_dir"
  printf '%s\n' "$tag" >"$target_dir/run-tag.txt"
  for filename in order.jsonl vllm.log; do
    [[ -f "$source_dir/$filename" ]] || { echo "missing vLLM evidence: $source_dir/$filename" >&2; return 1; }
    cp "$source_dir/$filename" "$target_dir/$filename"
  done
}

new_attempt_tag() {
  local base="$1" nonce
  nonce="$(python3 -c 'import time; print(time.time_ns())')"
  printf '%s-attempt-%s\n' "$base" "$nonce"
}

write_manifest() {
  local policy="$1" scheduler="$2" profile="$3" vllm_tag="$4" output="$5"
  local workload="$6" capacity="$7" model="$8" vllm_version="$9"
  python3 - "$REPO_ROOT" "$GATEWAY_REPO" "$ARTIFACTS/rank_quantiles.json" \
    "$LTR_ROOT/manifest.decision-latency.json" "$policy" "$scheduler" "$profile" "$vllm_tag" "$output" \
    "$workload" "$capacity" "$model" "$vllm_version" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
repo, gateway, quantile_path, latency_path = map(Path, sys.argv[1:5])
quantile_raw = quantile_path.read_bytes(); quantile = json.loads(quantile_raw)
latency = json.loads(latency_path.read_text())
payload = {
  "schema_version": "rental-run-manifest-v1",
  "repo_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
  "gateway_commit": subprocess.check_output(["git", "-C", str(gateway), "rev-parse", "HEAD"], text=True).strip(),
  "gateway_pin": (repo / "scripts/server/manifest/gateway-pin.txt").read_text().strip(),
  "mapping_version": quantile["mapping_version"],
  "approximation_notice": quantile["approximation_notice"],
  "quantile_manifest_sha256": hashlib.sha256(quantile_raw).hexdigest(),
  "decision_timeout_ms": latency["timeout_ms"],
  "policy": sys.argv[5], "scheduler_cls": sys.argv[6], "profile": sys.argv[7],
  "vllm_run_tag": sys.argv[8],
  "workload_path": str(Path(sys.argv[10]).resolve()),
  "workload_sha256": hashlib.sha256(Path(sys.argv[10]).read_bytes()).hexdigest(),
  "capacity_rps": float(sys.argv[11]),
  "model": sys.argv[12],
  "vllm_version": sys.argv[13],
  "shape_parameters": ["enable-auto-tool-choice", "tool-call-parser=qwen3_coder", "reasoning-parser=qwen3", "enable_thinking=false", "max-model-len=8192"],
  "limitation": "max-num-seqs and gpu-memory-utilization are throughput parameters intentionally not copied; greedy content invariance under batching is assumed",
}
path = Path(sys.argv[9])
if path.exists():
    prior = json.loads(path.read_text())
    immutable = set(payload) - {"vllm_run_tag"}
    mismatches = [key for key in immutable if prior.get(key) != payload[key]]
    if mismatches:
        raise SystemExit(f"resume provenance mismatch for {path}: {mismatches}")
    tags = list(prior.get("vllm_attempt_tags", []))
    legacy_tag = prior.get("vllm_run_tag")
    if isinstance(legacy_tag, str) and legacy_tag and legacy_tag not in tags:
        tags.append(legacy_tag)
    if payload["vllm_run_tag"] not in tags:
        tags.append(payload["vllm_run_tag"])
    prior["vllm_run_tag"] = payload["vllm_run_tag"]
    prior["vllm_attempt_tags"] = tags
    statuses = dict(prior.get("vllm_attempt_status", {}))
    statuses[payload["vllm_run_tag"]] = "pending"
    prior["vllm_attempt_status"] = statuses
    payload = prior
else:
    payload["vllm_attempt_tags"] = [payload["vllm_run_tag"]]
    payload["vllm_attempt_status"] = {payload["vllm_run_tag"]: "pending"}
temporary = path.with_suffix(path.suffix + ".partial")
temporary.write_text(json.dumps(payload, indent=2) + "\n")
temporary.replace(path)
PY
}

mark_attempt_status() {
  local manifest="$1" attempt_tag="$2" status="$3"
  python3 - "$manifest" "$attempt_tag" "$status" <<'PY'
import json, sys
from pathlib import Path
path, tag, status = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
if status not in {"complete", "failed"}:
    raise SystemExit(f"invalid attempt status: {status}")
payload = json.loads(path.read_text())
if tag not in payload.get("vllm_attempt_tags", []):
    raise SystemExit(f"attempt tag is not registered: {tag}")
statuses = dict(payload.get("vllm_attempt_status", {}))
statuses[tag] = status
payload["vllm_attempt_status"] = statuses
temporary = path.with_suffix(path.suffix + ".partial")
temporary.write_text(json.dumps(payload, indent=2) + "\n")
temporary.replace(path)
PY
}

policy_output_complete() {
  local output="$1" scheduler="$2" profile="$3" repeats="$4" workload="$5"
  local capacity="$6" model="$7" vllm_version="$8"
  python3 - "$output" "$scheduler" "$profile" "$repeats" "$workload" "$capacity" "$model" "$vllm_version" <<'PY'
import hashlib, json, sys
from pathlib import Path
path, scheduler, profile, repeats = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
workload, capacity, model, vllm_version = Path(sys.argv[5]), float(sys.argv[6]), sys.argv[7], sys.argv[8]
if not path.is_file():
    raise SystemExit(1)
try:
    payload = json.loads(path.read_text())
    scenarios = payload["scenarios"]
    valid = (
        payload.get("valid") is True
        and payload.get("scheduler_cls") == scheduler
        and payload.get("profiles") == [profile]
        and payload.get("repeats") == repeats
        and payload.get("capacity_rps") == capacity
        and payload.get("model") == model
        and payload.get("vllm_version") == vllm_version
        and payload.get("workload_sha256") == hashlib.sha256(workload.read_bytes()).hexdigest()
        and len(scenarios) == 1
        and scenarios[0].get("load_pct") == 90
        and scenarios[0].get("profile") == profile
        and scenarios[0].get("scenario", {}).get("name") == "saturation-90"
        and len(scenarios[0].get("runs", [])) == repeats
        and all(run.get("status") == "complete" for run in scenarios[0]["runs"])
    )
except (KeyError, TypeError, ValueError, json.JSONDecodeError):
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

require_completed_evidence() {
  local manifest="$1" scheduler="$2" profile="$3" workload="$4"
  local capacity="$5" model="$6" vllm_version="$7"
  python3 - "$manifest" "$RUN_ROOT/vllm-evidence" "$scheduler" "$profile" "$STOCK" \
    "$workload" "$capacity" "$model" "$vllm_version" "$REPO_ROOT" "$GATEWAY_REPO" \
    "$ARTIFACTS/rank_quantiles.json" "$LTR_ROOT/manifest.decision-latency.json" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
manifest, evidence_root = Path(sys.argv[1]), Path(sys.argv[2])
scheduler, profile, stock = sys.argv[3:6]
workload, capacity, model, vllm_version = Path(sys.argv[6]), float(sys.argv[7]), sys.argv[8], sys.argv[9]
repo, gateway, quantile_path, latency_path = map(Path, sys.argv[10:14])
if not manifest.is_file():
    raise SystemExit(f"complete output has no manifest: {manifest}")
payload = json.loads(manifest.read_text())
quantile_raw = quantile_path.read_bytes()
quantile = json.loads(quantile_raw)
latency = json.loads(latency_path.read_text())
identity_matches = (
    payload.get("scheduler_cls") == scheduler
    and payload.get("profile") == profile
    and payload.get("workload_path") == str(workload.resolve())
    and payload.get("workload_sha256") == hashlib.sha256(workload.read_bytes()).hexdigest()
    and payload.get("capacity_rps") == capacity
    and payload.get("model") == model
    and payload.get("vllm_version") == vllm_version
    and payload.get("repo_commit") == subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    and payload.get("gateway_commit") == subprocess.check_output(["git", "-C", str(gateway), "rev-parse", "HEAD"], text=True).strip()
    and payload.get("gateway_pin") == (repo / "scripts/server/manifest/gateway-pin.txt").read_text().strip()
    and payload.get("mapping_version") == quantile.get("mapping_version")
    and payload.get("approximation_notice") == quantile.get("approximation_notice")
    and payload.get("quantile_manifest_sha256") == hashlib.sha256(quantile_raw).hexdigest()
    and payload.get("decision_timeout_ms") == latency.get("timeout_ms")
)
if not identity_matches:
    raise SystemExit("complete output manifest identity mismatch")
tags = payload.get("vllm_attempt_tags")
statuses = payload.get("vllm_attempt_status", {})
if not isinstance(tags, list) or not tags or len(tags) != len(set(tags)):
    raise SystemExit("complete output manifest has no unique attempt tags")
completed = []
for tag in tags:
    directory = evidence_root / tag
    order, log = directory / "order.jsonl", directory / "vllm.log"
    if not order.is_file() or not log.is_file():
        raise SystemExit(f"attempt evidence missing for {tag}")
    if statuses.get(tag) == "complete":
        completed.append(tag)
if not completed:
    raise SystemExit("complete output has no completed evidence attempt")
if scheduler != stock and not any((evidence_root / tag / "order.jsonl").stat().st_size > 0 for tag in completed):
    raise SystemExit("complete custom scheduler output has no non-empty completed order log")
PY
}

run_policy() {
  local scheduler="$1" profile="$2" workload="$3" repeats="$4" output_dir="$5"
  local policy run_id output manifest attempt_tag runner_status runner_profile
  policy="$(policy_name "$scheduler")"
  run_id="${profile}-${policy}"
  # The runner only accepts id|ood|mixed for --profile; run labels like
  # "sentinel-1" or "mixed-round-a" are naming, not runner profiles.
  case "$profile" in
    id|ood|mixed) runner_profile="$profile" ;;
    *) runner_profile="mixed" ;;
  esac
  output="$output_dir/$policy.json"
  manifest="$RUN_ROOT/manifests/$run_id.json"
  if policy_output_complete "$output" "$scheduler" "$runner_profile" "$repeats" "$workload" "$capacity_rps" "$MODEL" "$VLLM_VERSION"; then
    # The runner OUTPUT records the enum profile; the MANIFEST records the label.
    require_completed_evidence "$manifest" "$scheduler" "$profile" "$workload" "$capacity_rps" "$MODEL" "$VLLM_VERSION"
    echo "resume: complete policy with archived evidence, skipping launch: $run_id"
    return 0
  fi
  attempt_tag="$(new_attempt_tag "$RUN_TAG-$run_id")"
  stop_vllm
  prepare_vllm_evidence "$attempt_tag" unique
  write_manifest "$policy" "$scheduler" "$profile" "$attempt_tag" "$manifest" \
    "$workload" "$capacity_rps" "$MODEL" "$VLLM_VERSION"
  ACTIVE_ATTEMPT_TAG="$attempt_tag"
  ACTIVE_ATTEMPT_MANIFEST="$manifest"
  ACTIVE_ATTEMPT_SCHEDULER="$scheduler"
  "$REPO_ROOT/scripts/server/launch_vllm.sh" "$scheduler" "$attempt_tag"
  runner_status=0
  if "$VENV/bin/python" -m scheduler_benchmark.runner \
      --endpoint "$ENDPOINT" --model "$MODEL" --workload "$workload" \
      --capacity-rps "$capacity_rps" --scheduler-cls "$scheduler" \
      --output "$output" --api-key vx-dev --scenario steady --load 90 \
      --profile "$runner_profile" --repeats "$repeats" --resume; then
    runner_status=0
  else
    runner_status=$?
  fi
  stop_vllm
  collect_vllm_evidence "$attempt_tag" unique
  if [[ "$runner_status" != 0 ]]; then
    mark_attempt_status "$manifest" "$attempt_tag" failed
    ACTIVE_ATTEMPT_TAG=""; ACTIVE_ATTEMPT_MANIFEST=""; ACTIVE_ATTEMPT_SCHEDULER=""
    return "$runner_status"
  fi
  policy_output_complete "$output" "$scheduler" "$runner_profile" "$repeats" "$workload" "$capacity_rps" "$MODEL" "$VLLM_VERSION" || {
    mark_attempt_status "$manifest" "$attempt_tag" failed
    ACTIVE_ATTEMPT_TAG=""; ACTIVE_ATTEMPT_MANIFEST=""; ACTIVE_ATTEMPT_SCHEDULER=""
    echo "runner returned success without an exact complete output: $run_id" >&2
    return 1
  }
  if [[ "$scheduler" != "$STOCK" && "$scheduler" != "scheduler_benchmark.vllm_scheduler.StockFCFSShim" && ! -s "$RUN_ROOT/vllm-evidence/$attempt_tag/order.jsonl" ]]; then
    mark_attempt_status "$manifest" "$attempt_tag" failed
    ACTIVE_ATTEMPT_TAG=""; ACTIVE_ATTEMPT_MANIFEST=""; ACTIVE_ATTEMPT_SCHEDULER=""
    echo "custom scheduler attempt produced an empty order log: $attempt_tag" >&2
    return 1
  fi
  mark_attempt_status "$manifest" "$attempt_tag" complete
  ACTIVE_ATTEMPT_TAG=""; ACTIVE_ATTEMPT_MANIFEST=""; ACTIVE_ATTEMPT_SCHEDULER=""
}
