#!/usr/bin/env bash
set -euo pipefail

LTR_ROOT="${LTR_ROOT:-/hy-tmp/ltr}"
REPO_ROOT="${REPO_ROOT:-$LTR_ROOT/repo}"
VENV="${VENV:-$LTR_ROOT/venv}"
ARTIFACTS="${ARTIFACTS:-$LTR_ROOT/artifacts/current}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_HOME="${HF_HOME:-/hy-tmp/hf}"
VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://127.0.0.1:8000/v1/chat/completions}"
VLLM_HEALTH_ENDPOINT="${VLLM_HEALTH_ENDPOINT:-http://127.0.0.1:8000/health}"
MODEL="${MODEL:-qwen3.5-9b}"
OOD_REPLAY_CONCURRENCY="${OOD_REPLAY_CONCURRENCY:-8}"

# Frozen R1 values: data-offline-spec ARTIFACT 1 requires about 400 rows per
# OOD stratum with a fixed seed; the committed R1 conversion evidence uses 17.
OOD_SAMPLE_SIZE=400
OOD_SAMPLE_SEED=17
WORKLOAD_SEED=42
OOD_RATIO=0.5

DECLARATIONS="$REPO_ROOT/configs/source-declarations.json"
HELPER="$REPO_ROOT/scripts/server/build_server_workloads.py"
SOURCE_ROOT="$LTR_ROOT/sources/ood-pinned"
WORK_DIR="$LTR_ROOT/workload-build"
REPLAY_PIDFILE="$WORK_DIR/ood-replay.pid"
REPLAY_STARTFILE="$REPLAY_PIDFILE.starttime"
REPLAY_LOG="$WORK_DIR/ood-replay.log"

ID_INPUT="${ID_INPUT:-$ARTIFACTS/tier2-toolace-sample-6000.jsonl}"
ID_MANIFEST="${ID_MANIFEST:-$ARTIFACTS/tier2-sample-manifest.json}"
ID_LEDGER="${ID_LEDGER:-$ARTIFACTS/tier2-toolace-6000-ledger.jsonl}"
BFCL_INPUT="$ARTIFACTS/bfcl-label-inputs.jsonl"
BFCL_MANIFEST="$ARTIFACTS/bfcl-label-inputs.manifest.json"
TOOLATHLON_INPUT="$ARTIFACTS/toolathlon-label-inputs.jsonl"
TOOLATHLON_MANIFEST="$ARTIFACTS/toolathlon-label-inputs.manifest.json"
OOD_INPUT="$ARTIFACTS/ood-label-inputs.jsonl"
OOD_LEDGER="$ARTIFACTS/ood-label-ledger.jsonl"
OOD_REPLAY_REPORT="$ARTIFACTS/ood-label-replay-report.json"
LENGTHS="$ARTIFACTS/combined-lengths.jsonl"
MIXED_WORKLOAD="$ARTIFACTS/mixed.v2.jsonl"
MIXED_MANIFEST="$ARTIFACTS/mixed.v2.manifest.json"
OOD_WORKLOAD="$ARTIFACTS/ood.v2.jsonl"
OOD_WORKLOAD_MANIFEST="$ARTIFACTS/ood.v2.manifest.json"
VERIFICATION_REPORT="$ARTIFACTS/server-workloads.verification.json"

require_file() {
  local path="$1"
  [[ -f "$path" ]] || { echo "required file missing: $path" >&2; exit 1; }
}

require_executable() {
  local path="$1"
  [[ -x "$path" ]] || { echo "required executable missing: $path" >&2; exit 1; }
}

validate_live_replay_pid() {
  local pid="$1"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  [[ -r "/proc/$pid/stat" && -r "/proc/$pid/cmdline" ]] || return 1
  local saved_start current_start cmdline
  saved_start="$(<"$REPLAY_STARTFILE")"
  current_start="$(awk '{print $22}' "/proc/$pid/stat")"
  cmdline="$(tr '\0' ' ' <"/proc/$pid/cmdline")"
  [[ "$saved_start" =~ ^[0-9]+$ && "$saved_start" == "$current_start" ]] || return 1
  [[ "$cmdline" == *"scripts/replay_tier2_labels.py"* ]] || return 1
  [[ "$cmdline" == *"$OOD_LEDGER"* ]] || return 1
}

wait_for_replay() {
  local pid="$1"
  while kill -0 "$pid" 2>/dev/null; do
    if ! validate_live_replay_pid "$pid"; then
      echo "NO-GO: OOD replay PID identity changed while running" >&2
      exit 1
    fi
    sleep 10
  done
  rm -f "$REPLAY_PIDFILE" "$REPLAY_STARTFILE"
}

require_executable "$VENV/bin/python"
require_executable "$VENV/bin/hf"
require_file "$DECLARATIONS"
require_file "$HELPER"
require_file "$REPO_ROOT/scripts/build_ood_label_inputs.py"
require_file "$REPO_ROOT/scripts/replay_tier2_labels.py"
require_file "$REPO_ROOT/scripts/build_offline_workload.py"
require_file "$ID_INPUT"
require_file "$ID_MANIFEST"
require_file "$ID_LEDGER"
curl -fsS "$VLLM_HEALTH_ENDPOINT" >/dev/null || {
  echo "NO-GO: labeling vLLM is not healthy at $VLLM_HEALTH_ENDPOINT" >&2
  exit 1
}
mkdir -p "$SOURCE_ROOT" "$WORK_DIR" "$ARTIFACTS"

mapfile -t source_pin < <("$VENV/bin/python" - "$DECLARATIONS" <<'PY'
import json
import sys

declarations = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {
    "bfcl": {
        "repository": "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        "revision": "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda",
    },
    "toolathlon": {
        "repository": "hkust-nlp/Toolathlon-Trajectories",
        "revision": "6194034105bc27fa438447172be0e7b4e35396e4",
    },
}
if declarations != expected:
    raise SystemExit("NO-GO: source-declarations.json does not match frozen OOD pins")
for source in ("bfcl", "toolathlon"):
    print(declarations[source]["repository"])
    print(declarations[source]["revision"])
PY
)
[[ "${#source_pin[@]}" -eq 4 ]] || { echo "NO-GO: failed to read OOD source pins" >&2; exit 1; }
BFCL_REPO="${source_pin[0]}"
BFCL_REVISION="${source_pin[1]}"
TOOLATHLON_REPO="${source_pin[2]}"
TOOLATHLON_REVISION="${source_pin[3]}"
BFCL_SNAPSHOT="$SOURCE_ROOT/bfcl-$BFCL_REVISION"
TOOLATHLON_SNAPSHOT="$SOURCE_ROOT/toolathlon-$TOOLATHLON_REVISION"

HF_ENDPOINT="$HF_ENDPOINT" HF_HOME="$HF_HOME" \
  "$VENV/bin/hf" download "$BFCL_REPO" --repo-type dataset \
  --revision "$BFCL_REVISION" --local-dir "$BFCL_SNAPSHOT"
HF_ENDPOINT="$HF_ENDPOINT" HF_HOME="$HF_HOME" \
  "$VENV/bin/hf" download "$TOOLATHLON_REPO" --repo-type dataset \
  --revision "$TOOLATHLON_REVISION" --local-dir "$TOOLATHLON_SNAPSHOT"

BFCL_RAW="$WORK_DIR/bfcl-pinned-raw.jsonl"
TOOLATHLON_RAW="$WORK_DIR/toolathlon-pinned-raw.jsonl"
"$VENV/bin/python" "$HELPER" materialize-source \
  --source bfcl --snapshot "$BFCL_SNAPSHOT" --output "$BFCL_RAW"
"$VENV/bin/python" "$HELPER" materialize-source \
  --source toolathlon --snapshot "$TOOLATHLON_SNAPSHOT" \
  --output "$TOOLATHLON_RAW"

BFCL_FUNCTION_DOCS="$BFCL_SNAPSHOT/multi_turn_func_doc"
[[ -d "$BFCL_FUNCTION_DOCS" ]] || {
  echo "NO-GO: BFCL multi_turn_func_doc missing from pinned snapshot" >&2
  exit 1
}
PYTHONPATH="$REPO_ROOT" "$VENV/bin/python" \
  "$REPO_ROOT/scripts/build_ood_label_inputs.py" \
  --source bfcl --input "$BFCL_RAW" --function-docs "$BFCL_FUNCTION_DOCS" \
  --category pinned-bfcl --sample-size "$OOD_SAMPLE_SIZE" \
  --seed "$OOD_SAMPLE_SEED" --output "$BFCL_INPUT" \
  --manifest "$BFCL_MANIFEST" --source-declarations "$DECLARATIONS"
PYTHONPATH="$REPO_ROOT" "$VENV/bin/python" \
  "$REPO_ROOT/scripts/build_ood_label_inputs.py" \
  --source toolathlon --input "$TOOLATHLON_RAW" \
  --category pinned-toolathlon --sample-size "$OOD_SAMPLE_SIZE" \
  --seed "$OOD_SAMPLE_SEED" --output "$TOOLATHLON_INPUT" \
  --manifest "$TOOLATHLON_MANIFEST" --source-declarations "$DECLARATIONS"

"$VENV/bin/python" "$HELPER" combine-label-inputs \
  --bfcl-input "$BFCL_INPUT" --bfcl-manifest "$BFCL_MANIFEST" \
  --toolathlon-input "$TOOLATHLON_INPUT" \
  --toolathlon-manifest "$TOOLATHLON_MANIFEST" \
  --expected-per-source "$OOD_SAMPLE_SIZE" --output "$OOD_INPUT"

if [[ -f "$REPLAY_PIDFILE" || -f "$REPLAY_STARTFILE" ]]; then
  [[ -f "$REPLAY_PIDFILE" && -f "$REPLAY_STARTFILE" ]] || {
    echo "NO-GO: incomplete OOD replay PID metadata" >&2
    exit 1
  }
  replay_pid="$(<"$REPLAY_PIDFILE")"
  if validate_live_replay_pid "$replay_pid"; then
    echo "OOD replay already active; waiting for pid=$replay_pid"
    wait_for_replay "$replay_pid"
  else
    rm -f "$REPLAY_PIDFILE" "$REPLAY_STARTFILE"
  fi
fi

if [[ ! -f "$OOD_REPLAY_REPORT" ]] || ! "$VENV/bin/python" "$HELPER" merge-lengths \
  --id-input "$ID_INPUT" --id-ledger "$ID_LEDGER" \
  --ood-input "$OOD_INPUT" --ood-ledger "$OOD_LEDGER" \
  --output "$LENGTHS" >/dev/null 2>&1; then
  echo "starting/resuming OOD labeling against direct vLLM"
  PYTHONPATH="$REPO_ROOT" nohup "$VENV/bin/python" \
    "$REPO_ROOT/scripts/replay_tier2_labels.py" \
    --labels "$OOD_INPUT" --ledger "$OOD_LEDGER" \
    --report "$OOD_REPLAY_REPORT" --endpoint "$VLLM_ENDPOINT" \
    --model "$MODEL" --max-tokens 4096 \
    --concurrency "$OOD_REPLAY_CONCURRENCY" \
    >>"$REPLAY_LOG" 2>&1 &
  replay_pid=$!
  printf '%s\n' "$replay_pid" >"$REPLAY_PIDFILE"
  if ! awk '{print $22}' "/proc/$replay_pid/stat" >"$REPLAY_STARTFILE"; then
    echo "NO-GO: could not record OOD replay process identity" >&2
    exit 1
  fi
  set +e
  wait "$replay_pid"
  replay_status=$?
  set -e
  rm -f "$REPLAY_PIDFILE" "$REPLAY_STARTFILE"
  if (( replay_status != 0 )); then
    tail -100 "$REPLAY_LOG" >&2 || true
    echo "NO-GO: OOD replay exited with status $replay_status" >&2
    exit 1
  fi
fi

require_file "$OOD_REPLAY_REPORT"
"$VENV/bin/python" "$HELPER" merge-lengths \
  --id-input "$ID_INPUT" --id-ledger "$ID_LEDGER" \
  --ood-input "$OOD_INPUT" --ood-ledger "$OOD_LEDGER" \
  --output "$LENGTHS"

build_workload() {
  local profile="$1"
  local output="$2"
  local manifest="$3"
  PYTHONPATH="$REPO_ROOT" "$VENV/bin/python" \
    "$REPO_ROOT/scripts/build_offline_workload.py" \
    --profile "$profile" \
    --id-input "$ID_INPUT" --id-manifest "$ID_MANIFEST" --id-split test \
    --ood-input "$OOD_INPUT" --lengths "$LENGTHS" --per-token-ms 2.5 \
    --ood-ratio "$OOD_RATIO" --seed "$WORKLOAD_SEED" \
    --output "$output" --manifest "$manifest"
}

build_workload mixed "$MIXED_WORKLOAD" "$MIXED_MANIFEST"
build_workload ood "$OOD_WORKLOAD" "$OOD_WORKLOAD_MANIFEST"

verification_tmp="$WORK_DIR/server-workloads.verification.json.partial"
"$VENV/bin/python" "$HELPER" verify-workloads \
  --mixed "$MIXED_WORKLOAD" --mixed-manifest "$MIXED_MANIFEST" \
  --ood "$OOD_WORKLOAD" --ood-manifest "$OOD_WORKLOAD_MANIFEST" \
  --ood-input "$OOD_INPUT" --lengths "$LENGTHS" \
  >"$verification_tmp"
mv "$verification_tmp" "$VERIFICATION_REPORT"
echo "server workloads ready: $MIXED_WORKLOAD $OOD_WORKLOAD"
