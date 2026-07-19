#!/usr/bin/env bash
set -euo pipefail

LTR_ROOT="${LTR_ROOT:-/hy-tmp/ltr}"
REPO_ROOT="${REPO_ROOT:-$LTR_ROOT/repo}"
VENV="${VENV:-$LTR_ROOT/venv}"
MANIFEST="${MANIFEST:-$REPO_ROOT/scripts/server/manifest/oss-objects.json}"
CACHE_DIR="$LTR_ROOT/downloads"
ARTIFACTS_DIR="$LTR_ROOT/artifacts"
LM_CACHE_OSS_URI="${LM_CACHE_OSS_URI:-oss://lmcache-labels.tar.gz}"
PREPARE_QUANTILES="${PREPARE_QUANTILES:-0}"
HF_CACHE="${HF_HOME:-/hy-tmp/hf}"

command -v oss >/dev/null || { echo "oss CLI is required" >&2; exit 1; }
[[ -f "$MANIFEST" ]] || { echo "manifest missing: $MANIFEST" >&2; exit 1; }
mkdir -p "$CACHE_DIR" "$ARTIFACTS_DIR"

sha256_file() {
  python3 - "$1" <<'PY'
import hashlib, sys
h = hashlib.sha256()
with open(sys.argv[1], "rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        h.update(chunk)
print(h.hexdigest())
PY
}

mapfile -t object_rows < <(python3 - "$MANIFEST" <<'PY'
import json, sys
for obj in json.load(open(sys.argv[1]))["objects"]:
    sha = obj.get("sha256")
    size = obj.get("size_bytes")
    members = obj.get("unpacks_to")
    if not isinstance(sha, str) or len(sha) != 64 or not isinstance(size, int) or size <= 0 or not isinstance(members, list) or not members or not all(isinstance(item, str) and item for item in members):
        raise SystemExit(f"manifest object has unfilled placeholders: {obj['name']}")
    print("\t".join((obj["name"], obj["oss_uri"], sha, str(size), obj["archive_format"])))
PY
)

for row in "${object_rows[@]}"; do
  IFS=$'\t' read -r name uri _sha _size _format <<<"$row"
  oss ls "$uri" >/dev/null || { echo "OSS preflight failed: $uri" >&2; exit 1; }
done

for row in "${object_rows[@]}"; do
  IFS=$'\t' read -r name uri sha size _format <<<"$row"
  destination="$CACHE_DIR/$name"
  cache_valid=0
  if [[ -f "$destination" ]] && [[ "$(stat -c %s "$destination")" == "$size" ]]; then
    [[ "$(sha256_file "$destination")" == "$sha" ]] && cache_valid=1
  fi
  if [[ "$cache_valid" != 1 ]]; then
    partial="$destination.partial-$$"
    rm -f "$partial"
    oss cp "$uri" "$partial"
    [[ "$(stat -c %s "$partial")" == "$size" ]] || { echo "size mismatch for $name" >&2; exit 1; }
    actual="$(sha256_file "$partial")"
    [[ "$actual" == "$sha" ]] || { echo "sha256 mismatch for $name" >&2; exit 1; }
    mv -f "$partial" "$destination"
  fi
done

release_id="$(python3 - "$MANIFEST" <<'PY'
import hashlib, json, sys
objects = json.load(open(sys.argv[1]))["objects"]
print(hashlib.sha256("".join(obj["sha256"] for obj in objects).encode()).hexdigest()[:8])
PY
)"
release="$ARTIFACTS_DIR/$release_id"
if [[ ! -f "$release/.restored" ]]; then
  partial_release="$ARTIFACTS_DIR/.partial-$release_id-$$"
  rm -rf "$partial_release"
  mkdir -p "$partial_release"
  for row in "${object_rows[@]}"; do
    IFS=$'\t' read -r name _uri _sha _size format <<<"$row"
    case "$format" in
      tar) tar -xf "$CACHE_DIR/$name" -C "$partial_release" ;;
      tar.gz) tar -xzf "$CACHE_DIR/$name" -C "$partial_release" ;;
      *) echo "unsupported archive format: $format" >&2; exit 1 ;;
    esac
  done
  for seed in 17 42 73; do
    checkpoint="$(find "$partial_release" -type d -path "*/bert-prompt_schema-tier2-seed${seed}/final" -print -quit)"
    if [[ -z "$checkpoint" ]]; then
      target="checkpoints_best_predictor"
      [[ "$seed" == 17 ]] || target="checkpoints_best_predictor_seed${seed}"
      checkpoint="$(find "$partial_release" -type d -name "$target" -print -quit)"
    fi
    [[ -n "$checkpoint" ]] || { echo "checkpoint seed $seed missing after restore" >&2; exit 1; }
    target="checkpoints_best_predictor"
    [[ "$seed" == 17 ]] || target="checkpoints_best_predictor_seed${seed}"
    [[ "$checkpoint" == "$partial_release/$target" ]] || cp -R "$checkpoint" "$partial_release/$target"
  done
  ledger="$(find "$partial_release" -type f -name 'tier2-toolace-6000-ledger.jsonl' -print -quit)"
  reference="$(find "$partial_release" -type f -name 'tier2-sample-manifest.json' -print -quit)"
  [[ -n "$ledger" && -n "$reference" ]] || { echo "ledger/reference manifest missing" >&2; exit 1; }
  [[ "$ledger" == "$partial_release/tier2-toolace-6000-ledger.jsonl" ]] || cp "$ledger" "$partial_release/tier2-toolace-6000-ledger.jsonl"
  [[ "$reference" == "$partial_release/tier2-sample-manifest.json" ]] || cp "$reference" "$partial_release/tier2-sample-manifest.json"
  touch "$partial_release/.restored"
  mv "$partial_release" "$release"
fi
ln -sfn "$release" "$ARTIFACTS_DIR/current"

source_path="$(find "$release" -type f -name 'toolace-6bda777-qwen35.jsonl' -print -quit)"
if [[ -z "$source_path" ]]; then
  lmcache_tar="$CACHE_DIR/lmcache-labels.tar.gz"
  if [[ ! -f "$lmcache_tar" ]]; then
    if oss ls "$LM_CACHE_OSS_URI" >/dev/null 2>&1; then
      oss cp "$LM_CACHE_OSS_URI" "$lmcache_tar.partial-$$"
      mv "$lmcache_tar.partial-$$" "$lmcache_tar"
    else
      echo "tier1 OSS source absent; rebuilding from pinned Team-ACE/ToolACE snapshot"
    fi
  fi
  if [[ -f "$lmcache_tar" ]]; then
    mkdir -p "$release/lmcache-labels"
    tar -xzf "$lmcache_tar" -C "$release/lmcache-labels"
  fi
  source_path="$(find "$release" -type f -name 'toolace-6bda777-qwen35.jsonl' -print -quit)"
fi
if [[ -z "$source_path" ]]; then
  toolace_dir="$release/toolace-6bda777c88d21e5a204703c1ee45597a8fa4f734"
  snapshot="$toolace_dir/data.json"
  mkdir -p "$toolace_dir"
  HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" HF_HOME="$HF_CACHE" \
    "$VENV/bin/python" - "$toolace_dir" <<'PY'
import sys
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="Team-ACE/ToolACE",
    repo_type="dataset",
    revision="6bda777c88d21e5a204703c1ee45597a8fa4f734",
    filename="data.json",
    local_dir=sys.argv[1],
)
print(path)
PY
  snapshot_sha="$(sha256_file "$snapshot")"
  [[ "$snapshot_sha" == "ba12c083fca7e8da48c67ad5b895e495447da7c66e39a2e19742c082e6cb537e" ]] || {
    echo "NO-GO: pinned ToolACE data.json sha256 mismatch: $snapshot_sha" >&2
    exit 1
  }
  PYTHONPATH="$REPO_ROOT" HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" HF_HOME="$HF_CACHE" \
    "$VENV/bin/python" "$REPO_ROOT/scripts/extract_tier1_labels.py" \
    --source toolace --toolace-snapshot "$snapshot" \
    --output "$release/toolace-6bda777-qwen35.jsonl" --cache-dir "$HF_CACHE"
  source_path="$release/toolace-6bda777-qwen35.jsonl"
fi
[[ -n "$source_path" ]] || { echo "NO-GO: pinned tier1 source still missing" >&2; exit 1; }

"$VENV/bin/python" "$REPO_ROOT/scripts/server/rebuild_tier2_sample.py" \
  --source "$source_path" \
  --reference-manifest "$release/tier2-sample-manifest.json" \
  --output "$release/tier2-toolace-sample-6000.jsonl" --seed 42

PYTHONPATH="$REPO_ROOT" "$VENV/bin/python" - "$release" <<'PY'
import sys
from pathlib import Path
from scheduler_benchmark.predictor import BertPredictor, PredictorInput
root = Path(sys.argv[1])
for name in ("checkpoints_best_predictor", "checkpoints_best_predictor_seed42", "checkpoints_best_predictor_seed73"):
    prediction = BertPredictor(root / name).predict(PredictorInput("smoke", (), {"prompt_text": "hello", "tool_schema_text": "[]"}))
    if not 0.0 <= prediction.score <= 1.0:
        raise SystemExit(f"checkpoint smoke failed: {name}")
PY

if [[ "$PREPARE_QUANTILES" == 1 ]]; then
  "$VENV/bin/python" "$REPO_ROOT/scripts/replay_tier2_labels.py" \
    --labels "$release/tier2-toolace-sample-6000.jsonl" \
    --ledger "$release/tier2-toolace-6000-ledger.jsonl" \
    --report "$release/tier2-replay-repair-report.json" \
    --endpoint "${VLLM_CHAT_ENDPOINT:-http://127.0.0.1:8000/v1/chat/completions}" \
    --model qwen3.5-9b --max-tokens 4096 --concurrency "${REPLAY_CONCURRENCY:-1}"
  # Derive structural exclusions: ONLY deterministic context-length 400s
  # (prompt + frozen max_tokens 4096 > frozen max-model-len 8192) qualify.
  # Any other lingering failure is a hard NO-GO, not an exclusion.
  "$VENV/bin/python" - "$release" <<'EXCL_PY'
import json, sys
release = sys.argv[1]
latest = {}
for line in open(f"{release}/tier2-toolace-6000-ledger.jsonl"):
    row = json.loads(line)
    latest[str(row.get("sample_id", ""))] = row
# Ledger only stores the terse HTTPError line; re-probe each failing row
# against the LIVE engine and read the 400 body. Only a verified
# context-length violation qualifies as structural.
import urllib.request, urllib.error
sys.path.insert(0, "/hy-tmp/ltr/repo")
from ltr_training.tier2 import build_request

samples = {}
for line in open(f"{release}/tier2-toolace-sample-6000.jsonl"):
    row = json.loads(line)
    samples[str(row["sample_id"])] = row

exclusions = []
for sample_id, row in latest.items():
    if row.get("status") == "ok":
        continue
    request_body = json.dumps(
        build_request(samples[sample_id], model="qwen3.5-9b")
    ).encode()
    try:
        urllib.request.urlopen(
            urllib.request.Request(
                "http://127.0.0.1:8000/v1/chat/completions",
                data=request_body,
                headers={"Content-Type": "application/json"},
            ),
            timeout=120,
        )
        raise SystemExit(
            f"NO-GO: {sample_id} succeeded on re-probe; it is labelable, not structural"
        )
    except urllib.error.HTTPError as probe:
        body = probe.read().decode(errors="replace")
        if probe.code != 400 or "maximum context length" not in body:
            raise SystemExit(
                f"NO-GO: non-structural failure for {sample_id}: HTTP {probe.code} {body[:160]}"
            )
        exclusions.append(
            {
                "sample_id": sample_id,
                "reason": "prompt tokens + frozen max_tokens 4096 exceed frozen max-model-len 8192",
                "http_status": 400,
                "error_snippet": body[:200],
            }
        )
if len(exclusions) > 5:
    raise SystemExit(f"NO-GO: too many structural exclusions: {len(exclusions)}")
with open(f"{release}/structural-exclusions.json", "w") as handle:
    json.dump(sorted(exclusions, key=lambda e: e["sample_id"]), handle, indent=2)
print(f"structural exclusions: {len(exclusions)}")
EXCL_PY
  exclusion_count="$("$VENV/bin/python" -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$release/structural-exclusions.json")"
  "$VENV/bin/python" "$REPO_ROOT/scripts/server/merge_quantile_labels.py" \
    --samples "$release/tier2-toolace-sample-6000.jsonl" \
    --ledger "$release/tier2-toolace-6000-ledger.jsonl" \
    --structural-exclusions "$release/structural-exclusions.json" \
    --output "$release/labels-merged-6k.jsonl"
  "$VENV/bin/python" "$REPO_ROOT/scripts/build_rank_quantiles.py" \
    --labels "$release/labels-merged-6k.jsonl" \
    --checkpoint "$release/checkpoints_best_predictor" \
    --sidecar-output "$release/replay-sidecar.jsonl" \
    --manifest-output "$release/rank_quantiles.json" \
    --model-version bert-prompt_schema-tier2-seed17 \
    --expected-count "$((6000 - exclusion_count))" \
    --structural-exclusions "$release/structural-exclusions.json"
fi

echo "artifacts restored at $release (current symlink updated)"
