#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
T7_ROOT="${T7_ROOT:-/Volumes/T7 Shield/vllm-ltr-results}"
OSS_PREFIX="${OSS_PREFIX:-oss://backup/vllm-ltr-rental-readiness}"
MANIFEST="${MANIFEST:-$REPO_ROOT/scripts/server/manifest/oss-objects.json}"
MIXED_WORKLOAD="${MIXED_WORKLOAD:-$REPO_ROOT/runs/workloads/mixed.v2.jsonl}"
OOD_WORKLOAD="${OOD_WORKLOAD:-$REPO_ROOT/runs/workloads/ood.v2.jsonl}"
WORKLOAD_MANIFEST_DIR="${WORKLOAD_MANIFEST_DIR:-$REPO_ROOT/runs/workloads/manifests}"
NORMALIZED_CHECKPOINT_DIR="${NORMALIZED_CHECKPOINT_DIR:-$T7_ROOT/extracted}"
CHECKPOINT_TAR="$T7_ROOT/tier2-checkpoints.tar"
RESULTS_TAR="$T7_ROOT/tier2-results.tar.gz"
BUNDLE="$T7_ROOT/benchmark-bundle.tar.gz"
STAGING="${STAGING:-$(mktemp -d "$T7_ROOT/repack-staging.XXXXXX")}"

for path in "$CHECKPOINT_TAR" "$RESULTS_TAR" "$MIXED_WORKLOAD" "$OOD_WORKLOAD"; do
  [[ -f "$path" ]] || { echo "missing required input: $path" >&2; exit 1; }
done
[[ -d "$WORKLOAD_MANIFEST_DIR" ]] || { echo "missing workload manifests: $WORKLOAD_MANIFEST_DIR" >&2; exit 1; }
command -v oss >/dev/null || { echo "oss CLI is required" >&2; exit 1; }

[[ "$STAGING" == "$T7_ROOT"/repack-staging.* ]] || { echo "STAGING must be a dedicated directory under $T7_ROOT" >&2; exit 1; }
[[ "$NORMALIZED_CHECKPOINT_DIR" == "$T7_ROOT"/* ]] || { echo "NORMALIZED_CHECKPOINT_DIR must be under $T7_ROOT" >&2; exit 1; }
mkdir -p "$STAGING/checkpoints-raw" "$STAGING/results-raw" "$STAGING/bundle/workload-manifests" "$NORMALIZED_CHECKPOINT_DIR"
tar -tf "$CHECKPOINT_TAR" >"$STAGING/checkpoint-members.txt"
tar -tzf "$RESULTS_TAR" >"$STAGING/results-members.txt"
tar -xf "$CHECKPOINT_TAR" -C "$STAGING/checkpoints-raw"

for seed in 17 42 73; do
  source_dir="$(find "$STAGING/checkpoints-raw" -type d -path "*/bert-prompt_schema-tier2-seed${seed}/final" -print -quit)"
  [[ -n "$source_dir" ]] || { echo "checkpoint seed $seed not found" >&2; exit 1; }
  [[ -f "$source_dir/config.json" ]] || { echo "checkpoint seed $seed has no config.json" >&2; exit 1; }
  target="checkpoints_best_predictor"
  [[ "$seed" == 17 ]] || target="checkpoints_best_predictor_seed${seed}"
  target_dir="$NORMALIZED_CHECKPOINT_DIR/$target"
  partial_target="$NORMALIZED_CHECKPOINT_DIR/.${target}.partial-$$"
  rm -rf "$partial_target"
  cp -R "$source_dir" "$partial_target"
  rm -rf "$target_dir"
  mv "$partial_target" "$target_dir"
done

tar -xzf "$RESULTS_TAR" -C "$STAGING/results-raw"
ledger="$(find "$STAGING/results-raw" -type f -name 'tier2-toolace-6000-ledger.jsonl' -print -quit)"
sample_manifest="$(find "$STAGING/results-raw" -type f -name 'tier2-sample-manifest.json' -print -quit)"
[[ -n "$ledger" && -n "$sample_manifest" ]] || { echo "ledger/sample manifest missing from results tar" >&2; exit 1; }
cp "$ledger" "$STAGING/bundle/tier2-toolace-6000-ledger.jsonl"
cp "$sample_manifest" "$STAGING/bundle/tier2-sample-manifest.json"
cp "$MIXED_WORKLOAD" "$STAGING/bundle/mixed.v2.jsonl"
cp "$OOD_WORKLOAD" "$STAGING/bundle/ood.v2.jsonl"
cp -R "$WORKLOAD_MANIFEST_DIR"/. "$STAGING/bundle/workload-manifests/"

python3 - "$STAGING/bundle/tier2-toolace-6000-ledger.jsonl" "$STAGING/ledger-inventory.json" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path
rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
latest = {}
for row in rows:
    latest[str(row["sample_id"])] = row
counts = Counter(str(row.get("status")) for row in latest.values())
if len(latest) != 6000 or counts != Counter({"ok": 5997, "error": 3}):
    raise SystemExit(f"unexpected ledger inventory: unique={len(latest)} statuses={dict(counts)}")
Path(sys.argv[2]).write_text(json.dumps({"raw_rows": len(rows), "unique_sample_ids": len(latest), "latest_status_counts": dict(counts)}, indent=2) + "\n")
PY
cp "$STAGING/ledger-inventory.json" "$STAGING/bundle/ledger-inventory.json"

tar -czf "$BUNDLE" -C "$STAGING/bundle" .

python3 - "$STAGING/checkpoint-members.txt" "$STAGING/results-members.txt" "$BUNDLE" "$STAGING/archive-members.json" <<'PY'
import json, re, subprocess, sys
from pathlib import Path
checkpoint_lines = Path(sys.argv[1]).read_text().splitlines()
result_lines = Path(sys.argv[2]).read_text().splitlines()
checkpoint_members = []
for seed in (17, 42, 73):
    pattern = re.compile(rf"(^|/)bert-prompt_schema-tier2-seed{seed}/final(?:/|$)")
    matches = [line for line in checkpoint_lines if pattern.search(line)]
    if not matches:
        raise SystemExit(f"checkpoint archive has no seed {seed} final member")
    match = matches[0]
    checkpoint_members.append(match[:pattern.search(match).end()].rstrip("/") + "/")
result_members = []
for filename in ("tier2-toolace-6000-ledger.jsonl", "tier2-sample-manifest.json"):
    matches = [line for line in result_lines if line.rstrip("/").endswith("/" + filename) or line.rstrip("/") == filename]
    if len(matches) != 1:
        raise SystemExit(f"expected one concrete {filename} member, got {matches}")
    result_members.append(matches[0])
bundle_members = [line for line in subprocess.check_output(["tar", "-tzf", sys.argv[3]], text=True).splitlines() if line not in ("./", "")]
Path(sys.argv[4]).write_text(json.dumps({
    "tier2-checkpoints.tar": checkpoint_members,
    "tier2-results.tar.gz": result_members,
    "benchmark-bundle.tar.gz": bundle_members,
}, indent=2) + "\n")
PY

python3 - "$MANIFEST" "$CHECKPOINT_TAR" "$RESULTS_TAR" "$BUNDLE" "$OSS_PREFIX" "$STAGING/archive-members.json" <<'PY'
import datetime, hashlib, json, os, sys
from pathlib import Path
manifest_path = Path(sys.argv[1])
paths = {Path(value).name: Path(value) for value in sys.argv[2:5]}
prefix = sys.argv[5].rstrip("/")
archive_members = json.loads(Path(sys.argv[6]).read_text())
payload = json.loads(manifest_path.read_text())
for obj in payload["objects"]:
    path = paths[obj["name"]]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    obj.update(source_path=str(path), oss_uri=f"{prefix}/{path.name}", size_bytes=os.stat(path).st_size, sha256=digest.hexdigest(), unpacks_to=archive_members[obj["name"]])
payload["generated_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
temporary = manifest_path.with_suffix(".json.partial")
temporary.write_text(json.dumps(payload, indent=2) + "\n")
temporary.replace(manifest_path)
PY

python3 - "$MANIFEST" <<'PY' | while IFS=$'\t' read -r source uri; do
import json, sys
for obj in json.load(open(sys.argv[1]))["objects"]:
    if not obj.get("sha256") or not obj.get("size_bytes"):
        raise SystemExit(f"manifest object is incomplete: {obj['name']}")
    print(f"{obj['source_path']}\t{obj['oss_uri']}")
PY
  oss cp "$source" "$uri"
  oss ls "$uri" >/dev/null
done

echo "inventory, repack, manifest fill, and OSS readback completed"
