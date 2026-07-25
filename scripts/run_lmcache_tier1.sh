#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/hy-tmp/hf
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export TMPDIR=/hy-tmp/tmp
export XDG_CACHE_HOME=/hy-tmp/xdg-cache

source /hy-tmp/venvs/vllm-ltr/bin/activate
mkdir -p /hy-tmp/work/tier1 /hy-tmp/results
work_output=/hy-tmp/work/tier1/lmcache-6e043b9-full.jsonl
result_output=/hy-tmp/results/lmcache-6e043b9-full.jsonl

python /hy-tmp/staging/scripts/extract_tier1_labels.py \
  --source lmcache \
  --output "$work_output" \
  --cache-dir /hy-tmp/hf

row_count=$(wc -l < "$work_output")
if [[ "$row_count" -ne 24880 ]]; then
  echo "expected 24880 LMCache rows, got $row_count" >&2
  exit 1
fi
install -m 0644 "$work_output" "$result_output"
sha256sum "$result_output"
