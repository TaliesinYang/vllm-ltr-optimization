#!/usr/bin/env bash
# 恒源云: run the baseline sweep, collect results, upload to OSS, then shutdown (stop billing).
# Run INSIDE tmux so it survives SSH disconnect. Run `oss login` FIRST (for the upload).
#   tmux new -s run ; oss login ; bash hy_run_and_upload.sh <run-id>
set -euo pipefail

RUN="${1:-baseline-$(date -u +%Y%m%d-%H%M)}"
export FORK_DIR="${FORK_DIR:-/hy-tmp/vllm-ltr}"
REPO=/hy-tmp/vllm-ltr-optimization

echo "=== run baseline sweep (FCFS / LTR / classification) ==="
bash "$REPO/scripts/run_baseline.sh"

echo "=== collect + manifest (persists to \$HOME/vllm-ltr-results) ==="
bash "$REPO/scripts/collect_results.sh" "$RUN"

echo "=== pack everything + upload to OSS (skill pattern) ==="
PACK="/hy-tmp/${RUN}-all.tar.gz"
tar czf "$PACK" -C "$HOME" vllm-ltr-results
oss cp "$PACK" oss://backup/        # set -e: if this fails, we DO NOT reach shutdown below
echo "Uploaded -> oss://backup/$(basename "$PACK")"

echo "=== upload OK -> shutting down to stop billing ==="
echo "Retrieve locally with:  oss login && oss cp oss://backup/$(basename "$PACK") . && tar xzf $(basename "$PACK")"
shutdown
