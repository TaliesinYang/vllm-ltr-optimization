#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/hy-tmp/hf
export HF_HUB_CACHE=/hy-tmp/hf
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR=/hy-tmp/tmp
export XDG_CACHE_HOME=/hy-tmp/xdg-cache
export TRITON_CACHE_DIR=/hy-tmp/triton-cache
export TORCHINDUCTOR_CACHE_DIR=/hy-tmp/torchinductor-cache
export VLLM_CACHE_ROOT=/hy-tmp/vllm-cache

source /hy-tmp/venvs/vllm-ltr/bin/activate
cd /hy-tmp/staging

python scripts/replay_censored_diagnostics.py \
  --source /hy-tmp/staging/tier1/toolace-6bda777-qwen35.jsonl \
  --pilot-ledger /hy-tmp/results/tier2-toolace-pilot-400.jsonl \
  --output-dir /hy-tmp/results/censored-texts \
  --report /hy-tmp/results/censored-texts-report.json \
  > /hy-tmp/logs/censored-texts.log 2>&1 &
DIAGNOSTIC_PID=$!

python scripts/replay_tier2_labels.py \
  --labels /hy-tmp/results/tier2-toolace-sample-6000.jsonl \
  --ledger /hy-tmp/results/tier2-toolace-6000-ledger.jsonl \
  --report /hy-tmp/results/tier2-toolace-6000-report.json \
  --concurrency 8 \
  > /hy-tmp/logs/tier2-toolace-6000.log 2>&1

wait "$DIAGNOSTIC_PID"

VLLM_PID=$(cat /hy-tmp/logs/vllm-serve.pid)
VLLM_CMD=$(tr '\0' ' ' < "/proc/$VLLM_PID/cmdline")
case "$VLLM_CMD" in
  *vllm*serve*Qwen/Qwen3.5-9B*) kill "$VLLM_PID" ;;
  *) echo "refusing unexpected vLLM process: $VLLM_CMD" >&2; exit 1 ;;
esac
for _ in $(seq 1 120); do
  kill -0 "$VLLM_PID" 2>/dev/null || break
  sleep 1
done
if kill -0 "$VLLM_PID" 2>/dev/null; then
  echo "vLLM did not stop before Tier-2 training" >&2
  exit 1
fi

python scripts/run_tier2_matrix.py \
  --sample /hy-tmp/results/tier2-toolace-sample-6000.jsonl \
  --ledger /hy-tmp/results/tier2-toolace-6000-ledger.jsonl \
  --tier1-results-dir /hy-tmp/results/tier1-matrix \
  --tier1-config-dir /hy-tmp/results/configs \
  --work-dir /hy-tmp/work/tier2 \
  --results-dir /hy-tmp/results \
  > /hy-tmp/logs/tier2-matrix-and-learning-curve.log 2>&1

sha256sum \
  /hy-tmp/results/tier2-toolace-6000-ledger.jsonl \
  /hy-tmp/results/tier2-matrix-summary.json \
  /hy-tmp/results/tier2-learning-curve.json \
  > /hy-tmp/results/tier2-final-artifacts.sha256
