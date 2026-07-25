#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/hy-tmp/hf
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export TMPDIR=/hy-tmp/tmp
export XDG_CACHE_HOME=/hy-tmp/xdg-cache
export TRITON_CACHE_DIR=/hy-tmp/triton-cache
export TORCHINDUCTOR_CACHE_DIR=/hy-tmp/torchinductor-cache
export VLLM_CACHE_ROOT=/hy-tmp/vllm-cache

source /hy-tmp/venvs/vllm-ltr/bin/activate
mkdir -p /hy-tmp/work/tier1-matrix /hy-tmp/results /hy-tmp/logs \
  "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT"

nohup bash /hy-tmp/staging/scripts/download_qwen.sh \
  > /hy-tmp/logs/qwen-download.log 2>&1 < /dev/null &
qwen_download_pid=$!

python /hy-tmp/staging/scripts/run_tier1_matrix.py \
  --base-config /hy-tmp/staging/configs/bert_prompt_only_seed42.json \
  --labels /hy-tmp/staging/tier1/toolace-6bda777-qwen35.jsonl \
  --config-dir /hy-tmp/staging/configs \
  --work-dir /hy-tmp/work/tier1-matrix \
  --results-dir /hy-tmp/results

wait "$qwen_download_pid"

nohup vllm serve Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --served-model-name qwen3.5-9b-tier2 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.95 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  > /hy-tmp/logs/vllm-serve.log 2>&1 < /dev/null &
echo "$!" > /hy-tmp/logs/vllm-serve.pid

for _ in $(seq 1 720); do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null; then
    break
  fi
  sleep 10
done
curl -fsS http://127.0.0.1:8000/health >/dev/null

nohup python /hy-tmp/staging/scripts/replay_tier2_labels.py \
  --labels /hy-tmp/staging/tier1/toolace-6bda777-qwen35.jsonl \
  --ledger /hy-tmp/results/vllm-smoke-toolace-3.jsonl \
  --report /hy-tmp/results/vllm-smoke-toolace-3-report.json \
  --limit 3 \
  > /hy-tmp/logs/vllm-smoke-toolace-3.log 2>&1 < /dev/null &
wait "$!"

nohup python /hy-tmp/staging/scripts/replay_tier2_labels.py \
  --labels /hy-tmp/staging/tier1/toolace-6bda777-qwen35.jsonl \
  --ledger /hy-tmp/results/tier2-toolace-pilot-400.jsonl \
  --report /hy-tmp/results/tier2-toolace-pilot-400-report.json \
  --limit 400 \
  > /hy-tmp/logs/tier2-toolace-pilot-400.log 2>&1 < /dev/null &
wait "$!"

python - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("/hy-tmp/results/tier2-toolace-pilot-400-report.json").read_text())
if not report["d4_passed"]:
    raise SystemExit(f"D4 failed: failure_rate={report['failure_rate']}")
PY

install -m 0644 \
  /hy-tmp/results/tier2-toolace-pilot-400.jsonl \
  /hy-tmp/results/tier2-toolace-full-ledger.jsonl
nohup python /hy-tmp/staging/scripts/replay_tier2_labels.py \
  --labels /hy-tmp/staging/tier1/toolace-6bda777-qwen35.jsonl \
  --ledger /hy-tmp/results/tier2-toolace-full-ledger.jsonl \
  --report /hy-tmp/results/tier2-toolace-full-report.json \
  > /hy-tmp/logs/tier2-toolace-full.log 2>&1 < /dev/null &
wait "$!"
