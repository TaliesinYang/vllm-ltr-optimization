#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/hy-tmp/hf
export HF_HUB_CACHE=/hy-tmp/hf
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR=/hy-tmp/tmp
export XDG_CACHE_HOME=/hy-tmp/xdg-cache
export TRITON_CACHE_DIR=/hy-tmp/triton-cache
export TORCHINDUCTOR_CACHE_DIR=/hy-tmp/torchinductor-cache
export VLLM_CACHE_ROOT=/hy-tmp/vllm-cache

source /hy-tmp/venvs/vllm-ltr/bin/activate

for _ in $(seq 1 2160); do
  if python - <<'PY'
import json
from pathlib import Path

path = Path("/hy-tmp/results/tier1-matrix-summary.json")
raise SystemExit(0 if path.exists() and json.loads(path.read_text()).get("completed_runs") == 10 else 1)
PY
  then
    break
  fi
  sleep 10
done

python - <<'PY'
import json
from pathlib import Path

path = Path("/hy-tmp/results/tier1-matrix-summary.json")
if not path.exists() or json.loads(path.read_text()).get("completed_runs") != 10:
    raise SystemExit("Tier-1 matrix did not complete within six hours")
PY

python - <<'PY'
from huggingface_hub import snapshot_download

print(
    snapshot_download(
        repo_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        cache_dir="/hy-tmp/hf",
        local_files_only=True,
    )
)
PY

nohup vllm serve Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --served-model-name qwen3.5-9b-tier2 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.90 \
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
