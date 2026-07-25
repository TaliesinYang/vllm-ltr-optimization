#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/hy-tmp/hf
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export TMPDIR=/hy-tmp/tmp
export XDG_CACHE_HOME=/hy-tmp/xdg-cache

source /hy-tmp/venvs/vllm-ltr/bin/activate
python - <<'PY'
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="Qwen/Qwen3.5-9B",
    revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    cache_dir="/hy-tmp/hf",
)
print(path, flush=True)
PY
