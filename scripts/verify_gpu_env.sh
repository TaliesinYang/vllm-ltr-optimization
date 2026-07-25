#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/hy-tmp/hf
export HF_ENDPOINT=https://hf-mirror.com
export TMPDIR=/hy-tmp/tmp
export XDG_CACHE_HOME=/hy-tmp/xdg-cache
export TOOLACE_SNAPSHOT=/hy-tmp/staging/fixtures/toolace-data.json
export LMCACHE_ROWS_SNAPSHOT=/hy-tmp/staging/fixtures/lmcache-first-rows.json

source /hy-tmp/venvs/vllm-ltr/bin/activate
cd /hy-tmp/staging
python -m pytest -q | tee /hy-tmp/logs/pytest.log
python - <<'PY'
import json
import platform
from pathlib import Path

import torch
import transformers
import vllm

payload = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "transformers": transformers.__version__,
    "vllm": vllm.__version__,
    "pytest_log": "/hy-tmp/logs/pytest.log",
    "pytest_summary": Path("/hy-tmp/logs/pytest.log").read_text().strip().splitlines()[-1],
}
Path("/hy-tmp/results").mkdir(parents=True, exist_ok=True)
Path("/hy-tmp/results/environment.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(payload, sort_keys=True))
PY
