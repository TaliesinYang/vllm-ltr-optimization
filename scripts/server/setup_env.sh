#!/usr/bin/env bash
set -euo pipefail

LTR_ROOT="${LTR_ROOT:-/hy-tmp/ltr}"
REPO_ROOT="${REPO_ROOT:-$LTR_ROOT/repo}"
VENV="${VENV:-$LTR_ROOT/venv}"
MODEL_DIR="${MODEL_DIR:-/hy-tmp/models/qwen3.5-9b}"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3}"

mkdir -p "$LTR_ROOT" "$MODEL_DIR" /hy-tmp/hf
available_kb="$(df -Pk /hy-tmp | awk 'NR==2 {print $4}')"
(( available_kb >= 60 * 1024 * 1024 )) || { echo "/hy-tmp needs at least 60 GiB free" >&2; exit 1; }

"$PYTHON_BOOTSTRAP" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python >=3.10 is required")
PY
[[ -d "$VENV" ]] || "$PYTHON_BOOTSTRAP" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r "$REPO_ROOT/requirements/server.in"
"$VENV/bin/python" -m pip freeze >"$LTR_ROOT/manifest.pip.txt"
"$VENV/bin/python" - <<'PY'
from importlib.metadata import version
actual = version("vllm")
if not actual.startswith("0.24."):
    raise SystemExit(f"vllm 0.24.x required, got {actual}")
PY

HF_ENDPOINT="https://hf-mirror.com" HF_HOME="/hy-tmp/hf" \
  "$VENV/bin/hf" download Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --local-dir "$MODEL_DIR"
"$VENV/bin/python" - <<'GPU_PY'
import json, sys
import torch, pydantic, vllm
info = {
    "torch": torch.__version__,
    "torch_cuda_build": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "pydantic": pydantic.__version__,
    "vllm": vllm.__version__,
}
print(json.dumps(info, indent=2))
open("/hy-tmp/ltr/manifest.cuda.json", "w").write(json.dumps(info, indent=2) + "\n")
if not info["cuda_available"]:
    sys.exit("NO-GO: torch cannot see the GPU — wheel/driver CUDA mismatch")
GPU_PY
echo "server environment ready: $VENV"
