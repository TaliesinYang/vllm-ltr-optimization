#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/hy-tmp/hf
export HF_ENDPOINT=https://hf-mirror.com
export PIP_CACHE_DIR=/hy-tmp/pip-cache
export PIP_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple
export TMPDIR=/hy-tmp/tmp
export VIRTUALENV_OVERRIDE_APP_DATA=/hy-tmp/virtualenv-cache
export XDG_CACHE_HOME=/hy-tmp/xdg-cache

mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR" "$VIRTUALENV_OVERRIDE_APP_DATA" "$XDG_CACHE_HOME" /hy-tmp/venvs /hy-tmp/bootstrap
if [[ ! -f /hy-tmp/bootstrap/virtualenv/__init__.py ]]; then
  python3 -m pip install --target /hy-tmp/bootstrap virtualenv
fi
PYTHONPATH=/hy-tmp/bootstrap python3 -m virtualenv --clear /hy-tmp/venvs/vllm-ltr
source /hy-tmp/venvs/vllm-ltr/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  vllm==0.19.1 transformers datasets pytest scipy scikit-learn lightgbm requests
python -m pip freeze
