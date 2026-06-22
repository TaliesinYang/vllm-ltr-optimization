#!/usr/bin/env bash
# Battle-tested setup for reproducing vllm-ltr on a 恒源云/gpushare RTX 4090 48GB instance.
# Encodes EVERY fix found on 2026-06-22 (instance i-1:19590). Re-runnable; ~10-15 min instead of ~1h of debugging.
#
# ASSUMPTION: the rented image already ships Python 3.11 + torch 2.2.1+cu121 + CUDA 12.1 toolkit
#             (pick a "PyTorch 2.x / CUDA 12.1" image when renting). If torch/CUDA differ, stop and re-pick.
#
#   bash scripts/setup.sh
set -euo pipefail

M="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"      # CN pip mirror (GitHub/PyPI slow/blocked)
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"        # HF blocked in CN -> mirror
export HF_HOME="${HF_HOME:-/hy-tmp/hf-cache}"
export PATH=/usr/local/cuda/bin:/usr/local/bin:$PATH             # FIX#2: nvcc is NOT on PATH in a non-login shell
export CUDA_HOME=/usr/local/cuda
FORK_DIR="${FORK_DIR:-/hy-tmp/vllm-ltr}"
MODEL_DIR="${MODEL_DIR:-/hy-tmp/models/Meta-Llama-3-8B-Instruct}"
mkdir -p "$HF_HOME"

echo "[0] sanity — torch 2.2.1 + CUDA 12.1 + nvcc must already be present"
python3 -c "import torch; assert torch.__version__.startswith('2.2.1'), torch.__version__; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
nvcc --version | grep release

echo "[1] FIX#1 — system cmake is 3.16 (< 3.21 required); install 3.30 (NOT cmake 4.x)"
pip install -q -i "$M" "cmake==3.30.5"; hash -r; cmake --version | head -1

echo "[2] FIX#3 — xformers matching torch 2.2.1, --no-deps so it doesn't re-download ~2GB of torch"
pip install -q --no-deps -i "$M" xformers==0.0.25

echo "[3] build deps + benchmark-CLIENT deps (FIX#6 aiohttp) + downloaders"
pip install -q -i "$M" ninja packaging setuptools wheel aiohttp tqdm modelscope huggingface_hub

echo "[4] base fork must be present (rsync it; GitHub is blocked on the instance)"
[ -f "$FORK_DIR/setup.py" ] || { echo "  base fork missing at $FORK_DIR. From a machine that has it:"; \
  echo "    rsync -az --exclude .git ~/develop/vllm-ltr/ root@<host>:$FORK_DIR/"; exit 1; }

echo "[5] FIX#4 — build vllm-ltr for Ada sm_89, --no-build-isolation so it uses the system torch 2.2.1  (~10-15min)"
cd "$FORK_DIR"
TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS="${MAX_JOBS:-16}" pip install -e . --no-build-isolation

echo "[6] FIX#5 — transformers 5.x disables PyTorch<2.4; pin 4.40.2 (+ tokenizers 0.19.1)"
pip install -q -i "$M" "transformers==4.40.2" "tokenizers==0.19.1"
python3 -c "import transformers; print('transformers', transformers.__version__)"

echo "[7] data — trace + pretrained predictors via hf-mirror (non-gated) -> train/MODEL/results/"
cd "$FORK_DIR/train"
hf download LLM-ltr/Llama3-Trace lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl \
  --repo-type dataset --local-dir jsonfiles
hf download LLM-ltr/OPT-Predictors --repo-type model --include "opt-125m-llama3-8b-lmsys-*" --local-dir MODEL/results

echo "[8] FIX#4(gated) — Llama-3-8B from ModelScope (HF gated denies it); exclude original/*.pth (~15G saved)"
[ -f "$MODEL_DIR/config.json" ] || modelscope download --model LLM-Research/Meta-Llama-3-8B-Instruct \
  --exclude "original/*" --local_dir "$MODEL_DIR"

echo "[9] FIX#7 — RESULTS dir must exist (benchmark torch.save() fails with 'Parent directory RESULTS does not exist')"
mkdir -p "$FORK_DIR/train/RESULTS"

echo
echo "DONE. Verified working 2026-06-22 on RTX 4090 48GB / CUDA 12.1."
echo "Smoke test the server, then run the sweep: MODEL=$MODEL_DIR bash scripts/run_baseline.sh"
