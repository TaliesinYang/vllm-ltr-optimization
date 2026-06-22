#!/usr/bin/env bash
# P0 + P1 — build the vllm-ltr env and download dataset + pretrained predictors.
# RUN ON THE RENTED LINUX GPU BOX (RTX 4090 48GB / L20 48GB / A800-80GB). NOT macOS — needs CUDA 12.1 + nvcc.
#
#   FORK_DIR=$HOME/vllm-ltr  TORCH_CUDA_ARCH_LIST=8.9  bash setup.sh
#
# Arch: 8.9 = Ada (4090 / L20) · 8.0 = Ampere (A800/A100) · 9.0 = Hopper (H800/H20).
# DO NOT use Blackwell (5090 / PRO 6000, sm_120) — the old fork's torch 2.2.1 / CUDA 12.1 stack won't build on it.
set -euo pipefail

FORK_DIR="${FORK_DIR:-$HOME/vllm-ltr}"
ENV_NAME="${ENV_NAME:-vllm-ltr}"
ARCH="${TORCH_CUDA_ARCH_LIST:-8.9}"

echo "[1/5] clone base fork -> $FORK_DIR"
[ -d "$FORK_DIR/.git" ] || git clone https://github.com/hao-ai-lab/vllm-ltr.git "$FORK_DIR"
cd "$FORK_DIR"

echo "[2/5] conda env: $ENV_NAME (python 3.10)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n "$ENV_NAME" python=3.10 -y 2>/dev/null || true
conda activate "$ENV_NAME"

echo "[3/5] torch 2.2.1 + xformers 0.0.25 (CUDA 12.1 wheels)"
pip install torch==2.2.1 xformers==0.0.25

echo "[4/5] build vllm-ltr CUDA kernels for arch=$ARCH (~20-40 min)"
export TORCH_CUDA_ARCH_LIST="$ARCH"
pip install -e .

echo "[5/5] HF login + download dataset + pretrained predictors (skip training)"
huggingface-cli whoami >/dev/null 2>&1 || huggingface-cli login   # needs Llama license access
cd train
huggingface-cli download LLM-ltr/Llama3-Trace   --local-dir jsonfiles --repo-type dataset
huggingface-cli download LLM-ltr/OPT-Predictors --local-dir MODEL     --repo-type model

echo
echo "DONE. Env ready, data + pretrained predictors downloaded."
echo "Next: bash <this-repo>/scripts/run_baseline.sh    # P3 baseline sweep (FCFS / LTR / classification)"
