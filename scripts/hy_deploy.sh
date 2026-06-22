#!/usr/bin/env bash
# 恒源云 (gpushare) instance bootstrap for the vllm-ltr reproduction.
# Follows the `gpu-cloud-deploy` skill pattern. Run AFTER SSH into the instance.
# Pick a 4090 48GB / L20 48GB / A800-80GB image (Ada/Ampere); NOT Blackwell (5090/PRO6000).
#
#   export OSS_DATA=vllm-ltr/data.tar.gz   # optional: pre-staged jsonfiles+MODEL (run `oss login` first)
#   bash hy_deploy.sh
set -euo pipefail

OPT_REPO="${OPT_REPO:-https://github.com/TaliesinYang/vllm-ltr-optimization.git}"
export FORK_DIR="${FORK_DIR:-/hy-tmp/vllm-ltr}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"   # 8.9 Ada · 8.0 Ampere(A800) · 9.0 Hopper
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"   # HF 国内镜像,免墙

echo "=== network acceleration ==="
source /etc/network_turbo 2>/dev/null || echo "(no network_turbo on this image)"

echo "=== clone our optimization repo (scripts + docs) into /hy-tmp ==="
cd /hy-tmp
[ -d vllm-ltr-optimization/.git ] || git clone "$OPT_REPO"

echo "=== optional: pull pre-staged data from OSS (skips HF) ==="
if [ -n "${OSS_DATA:-}" ]; then
  oss cp "oss://${OSS_DATA}" /hy-tmp/                       # needs `oss login` first
  mkdir -p "$FORK_DIR/train"
  tar xzf "/hy-tmp/$(basename "$OSS_DATA")" -C "$FORK_DIR/train"
  echo "  data restored from OSS -> $FORK_DIR/train"
fi

echo "=== build env + base fork + (HF-mirror) data — our setup.sh ==="
bash /hy-tmp/vllm-ltr-optimization/scripts/setup.sh

echo
echo "Env ready. Next, IN A TMUX SESSION:"
echo "  tmux new -s run ; oss login ; bash /hy-tmp/vllm-ltr-optimization/scripts/hy_run_and_upload.sh baseline-1"
echo
echo "(Optional, one-time) back up downloaded data to OSS for future runs:"
echo "  cd $FORK_DIR/train && tar czf /hy-tmp/data.tar.gz jsonfiles MODEL && oss cp /hy-tmp/data.tar.gz oss://vllm-ltr/"
