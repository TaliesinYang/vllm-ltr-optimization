#!/usr/bin/env bash
# Persist EVERYTHING before stopping the rented GPU. Ephemeral disk -> unsaved data = wasted run.
#   FORK_DIR=$HOME/vllm-ltr  bash collect_results.sh <run-id>
set -euo pipefail

FORK_DIR="${FORK_DIR:-$HOME/vllm-ltr}"
OUT="${OUT:-$HOME/vllm-ltr-results}"
RUN="${1:-run-$(date -u +%Y%m%d-%H%M)}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
DATASET="${DATASET:-lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl}"

mkdir -p "$OUT"
cd "$FORK_DIR/train"

echo "[1/3] raw benchmark results"
tar czf "$OUT/${RUN}-RESULTS.tgz" RESULTS/ 2>/dev/null && echo "  saved RESULTS/" || echo "  (no RESULTS/ yet)"

echo "[2/3] predictor configs (so we don't retrain)"
find MODEL -name 'usage_config.json' 2>/dev/null | tar czf "$OUT/${RUN}-predictor-configs.tgz" -T - 2>/dev/null \
  && echo "  saved usage_config.json files" || echo "  (no predictor configs)"

echo "[3/3] run manifest (versions / flags / trace — for reproducibility)"
{
  echo "run_id:      $RUN"
  echo "utc:         $(date -u +%FT%TZ)"
  echo "gpu:         $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
  echo "fork_commit: $(git -C "$FORK_DIR" rev-parse HEAD 2>/dev/null)"
  echo "vllm:        $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"
  echo "torch:       $(python -c 'import torch; print(torch.__version__, \"cuda\", torch.version.cuda)' 2>/dev/null)"
  echo "model:       $MODEL"
  echo "dataset:     $DATASET"
  echo "arch:        ${TORCH_CUDA_ARCH_LIST:-unset}"
} | tee "$OUT/${RUN}-manifest.txt"

echo
echo "Saved to: $OUT"
ls -la "$OUT"
echo
echo ">>> DOWNLOAD $OUT to your laptop, OPEN one file to verify, THEN stop the GPU. <<<"
