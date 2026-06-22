#!/usr/bin/env bash
# P3 — baseline serving sweep on Llama-3-8B / LMSYS. Reproduces the paper's Fig.3 (latency vs request-rate).
# Methods: FCFS (floor) -> LTR (ours, pretrained predictor) -> classification (optional).
# Run on the GPU box AFTER setup.sh. Results land in $FORK_DIR/train/RESULTS/.
#
#   FORK_DIR=$HOME/vllm-ltr  bash run_baseline.sh
set -euo pipefail

FORK_DIR="${FORK_DIR:-/hy-tmp/vllm-ltr}"            # data disk
export HF_HOME="${HF_HOME:-/hy-tmp/hf-cache}"       # serve step caches here, not the 20G system disk
# local model dir from ModelScope (China, no gated token). Set MODEL=meta-llama/Meta-Llama-3-8B-Instruct to use HF instead.
MODEL="${MODEL:-/hy-tmp/models/Meta-Llama-3-8B-Instruct}"
DATASET="${DATASET:-lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl}"
RATES="${RATES:-2 4 8 16 32 64}"
PORT="${PORT:-3343}"
LTR_CFG="MODEL/results/opt-125m-llama3-8b-lmsys-score-trainbucket10-b32/usage_config.json"
CLS_CFG="MODEL/results/opt-125m-llama3-8b-lmsys-class-trainbucket820-b32/usage_config.json"

cd "$FORK_DIR/train"   # benchmark_serving_real.py + dataset jsonl live relative to train/ (mirrors bench-lmsys.sh)

sweep () {  # $1=server schedule-type  $2=client schedule-type  $3...=extra server args
  local sty="$1" csty="$2"; shift 2
  echo "================ method: $sty ================"
  CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --disable-log-requests --schedule-type "$sty" \
    --enable-chunked-prefill --enforce-eager --port "$PORT" "$@" &
  local pid=$!
  sleep 90                                   # warmup (model load); bump if 8B loads slowly
  for r in $RATES; do
    echo "---- $sty @ rate $r ----"
    python ../benchmarks/benchmark_serving_real.py --backend vllm \
      --model "$MODEL" --tokenizer "$MODEL" --dataset "$DATASET" \
      --num-prompts -1 --request-time 60 --schedule-type "$csty" --output-len -1 \
      --request-rate "$r" --result-dir RESULTS --port "$PORT"
  done
  kill "$pid" 2>/dev/null || true; sleep 60
}

# --- core (must-have for Wednesday) ---
sweep fcfs    fcfs    --swap-space 16
sweep opt-xxx opt-xxx --swap-space 32 --prefill-predictor-model-config "$LTR_CFG"

# --- optional: classification baseline (needs the class predictor config in MODEL/) ---
if [ -f "$CLS_CFG" ]; then
  sweep tpt-class10-xxx tpt-class10-xxx --swap-space 100 --prefill-predictor-model-config "$CLS_CFG"
else
  echo "SKIP classification: $CLS_CFG not found (train it or fetch from OPT-Predictors)."
fi

echo
echo "DONE. Raw data -> $FORK_DIR/train/RESULTS/"
echo "NOW RUN scripts/collect_results.sh BEFORE stopping the GPU (ephemeral disk!)."
