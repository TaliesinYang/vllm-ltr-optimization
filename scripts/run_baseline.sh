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

# activate the conda env from setup.sh — a fresh tmux shell does NOT inherit it
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null \
  && conda activate "${ENV_NAME:-vllm-ltr}" 2>/dev/null \
  || echo "WARN: conda env '${ENV_NAME:-vllm-ltr}' not activated — make sure the vllm-ltr python is on PATH"

cd "$FORK_DIR/train"   # benchmark_serving_real.py + dataset jsonl live relative to train/ (mirrors bench-lmsys.sh)

# sanity: pretrained predictor config must exist (run-id names must match LLM-ltr/OPT-Predictors)
if [ ! -f "$LTR_CFG" ]; then
  echo "WARN: LTR predictor config NOT found: $LTR_CFG"
  echo "      predictor dirs actually present in MODEL/results/:"; ls MODEL/results/ 2>/dev/null || echo "      (none — download failed?)"
  echo "      → adjust LTR_CFG/CLS_CFG run-id to a real name, or train the predictor first. Skipping LTR."
fi

# resolve the dataset to a real path — benchmark_serving_real.py does open(dataset_path) literally,
# but Llama3-Trace downloads under jsonfiles/, so a bare filename in CWD won't be found.
DATASET_FILE="$(find "$FORK_DIR/train" -maxdepth 3 -name "$DATASET" 2>/dev/null | head -1)"
if [ -z "$DATASET_FILE" ]; then
  echo "ERROR: dataset '$DATASET' not found under $FORK_DIR/train. Files in jsonfiles/:"
  ls "$FORK_DIR/train/jsonfiles" 2>/dev/null | head
  exit 1
fi
echo "dataset resolved: $DATASET_FILE"

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
      --model "$MODEL" --tokenizer "$MODEL" --dataset "$DATASET_FILE" \
      --num-prompts -1 --request-time 60 --schedule-type "$csty" --output-len -1 \
      --request-rate "$r" --result-dir RESULTS --port "$PORT"
  done
  kill "$pid" 2>/dev/null || true; sleep 60
}

# --- core (must-have for Wednesday) ---
sweep fcfs fcfs --swap-space 16
if [ -f "$LTR_CFG" ]; then
  sweep opt-xxx opt-xxx --swap-space 32 --prefill-predictor-model-config "$LTR_CFG"
else
  echo "SKIP LTR: predictor config missing (see warning above) — FCFS still ran."
fi

# --- optional: classification baseline (needs the class predictor config in MODEL/) ---
if [ -f "$CLS_CFG" ]; then
  sweep tpt-class10-xxx tpt-class10-xxx --swap-space 100 --prefill-predictor-model-config "$CLS_CFG"
else
  echo "SKIP classification: $CLS_CFG not found (train it or fetch from OPT-Predictors)."
fi

echo
echo "DONE. Raw data -> $FORK_DIR/train/RESULTS/"
echo "NOW RUN scripts/collect_results.sh BEFORE stopping the GPU (ephemeral disk!)."
