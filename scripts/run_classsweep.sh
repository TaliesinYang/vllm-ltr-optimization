#!/bin/bash
set -x
export PATH=/usr/local/cuda/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/hy-tmp/hf-cache
cd /hy-tmp/vllm-ltr/train
ST=tpt-class10-xxx
CFG=MODEL/results/opt-125m-llama3-8b-lmsys-class-trainbucket820-b4-OURS/usage_config.json

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model /hy-tmp/models/Meta-Llama-3-8B-Instruct \
  --swap-space 16 --disable-log-requests \
  --schedule-type $ST --enable-chunked-prefill --enforce-eager \
  --prefill-predictor-model-config $CFG \
  --port 3344 > /hy-tmp/classsweep_server.log 2>&1 &
SERVER_PID=$!
echo "server pid $SERVER_PID"

READY=0
for i in $(seq 1 60); do
  if curl -s http://localhost:3344/v1/models >/dev/null 2>&1; then
    echo "server ready after $((i*5))s"; READY=1; break
  fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "SERVER DIED during startup"; tail -n 50 /hy-tmp/classsweep_server.log; exit 1
  fi
  sleep 5
done
if [ "$READY" != "1" ]; then echo "SERVER NOT READY after 300s"; tail -n 50 /hy-tmp/classsweep_server.log; kill $SERVER_PID 2>/dev/null; exit 1; fi

for r in 4 8 16 32; do
  echo "===== SWEEP RATE $r ====="
  python ../benchmarks/benchmark_serving_real.py --backend vllm \
    --model /hy-tmp/models/Meta-Llama-3-8B-Instruct \
    --tokenizer /hy-tmp/models/Meta-Llama-3-8B-Instruct \
    --dataset jsonfiles/lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl \
    --num-prompts -1 --request-time 60 --schedule-type $ST \
    --output-len -1 --request-rate $r --result-dir RESULTS --port 3344
done

kill $SERVER_PID 2>/dev/null
sleep 3
echo "CLASSSWEEP DONE"
