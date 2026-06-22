#!/bin/bash
cd /hy-tmp/vllm-ltr/train || exit 1
export PATH=/usr/local/cuda/bin:/usr/local/bin:$PATH CUDA_HOME=/usr/local/cuda
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/hy-tmp/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TRACE=jsonfiles/lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl
TOK=/hy-tmp/models/Meta-Llama-3-8B-Instruct

echo "=== A1: OPT-125M + marginRanking (isolate LOSS, same backbone as listMLE) $(date) ==="
python trainer.py --config configs/config_prefill_opt.txt --file $TRACE \
  --job-dir MODEL --run-id A1-opt125m-margin1.0-delta0.2-b4-OURS \
  --batch-size 4 --label-group-size 10 --loss marginRanking --margin 1.0 --delta 0.2 --tokenizer $TOK

echo "=== A2: BERT + marginRanking + delta=0 (isolate delta-FILTER) $(date) ==="
python trainer.py --config configs/config_prefill_bert.txt --file $TRACE \
  --job-dir MODEL --run-id A2-bert-margin1.0-delta0-b32-OURS \
  --batch-size 32 --label-group-size 10 --loss marginRanking --margin 1.0 --delta 0.0 --tokenizer $TOK

echo "=== ablations end $(date) ==="
