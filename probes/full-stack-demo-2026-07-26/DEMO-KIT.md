# Full-stack live chain + presentation demo kit (ticket #11)

**Status: PASS, 2026-07-26.** OpenCode tasks ran through the COMPLETE stack —
gateway → Decision Service (real BERT checkpoint + real quantile manifest + Rule C
gate) → model — with **zero fail-open events**: every real agent request was scored
by the Ranker and adjudicated by the Reliability Gate.

## Topology (all local, $0)

```
OpenCode (vanilla config) → VeloxMesh gateway :9100 ──ltr.Apply──▶ Decision Service :9200
                                   │                              (BertPredictor seed17,
                                   ▼                               rank_quantiles.json,
                        SSH tunnel :11435 → 201 Ollama              Rule C confidence,
                            qwen2.5:7b-instruct                     threshold 0.5)
```

## Live evidence captured

| Probe | Result |
|---|---|
| S4 request (all-new tools: glob+read) | `reliability_probability: 0.6233` (= Rule C S4), `prediction_reliable: true`, `estimated_tokens: 156` (plausible: real-trace completions p50=42/p99=328) |
| S2 request (two in-vocabulary ToolACE tools, new combination) | confidence **0.0 → abstain** — the gate refusing to vouch for the measured-hard stratum, live |
| OpenCode task through full stack | 3 agent-loop requests, all 200, **0 fail-open lines** |

## Three integration bugs found and fixed during bring-up (the point of T7)

1. `a7e8299` — decision service crashed (503 → permanent fail-open) on gateway-shaped
   requests: it required `tool_schema_text`, but gateways forward only the OpenAI
   `tools` array. Real traffic was NEVER scored before this fix.
2. `8808d84` — `tool_vocabulary` couldn't parse the OpenAI nested `function.name`
   format → every live request classified `unknown` → 0.0. Rule C was unreachable live.
3. `af8a7ed` — `DEFAULT_RELIABILITY_THRESHOLD = 0.8` exceeds Rule C's maximum
   confidence (0.6233) → the gate could never trust anything. Added
   `--reliability-threshold`; demo runs at 0.5 (S3/S4 pass, S1/S2 abstain — Rule C
   intent). The 0.5-vs-0.8 disagreement stays an open policy item.

## 7/29 presentation demo script (~2 min)

1. **Pre-flight (10 min before, in order):** tunnel up → Ollama warm (one direct
   curl; cold load = 20–60 s) → decision service up (BERT loads ~10 s) → gateway up →
   one smoke curl. Commands below.
2. **Screen layout:** left = `tail -f` gateway log + decision service log; right =
   terminal for OpenCode.
3. **Beat 1 (30 s):** run
   `opencode run -m vx/qwen2.5:7b-instruct "Count the markdown files here"` —
   audience watches requests traverse the gateway (request ids, latency, no fail-open).
4. **Beat 2 (30 s):** two prepared curls to `/v1/decision`: the S4 probe
   (reliable, estimated 156 tokens) and the S2 probe (confidence 0.0, abstain).
   One sentence: *"the gate knows what it doesn't know — novel compositions of
   familiar tools are where the model fails, so that's exactly where it steps aside."*
5. **Beat 3 (30 s):** open `probes/agent-traces-2026-07-26/MANIFEST.md` — 75 real
   requests, schema share, output-length distribution. *"This is the traffic we
   schedule, and this is the signal we schedule it with."*
6. **B-plan:** if anything is down, narrate over the committed logs and this file —
   same story, zero live risk.

## Launch commands

```bash
# 201 (once): OLLAMA_HOST=0.0.0.0:11434 setsid nohup ~/bin/ollama serve &
ssh -p 2222 -N -L 11435:localhost:11434 alex@192.168.8.201 &          # tunnel
curl -s http://127.0.0.1:11435/v1/chat/completions -d '{"model":"qwen2.5:7b-instruct","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' -H 'Content-Type: application/json'  # warm

cd ~/develop/vllm-ltr-optimization
PYTHONPATH=$PWD .worktrees/final-training-artifacts/.venv/bin/python \
  scripts/run_decision_service.py --host 127.0.0.1 --port 9200 --predictor bert \
  --checkpoint checkpoints_best_predictor \
  --quantile-manifest runs/full-stack-demo-2026-07-26/rank_quantiles.json \
  --max-concurrency 8 --reliability-threshold 0.5 &

cd ~/develop/VeloxMesh
GATEWAY_DATA_ADDR=127.0.0.1:9100 DEV_API_KEY=vx-dev DEFAULT_PROVIDER=openai-primary \
OPENAI_PRIMARY_BASE_URL=http://127.0.0.1:11435/v1 OPENAI_PRIMARY_API_KEY=unused \
OPENAI_PRIMARY_MODELS="qwen2.5:7b-instruct" OPENAI_PRIMARY_DEFAULT_MODEL="qwen2.5:7b-instruct" \
LTR_DECISION_ENDPOINT=http://127.0.0.1:9200 LTR_DECISION_TIMEOUT_MS=2000 ./bin/gateway &

# OpenCode project config: probes runbook + scratchpad gateway-project pattern
```

## Honest scope notes

- Quantile manifest built from the LOCAL 6k labels (3 ledger failure rows declared as
  exclusions, 5997 rows) — real data, but `estimated_tokens` remains the uncalibrated
  rank-lookup the manifest's own approximation_notice describes.
- Threshold 0.5 is a demo-recorded choice, not a tuned value.
- Single client, no queue contention (X-Queue-Wait-Ms = 0 throughout) — scheduling
  under load remains E5 (rental).
