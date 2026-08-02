# GPU validation — scoped Go/No-Go, 2026-08-01/02

Verdict: **Go**, with the direction narrowed. Three of the four scoped criteria
are met. The fourth (concurrency) was not run, and one sub-criterion (KV
footprint) could not be evaluated because of a measurement gap described below.

## Rig

| | |
|---|---|
| GPU | RTX 4090 Laptop, 16 GB, compute capability **8.9** (Ada), under WSL2 |
| Engine | vLLM **0.9.2, V1 engine**, FlashAttention backend, `awq_marlin` |
| Model | `Qwen/Qwen3-8B-AWQ`, float16, TP=1, `max_model_len` 32768, util 0.90 |
| KV capacity | **52,576 tokens** · max concurrency at 32768 tok/req: 1.60x |
| Trace | `agent-traces-2026-07-26/agent_trace_vanilla.jsonl.gz`, 14 multi-turn sessions, 21 parent-child pairs, ~20.6 K tokens/prompt |

vLLM 0.26.0 (current) is unusable here: its V1 worker requires UVA, which WSL2
does not provide. Four platform attempts were needed; see `HARDWARE-FINDINGS.md`
for the V100 results, which rule out anything below compute capability 8.0.

## Group 1 — cache sanity (288 measurements, 3 restarts, cache on/off)

| cache | policy | n | prompt tok | cached | new prefill | cached frac | wall ms |
|---|---|---:|---:|---:|---:|---:|---:|
| on | Original | 48 | 20,564 | 20,416 | 196 | 0.9889 | 136.5 |
| on | Stable Full | 48 | 20,546 | 20,384 | 201 | 0.9890 | 139.6 |
| on | Shuffled Full | 48 | 20,546 | 15,232 | 5,314 | 0.7389 | 2,155.8 |
| off | Original | 48 | 20,564 | 0 | 20,564 | 0.0000 | 6,760.1 |
| off | Stable Full | 48 | 20,546 | 0 | 20,546 | 0.0000 | 6,757.6 |
| off | Shuffled Full | 48 | 20,546 | 0 | 20,546 | 0.0000 | 6,769.3 |

**Positive control passes.** Shuffling the tool order costs 25.0 pp of cached
fraction, raises new prefill tokens 196 → 5,314 (+2611%), and wall time
137 → 2,156 ms. The chain measures caching.

**Cache OFF collapses the policies.** All three read 0.0000 cached with ~20.5 K
new prefill and ~6,760 ms; spread across policies 0.0000. The differences under
cache ON are caching, not something else.

**The offline block simulation is validated.**

| policy | offline predicted | measured | delta |
|---|---:|---:|---:|
| Original | 0.9894 | 0.9889 | −0.0005 |
| Stable Full | 0.9895 | 0.9890 | −0.0005 |
| Shuffled Full | 0.7388 | 0.7389 | +0.0001 |

**Stabilising the schema buys +0.01 pp**, exactly as offline. Prefix caching
itself is worth 49× on this workload (6,760 → 137 ms) and the unmodified client
already captures essentially all of it. There is no headroom for a schema-
stabilisation policy to claim, and the earlier No-Go now rests on a model that
real hardware confirms.

## Group 2 — cold start (112 requests, 0 errors, 0 OOM)

Session heads only — the first request of each multi-turn session.

| cache | policy | tools | prompt tok | new prefill | wall ms |
|---|---|---:|---:|---:|---:|
| off | Original | 10 | 20,418 | 20,418 | 6,604.6 |
| off | Frozen Thin | 1 | 15,670 | 15,670 | 4,420.0 |
| on | Original | 10 | 20,418 | 12,611 | 5,039.3 |
| on | Frozen Thin | 1 | 15,670 | 8,010 | 2,933.3 |

| | cache OFF | cache ON |
|---|---:|---:|
| prompt tokens | −23.3% | −23.3% |
| new prefill tokens | −23.3% | **−36.5%** |
| wall time | **−33.1%** | **−41.8%** |
| latency drop ÷ token drop | **1.42×** | **1.80×** |

The latency saving **exceeds** the token saving in both modes. That matters: it
means this is not simply "send fewer tokens, wait proportionally less". Prefill
attention is super-linear in sequence length, so trimming the schema shortens the
uncached span by more than its token share. This is a mechanism, not an
accounting identity, and it is the strongest result in the run.

## Scoped Go/No-Go

| # | Criterion | Result |
|---|---|---|
| 1 | Frozen Thin cold-start prefill or TTFT drop ≥ ~20% | ✅ −33.1% (cache off), −41.8% (cache on) |
| 2 | New prefill tokens or KV footprint drop ≥ ~30% | ✅ −36.5% (cache on) |
| 3 | Queue time or goodput ≥ ~15% near saturation | ⬜ not run |
| 4 | Original/Stable/Shuffled match the offline prediction | ✅ delta ≤ 0.0005 |

Three of four → **Go**. The direction narrows from *cache preservation* to
**tool working-set reduction**: the win is in prefill cost, not in cache reuse.

## What this does not show

- **Frozen Thin is an oracle, not a system.** It keeps the tools the session
  turned out to call, chosen with hindsight. Nothing here shows a deployable
  selector can pick that set in advance, and pruning a tool the model later needs
  could change what it is able to do. Correctness is untested.
- **KV footprint is unevaluated, not unmet.** `kv_cache_usage_perc` is scraped
  after the request returns, when the KV has already been freed, so it reads
  0.00030 for every arm. Peak KV needs in-flight sampling; that is a prerequisite
  for the concurrency probe, not an optional extra.
- **No concurrency result.** Group 3 was not run. Nothing here speaks to goodput,
  queueing, or capacity. On this 16 GB card a single 32 K request already
  occupies 1.60× concurrency worth of KV, so a meaningful sweep needs a larger
  card or tensor parallelism.
- **One client, one model, one trace.** OpenCode vanilla config only, Qwen3-8B-AWQ
  only, 14 sessions. The 170-tool configuration (~55 K tokens) does not fit in
  52,576 tokens of KV and was not run.
- **Not the current vLLM.** 0.9.2 rather than 0.26.0, forced by the WSL2/UVA
  issue. The V1 engine is the same generation, but the versions differ.
- **Wall time is not TTFT.** `max_tokens=1` makes wall time a close proxy for
  prefill plus one decode step, but it is measured client-side and includes HTTP
  overhead. Per-request TTFT from the server would be cleaner.

## Reproduce

```bash
# Group 1
VENV=~/vllm-schema-exp/.venv092 VLLM_USE_V1=1 \
  bash phase1/run_phase1.sh <trace> Qwen/Qwen3-8B-AWQ 1 32768 <outdir> 3
python phase1/analyze_phase1.py --csv <outdir>/phase1_measurements.csv

# Group 2
VENV=~/vllm-schema-exp/.venv092 VLLM_USE_V1=1 \
  bash phase2/run_phase2.sh <trace> vanilla_10tool Qwen/Qwen3-8B-AWQ 1 32768 <outdir> 2
python phase2/analyze_phase2.py --csv <outdir>/phase2_coldstart.csv
```
