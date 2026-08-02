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

## Group 1b — request layout, cross-session (84 measurements, 3 restarts)

Group 1 asked whether the schema can be made *more* reusable within a session;
the answer was no, because it is already reused. This asks the question the
offline audit raised instead: across sessions the earliest divergence sits in the
**system prompt**, so a byte-stable schema placed behind it is unreachable. Does
moving the schema in front recover the loss?

The chat endpoint renders server-side, so layout cannot be controlled through it.
Prompts are rendered locally with the same tokenizer, permuted (asserted
byte-preserving), and posted to `/v1/completions`. `prompt_tokens` returned by
the server matched the locally counted tokens exactly on every request, so the
rendering is faithful. Session heads are sent in arrival order within one server
lifetime — head N may reuse whatever heads 1..N-1 left behind.

| layout | n | prompt tok | cached | new prefill | cached frac | wall ms |
|---|---:|---:|---:|---:|---:|---:|
| as-is | 42 | 20,418 | 7,672 | 12,611 | 0.3808 | 4,806.3 |
| hoisted | 42 | 20,419 | 11,864 | 7,447 | 0.6303 | 3,211.4 |

**+24.9 pp cached fraction, −40.9% new prefill tokens, −33.2% wall time.**
Paired across the same 42 head/run cells: 33 better, 3 worse, 6 tied.

The offline prediction is reproduced group by group:

| group | offline | measured |
|---|---:|---:|
| 10 tools (shared toolset), n=33 | +25.5 pp | **+25.5 pp** |
| pooled | +25.0 pp | **+24.9 pp** |
| 8 tools (different toolset), n=3 | negative | **−25.0 pp** |

The sign flip survives: heads whose toolset differs from the majority are *hurt*
by hoisting, because a differing schema at offset 0 destroys the prefix
immediately instead of after the shared system preamble. With n=3 that is a
direction, not a measurement, but it is the direction the offline audit
predicted.

**Why this is the stronger of the two positive results.** Frozen Thin is an
oracle — it needs hindsight about which tools the session will call, and pruning
a tool the model later needs can change what it can do. Hoisting sends *exactly
the same bytes* in a different order: no information is removed, so there is no
correctness question of that kind, and any client or gateway can do it today.
It also has a genuine policy structure — the sign depends on whether sessions
share a schema — where Frozen Thin only has "cut more, go faster".

## Scoped Go/No-Go

| # | Criterion | Result |
|---|---|---|
| 1 | Frozen Thin cold-start prefill or TTFT drop ≥ ~20% | ✅ −33.1% (cache off), −41.8% (cache on) |
| 2 | New prefill tokens or KV footprint drop ≥ ~30% | ✅ −36.5% (cache on) |
| 3 | Queue time or goodput ≥ ~15% near saturation | ⬜ not run |
| 4 | Original/Stable/Shuffled match the offline prediction | ✅ delta ≤ 0.0005 |

Three of four → **Go**. The win is in prefill cost, not in cache reuse — but
after Group 1b the better framing is **request layout**, not tool working-set
reduction. Reordering recovers more new-prefill savings than trimming (−40.9% vs
−36.5%), needs no hindsight, removes no capability, and has a real policy
decision inside it. Tool trimming remains a legitimate second lever; it is just
the one with the correctness problem attached.

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
- **Hoisting is untested for output quality.** It puts 5 K tokens of schema ahead
  of the system prompt. The bytes are identical, but the model reads them in a
  different order, and nothing here measures whether tool selection or
  instruction-following changes. That is the first thing to check before treating
  it as deployable.
- **The layout result is one arrival order.** Heads were replayed in capture
  order. A different order changes which head is cold and how much each can
  inherit; the pooled number would move even though the mechanism would not.
- **The sign flip rests on 3 heads.** The negative group is directionally
  consistent with the offline audit but far too small to quantify.
- **The super-proportional latency effect is expected physics, not a discovery.**
  Prefill attention is quadratic in sequence length, so any token cut should beat
  its own proportion. Worth reporting; not worth claiming as a mechanism nobody
  knew about.

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
