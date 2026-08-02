# GPU validation — scoped Go/No-Go, 2026-08-01/02

**Verdict: Go**, with the direction reframed from *tool-schema stabilisation* to
**request layout for agentic serving**.

Three findings, in the order they were established:

1. Stabilising the tool schema buys **+0.01 pp** of cached prefix. It cannot buy
   more, because real agent traffic already sends a byte-identical schema every
   turn. The original hypothesis is dead.
2. Across sessions the earliest divergence sits in the **system prompt**, not the
   tools — so the stable schema is parked behind a volatile prefix and the cache
   can never reach it. Reordering the prompt so the schema leads recovers
   **+24.9 pp** of cached fraction and **−40.9%** of uncached prompt tokens,
   using the same constituent text in a different segment order.
3. Trimming the schema to the tools a session actually uses cuts prompt tokens
   23.3% and latency 33–42%. Real, but it is an oracle and it removes capability,
   so it is the weaker of the two positive levers.

An offline block-level simulation written before any GPU work predicted all three
outcomes; the hardware matched it to within 0.0005 on the policy comparison and
0.1 pp on the layout comparison.

## Rig

| | |
|---|---|
| GPU | RTX 4090 Laptop, 16 GB, compute capability **8.9** (Ada), under WSL2 |
| Engine | vLLM **0.9.2, V1 engine**, FlashAttention backend, `awq_marlin` |
| Model | `Qwen/Qwen3-8B-AWQ`, float16, TP=1, `max_model_len` 32768, util 0.90 |
| KV capacity | **52,576 tokens** · max concurrency at 32768 tok/req: 1.60× |
| Trace | `agent-traces-2026-07-26/agent_trace_vanilla.jsonl.gz` — 14 multi-turn sessions, 21 parent-child pairs, ~20.6 K tokens/prompt |

Every arm runs against a freshly started server; no server is reused across arms,
so a claimed cold cache is always an observed one.

vLLM 0.26.0 (current) is unusable here — its V1 worker requires UVA, which WSL2
does not provide. Four platform attempts were needed; `HARDWARE-FINDINGS.md` has
the V100 results, which rule out anything below compute capability 8.0.

---

## Group 1 — cache sanity (288 measurements, 3 restarts, cache on/off)

Does the offline block simulation correspond to real prefix-cache behaviour, and
does stabilising the schema help?

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
137 → 2,156 ms. The instrument measures caching.

**Cache OFF collapses the policies.** All three read 0.0000 cached, ~20.5 K new
prefill, ~6,760 ms; spread across policies 0.0000. The cache-on differences are
caching and nothing else.

**The offline simulation is validated.**

| policy | offline predicted | measured | delta |
|---|---:|---:|---:|
| Original | 0.9894 | 0.9889 | −0.0005 |
| Stable Full | 0.9895 | 0.9890 | −0.0005 |
| Shuffled Full | 0.7388 | 0.7389 | +0.0001 |

**Stabilising the schema buys +0.01 pp.** Prefix caching itself is worth 49× on
this workload (6,760 → 137 ms) and the unmodified client already captures
essentially all of it. There is no headroom left for a schema-stabilisation
policy, and the earlier No-Go now rests on a model real hardware confirms.

---

## Group 1b — request layout, cross-session (84 measurements, 3 restarts)

Group 1 asked whether the schema could be made *more* reusable within a session.
It could not — it is already reused. The offline audit pointed somewhere else:
**across** sessions the earliest divergence moves from the appended message to
the **system prompt** (90 of 91 session pairs). OpenCode puts per-session context
there, and the template puts the tool schema behind it, so a provably byte-stable
schema is unreachable. Does moving it in front recover the loss?

Layout cannot be set through the chat endpoint, which renders server-side.
Prompts are rendered locally with the same tokenizer, permuted (asserted
byte-preserving), and posted to `/v1/completions`. The server's `prompt_tokens`
matched the locally counted tokens on every request, confirming the rendering is
faithful. Session heads are replayed in arrival order within one server lifetime:
head N may reuse whatever heads 1..N−1 left behind.

| layout | n | prompt tok | cached | new prefill | cached frac | wall ms |
|---|---:|---:|---:|---:|---:|---:|
| as-is | 42 | 20,418 | 7,672 | 12,611 | 0.3808 | 4,806.3 |
| hoisted | 42 | 20,419 | 11,864 | 7,447 | 0.6303 | 3,211.4 |

**+24.9 pp cached fraction · −40.9% uncached prompt tokens.**
Paired over the same 42 head/run cells: **33 better, 3 worse, 6 tied**.

"Uncached prompt tokens" is `prompt_tokens − cached_tokens`, where
`cached_tokens` is the APC hit-counter delta. It is a serving-level proxy for the
KV that must be built, **not** a kernel-level count of recomputed tokens —
chunked prefill, attention kernels and scheduling all sit between the two.

**The wall-time reduction is not a stable estimate and should not be quoted as a
headline.** The pooled median falls 4,806 → 3,211 ms (−33.2%), but per restart:

| restart | as-is | hoisted | reduction |
|---|---:|---:|---:|
| 1 | 4,293.0 | 3,042.9 | −29.1% |
| 2 | 4,806.3 | 4,598.2 | **−4.3%** |
| 3 | 11,260.7 | 3,006.4 | **−73.3%** |

Restart 2 also contains a 22.4-second hoisted outlier. The cache-hit counts are
deterministic and repeat exactly across restarts; the timings do not. Treat the
latency effect as exploratory until server-side TTFT and prefill histograms are
collected with more repetitions.

The offline prediction is reproduced group by group:

`n` counts head×restart **cells, not independent sessions** — each head is
replayed once per restart, so a head's three values are repeat measurements of
one case:

| group | distinct heads | cells | offline | measured |
|---|---:|---:|---:|---:|
| 10 tools — shared toolset | 11 | 33 | +25.5 pp | **+25.5 pp** |
| 5 tools | 2 | 6 | — | **+15.9 pp** |
| 8 tools — different toolset | **1** | 3 | negative | **−25.0 pp** |
| pooled | 14 | 42 | +25.0 pp | **+24.9 pp** |

**The sign flips** in the predicted direction: the head whose toolset differs
from the majority is *hurt* by hoisting, because a differing schema at offset 0
destroys the prefix immediately instead of after the shared system preamble.

Be precise about how weak that is. The negative group is **one session head,
measured three times** — three repeat measurements of a single case, not three
cases. It agrees with the offline prediction and with the mechanism, and that is
all it does. Establishing the flip needs a trace with genuinely mixed toolsets.

If the flip holds up, it is what makes this a research question rather than a
tuning tip: the right layout would be conditional on whether co-resident sessions
share a schema, which is a property of the deployment rather than the model. On
this trace that conditionality is a hypothesis with one supporting case, not a
finding.

---

## Group 2 — cold start, schema trimming (112 requests, 0 errors, 0 OOM)

Session heads only, Original Full vs Frozen Thin (the tools the session turned
out to call, frozen from turn 1).

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
| latency drop ÷ token drop | 1.42× | 1.80× |

The latency saving exceeds the token saving in both modes. That is worth
recording but it is **not a discovery**: prefill attention is quadratic in
sequence length, so any token cut should beat its own proportion. The number to
take from this group is the raw saving, not the ratio.

---

## Scoped Go/No-Go

| # | Criterion | Result |
|---|---|---|
| 1 | Frozen Thin cold-start prefill or TTFT drop ≥ ~20% | ✅ −33.1% (cache off), −41.8% (cache on) |
| 2 | New prefill tokens or KV footprint drop ≥ ~30% | ✅ −36.5% (Thin), −40.9% (hoisting) |
| 3 | Queue time or goodput ≥ ~15% near saturation | ⬜ not run |
| 4 | Original/Stable/Shuffled match the offline prediction | ✅ delta ≤ 0.0005 |

Three of four → **Go**.

**Layout, not trimming, should lead.** Hoisting recovers more new-prefill saving
than trimming (−40.9% vs −36.5%), needs no hindsight, removes no capability, and
carries a genuine policy decision inside it. Trimming stays as a second lever —
it works, but it is the one with the correctness problem attached.

---

## Prior art — the general principle is already published

An external adversarial audit (GPT-5.6 Sol, 64 min, cloned this branch and read
the raw CSVs; transcript
`https://chatgpt.com/c/6a6f6d4f-d888-83e8-8a40-0be75261a6ec`) established that
**"put stable content before volatile content for prefix caching" is established
practice, not a new idea.** Closest prior work:

| Work | Relation |
|---|---|
| OpenAI *Prompt Caching* docs | Explicitly instructs putting static content at the beginning, variable content at the end |
| Anthropic *Prompt caching* / *Tool use with prompt caching* | Cache prefix is built `tools → system → messages`, i.e. tool definitions are already placed early by design |
| *Don't Break the Cache* (arXiv 2601.06007, 2026) | Closest paper: prompt caching for long-horizon agents, advises putting dynamic system content last |
| OpenClaw issue #27732 (2026) | Closest engineering antecedent: proposes reordering system-prompt sections so a static prefix precedes dynamic metadata — same mechanism, closed unimplemented |
| *CacheSlide* (FAST 2026) | Closest peer-reviewed systems work: models agent prompts as invariant + dynamic segments, **considers reordering and rejects it as semantically unsafe**, building an engine-side sliding cache instead |
| *ContextPilot* (MLSys 2026), *TokenPilot* (2026) | Automatic layout/reordering for context reuse |
| *EPIC* (ICML 2025), *Irminsul*, *MEPIC*, *Prompt Cache* (MLSys 2024) | Solve the same occlusion problem position-independently, without moving anything the model reads |

**What may still be new is narrow**: the measurement that this captured OpenCode
layout places volatile per-session context *in front of* a stable tool schema,
plus a controlled vLLM APC replay quantifying the recoverable loss. That is a
case study, not a mechanism contribution.

CacheSlide is the pointed one: it reached the reordering idea and declined it on
semantic-safety grounds — the exact question this work has not tested. Any
write-up must answer why reordering what the model reads is preferable to a
position-independent cache that preserves order.

## What this does not show

**About hoisting**

- **Output quality is untested.** Hoisting puts ~5 K tokens of schema ahead of the
  system prompt. The bytes are identical, the reading order is not, and nothing
  here measures tool-selection accuracy or instruction-following. This is the
  first thing to check before calling it deployable.
- **One arrival order.** Heads were replayed in capture order. A different order
  changes which head is cold and how much each can inherit; the pooled number
  would move, though the mechanism would not. The three restarts repeat the
  *same* trajectory, so their exact agreement demonstrates engine determinism,
  not workload robustness.
- **The hoist assertion is weaker than it reads.** `sorted(hoisted) ==
  sorted(text)` proves only that both strings hold the same multiset of
  characters. It does not prove that exactly one contiguous schema block moved,
  that every other region stayed byte-identical, or that delimiters kept their
  intended ownership. Slice-level SHA-256 equality would.
- **Groups are keyed by tool count, not schema identity.** `analyze_layout.py`
  buckets by `n_tools`; two 10-tool schemas can differ entirely. The correct
  sharing key is a hash of the rendered schema token sequence.
- **Group 1 and Group 1b are not two arms of one experiment.** 0.9889 is
  within-session via `/v1/chat/completions`; 0.3808 is cross-session-head via
  `/v1/completions` with locally rendered prompts. Different denominators — do
  not compare them directly. Matching `prompt_tokens` shows no tokens were added
  or dropped, but that is length equality, not token-ID identity with the
  server-side chat path.
- **The sign flip rests on one head.** The negative group is a single session
  head measured across 3 restarts, not 3 independent sessions. Directionally
  consistent with the offline audit; far too little to quantify or to build a
  conditional policy on.
- **Paired counts are over cells, not sessions.** "33 better, 3 worse, 6 tied" is
  over 42 head×restart cells drawn from 14 distinct heads. Restarts control for
  server-state noise; they do not add independent sessions.

**About trimming**

- **Frozen Thin is an oracle, not a system.** It keeps the tools the session
  turned out to call, chosen with hindsight. Nothing here shows a selector can
  pick that set in advance, and pruning a tool the model later needs could change
  what it is able to do. Correctness is untested.

**About the whole run**

- **KV footprint is unevaluated, not unmet.** `kv_cache_usage_perc` is scraped
  after the request returns, when the KV has already been freed, so it reads
  0.00030 for every arm. Peak KV needs in-flight sampling — a prerequisite for
  the concurrency probe, not an optional extra.
- **No concurrency result.** Group 3 was not run; nothing here speaks to goodput,
  queueing, or capacity. On this 16 GB card a single 32 K request already occupies
  1.60× concurrency worth of KV, so a meaningful sweep needs a larger card or
  tensor parallelism.
- **One client, one model, one trace.** OpenCode vanilla config, Qwen3-8B-AWQ,
  14 sessions. The 170-tool configuration (~55 K tokens) does not fit in 52,576
  tokens of KV and was not run.
- **Not the current vLLM.** 0.9.2 rather than 0.26.0, forced by the WSL2/UVA
  issue. Same engine generation, different version.
- **Wall time is not TTFT.** `max_tokens=1` makes it a close proxy for prefill
  plus one decode step, but it is client-side and includes HTTP overhead.
  Server-side per-request TTFT would be cleaner.

---

## Reproduce

All three groups assume the venv and trace layout created on the rig; `VENV`
points at the vLLM 0.9.2 environment and `VLLM_USE_V1=1` is set inside each
runner.

```bash
export VENV=~/vllm-schema-exp/.venv092 VLLM_USE_V1=1
TRACE=~/vllm-schema-exp/probes/agent-traces-2026-07-26/agent_trace_vanilla.jsonl.gz

# Group 1 — cache sanity (3 policies x cache on/off x 3 restarts)
bash phase1/run_phase1.sh "$TRACE" Qwen/Qwen3-8B-AWQ 1 32768 <outdir> 3
python phase1/analyze_phase1.py --csv <outdir>/phase1_measurements.csv

# Group 1b — layout (2 layouts x 3 restarts); driver hardcodes model and trace
bash phase1b/run_p1b.sh
python phase1b/analyze_layout.py --csv <outdir>/layout_measurements.csv

# Group 2 — cold start (2 policies x cache on/off x 2 restarts)
bash phase2/run_phase2.sh "$TRACE" vanilla_10tool Qwen/Qwen3-8B-AWQ 1 32768 <outdir> 2
python phase2/analyze_phase2.py --csv <outdir>/phase2_coldstart.csv
```

Raw measurements are committed under each group's `results/` directory
(force-added: the repo `.gitignore` has `RESULTS/`, which matches `results/` on a
case-insensitive filesystem).
