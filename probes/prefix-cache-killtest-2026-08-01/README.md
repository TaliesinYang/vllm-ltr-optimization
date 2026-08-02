# Probe: tool-schema prefix-cache kill test — 2026-08-01

**Question.** In real OpenCode agent requests, does keeping the tool schema stable in
content and order measurably increase reusable prefix tokens / KV blocks?

**Verdict: No-Go on the "stable schema improves prefix reuse" direction.**
The schema in real single-client traffic is *already* byte-stable, so freezing it buys
+0.01 pp. Shuffling it is catastrophic (−25 to −68 pp reuse), but no real client shuffles.

This is an **offline block simulation**: it measures theoretical cacheability only and
makes **no** claim about GPU latency. See "What this cannot say" below.

## Rig

| Item | Value |
|---|---|
| Tokenizer / chat template | `Qwen/Qwen2.5-7B-Instruct` (HF), template sha256 `cd8e9439f0570856…` |
| Rendering | `apply_chat_template(messages, tools=…, add_generation_prompt=True)`, then encode with `add_special_tokens=False` |
| KV block size | 16 |
| Shuffle seed | 20260801 |
| Scripts | `killtest.py` (render + metrics), `analyze.py` (Go/No-Go + figures) |

```bash
python killtest.py --trace ../agent-traces-2026-07-26/agent_trace_vanilla.jsonl.gz --out results
python analyze.py  --results results
```

> The repo `.gitignore` has `RESULTS/`, which matches `results/` on a case-insensitive
> filesystem. The committed outputs under `results/` were added with `git add -f`.

### Datasets

| Dir | Trace | Requests | Sessions ≥2 turns | Pairs | Tools/req |
|---|---|---:|---:|---:|---:|
| `results/` | `agent-traces-2026-07-26/agent_trace_vanilla.jsonl.gz` | 75 | 14 | **21** | 10 (vanilla) |
| `results_170tool/` | `schema-variability-2026-07-25/captured_requests_v2.jsonl` | 28 | 7 | **14** | 170 (full MCP config) |

The 170-tool set is the harsher regime — schema is 67.9 % of the prompt there vs 24.9 %
in vanilla — and is included as an independent replication.

### Session reconstruction

Neither capture carries a session id. Request *B* is treated as the continuation of *A*
iff `A.messages` is a **strict prefix** of `B.messages` — exactly what an agent loop
emits when it appends a tool result and re-sends. Grouping by (system, first user turn)
instead was tried first and is **wrong**: OpenCode's 25 title-generation requests share a
fixed two-message head and would collapse into one bogus 25-request "session".

### Policies

| Policy | Definition |
|---|---|
| Original | request body untouched |
| Stable Full | session-wide union of tools, name-sorted, recursively key-sorted JSON |
| Shuffled Full | same set/token count as Stable Full, order re-randomised each turn |
| Frozen Thin | only tools the session actually invoked, frozen from turn 1 |

Requests that carried no schema (title-gen) keep carrying none under every policy.

## Results

### Vanilla, 10 tools — 21 pairs (`results/`)

| Policy | Median total tok | Median schema tok | Median token LCP | Median reuse ratio | Median invalidated suffix |
|---|---:|---:|---:|---:|---:|
| Original | 20 606 | 5 128 | 20 418 | **0.9894** | 203 |
| Stable Full | 20 587 | 5 109 | 20 399 | **0.9895** | 203 |
| Shuffled Full | 20 587 | 5 109 | 15 239 | **0.7388** | 5 344 |
| Frozen Thin | 16 016 | 563 | 15 779 | **0.9868** | 202 |

### Full MCP config, 170 tools — 14 pairs (`results_170tool/`)

| Policy | Median total tok | Median schema tok | Median token LCP | Median reuse ratio | Median invalidated suffix |
|---|---:|---:|---:|---:|---:|
| Original | 55 559 | 37 964 | 55 425 | **0.9974** | 142 |
| Stable Full | 55 468 | 37 873 | 55 334 | **0.9974** | 143 |
| Shuffled Full | 55 468 | 37 873 | 17 381 | **0.3144** | 38 058 |
| Frozen Thin | 17 866 | 271 | 17 732 | **0.9919** | 143 |

### Go / No-Go criteria

| # | Criterion | Threshold | Vanilla | 170-tool | |
|---|---|---|---|---|---|
| 1 | Stable − Original median reuse | ≥ +20 pp | **+0.01 pp** | **+0.00 pp** | ✗ |
| 2 | Stable vs Shuffled invalidated-suffix cut | ≥ 30 % | 96.2 % | 99.6 % | ✓ |
| 3 | Shuffle moves earliest mutation earlier | consistent | 21/21 | 14/14 | ✓ |
| 4 | Pairs where Stable > Original blocks | ≥ 70 % | **0/21 (0 %)** | **0/14 (0 %)** | ✗ |
| 5 | Difference driven by the `tools` segment | — | 0/21 (all `messages`) | 0/14 (all `messages`) | ✗ |

Criteria 2 and 3 pass only for the **synthetic** Shuffled arm. Criteria 1, 4, 5 — the
ones that test whether a *real* client has anything to gain — all fail on both datasets.

## The five questions

1. **Where does the earliest change occur?** In `messages`, in **35/35 pairs** across both
   datasets. Never in `system`, never in `tools`. The mutation is the newly appended
   assistant turn + tool result — inherent to the agent loop, not a schema artifact.
2. **Where does the schema sit in the token stream?** Inside the system block, *after* the
   system prompt, contiguous, ending before the first `<|im_end|>`. Median token span
   `[15 229, 20 357)` of 20 427 (vanilla — starts at 74.6 % of the prompt) and
   `[17 371, 55 335)` of 55 437 (170-tool — starts at 31.3 %).
3. **How many extra full blocks does Stable Full buy?** Effectively zero — median token LCP
   moves 20 418 → 20 399 (it *drops* 19 tokens, because canonical key ordering re-renders
   slightly shorter). Reuse gain +0.01 pp; 0/21 and 0/14 pairs improve.
4. **What does reordering alone cost?** A lot: reuse 0.9894 → 0.7388 (vanilla) and
   0.9974 → 0.3144 (170-tool); invalidated suffix 203 → 5 344 and 142 → 38 058 tokens.
   Order sensitivity is real — it is simply not a problem any observed client has.
5. **Is it big enough for GPU experiments?** **No.** Original already reuses 98.9 % / 99.7 %
   of full blocks; the maximum headroom for any schema policy is ~1 pp.

## Conclusion

> We analysed 21 adjacent request pairs (14 sessions, 75 requests, vanilla 10-tool config)
> and replicated on 14 pairs (7 sessions, 28 requests, 170-tool MCP config).
>
> Under the real Qwen2.5 chat template, the median reusable-block ratio is 0.9894 for
> Original, 0.9895 for Stable Full and 0.7388 for Shuffled Full (170-tool: 0.9974 /
> 0.9974 / 0.3144).
>
> Stable Full improves on Original by +0.01 pp and reduces invalidated suffix vs Shuffled
> Full by 96.2 %. The earliest change lies in the `messages` segment in every pair.
>
> **Verdict: No-Go** for dynamic/stabilised schema selection as a prefix-cache
> optimisation. The Conditional-Go escape hatch does not apply either: the system prompt
> does *not* mutate, so the failure is not "layout blocks a stable schema" — there is
> simply nothing left to stabilise.

This is consistent with, and mechanistically explains,
`../schema-variability-2026-07-25/FINDINGS.md`, which found the schema byte-identical
across turns within a session.

## The one effect that did survive

Frozen Thin does **not** raise the reuse ratio, but it cuts prompt size hard:
median total tokens 20 606 → 16 016 (**−22.3 %**) vanilla and 55 559 → 17 866
(**−67.8 %**) at 170 tools; schema tokens 37 964 → 271 in the latter.

That is a **prefill-cost and KV-footprint** result, not a cache-reuse result — it pays off
on cold start and in memory pressure, where the prefix cache gives nothing. If this
direction is pursued, it must be framed and measured that way, and it inherits an
unanswered correctness question: pruning tools the session has not yet called can change
what the model is able to do next.

## What this cannot say

- **No GPU, no latency claim.** Full-block LCP is an upper bound on cacheability. Real
  vLLM hit rates are additionally bounded by eviction, block hashing, chunked prefill and
  scheduler behaviour. Nothing here supports a TTFT or throughput number.
- **Template mismatch with the capture.** The traces were served by Ollama
  (`qwen2.5:7b-instruct`) using Ollama's own Go template; we re-render with the HF/vLLM
  Jinja template because vLLM is the target system. Absolute token counts therefore differ
  from the captured `usage.prompt_tokens`; the paired comparison across policies is
  unaffected, since all four arms share one renderer.
- **Single client, single tenant.** Every session comes from one OpenCode instance. Nothing
  here speaks to multi-tenant queues where schemas genuinely differ across clients.
- **Small n.** 21 + 14 pairs. The effect sizes are large enough that n is not the binding
  constraint for the No-Go, but no significance testing was run.
- **Content-part flattening.** The 170-tool capture stores message content as OpenAI
  content-part lists; they are flattened to text before rendering (`flatten_content`),
  matching what vLLM's OpenAI server does. This is a no-op on the vanilla trace —
  verified by re-running it and diffing `aggregate_summary.csv`.
