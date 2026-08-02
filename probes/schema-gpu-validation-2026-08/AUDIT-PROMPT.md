# Adversarial audit prompt — self-contained

Paste everything below the line into a fresh model session. It carries all the
numbers, so the reviewer needs no repo access.

---

You are reviewing an empirical systems result before it is sent to a professor.
Your job is to **try to break it**, not to summarise it. Assume the author is
motivated to believe their own findings. Be specific: name the confound, say what
measurement would settle it, and say which stated claim would have to be
withdrawn if you are right. If a claim survives your scrutiny, say so plainly —
do not manufacture doubt to seem rigorous.

## Context

An LLM agent (OpenCode) sends requests to a vLLM server. Each request carries a
system prompt, a JSON tool schema, and a growing message list. vLLM's prefix
cache reuses KV for any exact token prefix shared with an earlier request. The
question was whether managing the tool schema can increase that reuse.

## Rig

- RTX 4090 Laptop, 16 GB, compute capability 8.9, running under WSL2
- vLLM 0.9.2, V1 engine, FlashAttention backend, `awq_marlin` quantisation
- `Qwen/Qwen3-8B-AWQ`, float16, TP=1, `max_model_len` 32768, `gpu_memory_utilization` 0.90
- Reported KV capacity 52,576 tokens; max concurrency at 32768 tok/req = 1.60×
- Trace: 75 captured requests from real OpenCode sessions, 14 multi-turn sessions,
  21 parent-child pairs, ~20.6 K tokens per prompt, ~10 tools per request
- `max_tokens=1` on every request; requests sent strictly serially
- A fresh server process per experimental arm — no arm reuses another's cache
- Sessions were reconstructed by strict message-prefix chaining (request B
  continues A iff A's message list is a strict prefix of B's), because the trace
  has no session id

## Measurement method

vLLM 0.9.2 returns `prompt_tokens_details = null`, so per-request cached tokens
are taken as the delta of the Prometheus counter
`vllm:gpu_prefix_cache_hits_total` across the request, with
`vllm:gpu_prefix_cache_queries_total` as the denominator check. `cached_frac =
cached_tokens / prompt_tokens`. Wall time is measured client-side and includes
HTTP overhead.

## Finding 1 — stabilising the tool schema does nothing (288 measurements, 3 restarts)

Policies: **Original** (request unchanged), **Stable Full** (session-wide tool
union, name-sorted, recursively key-sorted JSON), **Shuffled Full** (same tool
set, order re-randomised each turn — a deliberate negative control).

| cache | policy | n | prompt tok | cached | new prefill | cached frac | wall ms |
|---|---|---:|---:|---:|---:|---:|---:|
| on | Original | 48 | 20,564 | 20,416 | 196 | 0.9889 | 136.5 |
| on | Stable Full | 48 | 20,546 | 20,384 | 201 | 0.9890 | 139.6 |
| on | Shuffled Full | 48 | 20,546 | 15,232 | 5,314 | 0.7389 | 2,155.8 |
| off | Original | 48 | 20,564 | 0 | 20,564 | 0.0000 | 6,760.1 |
| off | Stable Full | 48 | 20,546 | 0 | 20,546 | 0.0000 | 6,757.6 |
| off | Shuffled Full | 48 | 20,546 | 0 | 20,546 | 0.0000 | 6,769.3 |

An offline block-level simulation, written before any GPU work, predicted cached
fractions of 0.9894 / 0.9895 / 0.7388. Measured: 0.9889 / 0.9890 / 0.7389.

Claim: real agent traffic already sends a byte-identical schema every turn, so
there is nothing left to stabilise. Across 35 adjacent request pairs the first
differing token is always inside the appended message, never in the system prompt
or the tools.

## Finding 2 — the schema sits behind a volatile prefix; hoisting it helps (84 measurements, 3 restarts)

Offline observation: **across** sessions (comparing the first request of
different sessions), the first differing token moves to the **system prompt** in
90 of 91 pairs, because OpenCode injects per-session context there. The template
places the tool schema after the system text, so a byte-stable schema is
unreachable.

Test: render each prompt locally with the same tokenizer, then either leave it
as-is or move the `# Tools` block to the front of the system message. The
transform is asserted byte-preserving (same multiset of characters). Because the
chat endpoint renders server-side, prompts are posted as raw strings to
`/v1/completions`. The server's reported `prompt_tokens` matched the locally
counted token length on every request. Session heads are replayed in capture
arrival order within one server lifetime.

| layout | n | prompt tok | cached | new prefill | cached frac | wall ms |
|---|---:|---:|---:|---:|---:|---:|
| as-is | 42 | 20,418 | 7,672 | 12,611 | 0.3808 | 4,806.3 |
| hoisted | 42 | 20,419 | 11,864 | 7,447 | 0.6303 | 3,211.4 |

Paired over the same 42 head/run cells: 33 better, 3 worse, 6 tied.

Split by tool-set size (offline predicted the sign would flip with whether
sessions share a toolset):

| group | offline predicted | measured |
|---|---:|---:|
| 10 tools — shared toolset, n=33 | +25.5 pp | +25.5 pp |
| pooled | +25.0 pp | +24.9 pp |
| 8 tools — different toolset, n=3 | negative | −25.0 pp |

## Finding 3 — trimming the schema to the tools actually used (112 requests)

**Frozen Thin** keeps only the tools the session turned out to call, frozen from
turn 1. Measured on session heads:

| | cache OFF | cache ON |
|---|---:|---:|
| prompt tokens | −23.3% | −23.3% |
| new prefill tokens | −23.3% | −36.5% |
| wall time | −33.1% | −41.8% |

The author notes the latency drop exceeds the token drop (1.42× and 1.80×) and
attributes it to quadratic prefill attention, explicitly declining to call it a
discovery.

## What the author already concedes

- Frozen Thin is an oracle chosen with hindsight; no selector is demonstrated and
  correctness is untested.
- Hoisting's effect on output quality is completely untested.
- KV footprint is unmeasured — the gauge is read after the request returns, when
  the KV has been freed.
- No concurrency or goodput data.
- One client, one model, one trace, 14 sessions; the 170-tool configuration was
  not run.
- The sign flip rests on 3 heads.
- The layout result reflects one arrival order.
- vLLM 0.9.2 rather than current stable, forced by a WSL2 limitation.

## What to attack

Work through these, and add anything they miss:

1. **Is the cached-token metric sound?** The counter delta is attributed to a
   single request on the assumption of serial execution. What would break that
   assumption, and would it bias toward or against the reported effects? Does the
   counter measure tokens, or blocks, and does it matter here?
2. **Is the hoisting comparison fair?** The two layouts contain identical bytes,
   but is anything else different — tokenisation at the seam, block alignment,
   chunked-prefill boundaries, or the `/v1/completions` path versus the chat path
   used in Finding 1?
3. **Is arrival order doing the work?** Heads are replayed in capture order in a
   single server lifetime. How much of the +24.9 pp could be an artifact of that
   order rather than the layout? What ordering would be a fair control?
4. **Is the offline/GPU agreement too good?** Three predictions matched to
   ≤0.0005 and a layout prediction matched to 0.1 pp. Is there a path by which the
   simulation and the measurement share an assumption, making the agreement
   circular rather than confirmatory?
5. **Does the negative result generalise?** Finding 1 rests on the schema being
   byte-identical in this client. What client behaviour would overturn it, and how
   likely is it in practice?
6. **Is the framing honest?** The author reframes from "tool working-set
   reduction" to "request layout". Given the evidence, is that the framing the
   data supports, or is it over-claiming from a 14-session single-client trace?
7. **What single experiment would most change your confidence** in the layout
   result — and is it cheap?

Finish with: which claims you would let stand as written, which need softening
and to what, and which should be cut.
