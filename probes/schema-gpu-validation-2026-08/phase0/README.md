# Phase 0A — offline audit of the kill test (no GPU)

Follow-up to `../../prefix-cache-killtest-2026-08-01/`. Answers the four offline
questions in RQ0 and adds one that the cross-session result forced.

```bash
python phase0_audit.py --out results          # default renderer: Qwen2.5-7B-Instruct
python phase0_audit.py --out results_qwen3 --model Qwen/Qwen3-8B
```

## 1. Pair audit — 35/35 clean

`results/audit_pairs.csv`, one row per parent-child pair with the evidence needed to
sign each off by hand. Machine-checkable flags, all zero:

| Check | Violations |
|---|---:|
| `strict_prefix_ok` false (parent messages not a strict prefix) | 0/35 |
| `system_hash_equal` false | 0/35 |
| `model_equal` false | 0/35 |
| `toolset_equal` false | 0/35 |
| `prev_has_siblings` true (branch mis-linked as a chain) | 0/35 |
| `body_is_duplicate` true (same request body seen twice) | 0/35 |

Every pair appends exactly the roles `assistant|tool` (35/35) — the agent-loop shape.
Inter-request gap: min 0.0 s, median 1.2 s, max 16.0 s.

The remaining manual step is semantic, not structural: confirm the appended tool result
belongs to the preceding tool call. The CSV carries `appended_roles`, `dt_seconds` and
both request ids for that read-through.

## 2. Block-size sensitivity — conclusion holds at 8/16/32

Median reuse ratio:

| Dataset | bs | Original | Stable Full | Shuffled Full | Frozen Thin |
|---|---:|---:|---:|---:|---:|
| vanilla_10tool | 8 | 0.9898 | 0.9899 | 0.7388 | 0.9868 |
| vanilla_10tool | 16 | 0.9894 | 0.9895 | 0.7388 | 0.9868 |
| vanilla_10tool | 32 | 0.9894 | 0.9887 | 0.7382 | 0.9866 |
| full_170tool | 8 | 0.9975 | 0.9975 | 0.3144 | 0.9922 |
| full_170tool | 16 | 0.9974 | 0.9974 | 0.3144 | 0.9919 |
| full_170tool | 32 | 0.9971 | 0.9971 | 0.3142 | 0.9912 |

Stable − Original stays within ±0.1 pp at every block size. Not a block-size artefact.

## 3. Renderer sensitivity — not a template artefact

Re-running the full kill test with the **Qwen3-8B** template (the model planned for the
GPU phase; different template sha256 from Qwen2.5) reproduces every number:

| Dataset | Policy | Qwen2.5-7B | Qwen3-8B |
|---|---|---:|---:|
| vanilla | Original | 0.9894 | 0.9897 |
| vanilla | Stable Full | 0.9895 | 0.9899 |
| vanilla | Shuffled Full | 0.7388 | 0.7390 |
| 170-tool | Original | 0.9974 | 0.9975 |
| 170-tool | Shuffled Full | 0.3144 | 0.3145 |

**Ollama vs vLLM.** The trace was actually served by Ollama. Its template for
`qwen2.5:7b-instruct` was recovered byte-exact from the 201 blob store
(`~/.ollama/models/blobs/sha256-eb4402837c78…`) and is structurally identical where it
matters: same ChatML frame, same `# Tools` preamble, schema emitted as one contiguous
`<tools>…</tools>` block inside the system message, after the system text. The
policy comparison is invariant to that difference.

One real divergence, worth carrying into the design: Ollama renders each tool as
`{"type": "function", "function": {{ .Function }}}` where `.Function` is re-marshalled
through a Go struct, so **Ollama normalises JSON key order server-side**; the HF/vLLM
Jinja template uses `tojson` and preserves whatever order the client sent. "Canonicalise
the schema" therefore means different things on the two backends.

## 4. Cross-session reuse — the schema is stable, the layout is not

This is the check that changes the direction.

Taking the **first request of every multi-turn session** and comparing all pairs:

| Dataset | n pairs | median token LCP | median reuse | first change in `system` |
|---|---:|---:|---:|---:|
| vanilla_10tool | 91 | 5 665 | 0.2820 | 90/91 |
| full_170tool | 21 | 1 910 | 0.0934 | 21/21 |

Across sessions the earliest mutation moves from `messages` to **`system`** — OpenCode's
system prompt carries per-session context that changes. Because the template places the
schema *behind* the system prompt, a schema that is provably byte-stable is nonetheless
**unreachable** by the prefix cache across sessions.

### What hoisting recovers

`hoist_schema()` reorders the rendered prompt so the `# Tools` block sits at the front of
the system message, before the volatile system text. It is a pure byte permutation
(asserted), i.e. exactly what a client or gateway could emit instead.

| Dataset | Sessions share a toolset | n | as-is | hoisted | Δ | improved |
|---|---|---:|---:|---:|---:|---:|
| vanilla_10tool | yes | 56 | 0.3706 | 0.6252 | **+25.5 pp** | 55/56 |
| vanilla_10tool | no | 35 | 0.0000 | 0.0674 | +6.7 pp | 24/35 |
| full_170tool | yes | 7 | 0.1034 | 0.7859 | **+68.2 pp** | 7/7 |
| full_170tool | no | 14 | 0.0933 | 0.0024 | **−9.1 pp** | 1/14 |

Pooled: vanilla 0.2820 → 0.5318 (+25.0 pp, 79/91); 170-tool 0.0934 → 0.0024 (−9.1 pp,
8/21).

The mechanism is clean and the sign flips: hoisting pays exactly when the schema is
shared between sessions, and **costs** when it is not, because a differing schema at
offset 0 destroys the prefix immediately instead of after the shared system preamble.
There is a real crossover here — a layout/ordering decision conditioned on schema
sharing — which is a policy question, not a canonicalisation question.

Note this is the same +20 pp bar the original kill test set for "Go", now met on one
axis (same-toolset, hoisted) and violated on another (different-toolset). It reframes
the direction as **request layout**, matching the Conditional-Go escape hatch — except
the trigger is cross-session divergence, not within-session, which the earlier
experiment could not see.

## What this still cannot say

- **No GPU, no latency claim.** Everything here is full-block LCP, an upper bound on
  cacheability. Real hit rates are additionally bounded by eviction, block hashing and
  scheduling.
- **Cross-session pairs are all-pairs combinations, not an arrival order.** They measure
  *potential* sharing between two session heads, not what a cache with finite capacity
  and a real arrival process would retain.
- **Not independent samples.** 91 and 21 pairs come from 14 and 7 sessions; the same
  session appears in many pairs. No CI is reported for that reason.
- **Hoisting changes what the model reads.** Putting 5 k–38 k tokens of schema before the
  system prompt may change tool-selection quality. Completely untested here.
- **Serving support is assumed.** Neither the HF nor the Ollama template emits this
  layout; it would have to come from the client or a gateway rewrite, and the server
  must accept it.
- **One client, two configs.** Still OpenCode only. A second client family remains the
  largest open gap in RQ0.
