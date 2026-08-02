# Draft — not sent (v2, layout result now leads)

**To:** j.ryoo@fdu.edu
**Subject:** Tool-schema caching: the original idea fails, but the layout does not

---

Hi Professor Ryoo,

I ran GPU validation on the tool-schema caching direction. The original
hypothesis is dead, but a different one came out of it that I think is stronger.

**What failed.** I had assumed that stabilising the tool schema — fixing its
order and serialisation — would increase reusable prefix across an agent's turns.
On real vLLM (V1 engine, Qwen3-8B-AWQ) it moves the cached fraction from 0.9889
to 0.9890. Real OpenCode traffic already sends a byte-identical schema every
turn, so there was nothing to stabilise. Across 35 adjacent request pairs the
first change is always in the appended message, never in the system prompt or the
tools.

I checked the instrument rather than trusting it. Deliberately shuffling the tool
order costs 25 percentage points of cached fraction and takes latency from 137 ms
to 2156 ms; turning the prefix cache off collapses every policy to identical
numbers. An offline block-level simulation I wrote first predicted 0.9894 /
0.9895 / 0.7388 for the three policies, and the GPU measured 0.9889 / 0.9890 /
0.7389.

**What replaced it.** Within a session the schema is already reused, so I looked
across sessions. There the earliest divergence moves from the appended message to
the *system prompt* — 90 of 91 session pairs. OpenCode puts per-session context
in the system prompt, and the template puts the tool schema behind it, so a
schema that is provably byte-stable is unreachable by the cache across sessions.

Moving the schema block to the front of the system message — the same bytes, only
reordered — recovers a lot of that:

- cached fraction 0.3808 → 0.6303 (+24.9 points)
- newly computed prefill tokens 12611 → 7447 (−40.9%)
- wall time 4806 → 3211 ms (−33.2%)
- 33 of 42 paired requests improve, 3 get worse

The offline model predicted +25.5 points for the sessions that share a toolset;
the GPU measured +25.5. It also predicted that sessions *not* sharing a toolset
would be hurt, and they are, by −25.0 points — hoisting puts a differing schema
at offset 0, which destroys the prefix immediately instead of after the shared
preamble. So the right layout is conditional on the deployment, which makes it a
policy question rather than a constant.

I also tested trimming the schema to the tools a session actually uses: −23.3%
tokens, −41.8% latency. It works, but it needs hindsight to pick the set and it
removes capability the model might need, so I would treat it as a second lever
rather than the main one.

So I would like to reframe the direction from *preserving cache across turns* to
**request layout for agentic serving** — where the tool schema sits relative to
the volatile part of the prompt, and when moving it pays.

Three things I have not shown:

1. Whether hoisting changes output quality. It puts 5K tokens of schema ahead of
   the system prompt; the bytes are identical but the reading order is not, and I
   have not measured tool-selection accuracy.
2. Any concurrency or goodput number. A single 32K request already fills most of
   the KV on the 16 GB card I have.
3. Generality — one client (OpenCode), one model, one trace, 14 sessions.

Before I invest in the quality and concurrency work: does the layout framing seem
worth pursuing to you? And is the negative result on schema stabilisation worth
writing up on its own, or only as motivation for the layout result?

Code and data: <repo URL>

Alex

---

## Notes for Alex before sending

- Replace `<repo URL>`.
- Every number is in `probes/schema-gpu-validation-2026-08/RESULTS.md`; the
  offline predictions are in `probes/schema-gpu-validation-2026-08/phase0/README.md`.
- Deliberately **not** claimed: KV footprint reduction (the gauge is sampled after
  the KV is freed, so it is unmeasured), concurrency, any deployable tool selector.
- Also not mentioned: the rig is vLLM 0.9.2 rather than current stable, running
  under WSL2, because 0.26.0's V1 worker needs UVA which WSL2 lacks. Add a line if
  you expect him to ask; `HARDWARE-FINDINGS.md` has the reasoning and the V100
  results that rule out anything below compute capability 8.0.
- If he pushes on novelty: the honest position is that the super-proportional
  latency effect is just quadratic attention, and the contribution is the
  measurement that real agent traffic has its volatile segment *in front of* its
  stable segment, plus the conditional layout policy that follows.
