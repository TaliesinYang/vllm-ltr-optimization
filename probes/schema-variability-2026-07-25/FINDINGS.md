# Probe: tool-schema variability across a real agent's requests (Hole 2)

**Date:** 2026-07-25 · **Rig:** OpenCode 1.18.5 → local echo server (`echo_server_v2.py`,
port 8099, SSE-capable, forces 2 tool-call turns per run via a read-only `glob` call).
**Raw capture:** `captured_requests_v2.jsonl` (28 requests: 7 runs × ~4 requests).

## Question

If a client resends the same fixed tool set on every request, schema text cannot
discriminate between requests in the same queue — which would kill "schema text as a
per-request scheduling feature" for single-tenant traffic.

## Setup

| Config | Description | Tools |
|---|---|---|
| full | user's real OpenCode setup (6+ MCP servers) | 170 |
| vanilla | isolated `XDG_CONFIG_HOME`, project config only (native tools) | 10 |

Runs: full × {3 build tasks, 1 plan task}, vanilla × {2 build tasks, 1 plan task}.
Each run = 3 agent-loop turns (turn 1, after tool result 1, after tool result 2)
+ 1 title-generation side request.

## Results (sha256 over sorted-key JSON of `tools` array)

| Axis | Result |
|---|---|
| Across turns within one session (1→3) | **byte-identical**, same hash |
| Across different tasks, same config+mode | **byte-identical**, same hash |
| build vs plan mode, vanilla | different hash: 21,188 B vs 21,026 B (10 tools both) |
| build vs plan mode, full config | **identical** (147,235 B both) |
| full vs vanilla config | 147,235 B / 170 tools vs 21,188 B / 10 tools (~7×) |
| agent-loop request vs title-gen request | tools present vs **tools = []** — ~1 in 4 requests per run carries zero schema |

Distinct schema hashes over 21 tool-bearing requests: **3** (= one per config×mode cell,
minus the full-config mode collision).

## Interpretation

1. **Within a single deployment (config + agent mode), the schema is a constant.**
   It carries zero per-request information for ranking requests from the same client.
   Hole 2 is confirmed for single-tenant traffic — worse under vanilla (fixed 10 built-ins).
2. Schema text discriminates only **across** deployments/configs (147 KB vs 21 KB is a huge
   signal) and between **request kinds** within a session (agent turn vs no-tool utility
   request such as title generation).
3. Consequently the offline +0.203 τ (schema text vs scalar tool features, ToolACE) maps to
   a **multi-tenant / mixed-client queue**, because ToolACE varies tools per sample. It does
   NOT support per-request ranking inside one client's stream.
4. Open counter-argument (next cheap experiment): if schema is constant per client, a
   per-client hash + running output-length statistics might match schema text on warm
   clients. Schema text's residual value is **cold-start / unseen-tool-set generalization**
   — testable offline with an unseen-tool-set split, no GPU.

## Reproduce

```bash
python3 echo_server_v2.py &   # port 8099
cd <probe-project>            # opencode.json points probe provider at :8099
opencode run -m probe/probe-model "<task>"          # -m flag REQUIRED
XDG_CONFIG_HOME=<empty-dir> opencode run -m probe/probe-model "<task>"   # vanilla
```
