# Agent trace collection — 2026-07-26 (ticket #7)

Real multi-turn agent traffic captured at the gateway boundary. **Replay/evaluation
material ONLY — never training data** (ToolACE remains the training set).

## Topology

OpenCode (vanilla config, native tools) → trace proxy `:9101` (`trace_proxy.py`) →
VeloxMesh gateway `:9100` → SSH tunnel → 201 Ollama `qwen2.5:7b-instruct`.
Chain runbook: `../gateway-live-chain-2026-07-26/NOTES.md`.

## Contents

- `agent_trace_vanilla.jsonl.gz` — 75 requests (sha256 a530c4ea54616002…). One row per
  request: full request body (messages, tools), status, e2e_ms, `X-Queue-Wait-Ms`,
  request id, `usage` (**true completion_tokens**), finish_reason, tool-call count.
- `trace_proxy.py` — the capture proxy (buffered SSE relay, usage parsed from stream).
- `run_trace_batch.sh` — the 25-task batch driver (tasks listed inline).

## Batch outcome

25 varied coding-agent tasks (read/summarize/edit/create/search/refactor) against a
seeded toy Python project; 24 rc=0, 1 timeout (rc≠0). All 75 HTTP 200.

## Characterization (extends the schema-variability probe)

| Metric | Value |
|---|---|
| Requests | 75 (50 tool-bearing, **25 = 33% zero-tool** — title-gen/utility) |
| Distinct tool-schema hashes | **3** (vanilla constancy again; build/plan/subagent variants) |
| completion_tokens | p50 = 42, p90 = 143, p99 = 328, max = 328, min = 4 |
| prompt_tokens | p50 = 2050, max = 4072 |
| Schema share of payload | 26.8% mean (vanilla 10-tool config; 68.9% was the 170-tool config) |
| Turn depth (msgs/request) | p50 = 3, max = 10 |
| Requests emitting tool_calls | 34/75 |
| e2e_ms through full chain | p50 = 1473, p90 = 3197, max = 5517 |
| X-Queue-Wait-Ms | 0 on all rows (single client, no contention — expected) |

## Deliberate scope cut (recorded, not silent)

**MCP-heavy live execution was skipped.** Driving the user's real MCP servers (memory
graphs, notebook services, paper tools) with a 7B model has real side-effect risk, and
the 170-tool schema payload for that config is already captured byte-exact in
`../schema-variability-2026-07-25/`. If a live MCP-heavy trace is ever needed, run it
against a sandboxed MCP set, not the real one.

## Reproduce

```bash
python3 trace_proxy.py <capture.jsonl>          # :9101 → :9100
bash run_trace_batch.sh                          # 25 tasks, vanilla XDG config
```
