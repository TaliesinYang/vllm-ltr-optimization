# Benchmark Harness R1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. The user explicitly requested inline autonomous execution and no commits.

**Goal:** Upgrade the live scheduler harness to scheduled-arrival latency, warmup-aware measurement, resumable selectable runs, two additional scheduler policies, offline paired analysis, and schema-v2 plotting.

**Architecture:** Keep `scheduler_benchmark.runner` as the single live-run authority. Each requested `(scenario, load, profile, repeat)` becomes one deterministic subrun with a derived seed, atomic JSON/CSV evidence, and a content fingerprint; the final output only aggregates those subruns. Policy additions remain pure ordering functions plus thin vLLM scheduler adapters.

**Tech Stack:** Python 3.11, asyncio/aiohttp, dataclasses, JSON/CSV, hashlib, matplotlib, pytest.

**Implementation status:** Complete. Focused harness tests: 64 passed. Environment-safe broader suite: 94 passed. Full collection remains environment-blocked by missing torch; additional pre-existing suites require loopback sockets, network access, or a compatible protobuf runtime.

---

### Task 1: Scheduled timing and warmup measurement

**Files:**
- Modify: `scheduler_benchmark/runner.py`
- Modify: `scheduler_benchmark/gateway_overhead.py`
- Modify: `tests/test_runner.py`
- Modify: `tests/test_gateway_overhead.py`

- [ ] Add failing tests proving `ttft_ms/ttlt_ms` include dispatch lag while `send_ttft_ms/send_ttlt_ms` preserve send-relative values.
- [ ] Add failing tests for count/ratio warmup resolution, full-run completeness, measured-window summaries, and measured-window throughput denominator.
- [ ] Run `python -m pytest tests/test_runner.py tests/test_gateway_overhead.py -q` and confirm failures are caused by missing fields/helpers.
- [ ] Extend `ResponseSample` with `send_ttft_ms`, `send_ttlt_ms`, `dispatch_lag_ms`, `category`, `policy`, `profile`, and absolute scheduled/dispatch/first-token/completion timestamps.
- [ ] Enrich samples in `run_replay()` from the scheduled timestamp and actual dispatch timestamp; keep `stream_completion()` responsible only for send-relative measurements.
- [ ] Implement `resolve_warmup_requests()` and `measurement_window()` so completeness sees every request while `summarize_samples()` sees only measured samples and the truncated duration.
- [ ] Run the focused tests and confirm green.

### Task 2: Selectable, resumable schema-v2 orchestration

**Files:**
- Modify: `scheduler_benchmark/runner.py`
- Modify: `tests/test_runner.py`

- [ ] Add failing tests for `--scenario`, `--load`, `--profile`, arbitrary `--repeats`, `--warmup-requests`/`--warmup-ratio`, and `--resume`.
- [ ] Add failing tests proving seeds are identical for a fixed `(profile, load, repeat)` regardless of selection order.
- [ ] Add failing tests for aggregate scatter output:

```python
assert aggregate["metrics"]["p99_ttlt_ms"] == {
    "values": [90.0, 100.0, 110.0],
    "mean": 100.0,
    "min": 90.0,
    "max": 110.0,
}
```

- [ ] Add failing tests for canonical fingerprint verification and atomic subrun JSON plus sample CSV output.
- [ ] Implement scenario/load selection with legacy defaults `(steady,40/70/90)` plus `(burst,90)`, and profile filtering for `id`, `ood`, or `mixed`.
- [ ] Derive seeds from SHA-256 of the canonical `(profile, load, repeat)` tuple.
- [ ] Emit schema version 2 subrun records, compute the required fingerprint over schema/workload/policy/load/profile/seed/warmup/completed/errors, and write JSON/CSV through same-directory temp files followed by `Path.replace()`.
- [ ] Implement resume by scanning and fingerprint-verifying completed subrun records before skipping them.
- [ ] Aggregate arbitrary repeat counts as per-repeat values plus mean/min/max, with no t-critical logic.
- [ ] Run `python -m pytest tests/test_runner.py -q` and confirm green.

### Task 3: Prompt-length SJF and LTR+aging policies

**Files:**
- Modify: `scheduler_benchmark/policies.py`
- Modify: `scheduler_benchmark/vllm_scheduler.py`
- Modify: `scheduler_benchmark/runner.py`
- Modify: `tests/test_policies.py`
- Modify: `tests/test_vllm_scheduler.py`

- [ ] Add failing policy tests proving prompt-length ordering, queue-wide FCFS fallback for an empty prompt, and aging promotion for an old expensive LTR request.
- [ ] Add failing adapter tests proving `PromptLengthSJFScheduler.uses_predictor is False`, `LTRAgingScheduler.uses_predictor is True`, and both exact class paths map to policies.
- [ ] Add `RequestContext.prompt_token_count`, `prompt_sjf`, and `ltr_aging`; use `_ltr_score()` only for the aging variant while preserving pure LTR.
- [ ] Add both scheduler classes and `SCHEDULER_CLASS_TO_POLICY` entries. Describe prompt SJF as having zero predictor inference overhead.
- [ ] Run `python -m pytest tests/test_policies.py tests/test_vllm_scheduler.py tests/test_runner.py -q` and confirm green.

### Task 4: Offline paired analysis and Figure 6

**Files:**
- Create: `scripts/analyze_paired_deltas.py`
- Create: `tests/test_paired_deltas.py`
- Modify: `scripts/plot_fig6.py`
- Modify: `tests/test_plot_fig6.py`

- [ ] Add failing tests for matching subruns by scenario/load/profile/repeat/seed and reporting policy-B minus policy-A raw differences plus mean/min/max.
- [ ] Implement the independent analysis CLI; do not add paired statistics to the live runner.
- [ ] Replace Figure 6's fixed four-policy/three-repeat validation with schema-v2 group loading, arbitrary policy sets, arbitrary repeat counts, and dynamic styles.
- [ ] Add failing then passing render tests using two policies and five repeats.
- [x] Run `python -m pytest tests/test_paired_deltas.py tests/test_plot_fig6.py -q` and confirm green.

### Task 5: Parity compatibility and full verification

**Files:**
- Modify: `scheduler_benchmark/vllm_scheduler.py`
- Modify: `ltr_training/fcfs_replay.py`
- Modify: `ltr_training/fcfs_parity.py`
- Modify: `tests/test_fcfs_parity.py`
- Modify: `tests/test_runner.py`

- [ ] Add failing parity tests against schema-v2 aggregate means and scheduled-primary metric names.
- [ ] Update both parity readers to consume the scheduled-primary `ttft_ms/ttlt_ms` aggregates while retaining send-relative values only as diagnostics.
- [ ] Run all focused harness tests.
- [ ] Run `python -m pytest tests/ -q` and record the exact pass/fail count.
- [ ] Run both CLI help smokes plus a local synthetic resume/plot/paired-analysis smoke without contacting a live endpoint.
- [ ] Review the final file list and report `ARTIFACT n: done/blocked-因为X`; do not commit.
