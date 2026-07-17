# Gateway Main-Path Artifacts 6-8 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make VeloxMesh gateway the benchmark main path by adding the `/v1/decision` stub service, a CPU-testable gateway-to-vLLM chain, and an isolated FCFS direct-versus-gateway overhead run.

**Architecture:** Keep the committed predictor, policy, scheduler, and replay logic unchanged. Add a small standard-library decision HTTP service around the Artifact 1 predictor, a local mock stack that models `gateway -> decision -> vLLM`, and a separate overhead runner so direct-engine measurements cannot enter the four-policy comparison.

**Tech Stack:** Python 3.11+, `http.server`, `urllib.request`, existing `aiohttp` replay client, pytest.

---

## File map

- `scheduler_benchmark/decision_service.py`: schema validation, reason precedence, predictor adaptation, HTTP application, readiness and bounded concurrency.
- `scripts/run_decision_service.py`: executable `/v1/decision` service entrypoint.
- `tests/test_decision_service.py`: pure contract and real loopback HTTP tests.
- `scheduler_benchmark/gateway_transport.py`: validate decision responses and inject only reliable prediction metadata into `vllm_xargs`.
- `scheduler_benchmark/mock_stack.py`: local decision service, mock VeloxMesh gateway, and mock SSE engine for CPU E2E.
- `scripts/run_mock_gateway_stack.py`: executable local stack entrypoint.
- `tests/test_gateway_transport.py`: reliable/fallback transport tests and full loopback chain.
- `scheduler_benchmark/runner.py`: gateway-path manifest field; no policy/scheduler changes.
- `tests/test_runner.py`: prove the main runner labels its endpoint as gateway.
- `scheduler_benchmark/gateway_overhead.py`: one-repeat matched FCFS direct/gateway measurement and absolute deltas.
- `scripts/run_gateway_overhead.py`: isolated overhead-mode CLI.
- `tests/test_gateway_overhead.py`: matched input, completeness, and absolute delta tests.
- `README.md`: Artifact 6 curl contract and Artifact 7/8 local commands/non-claims.

### Task 1: Artifact 6 decision contract and HTTP service

**Files:**
- Create: `scheduler_benchmark/decision_service.py`
- Create: `scripts/run_decision_service.py`
- Create: `tests/test_decision_service.py`

- [ ] **Step 1: Write contract tests first**

Cover a reliable response, all four reason codes in precedence order, omission of `estimated_tokens` for unreliable predictions, `decision_id` echo, and the spec errors:

```python
def test_ood_takes_reason_precedence() -> None:
    app = make_app(confidence=0.1, ood=True)
    response = app.decide(valid_request(with_optional=False))
    assert response["decision_id"] == "decision-1"
    assert response["prediction_reliable"] is False
    assert response["reason_code"] == "ood_rejected"
    assert "estimated_tokens" not in response

@pytest.mark.parametrize(
    ("mutation", "status", "code"),
    [
        (lambda row: row.update(schema_version="2.0"), 400, "invalid_schema"),
        (lambda row: row.pop("request_id"), 422, "invalid_request"),
        (lambda row: row["generation_controls"].update(temperature=0.7), 422, "unsupported_controls"),
    ],
)
def test_typed_errors(mutation, status, code) -> None:
    request = valid_request()
    mutation(request)
    with pytest.raises(DecisionError) as raised:
        make_app().decide(request)
    assert raised.value.status == status
    assert raised.value.error_code == code
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_decision_service.py -q`

Expected: collection fails because `scheduler_benchmark.decision_service` does not exist.

- [ ] **Step 3: Implement the pure decision application**

Implement `DecisionError(status, error_code, retryable)`, bounded request validation, predictor input serialization, feature-variant missing-field checks, and this precedence:

```python
def reason_code(prediction, missing_optional_features, threshold):
    if prediction.ood:
        return "ood_rejected"
    if prediction.confidence < threshold:
        return "low_reliability"
    if missing_optional_features:
        return "missing_optional_features"
    return "prediction_reliable"
```

The development adapter maps normalized predictor score to `[1, 2048]`; this is explicitly a stub and is replaced with the real checkpoint adapter later. Only `prediction_reliable` responses contain `estimated_tokens`.

- [ ] **Step 4: Add real loopback HTTP tests, then implement HTTP adapter**

Test `POST /v1/decision`, `GET /healthz`, invalid JSON, oversized `Content-Length`, not-ready, saturated concurrency, and predictor exception. HTTP error bodies must be exactly shaped as `{schema_version, error_code, retryable}`.

- [ ] **Step 5: Verify GREEN**

Run: `python -m pytest tests/test_decision_service.py -q`

Expected: all decision service tests pass.

- [ ] **Step 6: Add executable service entrypoint**

`scripts/run_decision_service.py` accepts `--host`, `--port`, `--score`, `--confidence`, `--ood`, `--feature-variant`, `--max-body-bytes`, and `--max-concurrency`; it creates `ConstantPredictor` from Artifact 1 and serves until interrupted.

### Task 2: Artifact 7 gateway transport and CPU E2E

**Files:**
- Create: `scheduler_benchmark/gateway_transport.py`
- Create: `scheduler_benchmark/mock_stack.py`
- Create: `scripts/run_mock_gateway_stack.py`
- Create: `tests/test_gateway_transport.py`
- Modify: `scheduler_benchmark/runner.py`
- Modify: `tests/test_runner.py`

- [ ] **Step 1: Write transport RED tests**

```python
def test_reliable_decision_injects_namespaced_vllm_xargs() -> None:
    payload = apply_decision_to_payload(base_payload(), reliable_bundle())
    assert payload["vllm_xargs"]["workflow_estimated_tokens"] == 321
    assert payload["vllm_xargs"]["prediction_reliable"] is True
    assert payload["vllm_xargs"]["decision_id"] == "decision-1"

def test_unreliable_decision_uses_native_fallback() -> None:
    payload, audit = apply_decision_to_payload(base_payload(), unreliable_bundle())
    assert "workflow_estimated_tokens" not in payload.get("vllm_xargs", {})
    assert audit.fallback_source == "fallback_native"
```

- [ ] **Step 2: Verify RED and implement transport**

Run: `python -m pytest tests/test_gateway_transport.py -q`

Expected RED: missing module. Implement response validation and immutable payload copying. Malformed/error/timeout responses return the original payload plus a typed `fallback_native` audit; no retry.

- [ ] **Step 3: Add full loopback chain test before mock-stack code**

Start three ephemeral loopback servers and call them in this order:

```text
runner client -> mock gateway /v1/completions
              -> decision service /v1/decision
              -> mock engine /v1/completions
```

Assert the engine receives `workflow_estimated_tokens` only for reliable decisions and returns valid OpenAI-style SSE usage to `stream_completion`.

- [ ] **Step 4: Implement the reusable local mock stack**

The gateway derives the decision request from incoming completion JSON and request headers, performs exactly one decision RPC, applies `gateway_transport`, forwards once to the engine, and streams the engine body/status/content type back. The mock engine records its last request for assertions and emits deterministic SSE.

- [ ] **Step 5: Wire runner endpoint semantics without changing benchmark logic**

Keep `--endpoint` required, change help text to `VeloxMesh OpenAI-compatible gateway endpoint`, and emit:

```python
"request_path": "client->gateway->decision->vllm",
"gateway_endpoint": args.endpoint,
```

No direct-engine option is added to the four-policy runner.

- [ ] **Step 6: Verify GREEN**

Run: `python -m pytest tests/test_gateway_transport.py tests/test_runner.py -q`

Expected: all tests pass, including real HTTP CPU E2E.

### Task 3: Artifact 8 isolated gateway-overhead mode

**Files:**
- Create: `scheduler_benchmark/gateway_overhead.py`
- Create: `scripts/run_gateway_overhead.py`
- Create: `tests/test_gateway_overhead.py`

- [ ] **Step 1: Write matched-run RED tests**

```python
def test_gateway_overhead_uses_same_requests_and_arrivals_once() -> None:
    result = asyncio.run(
        run_overhead_pair(
            workload=workload,
            offsets=[0.0],
            direct_sender=recording_sender(direct_seen),
            gateway_sender=recording_sender(gateway_seen),
        )
    )
    assert result["mode"] == "gateway_overhead_fcfs"
    assert result["repeats"] == 1
    assert direct_seen == gateway_seen

def test_absolute_overhead_is_gateway_minus_direct() -> None:
    report = absolute_overhead(direct_metrics(), gateway_metrics())
    assert report["p95_ttlt_ms"] == 3.5
```

- [ ] **Step 2: Verify RED and implement calculations**

Run: `python -m pytest tests/test_gateway_overhead.py -q`

Expected RED: missing module. Implement one shared arrival schedule, sequential direct/gateway replays, complete-request gates, per-route raw metrics, and signed absolute deltas for TTLT/TTFT/throughput. Never merge this result with policy result schema or Fig. 6 inputs.

- [ ] **Step 3: Add isolated CLI**

`scripts/run_gateway_overhead.py` requires `--direct-endpoint`, `--gateway-endpoint`, `--model`, `--workload`, `--capacity-rps`, and `--output`. It writes one JSON report labelled `gateway_overhead_fcfs`, with no policy comparison label.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_gateway_overhead.py -q`

Expected: all overhead tests pass.

### Task 4: Mingye handoff docs and completion audit

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add the Artifact 6 contract example**

Document service startup and a complete curl body containing schema, IDs, messages, controls, and optional workflow fields. Show reliable response and state that unreliable/non-200 paths omit `workflow_estimated_tokens` and use `fallback_native` in gateway transport.

- [ ] **Step 2: Add CPU E2E and overhead commands**

Document mock-stack startup, runner `--endpoint` pointing to mock gateway, and the separate direct/gateway overhead command. Mark mock results as wiring evidence only; real VeloxMesh, real checkpoint, vLLM 0.24 queue-order proof, GPU latency, and two-turn tool calls remain rental-day gates.

- [ ] **Step 3: Run full verification**

Run:

```bash
python -m pytest -q
python scripts/run_decision_service.py --help
python scripts/run_mock_gateway_stack.py --help
python scripts/run_gateway_overhead.py --help
git diff --check
```

Expected: full suite passes, all CLIs exit 0, and `git diff --check` is clean.

- [ ] **Step 4: Requirement audit**

Confirm each explicit Artifact 6–8 requirement against current files/test output. Report GPU/real VeloxMesh checks as blocked by unavailable environment, not as passed.
