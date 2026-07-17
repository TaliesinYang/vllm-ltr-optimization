# vllm-ltr-optimization

CSCI 6806 capstone — reproduce + optimize **Learning-to-Rank scheduling** for low-latency LLM serving.

**Thesis:** *schedule by a predictable property → lower latency.*

## Three threads
- **Scheduling** (Dazhi) — reproduce the LTR scheduler; fix its ranker overfitting with **PARS** (pairwise ranking + BERT backbone).
- **Gateway** (Mingye) — latency-aware routing / two-layer semantic cache / admission control ("Velox" design).
- **Evaluation** (Yibo) — a reusable benchmark: MMLU quality gate + serving metrics (TTFT / TPOT / E2E / throughput).

Built on the base paper (Prof. A. S. Kumar, reproducing Fu et al. NeurIPS'24). Base fork: `hao-ai-lab/vllm-ltr`.

## Where to start
| Doc | Purpose |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | Full project context (read first) |
| [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) | Environment setup + how to reproduce (**start here to run**) |
| [`docs/references.md`](docs/references.md) | Papers to reproduce / compare, with `schedule-type` mapping |
| [`docs/presentation-plan.md`](docs/presentation-plan.md) | Midterm presentation outline (Wed 2026-06-24) |

## Status
Reproduction in progress — **baseline first** (FCFS / classification / LTR), PARS after.
GPU: RTX 4090 48GB.

## VeloxMesh `/v1/decision` development service

VeloxMesh is the public benchmark entry point. It calls this repository's
request-time predictor once, then forwards reliable metadata to vLLM. Start the
CPU development service:

```bash
python scripts/run_decision_service.py \
  --host 127.0.0.1 \
  --port 8081 \
  --score 0.5 \
  --confidence 0.9
```

Call the frozen v1 contract:

```bash
curl --fail-with-body http://127.0.0.1:8081/v1/decision \
  -H 'Content-Type: application/json' \
  -d '{
    "schema_version": "1.0",
    "request_id": "request-1",
    "decision_id": "decision-1",
    "model_id": "Qwen/Qwen3.5-9B",
    "workflow_id": "workflow-1",
    "step_id": "step-1",
    "conversation_id": "conversation-1",
    "request_age_ms": 0,
    "messages": [{"role": "user", "content": "Book a flight"}],
    "tools": [{
      "type": "function",
      "function": {"name": "search", "parameters": {}}
    }],
    "generation_controls": {
      "temperature": 0.0,
      "top_p": 1.0,
      "seed": 42,
      "max_tokens": 2048
    },
    "previous_tool_gap_ms": 25
  }'
```

Reliable stub response:

```json
{
  "schema_version": "1.0",
  "decision_id": "decision-1",
  "estimated_tokens": 1024,
  "reliability_probability": 0.9,
  "ood_score": 0.0,
  "prediction_reliable": true,
  "predictor_revision": "stub-constant-v1",
  "feature_variant": "prompt_schema_history_workflow",
  "reason_code": "prediction_reliable"
}
```

Reason precedence is `ood_rejected`, `low_reliability`,
`missing_optional_features`, then `prediction_reliable`. An unreliable HTTP 200
omits `estimated_tokens`. The gateway then selects `fallback_native` and omits
`workflow_estimated_tokens`; the optimizer service does not claim ownership of
that gateway action. Typed errors are `invalid_schema`, `body_too_large`,
`invalid_request`, `unsupported_controls`, `rate_limited`, `not_ready`, and
`internal_error`.

This endpoint currently adapts Artifact 1's constant predictor stub and uses a
deterministic byte serializer. It proves the HTTP contract only. Replace the
stub adapter with the trained checkpoint and exact training serializer before
reporting predictor quality or live latency.

## Gateway-path benchmark wiring

Run the dependency-light CPU E2E first:

```bash
python -m pytest tests/test_mock_stack.py -q
```

That test starts three loopback HTTP servers and exercises the real client
parser through this exact path:

```text
runner client -> mock VeloxMesh -> /v1/decision -> mock SSE vLLM
```

For manual adapter work, keep the local stack running with:

```bash
python scripts/run_mock_gateway_stack.py \
  --gateway-port 8080 \
  --decision-port 8081 \
  --engine-port 8082
```

The mock stack is wiring evidence only. It is not a vLLM scheduler benchmark,
GPU result, real VeloxMesh measurement, real-checkpoint result, or two-turn
tool-call proof.

On the rented GPU host, every four-policy benchmark invocation points
`--endpoint` at the VeloxMesh OpenAI-compatible endpoint, never directly at
vLLM:

```bash
python scripts/run_scheduler_benchmark.py \
  --endpoint http://127.0.0.1:8080/v1/completions \
  --model /path/to/Qwen3.5-9B \
  --workload /path/to/workload.jsonl \
  --capacity-rps 8 \
  --scheduler-cls scheduler_benchmark.vllm_scheduler.GatedHybridScheduler \
  --output results/gated-hybrid.json
```

The result manifest records
`client->gateway->decision->vllm`. Reliable predictions arrive at the engine
through `vllm_xargs.workflow_estimated_tokens`; unreliable predictions omit
that field and use native fallback. Policy and scheduler implementations remain
the committed Artifact 1-5 code.

## Isolated gateway-overhead run

Gateway overhead is not a fifth policy and does not enter Fig. 6. Run the same
FCFS workload and arrival schedule once against the engine and once through the
gateway:

```bash
python scripts/run_gateway_overhead.py \
  --direct-endpoint http://127.0.0.1:8000/v1/completions \
  --gateway-endpoint http://127.0.0.1:8080/v1/completions \
  --model /path/to/Qwen3.5-9B \
  --workload /path/to/workload.jsonl \
  --capacity-rps 8 \
  --scheduler-cls vllm.v1.core.sched.scheduler.Scheduler \
  --output results/gateway-overhead-fcfs.json
```

Output mode is `gateway_overhead_fcfs`, `repeats` is `1`, and
`absolute_gateway_minus_direct` contains signed TTLT, TTFT, throughput, and
token-rate deltas. Report these as absolute request-path overhead on this
testbed, not as a scheduling-policy improvement.
