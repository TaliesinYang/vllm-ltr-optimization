# vllm-ltr-optimization

CSCI 6806 student coursework (Group 05). Not peer reviewed, not submitted
anywhere, not reviewed by anyone outside the course. A single semester of work
by three students on one rented GPU.

**This is an unfinished line of enquiry, not a result we would ask anyone to
build on.** We could not establish whether the approach is worth pursuing, and
the measurements below are the reason.

## Read this before the rest

The short version of what we found: **on our setup, the thing we built did not
pay off**, and the two changes that did were not ours.

- At vLLM's default running-batch cap of 256 slots, no ordering policy we
  tested separated from any other by more than 1.2%. The queue a scheduler
  would reorder never formed. Three rounds of serving experiments produced
  nothing.
- Reducing the cap to 16 does make ordering matter, and there the learned
  ranker beats arrival order by 4.2%. But we chose that cap ourselves to
  create the condition. Whether any real deployment sits in that regime is
  not something we measured.
- Our own reliability gate — the abstaining mechanism in the report's title —
  is **4.9% slower than doing nothing at all** once a queue exists. Requests
  it declines to vouch for hold their slots and block the ones behind them.
- Ordering by prompt length, which costs nothing, is **8.5% worse than not
  reordering**. A free heuristic loses to arrival order.
- The two largest effects we measured came from neither prediction nor
  ordering: swapping the scheduler implementation with arrival order held
  fixed cut mean latency 44%, and prefix caching a further 17-23%.
- The predictor does not fit the gateway's documented 15 ms decision budget.
  A third of decisions time out and fall back to arrival order.

The offline signal is real — reading tool-schema text improves output-length
ranking by +0.044 Kendall's tau over the same encoder without it. Turning that
into latency is where it stops.

## What is not established

- **No equivalence test was run.** Where the report says quality "does not
  drop" across unseen-tool strata, that is an absence of detected difference,
  not a demonstration of equivalence.
- **No machine-level replication.** Every serving comparison resamples within
  a single engine launch per arm. Nothing here separates our effects from
  whatever that one machine was doing that day.
- **One workload family, one client.** Offline work is ToolACE only; the live
  traces are 75 requests from a single agent product. The replay inherits that
  capture's compressed job-size variance, which may leave ordering little room
  to separate in the first place.
- **One label model, one card.** Labels come from Qwen3.5-9B; serving runs on
  one vendor-modified 48 GB RTX 4090.
- **Contention was created one way only.** We constrained the batch. Long
  contexts exhausting the KV cache would reach the same condition differently
  and might not behave the same.
- **S2 is underpowered** at n=78, and the gate's confidence is a measured tau
  floor, not a calibrated probability.
- **Prefill is unmodeled.** The ranker predicts decode length only.

## Would we build on this?

Not as it stands, and we would not suggest anyone else does either. The
question the project set out to answer -- does a better-informed length
predictor make an LLM serving stack faster -- came back as *it depends on a
setting we chose ourselves*, which is not an answer. Before this direction is
worth more effort, at least these would have to be settled:

- Whether any production deployment actually runs in the contended regime
  where ordering pays. We did not measure one. We manufactured the condition.
- Whether the effect survives on a second machine. Nothing here is replicated
  at the machine level, so the 4.2% could be a property of one box on one day.
- Whether the gate can be made to cost less than it saves. Right now it costs
  more, and we do not have a design that fixes that.
- Whether the predictor can be made to fit a realistic decision budget. Two
  pre-registered attempts to shrink it both failed.
- Whether any of it generalises past one workload family and one agent client.

Someone starting from scratch on this problem would probably not want our
scheduler. They might want the negative results, and the experimental
discipline that produced them, which is the part below.

## The mistake worth copying

An early round benchmarked every policy against vLLM's stock scheduler and
found uniform gains across all of them. That looked like success. It should
have looked wrong: policies that order differently should not improve
identically. The gains came from the implementation underneath, not from
ordering. Anyone benchmarking a scheduler against a serving engine's default
will hit this — the control has to share your implementation and differ only
in the variable you claim.

## Reproducing

Every number in the report is re-derived from a committed artifact at build
time; `latex_source/scripts/build_evidence_map.py` fails if an artifact
disagrees with what is printed. Figure generators are under
`scripts/report_figures/`, run data under `runs/`.

| Doc | Purpose |
|---|---|
| [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) | Environment setup and how to re-run |
| [`latex_source/`](latex_source/) | LaTeX source of the report |
| [`latex_source/EVIDENCE-MAP.md`](latex_source/EVIDENCE-MAP.md) | Every printed number and the artifact it came from |
| [`docs/references.md`](docs/references.md) | Papers compared against |

Base fork: `hao-ai-lab/vllm-ltr`, reproducing Fu et al., NeurIPS 2024.

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
