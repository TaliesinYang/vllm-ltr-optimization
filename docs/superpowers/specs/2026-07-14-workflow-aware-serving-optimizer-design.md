# Workflow-Aware Serving Optimizer Design

Date: 2026-07-14  
Status: Approved design for implementation planning  
Primary repository: `TaliesinYang/vllm-ltr-optimization`  
Initial execution backend: local `hao-ai-lab/vllm-ltr` fork (vLLM 0.4.1)  
Later integrations: modern vLLM backend and Mingye's VeloxMesh gateway

## 1. Decision

Extend `vllm-ltr-optimization` into the primary research artifact for a workflow-aware LLM serving optimizer. The optimizer jointly uses predictor outputs, workflow progress, queue state, and KV-cache state to make two coupled decisions:

1. which ready request or workflow step should execute next; and
2. whether KV state should be preserved on GPU, swapped to CPU, discarded for recomputation, evicted, or prefetched.

The policy implementation and state model live in this repository. vLLM-LTR and future serving engines expose only narrow event/action hooks. VeloxMesh will later use a thin adapter and will not own the optimization logic.

This is a real serving implementation. A deterministic CPU simulator is required for testing and policy development, but simulator-only results do not satisfy completion.

## 2. Motivation and Evidence

The existing project demonstrated that chat-trained length predictors do not transfer reliably to BFCL tool-call requests. The optimizer must therefore use prediction conservatively and provide a prediction-free fallback.

The design combines mechanisms that are established in recent serving research:

- **AugServe (2025)**: output-length bucket prediction and external API-latency prediction for augmented LLM serving: <https://arxiv.org/abs/2512.04013>
- **Autellix (2025)**: program-level attained-service scheduling instead of isolated request scheduling: <https://arxiv.org/abs/2502.13965>
- **InferCept (ICML 2024)**: preserve, swap, or discard KV state during external interceptions: <https://proceedings.mlr.press/v235/abhyankar24a.html>
- **KVFlow (NeurIPS 2025)**: workflow-aware prefix-cache eviction, prefetch, and cache-status-aware scheduling: <https://proceedings.neurips.cc/paper_files/paper/2025/file/b7971d31a7d5eb0f1eed2f8f6f368195-Paper-Conference.pdf>
- **PBKV (2026 preprint)**: conservative future-agent prediction for dynamic workflow KV management: <https://arxiv.org/abs/2605.06472>
- **Pythia (2026 preprint)**: profile-derived workflow structure used for cache management and look-ahead scheduling: <https://arxiv.org/abs/2604.25899>
- **SAGA (2026 preprint)**: workflow-atomic scheduling, tool-aware cache lifetime, affinity, and workflow fairness: <https://arxiv.org/abs/2605.00528>
- **Orla (2026 preprint)**: separation of workflow policy from interchangeable inference backends: <https://arxiv.org/abs/2603.13605>

These works are references and baselines, not components invented by this project. The project contribution is the implemented, confidence-gated composition evaluated on the existing chat, BFCL, and mixed-workload evidence chain.

## 3. Scope

### 3.1 In scope

- A shared workflow and cache state model.
- A predictor bundle for output size, tool latency, next-step/reuse likelihood, and calibrated confidence.
- Program-aware request scheduling with aging and safe fallback.
- Tool-wait KV lifecycle decisions: preserve, swap, discard/recompute.
- Workflow-aware cache value, eviction, and asynchronous prefetch decisions.
- A backend-neutral optimizer interface.
- A deterministic event-driven simulator and trace replay.
- Real vLLM-LTR hooks for queue selection, prefix-cache observation, swap, recompute, and telemetry.
- A later modern-vLLM adapter using the same contracts.
- Reproducible ablations and manifests.

### 3.2 Out of scope for the first complete single-node artifact

- Multi-node or heterogeneous GPU placement.
- Prefill/decode disaggregation.
- Training or fine-tuning the served 8B model.
- A new agent framework or tool runtime.
- A new CUDA KV-cache implementation.
- Reproducing SAGA's 64-GPU deployment, HexAGenT, or cluster autoscaling.
- Making VeloxMesh the primary repository.

## 4. Repository Architecture

The repository will evolve from orchestration-only scripts into a Python package plus backend patches/adapters:

```text
vllm-ltr-optimization/
  src/workflow_optimizer/
    contracts.py          # events, snapshots, predictions, decisions
    state.py              # workflow/session/cache state transitions
    predictor/
      bundle.py           # combines independently replaceable predictors
      length.py           # existing PARS/listMLE-compatible adapter
      tool_latency.py     # online per-tool quantile estimator
      next_step.py        # workflow transition/reuse estimator
      calibration.py      # confidence and drift checks
    scheduler/
      program.py          # program-level attained-service scheduling
      aging.py            # starvation and deadline protection
      joint.py            # combines scheduling and cache readiness
    cache/
      lifecycle.py        # preserve/swap/discard expected-cost policy
      value.py            # workflow-aware cache value
      eviction.py         # value-density eviction
      prefetch.py         # conservative prefetch decisions
    backends/
      base.py             # backend protocol
      vllm_ltr.py         # vLLM-LTR adapter
      modern_vllm.py      # later adapter, same contracts
    telemetry.py          # observations and decision audit log
  simulator/
    engine.py             # deterministic event-driven simulation
    trace.py              # chat/BFCL/workflow trace ingestion
  tests/
  benchmarks/
  patches/vllm_ltr/
  configs/
  runs/
  docs/
```

The initial package is Python because both the existing predictor and vLLM scheduler are Python. Hot-path operations remain in vLLM; the optimizer must not copy KV tensors itself.

## 5. Core Contracts

### 5.1 WorkflowEvent

Every state change is expressed as an event:

```text
ARRIVE
PREFILL_START / PREFILL_END
DECODE_START / TOKEN / DECODE_END
TOOL_WAIT_START / TOOL_WAIT_END
KV_GPU / KV_CPU / KV_LOADING / KV_OFFLOADING / KV_DROPPED
COMPLETE / CANCEL / ERROR
```

Required identity fields are `request_id`, `workflow_id`, `step_id`, `model_id`, and monotonic timestamp. Optional fields describe tool identity, prefix lineage, token counts, KV blocks, and measured transfer time.

### 5.2 PredictionBundle

```text
output_length_bucket
output_rank_score
tool_latency_p50_ms
tool_latency_p95_ms
next_step_distribution
kv_reuse_probability
time_to_reuse_p50_ms
confidence_by_head
domain_shift_score
```

Missing predictions are valid. Each consumer must have a defined fallback.

### 5.3 SystemSnapshot

The optimizer receives a read-only snapshot containing ready, running, waiting-on-tool, swapped, and blocked requests; GPU/CPU cache capacity; cache-entry status; and recent bandwidth/latency telemetry.

### 5.4 JointDecision

```text
selected_request_ids
request_priorities
kv_actions: PRESERVE | SWAP | DISCARD | PREFETCH | NONE
cache_victims
policy_name
fallback_reason
prediction_fields_used
decision_timestamp
```

All decisions are auditable. Backend adapters validate that requested actions are supported before execution.

## 6. Predictor Design

The predictor and scheduler are co-designed and evaluated together, but they are not one inseparable neural network.

### 6.1 Output length

Reuse the existing BERT/PARS and listMLE predictors through adapters. The primary model produces a rank score and coarse output-length bucket. Ranking is more important than exact token regression for queue order. Existing classification, listMLE, PARS, and oracle outputs remain baselines.

### 6.2 Tool latency

Use an online per-tool quantile estimator rather than asking a text encoder to predict volatile network latency. The estimator maintains p50 and p95 latency by tool identity and optional outcome category, with EWMA updates and minimum-sample guards. Unknown tools use a conservative global p95.

### 6.3 Next-step and KV reuse

Start with an interpretable transition model over observed workflow steps. It estimates next-step probability and time-to-reuse from historical transitions, current step, tool identity, and prefix lineage. A context model may be added behind the same interface only after the transition baseline is measured.

### 6.4 Confidence and drift

Each prediction head reports confidence. Reliability is rejected when samples are insufficient, calibration error exceeds its configured bound, or the observed workload class differs from the training domain. Rejected fields are excluded from the decision instead of replaced with false precision.

## 7. Joint Scheduling Algorithm

Scheduling is hierarchical and event-driven:

1. Remove requests that are blocked on a tool or whose required KV blocks are still loading.
2. Apply an aging/deadline guard so no request can be postponed indefinitely.
3. Select a workflow using attained service, following the program-level principle from Autellix.
4. Within the selected fairness tier, rank ready steps by confidence-gated predicted remaining service.
5. Form a batch subject to the existing vLLM token and sequence budgets.

The exact production ordering is lexicographic rather than a fragile weighted sum:

```text
urgent deadline/aging class
  -> least workflow attained service
  -> shortest trusted remaining service
  -> oldest arrival time
```

When remaining-service prediction is rejected, the third key is omitted. When workflow identity is absent, each request is treated as a one-step workflow. This guarantees compatibility with ordinary chat traffic.

## 8. KV Lifecycle and Cache Algorithm

### 8.1 Tool-wait lifecycle

When a request enters tool wait, calculate three expected costs:

```text
preserve_cost = pressure_price * kv_bytes * predicted_wait_time
swap_cost     = measured_d2h_cost + reuse_probability * measured_h2d_cost
discard_cost  = reuse_probability * measured_recompute_cost
```

Choose the supported action with minimum expected cost. Under low confidence, use conservative thresholds:

- preserve only when the measured/estimated wait is short and memory pressure is low;
- otherwise use the backend's native preemption decision;
- do not speculative-prefetch.

This policy generalizes InferCept's three lifecycle actions without treating tool latency or reuse predictions as oracle values.

### 8.2 Cache value and eviction

For each evictable prefix entry:

```text
cache_value = reuse_probability * recompute_saving_ms
              / (kv_bytes * max(time_to_reuse_ms, epsilon))
```

Evict the lowest-value entries first, excluding entries that are running, loading, offloading, or pinned by the backend. Baselines include native LRU, size-aware LRU, KVFlow-style steps-to-execution, and an offline Bélády oracle in simulation only.

### 8.3 Prefetch

Prefetch a CPU-resident entry only when:

- reuse confidence exceeds the configured threshold;
- predicted time-to-reuse is close enough to cover measured transfer time;
- GPU headroom remains after active-batch reservation; and
- no duplicate loading operation exists.

Requests whose required cache is loading remain ready-but-not-dispatchable. Other ready work may execute while transfer completes.

## 9. Backend Integration

### 9.1 vLLM-LTR adapter

The existing vLLM 0.4.1 fork already contains automatic prefix caching, `BlockSpaceManagerV1`, swap-in/out, recompute preemption, scheduling budgets, and cache status paths. The adapter/patch must expose narrow hooks:

- emit workflow events from request and scheduler transitions;
- attach workflow metadata to `SequenceGroup` without changing ordinary requests;
- request validated lifecycle actions through existing block-manager methods;
- let the joint scheduler provide ordering keys and readiness filters;
- export observed KV blocks, prefix hits, swap bytes/time, recompute tokens, TTFT, TPOT, and completion time.

Unsupported actions must return a typed capability error and trigger native behavior. No optimizer exception may terminate the engine loop.

### 9.2 Modern vLLM adapter

The modern adapter is implemented after the vLLM-LTR path works. It uses the same contracts and translates them to current engine APIs. No policy code is duplicated. Compatibility tests replay identical snapshots through both adapters and compare normalized decisions.

### 9.3 VeloxMesh adapter

VeloxMesh later supplies normalized tool/workflow metadata and receives policy labels or backend hints. It does not own predictor training, cache state, or scheduling policy.

## 10. Failure Handling

- Predictor unavailable: run program-aware FCFS/aging without prediction.
- Unknown workload/tool: use conservative latency priors and disable speculative prefetch.
- Drift or low confidence: remove affected prediction fields.
- Optimizer timeout/exception: use the native scheduler and native cache policy for that iteration.
- Unsupported backend action: log capability fallback and continue natively.
- Transfer failure: mark the entry unavailable and recompute if the request resumes.
- Cancelled/failed workflow: release unshared cache entries and preserve reference-counted shared prefixes.
- Inconsistent event ordering: reject the event, increment an error metric, and retain the last valid state.

Fallback behavior is observable through `policy_name` and `fallback_reason`; it is not silently counted as optimizer success.

## 11. Telemetry

Record both decisions and outcomes:

- predictor outputs, confidence, calibration bin, and fields actually used;
- queue position, attained service, selected batch, and aging overrides;
- KV location/status, bytes, prefix hits, victims, transfer/recompute costs;
- tool wait duration and prediction error;
- TTFT, TPOT, end-to-end request latency, workflow JCT, throughput, starvation, and fairness;
- optimizer overhead and fallback count.

Raw per-request identifiers must be local pseudonymous IDs. Prompt text and tool credentials are never written to decision logs.

## 12. Verification Strategy

### 12.1 Unit and property tests

- State-machine transitions and invalid-event rejection.
- Predictor fallback and confidence gating.
- Starvation protection under adversarial arrivals.
- No eviction of pinned/loading/offloading entries.
- Lifecycle choice under controlled cost cases.
- Deterministic decisions from identical snapshots.

### 12.2 Simulator and trace replay

- Synthetic microcases with known optimum.
- Existing chat traces.
- BFCL tool-call traces.
- Mixed chat/tool ratios and controlled tool-latency distributions.
- Oracle output length, oracle next use, and Bélády cache baselines used only as upper bounds.

### 12.3 CPU integration

Use a fake backend implementing the real backend protocol. Verify event ordering, capability fallback, decision logs, and end-to-end batch/cache actions without CUDA.

### 12.4 GPU integration

On the 201 server or equivalent supported GPU:

- demonstrate real prefix-cache hits;
- demonstrate preserve, swap, discard/recompute, and prefetch paths;
- verify no crash or deadlock during concurrent tool waits;
- measure policy overhead and actual transfer/recompute costs;
- compare behavior with optimizer disabled.

GPU-dependent claims remain unverified until these runs complete.

## 13. Evaluation and Ablations

Required systems baselines:

1. native FCFS + native cache policy;
2. existing LTR/PARS scheduler + native cache policy;
3. program-aware scheduler without prediction;
4. confidence-gated predictor scheduler without new KV policy;
5. InferCept-style static preserve/swap/discard;
6. workflow-aware cache value without prefetch;
7. workflow-aware cache value with prefetch;
8. full joint optimizer;
9. full optimizer with each prediction head removed;
10. oracle predictors and Bélády simulation upper bounds, clearly labelled non-deployable.

Required interaction studies:

- tool-call ratio × memory pressure;
- tool-latency distribution × lifecycle action;
- predictor error/confidence × scheduler fallback;
- prefix reuse × eviction policy;
- offered load × batching/preemption;
- chat-only, tool-only, and mixed workloads.

Report absolute values in the testbed's units and relative comparisons only within matched hardware/configuration. Do not claim free preemption, stable sawtooth behavior, or solved cluster scheduling.

## 14. Delivery Sequence and Estimate

### Phase A — contracts, state, simulator, baseline policies

Estimated focused effort: 0.5–1 day. Completion requires unit tests and deterministic replay, not GPU claims.

### Phase B — real vLLM-LTR lifecycle and telemetry hooks

Estimated focused effort: 1–2 days. Completion requires CPU tests plus GPU demonstrations of prefix hit, swap, recompute, and release.

### Phase C — predictor bundle and joint scheduler

Estimated focused effort: 1–2 days. Completion requires calibration/fallback tests and end-to-end decision logging.

### Phase D — modern backend and integration adapters

Estimated focused effort: 1–2 days after the first backend is stable. VeloxMesh integration follows via a thin adapter.

### Phase E — GPU ablations and artifact verification

Estimated experiment time: 1–3 GPU days, depending on failures and repeated seeds. Failed or incomplete runs are reported as unverified.

The estimates are engineering ranges, not deadlines or promises. The first CPU-complete artifact can land earlier than the full GPU evidence package.

## 15. Acceptance Criteria

The implementation is complete only when all of the following hold:

- The primary repository contains the optimizer package, simulator, tests, configs, benchmark entry points, and reproduction documentation.
- Predictor, scheduler, and cache policies share the defined contracts and can be ablated independently.
- Ordinary non-tool requests work without workflow metadata.
- Low-confidence or failed prediction takes the documented native fallback path.
- At least one real vLLM backend executes scheduling and KV lifecycle decisions on GPU.
- Preserve, swap, discard/recompute, eviction, and prefetch outcomes are auditable.
- Required baselines and interaction ablations have commands and result directories.
- No paper-reported number is presented as a project measurement.
- GPU-dependent claims are marked unverified until a successful run artifact exists.

## 16. Implementation Boundary

This design intentionally avoids both extremes:

- It is not a small proof that only calculates policy scores.
- It is not an attempt to rebuild a full distributed agent-serving platform.

It is a complete, single-node, workflow-aware serving optimizer with real engine integration, safe degradation, and a path to later gateway and modern-backend integration.
