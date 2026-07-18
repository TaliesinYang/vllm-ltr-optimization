# VeloxMesh Replay gRPC Predictor Worker Design

## Scope

Expose the committed CPU `BertPredictor` through Mingye's
`predictor.v1.OutputTokenPredictor` gRPC contract without changing VeloxMesh Go
or protobuf code. This integration is explicitly limited to the replay harness:
VeloxMesh supplies the replay request identifier in `TaskFeature.task_id`, and
the worker retrieves the admission-time prompt and tool schema from a local
sidecar before invoking `BertPredictor`.

The three accepted deliverables are:

1. a gRPC worker implementing `Health` and `BatchPredict`;
2. a rank-score-to-quantiles mapping artifact built from the 6k training labels;
3. a CPU smoke client that sends one real replay invocation and prints the
   response and latency.

The implementation does not modify VeloxMesh, the existing `BertPredictor`, its
checkpoint, scheduler policies, or the Go gateway. It must not claim support for
arbitrary live requests whose text is absent from the replay sidecar.

## Authoritative Contract

Copy VeloxMesh's generated `predictor_pb2.py` and `predictor_pb2_grpc.py` into a
local `predictorv1` package without editing generated definitions. The worker
implements `OutputTokenPredictorServicer` and listens on `--addr`, defaulting to
`127.0.0.1:50052`.

`BatchPredict` preserves request order and returns exactly one `Prediction` for
each input `TaskFeature`. `TaskFeature.task_id` is interpreted as the replay
`request_id`; no other protobuf field is overloaded to transport prompt text.
Successful predictions contain quantile keys `50`, `70`, and `90`, the required
signals, and the manifest's `model_version`.

`Health` reports `ready=true` only after the checkpoint, replay sidecar, and
quantile manifest have all loaded and passed validation. Startup fails instead
of serving with a partial artifact set.

## Replay Text Sidecar

The replay sidecar is UTF-8 JSONL with one record per replay request:

```json
{
  "request_id": "toolace-000001:0000",
  "prompt_text": "...",
  "tool_schema_text": "...",
  "output_length": 137,
  "split": "train"
}
```

`request_id`, `prompt_text`, and `tool_schema_text` must be non-empty strings.
`output_length` must be a non-negative integer. The mapping builder accepts only
records with `split="train"`; the worker may resolve any replay record present
in its runtime sidecar. Duplicate identifiers, malformed rows, or missing
required fields are fatal load errors.

For a successful lookup the worker creates the existing `PredictorInput` with:

- `request_id=task.task_id`;
- `prompt_token_ids=()` because serving-model token IDs are neither present nor
  valid BERT token IDs;
- `metadata.prompt_text` and `metadata.tool_schema_text` copied exactly from the
  sidecar.

The report and smoke output must state: "the predictor retrieves admission-time
text by request ID in the replay harness." This is not a live gateway text
transport mechanism.

## Empirical Percentile Table

The mapping builder reads the 6k training records' real `output_length` values,
sorts them in ascending order, and writes an immutable JSON manifest. It stores
nearest-rank empirical lengths for every integer percentile from p10 through
p99, plus `global_p50`, `global_p70`, `global_p90`, sample count, source SHA-256,
checkpoint SHA-256, mapping version, model version, and the approximation label.

For percentile `p` and `N` sorted lengths, nearest-rank lookup uses:

```text
index = ceil((p / 100) * N) - 1
length[p] = sorted_lengths[clamp(index, 0, N - 1)]
```

The builder rejects an empty dataset, a non-positive `global_p50`, fewer than
6,000 accepted training rows, or a dataset count other than the declared
manifest count. It records the actual count so the artifact remains auditable;
the initial production artifact requires exactly 6,000 accepted rows.

The manifest includes this exact semantic statement:

> Quantiles are approximations obtained by mapping a rank score through the
> empirical training-length distribution, not calibrated intervals. Scheduling
> is driven by the rank score in signals.

## Runtime Score Mapping

`BertPredictor.predict()` remains unchanged. Its sigmoid score `s` in `[0, 1]`
is treated as a position in the empirical training-length distribution, not as
a calibrated probability.

Runtime mapping is deterministic:

1. compute `percentile = clamp(100 * s, 10, 99)`;
2. linearly interpolate between the two surrounding integer percentile entries
   in the p10..p99 table to obtain point estimate `L`;
3. return:

```text
p50 = L
p70 = L * (global_p70 / global_p50)
p90 = L * (global_p90 / global_p50)
```

Quantile values remain protobuf doubles and are not rounded to integer tokens.
Manifest validation requires finite non-negative percentile values and
`global_p50 <= global_p70 <= global_p90`, which guarantees
`p50 <= p70 <= p90` for every successful response.

The response signals are:

```text
quantile_spread  = global_p90 - global_p50
ood_distance     = 0.0
feature_coverage = 1.0
rank_score       = s
```

`quantile_spread` is a constant global token spread, as approved. `rank_score`
is the raw BERT sigmoid score and is the signal intended to drive scheduling.
`ood_distance=0.0` is an unavailable placeholder inherited from the current
predictor, not evidence that a request is in distribution. `feature_coverage`
is `1.0` only for successful sidecar records containing both exact BERT input
features.

The worker and manifest must use `uncalibrated-rank-lookup-v1` in their mapping
identity. Comments, CLI help, smoke output, and reports must call the quantiles
"rank-derived approximations" and must not describe them as calibrated,
probabilistic coverage, prediction intervals, or direct length-model outputs.

## Worker Errors and Concurrency

Errors are isolated per task so one malformed task does not block siblings in a
batch. For a missing or empty `task_id`, unknown sidecar identifier, invalid
sidecar features, predictor exception, or non-finite score, the corresponding
`Prediction` contains `model_version` and a stable error string but no quantiles
or success signals. Response cardinality and order remain unchanged.

The server uses a bounded thread pool. A single `BertPredictor` instance is
loaded at startup and shared. Model inference is serialized with a worker-side
lock because the first integration targets correctness and CPU smoke behavior;
batch-level parallel model execution is outside scope. gRPC transport remains
insecure localhost transport, matching VeloxMesh's current Python client.

The Go client currently defaults to a 15 ms deadline. This replay integration
does not change that client; harness configuration must set a timeout above the
measured CPU predictor latency. The smoke client owns its explicit timeout and
reports end-to-end RPC latency separately from the predictor's internal latency.

## CLI and Artifact Layout

The worker CLI accepts:

- `--addr` with default `127.0.0.1:50052`;
- `--checkpoint`, required;
- `--replay-sidecar`, required;
- `--quantile-manifest`, required;
- `--max-workers`, default `4`;
- `--model-version`, optional only when it exactly matches the manifest value.

The mapping builder accepts the training JSONL, checkpoint directory, output
manifest path, and model version. It writes the manifest atomically after all
validation succeeds.

The CPU smoke accepts the same three runtime artifact paths, starts the worker
on an ephemeral localhost port, selects one real sidecar request, calls
`Health`, sends one `BatchPredict` request, and prints:

- request ID and the replay-harness admission-text retrieval disclosure;
- p50, p70, and p90;
- all signals including `rank_score`;
- model version;
- end-to-end RPC latency measured by the client;
- the uncalibrated rank-derived approximation warning.

It exits non-zero when health is not ready, the response count differs from one,
the prediction contains an error, any required key is missing, values are
non-finite, or quantiles are unordered.

## Tests

Add five focused CPU unit/integration tests:

1. the mapping builder creates p10..p99 using the exact nearest-rank rule and
   records hashes plus the uncalibrated semantic statement;
2. runtime mapping clamps score tails, interpolates interior percentiles, applies
   the approved global ratios, and emits the four signals;
3. the worker resolves `task_id` to exact sidecar prompt/schema and passes them
   to the existing predictor without modifying its core;
4. `BatchPredict` preserves order and isolates an unknown request ID while a
   valid sibling succeeds;
5. a loopback gRPC test covers `Health` and one successful real-shaped request
   using the copied VeloxMesh bindings.

The separate CPU smoke uses the real 417 MiB checkpoint and one real replay
invocation. Unit tests use a fake predictor so routine verification does not
reload the checkpoint.

## Acceptance Criteria

- VeloxMesh Go and protobuf repositories remain unchanged.
- Existing `scheduler_benchmark/predictor.py` behavior remains unchanged.
- Generated Python bindings match the referenced VeloxMesh contract.
- The mapping manifest is built from exactly 6,000 accepted training labels and
  contains p10..p99 plus global p50/p70/p90 and reproducibility hashes.
- Every successful prediction returns ordered p50/p70/p90, the four signals,
  and the manifest model version.
- Missing replay IDs fail per prediction without corrupting batch order.
- Five focused tests pass.
- The real CPU smoke starts the worker, sends one real replay invocation, prints
  quantiles/signals/model version/latency, and exits zero.
- All user-facing text labels quantiles as rank-derived, uncalibrated
  approximations and states that replay text is retrieved by request ID.
