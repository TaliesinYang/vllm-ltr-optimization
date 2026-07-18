# VeloxMesh Replay gRPC Predictor Worker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a replay-only gRPC worker that retrieves admission-time ToolACE text by request ID, calls the committed CPU BERT ranker, and returns explicitly uncalibrated rank-derived p50/p70/p90 approximations plus the raw rank score.

**Architecture:** A mapping module validates 6,000 canonical training labels, emits a normalized request-ID sidecar and immutable empirical percentile manifest, and maps BERT sigmoid scores through p10..p99. A gRPC service reuses VeloxMesh's generated Python bindings, resolves `TaskFeature.task_id` in the sidecar, calls the unchanged `BertPredictor`, and returns one ordered prediction per task. Thin CLIs build artifacts, run the worker on `:50052` by default, and execute a real-checkpoint loopback smoke.

**Tech Stack:** Python 3.11, PyTorch CPU, Transformers, grpcio 1.81.1, protobuf 6.33.5, pytest, generated VeloxMesh predictor.v1 bindings

---

## File map

- `scheduler_benchmark/predictorv1/predictor_pb2.py`: verbatim VeloxMesh generated message descriptors.
- `scheduler_benchmark/predictorv1/predictor_pb2_grpc.py`: verbatim VeloxMesh generated client/server bindings.
- `scheduler_benchmark/rank_quantiles.py`: label validation, normalized replay sidecar, empirical manifest, runtime score mapping, and replay lookup.
- `scheduler_benchmark/grpc_worker.py`: `Health`, `BatchPredict`, per-task errors, inference lock, and gRPC server creation.
- `scripts/build_rank_quantiles.py`: CLI for the 6,000-label manifest and sidecar build.
- `scripts/run_grpc_worker.py`: production worker CLI with default `--addr :50052`.
- `scripts/smoke_grpc_worker.py`: real checkpoint/request loopback smoke and honest output report.
- `tests/test_rank_quantiles.py`: two mapping tests.
- `tests/test_grpc_worker.py`: three worker/loopback tests.

Exactly five new test functions are added, matching the approved 3–5 test limit.

## Runtime paths

- Python: `.worktrees/final-training-artifacts/.venv/bin/python`
- ToolACE snapshot: `/Users/alex/.cache/vllm-ltr-optimization/datasets/toolace/6bda777c88d21e5a204703c1ee45597a8fa4f734/data.json`
- Snapshot SHA-256: `ba12c083fca7e8da48c67ad5b895e495447da7c66e39a2e19742c082e6cb537e`
- Checkpoint: `checkpoints_best_predictor/`
- Model SHA-256: `8a00b458fe7d983a1709bd4b617e4613da9b1d0f24158bc8b39c4befabb03519`
- Generated labels: `/Users/alex/.cache/vllm-ltr-optimization/replay-grpc/toolace-train-6000.labels.jsonl`
- Replay sidecar: `/Users/alex/.cache/vllm-ltr-optimization/replay-grpc/toolace-train-6000.sidecar.jsonl`
- Quantile manifest: `/Users/alex/.cache/vllm-ltr-optimization/replay-grpc/uncalibrated-rank-lookup-v1.json`

### Task 1: Prepare the compatible Python runtime

**Files:**
- No repository changes.

- [ ] **Step 1: Verify the existing CPU model environment**

Run:

```bash
.worktrees/final-training-artifacts/.venv/bin/python -c 'import torch, transformers; print(torch.__version__, transformers.__version__)'
```

Expected: `2.13.0 5.14.1`.

- [ ] **Step 2: Install only the generated-binding requirements**

Run:

```bash
uv pip install --python .worktrees/final-training-artifacts/.venv/bin/python grpcio==1.81.1 protobuf==6.33.5
```

Expected: successful installation without changing repository files.

- [ ] **Step 3: Verify exact runtime versions**

Run:

```bash
.worktrees/final-training-artifacts/.venv/bin/python -c 'import grpc, google.protobuf; print(grpc.__version__, google.protobuf.__version__)'
```

Expected: `1.81.1 6.33.5`.

### Task 2: Vendor the unchanged VeloxMesh Python bindings

**Files:**
- Create: `scheduler_benchmark/predictorv1/predictor_pb2.py`
- Create: `scheduler_benchmark/predictorv1/predictor_pb2_grpc.py`

- [ ] **Step 1: Copy both generated files verbatim**

Use `apply_patch` with the already inspected contents from:

```text
https://github.com/zardonc/VeloxMesh/blob/main/tools/scheduler_training/scheduler_training/predictorv1/predictor_pb2.py
https://github.com/zardonc/VeloxMesh/blob/main/tools/scheduler_training/scheduler_training/predictorv1/predictor_pb2_grpc.py
```

Do not change the serialized descriptor, service path, generated version checks,
or absolute `import predictor_pb2` used by the upstream generator.

- [ ] **Step 2: Verify the copied descriptor**

Run:

```bash
.worktrees/final-training-artifacts/.venv/bin/python -c 'import sys; sys.path.insert(0, "scheduler_benchmark/predictorv1"); import predictor_pb2 as pb; service=pb.DESCRIPTOR.services_by_name["OutputTokenPredictor"]; assert list(service.methods_by_name)==["Health", "BatchPredict"]; assert set(pb.Prediction.DESCRIPTOR.fields_by_name)=={"quantiles", "model_version", "signals", "error"}; print("predictor-v1 contract ok")'
```

Expected: `predictor-v1 contract ok`.

- [ ] **Step 3: Commit the generated contract as one indivisible upstream unit**

```bash
git add scheduler_benchmark/predictorv1/predictor_pb2.py scheduler_benchmark/predictorv1/predictor_pb2_grpc.py
git commit -m "chore(gateway): vendor VeloxMesh predictor bindings"
```

### Task 3: Build the empirical manifest and normalized replay sidecar

**Files:**
- Create: `tests/test_rank_quantiles.py`
- Create: `scheduler_benchmark/rank_quantiles.py`

- [ ] **Step 1: Write the first failing test**

Create `tests/test_rank_quantiles.py` with a helper that writes exactly 6,000 canonical Tier-1 rows and this test:

```python
import hashlib
import json

from scheduler_benchmark.rank_quantiles import (
    APPROXIMATION_NOTICE,
    ReplayStore,
    build_rank_quantile_artifacts,
)


def write_training_labels(path, count=6_000):
    with path.open("w", encoding="utf-8") as handle:
        for index in range(1, count + 1):
            handle.write(
                json.dumps(
                    {
                        "sample_id": f"toolace-{index:06d}:0000",
                        "prompt": f"prompt {index}",
                        "tool_schema": f"schema {index}",
                        "output_length": index,
                    }
                )
                + "\n"
            )


def test_builder_writes_nearest_rank_manifest_and_sidecar(tmp_path):
    labels = tmp_path / "labels.jsonl"
    sidecar = tmp_path / "sidecar.jsonl"
    manifest = tmp_path / "manifest.json"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"checkpoint")
    write_training_labels(labels)

    built = build_rank_quantile_artifacts(
        labels_path=labels,
        checkpoint=checkpoint,
        sidecar_path=sidecar,
        manifest_path=manifest,
        model_version="bert-prompt_schema-tier2-seed17",
        expected_count=6_000,
    )

    assert built["sample_count"] == 6_000
    assert built["percentiles"]["10"] == 600
    assert built["percentiles"]["99"] == 5_940
    assert built["global_quantiles"] == {"50": 3_000, "70": 4_200, "90": 5_400}
    assert built["source_sha256"] == hashlib.sha256(labels.read_bytes()).hexdigest()
    assert built["checkpoint_sha256"] == hashlib.sha256(b"checkpoint").hexdigest()
    assert built["approximation_notice"] == APPROXIMATION_NOTICE
    rows = [json.loads(line) for line in sidecar.read_text().splitlines()]
    assert rows[0] == {
        "request_id": "toolace-000001:0000",
        "prompt_text": "prompt 1",
        "tool_schema_text": "schema 1",
        "output_length": 1,
        "split": "train",
    }
    stored = ReplayStore.from_path(sidecar).get("toolace-000001:0000")
    assert stored.prompt_text == "prompt 1"
    assert stored.tool_schema_text == "schema 1"
```

- [ ] **Step 2: Run the test to verify RED**

Run:

```bash
.worktrees/final-training-artifacts/.venv/bin/python -m pytest tests/test_rank_quantiles.py::test_builder_writes_nearest_rank_manifest_and_sidecar -q
```

Expected: FAIL because `scheduler_benchmark.rank_quantiles` does not exist.

- [ ] **Step 3: Implement canonical label validation and artifact building**

Create `scheduler_benchmark/rank_quantiles.py` with these public constants and types:

```python
APPROXIMATION_NOTICE = (
    "Quantiles are approximations obtained by mapping a rank score through "
    "the empirical training-length distribution, not calibrated intervals. "
    "Scheduling is driven by the rank score in signals."
)
MAPPING_VERSION = "uncalibrated-rank-lookup-v1"
MIN_PERCENTILE = 10
MAX_PERCENTILE = 99


@dataclass(frozen=True)
class ReplayRecord:
    request_id: str
    prompt_text: str
    tool_schema_text: str
    output_length: int
```

Implement `load_training_records(path)` so every non-empty JSONL row must have a unique non-empty `sample_id`, non-empty `prompt`, non-empty `tool_schema`, and non-negative integer `output_length`. Convert `sample_id` to `ReplayRecord.request_id`; reject booleans as integer lengths.

Implement nearest rank exactly:

```python
def nearest_rank(sorted_lengths: list[int], percentile: int) -> int:
    index = math.ceil(percentile / 100 * len(sorted_lengths)) - 1
    return sorted_lengths[max(0, min(index, len(sorted_lengths) - 1))]
```

Implement streamed SHA-256 for `labels_path` and `checkpoint / "model.safetensors"`. `build_rank_quantile_artifacts(...)` must:

```python
records = load_training_records(labels_path)
if len(records) != expected_count:
    raise ValueError(
        f"training label count mismatch: got {len(records)} want {expected_count}"
    )
lengths = sorted(record.output_length for record in records)
percentiles = {
    str(percentile): nearest_rank(lengths, percentile)
    for percentile in range(MIN_PERCENTILE, MAX_PERCENTILE + 1)
}
global_quantiles = {
    str(percentile): nearest_rank(lengths, percentile)
    for percentile in (50, 70, 90)
}
if global_quantiles["50"] <= 0:
    raise ValueError("global p50 must be positive")
manifest = {
    "mapping_version": MAPPING_VERSION,
    "model_version": model_version,
    "approximation_notice": APPROXIMATION_NOTICE,
    "sample_count": len(records),
    "source_sha256": sha256_file(labels_path),
    "checkpoint_sha256": sha256_file(checkpoint / "model.safetensors"),
    "percentiles": percentiles,
    "global_quantiles": global_quantiles,
}
```

Reject pre-existing output paths. Write the sidecar and manifest to sibling `.tmp` paths, flush and close them, then use `Path.replace()` only after both complete. Always remove remaining temporary files on failure. Serialize the manifest with `indent=2`, `sort_keys=True`, and a trailing newline. Preserve input order in the sidecar.

Implement `ReplayStore.from_path()` in the same module. It validates non-empty
`request_id`, `prompt_text`, and `tool_schema_text`, non-negative integer
`output_length`, and `split="train"`; rejects duplicate IDs; preserves input
order; exposes `get(request_id)` and `first_request_id()`; and raises `KeyError`
for an unknown ID.

- [ ] **Step 4: Run the first test to verify GREEN**

Run:

```bash
.worktrees/final-training-artifacts/.venv/bin/python -m pytest tests/test_rank_quantiles.py::test_builder_writes_nearest_rank_manifest_and_sidecar -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit builder core and its test together**

```bash
git add scheduler_benchmark/rank_quantiles.py tests/test_rank_quantiles.py
git commit -m "feat(gateway): build replay rank quantile artifacts"
```

### Task 4: Map a BERT rank score to approved approximate quantiles

**Files:**
- Modify: `tests/test_rank_quantiles.py`
- Modify: `scheduler_benchmark/rank_quantiles.py`

- [ ] **Step 1: Add the second failing test**

Append:

```python
from scheduler_benchmark.rank_quantiles import RankQuantileMapper


def test_mapper_clamps_interpolates_ratios_and_signals(tmp_path):
    labels = tmp_path / "labels.jsonl"
    sidecar = tmp_path / "sidecar.jsonl"
    manifest = tmp_path / "manifest.json"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"checkpoint")
    write_training_labels(labels)
    build_rank_quantile_artifacts(
        labels_path=labels,
        checkpoint=checkpoint,
        sidecar_path=sidecar,
        manifest_path=manifest,
        model_version="bert-prompt_schema-tier2-seed17",
        expected_count=6_000,
    )
    mapper = RankQuantileMapper.from_path(manifest)

    middle = mapper.map_score(0.105)

    assert middle.quantiles == {50: 630.0, 70: 882.0, 90: 1_134.0}
    assert middle.signals == {
        "quantile_spread": 2_400.0,
        "ood_distance": 0.0,
        "feature_coverage": 1.0,
        "rank_score": 0.105,
    }
    assert mapper.map_score(0.0).quantiles[50] == 600.0
    assert mapper.map_score(1.0).quantiles[50] == 5_940.0
```

- [ ] **Step 2: Run the test to verify RED**

Run:

```bash
.worktrees/final-training-artifacts/.venv/bin/python -m pytest tests/test_rank_quantiles.py::test_mapper_clamps_interpolates_ratios_and_signals -q
```

Expected: FAIL because `RankQuantileMapper` does not exist.

- [ ] **Step 3: Implement manifest loading and score mapping**

Keep the approved formulas exact: `p50=L`,
`p70=L*(global_p70/global_p50)`, and
`p90=L*(global_p90/global_p50)`. Add:

```python
@dataclass(frozen=True)
class RankMappedPrediction:
    quantiles: dict[int, float]
    signals: dict[str, float]


class RankQuantileMapper:
    def __init__(self, manifest: Mapping[str, object]) -> None:
        validate_manifest(manifest)
        self.model_version = str(manifest["model_version"])
        self.approximation_notice = str(manifest["approximation_notice"])
        self._percentiles = {
            int(key): float(value)
            for key, value in dict(manifest["percentiles"]).items()
        }
        self._globals = {
            int(key): float(value)
            for key, value in dict(manifest["global_quantiles"]).items()
        }

    @classmethod
    def from_path(cls, path: Path) -> "RankQuantileMapper":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def map_score(self, score: float) -> RankMappedPrediction:
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("rank score must be finite and between zero and one")
        position = max(float(MIN_PERCENTILE), min(100.0 * score, float(MAX_PERCENTILE)))
        lower = math.floor(position)
        upper = math.ceil(position)
        weight = position - lower
        point = self._percentiles[lower] + (
            self._percentiles[upper] - self._percentiles[lower]
        ) * weight
        p50 = self._globals[50]
        quantiles = {
            50: point,
            70: point * self._globals[70] / p50,
            90: point * self._globals[90] / p50,
        }
        signals = {
            "quantile_spread": self._globals[90] - self._globals[50],
            "ood_distance": 0.0,
            "feature_coverage": 1.0,
            "rank_score": score,
        }
        return RankMappedPrediction(quantiles=quantiles, signals=signals)
```

`validate_manifest()` must require mapping version and approximation notice exact equality, model version non-empty, sample count exactly 6,000, percentile keys exactly 10..99, finite non-negative values, global keys exactly 50/70/90, positive p50, and ordered globals.

- [ ] **Step 4: Run both mapping tests**

Run:

```bash
.worktrees/final-training-artifacts/.venv/bin/python -m pytest tests/test_rank_quantiles.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit runtime mapping**

```bash
git add scheduler_benchmark/rank_quantiles.py tests/test_rank_quantiles.py
git commit -m "feat(gateway): map rank scores to approximate quantiles"
```

### Task 5: Add the artifact-builder CLI

**Files:**
- Create: `scripts/build_rank_quantiles.py`

- [ ] **Step 1: Implement the thin CLI**

Use the repository's existing `ROOT`/`sys.path` script pattern. Parse required `--labels`, `--checkpoint`, `--sidecar-output`, `--manifest-output`, and `--model-version`; parse `--expected-count` with default `6000`. Call `build_rank_quantile_artifacts()` and print this JSON summary:

```python
print(
    json.dumps(
        {
            "manifest": str(args.manifest_output),
            "sidecar": str(args.sidecar_output),
            "sample_count": manifest["sample_count"],
            "mapping_version": manifest["mapping_version"],
            "approximation_notice": manifest["approximation_notice"],
        },
        sort_keys=True,
    )
)
```

Return zero only after both artifacts exist.

- [ ] **Step 2: Verify CLI help**

Run:

```bash
.worktrees/final-training-artifacts/.venv/bin/python scripts/build_rank_quantiles.py --help
```

Expected: exit zero and all six flags appear.

- [ ] **Step 3: Commit the CLI separately**

```bash
git add scripts/build_rank_quantiles.py
git commit -m "feat(gateway): add rank quantile artifact CLI"
```

### Task 6: Implement replay lookup and the gRPC service

**Files:**
- Create: `tests/test_grpc_worker.py`
- Create: `scheduler_benchmark/grpc_worker.py`

- [ ] **Step 1: Write the final three failing tests**

Create the imports, fake predictor, and fixture explicitly:

```python
import json

import grpc
import pytest

from scheduler_benchmark.grpc_worker import (
    GrpcPredictorWorker,
    predictor_pb2,
    predictor_pb2_grpc,
    start_server,
)
from scheduler_benchmark.predictor import Prediction
from scheduler_benchmark.rank_quantiles import (
    APPROXIMATION_NOTICE,
    MAPPING_VERSION,
    RankQuantileMapper,
    ReplayStore,
)


class RecordingPredictor:
    def __init__(self, score=0.105):
        self.score = score
        self.inputs = []

    def predict(self, predictor_input):
        self.inputs.append(predictor_input)
        return Prediction(
            score=self.score,
            confidence=0.9,
            ood=False,
            latency_ms=1.0,
        )


@pytest.fixture
def worker_fixture(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "mapping_version": MAPPING_VERSION,
                "model_version": "bert-prompt_schema-tier2-seed17",
                "approximation_notice": APPROXIMATION_NOTICE,
                "sample_count": 6_000,
                "source_sha256": "a" * 64,
                "checkpoint_sha256": "b" * 64,
                "percentiles": {
                    str(percentile): percentile * 10
                    for percentile in range(10, 100)
                },
                "global_quantiles": {"50": 500, "70": 700, "90": 900},
            }
        )
    )
    sidecar_path = tmp_path / "sidecar.jsonl"
    sidecar_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "request_id": "req-1",
                    "prompt_text": "exact prompt",
                    "tool_schema_text": "exact schema",
                    "output_length": 10,
                    "split": "train",
                },
                {
                    "request_id": "req-2",
                    "prompt_text": "second prompt",
                    "tool_schema_text": "second schema",
                    "output_length": 20,
                    "split": "train",
                },
            )
        )
        + "\n"
    )
    predictor = RecordingPredictor()
    worker = GrpcPredictorWorker(
        predictor=predictor,
        replay_store=ReplayStore.from_path(sidecar_path),
        mapper=RankQuantileMapper.from_path(manifest_path),
    )
    return worker, predictor, predictor_pb2
```

Add exactly these tests:

```python
def test_worker_resolves_exact_replay_text(worker_fixture):
    worker, predictor, pb = worker_fixture
    response = worker.BatchPredict(
        pb.BatchPredictRequest(tasks=[pb.TaskFeature(task_id="req-1")]), None
    )

    assert predictor.inputs[0].request_id == "req-1"
    assert predictor.inputs[0].prompt_token_ids == ()
    assert predictor.inputs[0].metadata == {
        "prompt_text": "exact prompt",
        "tool_schema_text": "exact schema",
    }
    assert response.predictions[0].signals["rank_score"] == 0.105


def test_batch_predict_preserves_order_and_isolates_missing_id(worker_fixture):
    worker, _, pb = worker_fixture
    response = worker.BatchPredict(
        pb.BatchPredictRequest(
            tasks=[
                pb.TaskFeature(task_id="req-1"),
                pb.TaskFeature(task_id="missing"),
                pb.TaskFeature(task_id="req-2"),
            ]
        ),
        None,
    )

    assert len(response.predictions) == 3
    assert response.predictions[0].error == ""
    assert response.predictions[1].error == "unknown_request_id"
    assert response.predictions[2].error == ""


def test_loopback_health_and_prediction_use_veloxmesh_bindings(worker_fixture):
    worker, _, pb = worker_fixture
    server, port = start_server(worker, "127.0.0.1:0", max_workers=2)
    try:
        channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        stub = predictor_pb2_grpc.OutputTokenPredictorStub(channel)
        health = stub.Health(pb.HealthRequest(), timeout=5)
        response = stub.BatchPredict(
            pb.BatchPredictRequest(tasks=[pb.TaskFeature(task_id="req-1")]),
            timeout=5,
        )
        assert health.ready is True
        assert health.model_version == "bert-prompt_schema-tier2-seed17"
        assert set(response.predictions[0].quantiles) == {50, 70, 90}
    finally:
        channel.close()
        server.stop(0).wait()
```

- [ ] **Step 2: Run worker tests to verify RED**

Run:

```bash
.worktrees/final-training-artifacts/.venv/bin/python -m pytest tests/test_grpc_worker.py -q
```

Expected: FAIL because `scheduler_benchmark.grpc_worker` does not exist.

- [ ] **Step 3: Implement generated-binding imports**

In `grpc_worker.py`, preserve the upstream absolute generated import:

```python
_BINDINGS_DIR = Path(__file__).with_name("predictorv1")
if str(_BINDINGS_DIR) not in sys.path:
    sys.path.insert(0, str(_BINDINGS_DIR))

import predictor_pb2  # noqa: E402
import predictor_pb2_grpc  # noqa: E402
```

- [ ] **Step 4: Implement the service and server factory**

Implement:

```python
class GrpcPredictorWorker(predictor_pb2_grpc.OutputTokenPredictorServicer):
    def __init__(self, predictor, replay_store, mapper):
        self._predictor = predictor
        self._replay_store = replay_store
        self._mapper = mapper
        self._inference_lock = threading.Lock()

    @property
    def model_version(self):
        return self._mapper.model_version

    @property
    def mapping_version(self):
        return MAPPING_VERSION

    @property
    def approximation_notice(self):
        return self._mapper.approximation_notice

    @property
    def first_request_id(self):
        return self._replay_store.first_request_id()

    def Health(self, request, context):
        del request, context
        return predictor_pb2.HealthResponse(
            ready=True,
            model_version=self._mapper.model_version,
        )

    def BatchPredict(self, request, context):
        del context
        return predictor_pb2.BatchPredictResponse(
            predictions=[self._prediction(task) for task in request.tasks]
        )

    def _prediction(self, task):
        task_id = str(task.task_id).strip()
        if not task_id:
            return self._error("missing_task_id")
        try:
            record = self._replay_store.get(task_id)
        except KeyError:
            return self._error("unknown_request_id")
        predictor_input = PredictorInput(
            request_id=task_id,
            prompt_token_ids=(),
            metadata={
                "prompt_text": record.prompt_text,
                "tool_schema_text": record.tool_schema_text,
            },
        )
        try:
            with self._inference_lock:
                prediction = self._predictor.predict(predictor_input)
            mapped = self._mapper.map_score(prediction.score)
        except Exception:
            logger.exception("predictor failed for replay task_id=%s", task_id)
            return self._error("predictor_error")
        return predictor_pb2.Prediction(
            quantiles=mapped.quantiles,
            signals=mapped.signals,
            model_version=self._mapper.model_version,
        )

    def _error(self, message):
        return predictor_pb2.Prediction(
            model_version=self._mapper.model_version,
            error=message,
        )
```

`load_worker(checkpoint, replay_sidecar, quantile_manifest)` loads one unchanged `BertPredictor`, one validated `ReplayStore`, and one `RankQuantileMapper`. `start_server(worker, address, max_workers)` validates positive worker count, binds with `add_insecure_port`, rejects port zero, starts the server, and returns `(server, port)`.

- [ ] **Step 5: Run all five new tests**

Run:

```bash
.worktrees/final-training-artifacts/.venv/bin/python -m pytest tests/test_rank_quantiles.py tests/test_grpc_worker.py -q
```

Expected: `5 passed`.

- [ ] **Step 6: Commit service and tests together**

```bash
git add scheduler_benchmark/grpc_worker.py tests/test_grpc_worker.py
git commit -m "feat(gateway): serve replay BERT predictions over gRPC"
```

### Task 7: Add the worker CLI

**Files:**
- Create: `scripts/run_grpc_worker.py`

- [ ] **Step 1: Implement CLI parsing and startup**

Use this interface:

```python
parser.add_argument("--addr", default=":50052")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--replay-sidecar", type=Path, required=True)
parser.add_argument("--quantile-manifest", type=Path, required=True)
parser.add_argument("--max-workers", type=int, default=4)
parser.add_argument("--model-version")
```

Load the worker before binding. When `--model-version` is supplied, compare it to `worker.model_version` and exit with `model version mismatch` on inequality. Start the server, print a single JSON readiness record containing address, bound port, model version, mapping version, and `calibrated=false`, then wait for termination. On `KeyboardInterrupt`, stop with a five-second grace period.

- [ ] **Step 2: Verify the required default and flags**

Run:

```bash
.worktrees/final-training-artifacts/.venv/bin/python scripts/run_grpc_worker.py --help
```

Expected: exit zero; output includes `--addr`, default `:50052`, and all artifact flags.

- [ ] **Step 3: Commit the worker CLI**

```bash
git add scripts/run_grpc_worker.py
git commit -m "feat(gateway): add replay gRPC worker CLI"
```

### Task 8: Add and run the real CPU smoke

**Files:**
- Create: `scripts/smoke_grpc_worker.py`

- [ ] **Step 1: Implement the loopback smoke client**

Parse required `--checkpoint`, `--replay-sidecar`, and `--quantile-manifest`, plus optional `--request-id` and `--timeout` defaulting to 30 seconds. Load the production worker, select the explicit request or the first sidecar record, and start it on `127.0.0.1:0`. Measure `time.perf_counter()` only around the `BatchPredict` RPC.

Validate one ready health response, exactly one prediction, empty error, exact quantile keys, exact signal keys `quantile_spread`, `ood_distance`, `feature_coverage`, and `rank_score`, finite values, and ordered quantiles. Print one JSON object containing:

```python
{
    "request_id": request_id,
    "retrieval_scope": (
        "predictor retrieves admission-time text by request ID "
        "in the replay harness"
    ),
    "quantiles": dict(prediction.quantiles),
    "signals": dict(prediction.signals),
    "model_version": prediction.model_version,
    "rpc_latency_ms": rpc_latency_ms,
    "calibrated": False,
    "warning": worker.approximation_notice,
}
```

Always close the channel and stop the server. Return non-zero by raising a clear exception on any failed invariant.

- [ ] **Step 2: Commit the smoke artifact**

```bash
git add scripts/smoke_grpc_worker.py
git commit -m "test(gateway): add real replay gRPC CPU smoke"
```

- [ ] **Step 3: Generate the real 6,000 training labels**

First prove all three explicit outputs are absent; do not delete or overwrite an
existing artifact:

```bash
for path in \
  /Users/alex/.cache/vllm-ltr-optimization/replay-grpc/toolace-train-6000.labels.jsonl \
  /Users/alex/.cache/vllm-ltr-optimization/replay-grpc/toolace-train-6000.sidecar.jsonl \
  /Users/alex/.cache/vllm-ltr-optimization/replay-grpc/uncalibrated-rank-lookup-v1.json; do
  test ! -e "$path" || { echo "existing artifact: $path"; exit 1; }
done
```

Then run:

```bash
.worktrees/final-training-artifacts/.venv/bin/python scripts/extract_tier1_labels.py \
  --source toolace \
  --toolace-snapshot /Users/alex/.cache/vllm-ltr-optimization/datasets/toolace/6bda777c88d21e5a204703c1ee45597a8fa4f734/data.json \
  --limit 6000 \
  --cache-dir /Users/alex/.cache/huggingface \
  --output /Users/alex/.cache/vllm-ltr-optimization/replay-grpc/toolace-train-6000.labels.jsonl
```

Expected: `source=toolace labels=6000`.

- [ ] **Step 4: Build the real manifest and sidecar**

Run:

```bash
.worktrees/final-training-artifacts/.venv/bin/python scripts/build_rank_quantiles.py \
  --labels /Users/alex/.cache/vllm-ltr-optimization/replay-grpc/toolace-train-6000.labels.jsonl \
  --checkpoint checkpoints_best_predictor \
  --sidecar-output /Users/alex/.cache/vllm-ltr-optimization/replay-grpc/toolace-train-6000.sidecar.jsonl \
  --manifest-output /Users/alex/.cache/vllm-ltr-optimization/replay-grpc/uncalibrated-rank-lookup-v1.json \
  --model-version bert-prompt_schema-tier2-seed17 \
  --expected-count 6000
```

Expected: JSON reports sample count 6000, mapping version `uncalibrated-rank-lookup-v1`, and the uncalibrated approximation notice.

- [ ] **Step 5: Run the real checkpoint smoke**

Run:

```bash
.worktrees/final-training-artifacts/.venv/bin/python scripts/smoke_grpc_worker.py \
  --checkpoint checkpoints_best_predictor \
  --replay-sidecar /Users/alex/.cache/vllm-ltr-optimization/replay-grpc/toolace-train-6000.sidecar.jsonl \
  --quantile-manifest /Users/alex/.cache/vllm-ltr-optimization/replay-grpc/uncalibrated-rank-lookup-v1.json \
  --request-id toolace-000000:0000
```

Expected: exit zero; JSON includes ordered p50/p70/p90, all four signals, finite RPC latency, model version, `calibrated=false`, and the replay request-ID disclosure.

### Task 9: Verification and completion audit

**Files:**
- Review only: all files above plus `scheduler_benchmark/predictor.py`

- [ ] **Step 1: Run the five focused tests again**

```bash
.worktrees/final-training-artifacts/.venv/bin/python -m pytest tests/test_rank_quantiles.py tests/test_grpc_worker.py -q
```

Expected: exactly `5 passed`.

- [ ] **Step 2: Run the complete existing test suite**

```bash
.worktrees/final-training-artifacts/.venv/bin/python -m pytest -q
```

Expected: all collected tests pass. Network-backed tests may be run separately only if collection marks them; do not hide or relabel failures.

- [ ] **Step 3: Prove no forbidden core or VeloxMesh changes**

Run:

```bash
git diff d11314c..HEAD -- scheduler_benchmark/predictor.py
git status --short
```

Expected: predictor diff empty and working tree clean. No VeloxMesh checkout is modified because all work occurs in this repository.

- [ ] **Step 4: Inspect artifact identities and honesty fields**

Run:

```bash
.worktrees/final-training-artifacts/.venv/bin/python -c 'import json; from pathlib import Path; p=Path("/Users/alex/.cache/vllm-ltr-optimization/replay-grpc/uncalibrated-rank-lookup-v1.json"); m=json.loads(p.read_text()); assert m["sample_count"]==6000; assert set(map(int,m["percentiles"]))==set(range(10,100)); assert m["mapping_version"]=="uncalibrated-rank-lookup-v1"; assert "not calibrated intervals" in m["approximation_notice"]; print(m["source_sha256"], m["checkpoint_sha256"])'
```

Expected: two hashes print and all assertions pass.

- [ ] **Step 5: Requirement-by-requirement audit**

Confirm with direct evidence:

- worker default is `:50052` and implements `OutputTokenPredictorServicer.BatchPredict`;
- generated bindings match VeloxMesh and Go/proto were not changed;
- task ID resolves exact admission-time replay prompt/schema;
- the unchanged `BertPredictor` supplies raw sigmoid rank score;
- p10..p99 and global ratios implement the approved mapping;
- responses contain p50/p70/p90, four signals, and model version;
- all five tests pass;
- real CPU smoke prints quantiles, signals, latency, disclosure, and warning;
- no output calls quantiles calibrated or treats them as prediction intervals.
