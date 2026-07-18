import importlib
import json

import grpc

from scheduler_benchmark.predictor import Prediction
from scheduler_benchmark.rank_quantiles import (
    APPROXIMATION_NOTICE,
    MAPPING_VERSION,
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
            latency_ms=1.25,
        )


def write_runtime_artifacts(tmp_path):
    sidecar = tmp_path / "sidecar.jsonl"
    sidecar.write_text(
        "\n".join(
            json.dumps(
                {
                    "request_id": request_id,
                    "prompt_text": prompt,
                    "tool_schema_text": schema,
                    "output_length": output_length,
                    "split": "train",
                }
            )
            for request_id, prompt, schema, output_length in (
                ("req-a", "admission prompt A", "schema A", 10),
                ("req-b", "admission prompt B", "schema B", 20),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "mapping_version": MAPPING_VERSION,
                "model_version": "bert-test-v1",
                "approximation_notice": APPROXIMATION_NOTICE,
                "sample_count": 6_000,
                "source_sha256": "0" * 64,
                "checkpoint_sha256": "1" * 64,
                "percentiles": {
                    str(percentile): percentile * 10
                    for percentile in range(10, 100)
                },
                "global_quantiles": {"50": 500, "70": 700, "90": 900},
            }
        ),
        encoding="utf-8",
    )
    return sidecar, manifest


def test_worker_retrieves_exact_admission_text_and_returns_rank_score(tmp_path):
    worker_module = importlib.import_module("scheduler_benchmark.grpc_worker")
    sidecar, manifest = write_runtime_artifacts(tmp_path)
    predictor = RecordingPredictor()
    worker = worker_module.GrpcPredictorWorker.from_paths(
        predictor=predictor,
        sidecar_path=sidecar,
        manifest_path=manifest,
    )

    response = worker.BatchPredict(
        worker_module.predictor_pb2.BatchPredictRequest(
            tasks=[worker_module.predictor_pb2.TaskFeature(task_id="req-a")]
        ),
        None,
    )

    assert predictor.inputs[0].request_id == "req-a"
    assert predictor.inputs[0].metadata == {
        "prompt_text": "admission prompt A",
        "tool_schema_text": "schema A",
    }
    assert dict(response.predictions[0].quantiles) == {
        50: 105.0,
        70: 147.0,
        90: 189.0,
    }
    assert response.predictions[0].signals["rank_score"] == 0.105


def test_worker_preserves_batch_order_and_isolates_unknown_ids(tmp_path):
    worker_module = importlib.import_module("scheduler_benchmark.grpc_worker")
    sidecar, manifest = write_runtime_artifacts(tmp_path)
    predictor = RecordingPredictor(score=0.5)
    worker = worker_module.GrpcPredictorWorker.from_paths(
        predictor=predictor,
        sidecar_path=sidecar,
        manifest_path=manifest,
    )
    pb = worker_module.predictor_pb2

    response = worker.BatchPredict(
        pb.BatchPredictRequest(
            tasks=[
                pb.TaskFeature(task_id="req-b"),
                pb.TaskFeature(task_id="missing"),
                pb.TaskFeature(task_id="req-a"),
            ]
        ),
        None,
    )

    assert len(response.predictions) == 3
    assert response.predictions[0].error == ""
    assert response.predictions[1].error == "unknown_request_id"
    assert dict(response.predictions[1].quantiles) == {}
    assert dict(response.predictions[1].signals) == {}
    assert response.predictions[2].error == ""
    assert [item.request_id for item in predictor.inputs] == ["req-b", "req-a"]


def test_loopback_health_and_batch_predict_use_generated_contract(tmp_path):
    worker_module = importlib.import_module("scheduler_benchmark.grpc_worker")
    sidecar, manifest = write_runtime_artifacts(tmp_path)
    worker = worker_module.GrpcPredictorWorker.from_paths(
        predictor=RecordingPredictor(),
        sidecar_path=sidecar,
        manifest_path=manifest,
    )
    server, port = worker_module.start_server(worker, "127.0.0.1:0")
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    try:
        grpc.channel_ready_future(channel).result(timeout=5)
        stub = worker_module.predictor_pb2_grpc.OutputTokenPredictorStub(channel)
        health = stub.Health(worker_module.predictor_pb2.HealthRequest(), timeout=5)
        response = stub.BatchPredict(
            worker_module.predictor_pb2.BatchPredictRequest(
                tasks=[worker_module.predictor_pb2.TaskFeature(task_id="req-a")]
            ),
            timeout=5,
        )
    finally:
        channel.close()
        server.stop(grace=0).wait()

    assert health.ready is True
    assert health.model_version == "bert-test-v1"
    assert len(response.predictions) == 1
    assert response.predictions[0].model_version == "bert-test-v1"
