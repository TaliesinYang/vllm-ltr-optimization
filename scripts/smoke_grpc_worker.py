#!/usr/bin/env python3
"""Run one real-checkpoint loopback request through the production worker."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import grpc


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scheduler_benchmark.grpc_worker import (  # noqa: E402
    load_worker,
    predictor_pb2,
    predictor_pb2_grpc,
    start_server,
)


RETRIEVAL_SCOPE = (
    "predictor retrieves admission-time text by request ID in the replay harness"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--request-id")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    worker = load_worker(
        checkpoint=args.checkpoint,
        sidecar_path=args.sidecar,
        manifest_path=args.manifest,
    )
    request_id = args.request_id or worker.first_request_id
    server, port = start_server(worker, "127.0.0.1:0", max_workers=1)
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    try:
        grpc.channel_ready_future(channel).result(timeout=10)
        stub = predictor_pb2_grpc.OutputTokenPredictorStub(channel)
        started = time.perf_counter()
        response = stub.BatchPredict(
            predictor_pb2.BatchPredictRequest(
                tasks=[predictor_pb2.TaskFeature(task_id=request_id)]
            ),
            timeout=60,
        )
        grpc_latency_ms = (time.perf_counter() - started) * 1_000.0
    finally:
        channel.close()
        server.stop(grace=0).wait()

    if len(response.predictions) != 1:
        raise SystemExit("smoke expected exactly one prediction")
    prediction = response.predictions[0]
    if prediction.error:
        raise SystemExit(f"smoke prediction failed: {prediction.error}")
    quantiles = dict(prediction.quantiles)
    signals = dict(prediction.signals)
    if set(quantiles) != {50, 70, 90}:
        raise SystemExit(f"unexpected quantile keys: {sorted(quantiles)}")
    expected_signals = {
        "quantile_spread",
        "ood_distance",
        "feature_coverage",
        "rank_score",
    }
    if set(signals) != expected_signals:
        raise SystemExit(f"unexpected signal keys: {sorted(signals)}")
    print(
        json.dumps(
            {
                "request_id": request_id,
                "model_version": prediction.model_version,
                "mapping_version": worker.mapping_version,
                "quantiles": {str(key): quantiles[key] for key in sorted(quantiles)},
                "signals": {key: signals[key] for key in sorted(signals)},
                "grpc_latency_ms": grpc_latency_ms,
                "retrieval_scope": RETRIEVAL_SCOPE,
                "calibrated": False,
                "warning": worker.approximation_notice,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
