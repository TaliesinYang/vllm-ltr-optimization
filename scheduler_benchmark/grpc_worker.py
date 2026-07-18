"""Replay-harness gRPC adapter for the unchanged CPU BERT predictor."""

from __future__ import annotations

import logging
import sys
import threading
from concurrent import futures
from pathlib import Path

import grpc

from scheduler_benchmark.predictor import BertPredictor, Predictor, PredictorInput
from scheduler_benchmark.rank_quantiles import RankQuantileMapper, ReplayStore


_BINDINGS_DIR = Path(__file__).with_name("predictorv1")
if str(_BINDINGS_DIR) not in sys.path:
    sys.path.insert(0, str(_BINDINGS_DIR))

import predictor_pb2  # noqa: E402
import predictor_pb2_grpc  # noqa: E402


LOGGER = logging.getLogger(__name__)


class GrpcPredictorWorker(predictor_pb2_grpc.OutputTokenPredictorServicer):
    """Resolve task IDs to admission-time replay text and run one predictor."""

    def __init__(
        self,
        *,
        predictor: Predictor,
        replay_store: ReplayStore,
        mapper: RankQuantileMapper,
    ) -> None:
        self._predictor = predictor
        self._replay_store = replay_store
        self._mapper = mapper
        self._inference_lock = threading.Lock()

    @classmethod
    def from_paths(
        cls,
        *,
        predictor: Predictor,
        sidecar_path: Path,
        manifest_path: Path,
    ) -> "GrpcPredictorWorker":
        return cls(
            predictor=predictor,
            replay_store=ReplayStore.from_path(sidecar_path),
            mapper=RankQuantileMapper.from_path(manifest_path),
        )

    @property
    def model_version(self) -> str:
        return self._mapper.model_version

    @property
    def mapping_version(self) -> str:
        return self._mapper.mapping_version

    @property
    def approximation_notice(self) -> str:
        return self._mapper.approximation_notice

    @property
    def first_request_id(self) -> str:
        return self._replay_store.first_request_id()

    def Health(self, request, context):  # noqa: N802
        del request, context
        return predictor_pb2.HealthResponse(
            ready=True,
            model_version=self.model_version,
            reason="",
        )

    def BatchPredict(self, request, context):  # noqa: N802
        del context
        response = predictor_pb2.BatchPredictResponse()
        for task in request.tasks:
            output = response.predictions.add(model_version=self.model_version)
            try:
                record = self._replay_store.get(task.task_id)
            except KeyError:
                output.error = "unknown_request_id"
                continue

            predictor_input = PredictorInput(
                request_id=task.task_id,
                prompt_token_ids=(),
                metadata={
                    "prompt_text": record.prompt_text,
                    "tool_schema_text": record.tool_schema_text,
                },
            )
            try:
                with self._inference_lock:
                    raw_prediction = self._predictor.predict(predictor_input)
                mapped = self._mapper.map_score(raw_prediction.score)
                output.quantiles.update(mapped.quantiles)
                output.signals.update(mapped.signals)
            except Exception:
                LOGGER.exception("predictor failed for request_id=%s", task.task_id)
                output.error = "predictor_error"
        return response


def load_worker(
    *,
    checkpoint: Path,
    sidecar_path: Path,
    manifest_path: Path,
) -> GrpcPredictorWorker:
    return GrpcPredictorWorker.from_paths(
        predictor=BertPredictor(checkpoint),
        sidecar_path=sidecar_path,
        manifest_path=manifest_path,
    )


def start_server(
    worker: GrpcPredictorWorker,
    addr: str,
    *,
    max_workers: int = 4,
):
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    predictor_pb2_grpc.add_OutputTokenPredictorServicer_to_server(worker, server)
    port = server.add_insecure_port(addr)
    if port == 0:
        raise RuntimeError(f"failed to bind gRPC worker to {addr}")
    server.start()
    return server, port
