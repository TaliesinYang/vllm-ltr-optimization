"""Admission-time predictor contract and implementations."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

from .contracts import MAX_ESTIMATED_TOKENS, RELIABLE


@dataclass(frozen=True)
class PredictorInput:
    """Admission-visible input; no future completion tokens are allowed."""

    request_id: str
    prompt_token_ids: tuple[int, ...]
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class Prediction:
    """Normalized scheduler prediction.

    ``score`` is a rank-space cost in [0, 1], where lower schedules earlier.
    ``confidence`` is a reliability signal in [0, 1]; each predictor documents
    whether it is calibrated. ``ood`` marks an out-of-distribution input.
    ``latency_ms`` is predictor wall-clock cost.
    """

    score: float
    confidence: float
    ood: bool
    latency_ms: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be normalized between 0 and 1")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0.0:
            raise ValueError("latency_ms must be finite and non-negative")


class Predictor(Protocol):
    def predict(self, predictor_input: PredictorInput) -> Prediction: ...


class BertPredictor:
    """CPU predictor for the trained ToolACE prompt-schema ranker."""

    MAX_LENGTH = 512
    PLACEHOLDER_CONFIDENCE = 0.9

    def __init__(self, checkpoint: Path) -> None:
        import os
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # CPU thread cap: with the default intra-op pool = all cores, N
        # concurrent HTTP handlers each launch a full-core forward and
        # oversubscribe the machine (e.g. 8 handlers x 96 threads on 192
        # vCPUs), exploding tail latency. Bound intra-op threads so
        # concurrent decision calls stay fast and predictable.
        _threads = int(os.environ.get("LTR_DECISION_TORCH_THREADS", "2"))
        torch.set_num_threads(max(1, _threads))
        self._torch = torch
        self._device = torch.device("cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, local_files_only=True
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, local_files_only=True
        )
        self._model.to(self._device)
        self._model.eval()

    def predict(self, predictor_input: PredictorInput) -> Prediction:
        started = time.perf_counter()
        prompt = _required_metadata_text(predictor_input, "prompt_text")
        tool_schema = _required_metadata_text(predictor_input, "tool_schema_text")
        rendered = f"[USER]\n{prompt}\n[TOOLS]\n{tool_schema}"
        inputs = self._tokenizer(
            [rendered],
            padding=True,
            truncation=True,
            max_length=self.MAX_LENGTH,
            return_tensors="pt",
        ).to(self._device)
        with self._torch.inference_mode():
            logits = self._model(**inputs).logits.reshape(-1)
        if logits.numel() != 1:
            raise ValueError("BERT predictor must return exactly one logit")
        score = float(self._torch.sigmoid(logits[0]).item())
        latency_ms = (time.perf_counter() - started) * 1000.0
        # Placeholder only; not a calibrated probability. Ensemble-based
        # confidence is future work.
        confidence = self.PLACEHOLDER_CONFIDENCE
        # No evaluated OOD detector is implemented yet.
        ood = False
        return Prediction(score, confidence, ood, latency_ms)


class GatewayMetadataPredictor:
    """Normalize the gateway's admission-time token estimate for scheduling."""

    def predict(self, predictor_input: PredictorInput) -> Prediction:
        reliable = predictor_input.metadata.get("prediction_reliable")
        estimated_tokens = predictor_input.metadata.get(
            "workflow_estimated_tokens"
        )
        is_reliable = (
            isinstance(reliable, int)
            and not isinstance(reliable, bool)
            and reliable == RELIABLE
        )
        has_valid_estimate = (
            isinstance(estimated_tokens, int)
            and not isinstance(estimated_tokens, bool)
            and estimated_tokens >= 1
        )
        if not is_reliable or not has_valid_estimate:
            return Prediction(1.0, 0.0, True, 0.0)
        score = (
            min(estimated_tokens, MAX_ESTIMATED_TOKENS)
            / MAX_ESTIMATED_TOKENS
        )
        return Prediction(score, 0.9, False, 0.0)


def _required_metadata_text(predictor_input: PredictorInput, key: str) -> str:
    value = predictor_input.metadata.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"BERT predictor requires non-empty {key}")
    return value


class ConstantPredictor:
    def __init__(self, score: float, confidence: float, ood: bool) -> None:
        self._score = score
        self._confidence = confidence
        self._ood = ood

    def predict(self, predictor_input: PredictorInput) -> Prediction:
        del predictor_input
        started = time.perf_counter()
        latency_ms = (time.perf_counter() - started) * 1000.0
        return Prediction(self._score, self._confidence, self._ood, latency_ms)


class RandomPredictor:
    def __init__(self, seed: int) -> None:
        self._random = random.Random(seed)

    def predict(self, predictor_input: PredictorInput) -> Prediction:
        del predictor_input
        started = time.perf_counter()
        score = self._random.random()
        latency_ms = (time.perf_counter() - started) * 1000.0
        return Prediction(score, confidence=0.0, ood=True, latency_ms=latency_ms)


class OracleFromFilePredictor:
    def __init__(self, path: Path) -> None:
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError("oracle file must contain a request-id mapping")
        self._rows = raw

    def predict(self, predictor_input: PredictorInput) -> Prediction:
        started = time.perf_counter()
        try:
            row = self._rows[predictor_input.request_id]
        except KeyError as exc:
            raise KeyError(
                f"oracle score missing for {predictor_input.request_id}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError("oracle row must be an object")
        latency_ms = (time.perf_counter() - started) * 1000.0
        return Prediction(
            score=float(row["score"]),
            confidence=float(row.get("confidence", 1.0)),
            ood=bool(row.get("ood", False)),
            latency_ms=latency_ms,
        )
