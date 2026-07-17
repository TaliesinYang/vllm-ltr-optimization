"""Admission-time predictor contract and development stubs."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol


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
    ``confidence`` is calibrated reliability in [0, 1]. ``ood`` marks an
    out-of-distribution input. ``latency_ms`` is predictor wall-clock cost.
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
