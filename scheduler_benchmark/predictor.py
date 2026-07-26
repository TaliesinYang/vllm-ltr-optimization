"""Admission-time predictor contract and implementations."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Protocol

from .contracts import MAX_ESTIMATED_TOKENS, RELIABLE
from .micro_batcher import MicroBatcher

if TYPE_CHECKING:
    from .tool_vocabulary import GateVocabulary


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
    """Predictor for the trained ToolACE prompt-schema ranker, CPU or CUDA.

    With ``batch_max > 1`` concurrent requests are coalesced into one forward by
    a single worker thread (see ``micro_batcher``). On GPU this is the
    difference between contending forwards and one padded batch; the 201-box
    prototype measured p99 687 ms naive against 53 ms batched at concurrency 8.
    """

    MAX_LENGTH = 512

    def __init__(
        self,
        checkpoint: Path,
        *,
        vocabulary: "GateVocabulary | None" = None,
        device: str = "cpu",
        batch_max: int = 1,
        batch_window_ms: float = 0.0,
        fp16: bool | None = None,
    ) -> None:
        import os
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from .tool_vocabulary import GateVocabulary as _GateVocabulary

        # Confidence comes from the Cold-Start stratum the request falls into,
        # measured offline (T5, issue #9) and loaded from a committed artifact
        # so the served values cannot drift from the evaluated ones.
        self._vocabulary = (
            vocabulary if vocabulary is not None else _GateVocabulary.from_artifact()
        )
        # Escape hatch: pin confidence for A/B runs and gate-disabled baselines.
        override = os.environ.get("LTR_CONSTANT_CONFIDENCE")
        self._confidence_override = float(override) if override is not None else None

        # CPU thread cap: with the default intra-op pool = all cores, N
        # concurrent HTTP handlers each launch a full-core forward and
        # oversubscribe the machine (e.g. 8 handlers x 96 threads on 192
        # vCPUs), exploding tail latency. Bound intra-op threads so
        # concurrent decision calls stay fast and predictable.
        _threads = int(os.environ.get("LTR_DECISION_TORCH_THREADS", "2"))
        torch.set_num_threads(max(1, _threads))
        self._torch = torch

        requested = str(device).lower()
        if requested.startswith("cuda") and not torch.cuda.is_available():
            print(
                "[predictor] cuda requested but torch.cuda.is_available() is False; "
                "falling back to cpu",
                flush=True,
            )
            requested = "cpu"
        self._device = torch.device(requested)
        self._device_name = requested

        # fp16 on CUDA only: it is what the 201-box prototype measured, and it
        # roughly halves the forward. On CPU fp16 is slower, not faster.
        self._fp16 = self._device_name.startswith("cuda") if fp16 is None else fp16
        if self._fp16 and not self._device_name.startswith("cuda"):
            print("[predictor] fp16 requested off-CUDA; ignoring", flush=True)
            self._fp16 = False

        self._tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, local_files_only=True
        )
        model_kwargs: dict[str, object] = {"local_files_only": True}
        if self._fp16:
            model_kwargs["dtype"] = torch.float16
        self._model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, **model_kwargs
        )
        self._model.to(self._device)
        self._model.eval()

        if self._device_name.startswith("cuda"):
            # Compile the kernels before the first real request, so the tail of
            # a short benchmark is not dominated by one-off CUDA warm-up.
            with torch.inference_mode():
                warm = self._tokenizer(
                    ["warmup"],
                    padding=True,
                    truncation=True,
                    max_length=self.MAX_LENGTH,
                    return_tensors="pt",
                ).to(self._device)
                self._model(**warm)
            torch.cuda.synchronize()

        # One worker owns the tokenizer and the device, so batched forwards
        # need no locking and cannot race the HF fast tokenizer.
        self._batcher: MicroBatcher | None = None
        if batch_max > 1:
            self._batcher = MicroBatcher(
                self._score_batch,
                batch_max=batch_max,
                window_s=max(0.0, batch_window_ms) / 1000.0,
            )
        print(
            f"[predictor] device={self._device_name} fp16={self._fp16} "
            f"batch_max={batch_max} batch_window_ms={batch_window_ms}",
            flush=True,
        )

    @property
    def gate_vocabulary(self) -> "GateVocabulary":
        """Exposed so the Decision Service can classify before calling the model.

        Sharing this object rather than a copy is what makes the gate's
        short-circuit and this predictor's confidence agree by construction.
        """
        return self._vocabulary

    @property
    def device(self) -> str:
        return self._device_name

    @property
    def batcher(self) -> "MicroBatcher | None":
        return self._batcher

    def close(self) -> None:
        if self._batcher is not None:
            self._batcher.close()

    def _score_batch(self, texts: list[str]) -> list[float]:
        """Tokenize and run one forward for the whole batch. Worker thread only."""
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.MAX_LENGTH,
            return_tensors="pt",
        ).to(self._device)
        with self._torch.inference_mode():
            logits = self._model(**inputs).logits.reshape(-1)
        if logits.numel() != len(texts):
            raise ValueError(
                f"BERT predictor returned {logits.numel()} logits for {len(texts)} rows"
            )
        # .float() so an fp16 forward still yields full-precision Python floats.
        return [
            float(value) for value in self._torch.sigmoid(logits).float().tolist()
        ]

    def predict(self, predictor_input: PredictorInput) -> Prediction:
        started = time.perf_counter()
        prompt = _required_metadata_text(predictor_input, "prompt_text")
        tool_schema = _required_metadata_text(predictor_input, "tool_schema_text")
        rendered = f"[USER]\n{prompt}\n[TOOLS]\n{tool_schema}"
        if self._batcher is not None:
            score = self._batcher.submit(rendered)
        else:
            score = self._score_batch([rendered])[0]
        latency_ms = (time.perf_counter() - started) * 1000.0
        # A measured Kendall tau-b lower bound for this request's Cold-Start
        # stratum, NOT a calibrated probability. Requests whose tool set cannot
        # be read get the artifact's conservative value rather than a guess.
        confidence = (
            self._confidence_override
            if self._confidence_override is not None
            else self._vocabulary.confidence(tool_schema)
        )
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
