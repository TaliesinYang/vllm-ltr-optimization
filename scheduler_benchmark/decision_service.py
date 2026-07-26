"""Schema-bound request-time prediction service for VeloxMesh."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Mapping

from scheduler_benchmark.contracts import MAX_ESTIMATED_TOKENS
from scheduler_benchmark.predictor import Predictor, PredictorInput
from scheduler_benchmark.rank_quantiles import RankQuantileMapper

if TYPE_CHECKING:
    from scheduler_benchmark.tool_vocabulary import GateVocabulary

SCHEMA_VERSION = "1.0"
DEFAULT_RELIABILITY_THRESHOLD = 0.8
MAX_TOOL_SCHEMA_TEXT_BYTES = 262_144
FEATURE_VARIANTS = {
    "prompt": (),
    "prompt_schema": ("tools",),
    "prompt_schema_history": ("tools", "conversation_id"),
    "prompt_schema_history_workflow": (
        "tools",
        "conversation_id",
        "workflow_id",
        "step_id",
        "previous_tool_gap_ms",
    ),
}
REQUIRED_FIELDS = (
    "request_id",
    "decision_id",
    "model_id",
    "request_age_ms",
    "messages",
    "generation_controls",
)
ID_LIMITS = {
    "request_id": 128,
    "decision_id": 128,
    "model_id": 256,
    "workflow_id": 128,
    "step_id": 128,
    "conversation_id": 128,
}
SUPPORTED_CONTROL_KEYS = {
    "temperature",
    "top_p",
    "seed",
    "max_tokens",
    "stop",
    "response_format",
    "stream",
}


@dataclass(frozen=True)
class DecisionError(Exception):
    status: int
    error_code: str
    retryable: bool = False

    @property
    def body(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "error_code": self.error_code,
            "retryable": self.retryable,
        }


class DecisionApplication:
    """Validate one decision request and adapt Artifact 1 predictor output."""

    def __init__(
        self,
        *,
        predictor: Predictor,
        predictor_revision: str,
        feature_variant: str,
        reliability_threshold: float = DEFAULT_RELIABILITY_THRESHOLD,
        ready: bool = True,
        max_concurrency: int = 8,
        quantile_mapper: RankQuantileMapper | None = None,
        quantile_manifest_sha256: str | None = None,
        gate_vocabulary: "GateVocabulary | None" = None,
    ) -> None:
        if feature_variant not in FEATURE_VARIANTS:
            raise ValueError(f"unknown feature variant: {feature_variant}")
        if not 0.0 <= reliability_threshold <= 1.0:
            raise ValueError("reliability_threshold must be between 0 and 1")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        if quantile_mapper is None and quantile_manifest_sha256 is not None:
            raise ValueError("quantile manifest SHA requires a quantile mapper")
        if quantile_mapper is not None and (
            not isinstance(quantile_manifest_sha256, str)
            or not quantile_manifest_sha256
        ):
            raise ValueError("quantile mapper requires non-empty manifest provenance")
        self._predictor = predictor
        self._predictor_revision = predictor_revision
        self._feature_variant = feature_variant
        self._reliability_threshold = reliability_threshold
        self._ready = ready
        self._capacity = threading.BoundedSemaphore(max_concurrency)
        self._quantile_mapper = quantile_mapper
        self._quantile_manifest_sha256 = quantile_manifest_sha256
        # Gate-first short-circuit. When the predictor carries a training
        # vocabulary, requests whose stratum earns zero confidence are answered
        # without running the model: the answer cannot depend on the score,
        # because an unreliable decision never reports estimated_tokens.
        # Predictors without a vocabulary keep the original flow exactly.
        self._gate_vocabulary = (
            gate_vocabulary
            if gate_vocabulary is not None
            else getattr(predictor, "gate_vocabulary", None)
        )

    @property
    def is_ready(self) -> bool:
        return self._ready

    def decide(self, request: Mapping[str, object]) -> dict[str, object]:
        if not self._ready:
            raise DecisionError(503, "not_ready", retryable=True)
        validated = _validate_request(request)
        predictor_input = _predictor_input(validated)
        missing_optional = _has_missing_optional_features(
            validated, self._feature_variant
        )

        abstained = self._abstain_confidence(predictor_input)
        if abstained is not None:
            # No model call, no capacity slot: the gate answers on its own.
            return self._build_response(
                validated,
                confidence=abstained,
                ood=False,
                score=None,
                missing_optional=missing_optional,
            )

        if not self._capacity.acquire(blocking=False):
            raise DecisionError(429, "rate_limited", retryable=True)
        try:
            prediction = self._predictor.predict(predictor_input)
        except DecisionError:
            raise
        except Exception as exc:
            raise DecisionError(503, "internal_error", retryable=True) from exc
        finally:
            self._capacity.release()

        return self._build_response(
            validated,
            confidence=prediction.confidence,
            ood=prediction.ood,
            score=prediction.score,
            missing_optional=missing_optional,
        )

    def _abstain_confidence(self, predictor_input: PredictorInput) -> float | None:
        """The gate's confidence when it is zero, else None to run the model."""
        if self._gate_vocabulary is None:
            return None
        tool_schema = predictor_input.metadata.get("tool_schema_text")
        if not isinstance(tool_schema, str) or not tool_schema:
            # Nothing to classify; let the predictor decide (and fail) as before.
            return None
        confidence = self._gate_vocabulary.confidence(tool_schema)
        return confidence if confidence <= 0.0 else None

    def _build_response(
        self,
        validated: Mapping[str, object],
        *,
        confidence: float,
        ood: bool,
        score: float | None,
        missing_optional: bool,
    ) -> dict[str, object]:
        reason = _reason_code(
            ood=ood,
            confidence=confidence,
            missing_optional=missing_optional,
            threshold=self._reliability_threshold,
        )
        is_reliable = reason == "prediction_reliable"
        response: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "decision_id": validated["decision_id"],
            "reliability_probability": confidence,
            "ood_score": 1.0 if ood else 0.0,
            "prediction_reliable": is_reliable,
            "predictor_revision": self._predictor_revision,
            "feature_variant": self._feature_variant,
            "reason_code": reason,
        }
        if self._quantile_mapper is not None:
            response.update(
                {
                    "mapping_version": self._quantile_mapper.mapping_version,
                    "approximation_notice": (
                        self._quantile_mapper.approximation_notice
                    ),
                    "quantile_manifest_sha256": self._quantile_manifest_sha256,
                }
            )
        if is_reliable:
            if score is None:
                raise ValueError("a reliable decision requires a predictor score")
            if self._quantile_mapper is None:
                response["estimated_tokens"] = _score_to_estimated_tokens(score)
            else:
                mapped = self._quantile_mapper.map_score(score)
                response["estimated_tokens"] = max(
                    1,
                    min(
                        MAX_ESTIMATED_TOKENS,
                        round(mapped.quantiles[50]),
                    ),
                )
        return response


def _validate_request(request: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(request, Mapping):
        raise DecisionError(422, "invalid_request")
    if request.get("schema_version") != SCHEMA_VERSION:
        raise DecisionError(400, "invalid_schema")
    if "queue_summary" in request or "cache_pressure" in request:
        raise DecisionError(422, "invalid_request")
    for field in REQUIRED_FIELDS:
        if field not in request:
            raise DecisionError(422, "invalid_request")
    for field, limit in ID_LIMITS.items():
        value = request.get(field)
        if value is None and field not in REQUIRED_FIELDS:
            continue
        if not isinstance(value, str) or not 1 <= len(value.encode("utf-8")) <= limit:
            raise DecisionError(422, "invalid_request")
    age = request["request_age_ms"]
    if isinstance(age, bool) or not isinstance(age, int) or age < 0:
        raise DecisionError(422, "invalid_request")
    _validate_messages(request["messages"])
    tools = request.get("tools")
    if tools is not None and not isinstance(tools, list):
        raise DecisionError(422, "invalid_request")
    if "tool_schema_text" in request:
        tool_schema_text = request["tool_schema_text"]
        if not isinstance(tool_schema_text, str):
            raise DecisionError(422, "invalid_request")
        try:
            tool_schema_bytes = tool_schema_text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise DecisionError(422, "invalid_request") from exc
        if not 1 <= len(tool_schema_bytes) <= MAX_TOOL_SCHEMA_TEXT_BYTES:
            raise DecisionError(422, "invalid_request")
    gap = request.get("previous_tool_gap_ms")
    if gap is not None and (
        isinstance(gap, bool) or not isinstance(gap, int) or gap < 0
    ):
        raise DecisionError(422, "invalid_request")
    _validate_generation_controls(request["generation_controls"])
    return dict(request)


def _validate_messages(value: object) -> None:
    if not isinstance(value, list) or not value:
        raise DecisionError(422, "invalid_request")
    for message in value:
        if not isinstance(message, Mapping):
            raise DecisionError(422, "invalid_request")
        if message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise DecisionError(422, "invalid_request")
        if "content" not in message and "tool_calls" not in message:
            raise DecisionError(422, "invalid_request")


def _validate_generation_controls(value: object) -> None:
    if not isinstance(value, Mapping):
        raise DecisionError(422, "invalid_request")
    if set(value) - SUPPORTED_CONTROL_KEYS:
        raise DecisionError(422, "unsupported_controls")
    supported_profile = {
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
    }
    if any(value.get(key) != expected for key, expected in supported_profile.items()):
        raise DecisionError(422, "unsupported_controls")
    max_tokens = value.get("max_tokens")
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or not 1 <= max_tokens <= MAX_ESTIMATED_TOKENS
    ):
        raise DecisionError(422, "unsupported_controls")


def _predictor_input(request: Mapping[str, object]) -> PredictorInput:
    serialized = json.dumps(
        {
            "messages": request["messages"],
            "tools": request.get("tools"),
            "workflow_id": request.get("workflow_id"),
            "step_id": request.get("step_id"),
            "conversation_id": request.get("conversation_id"),
            "previous_tool_gap_ms": request.get("previous_tool_gap_ms"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    metadata = {
        "model_id": request["model_id"],
        "request_age_ms": request["request_age_ms"],
        "generation_controls": request["generation_controls"],
    }
    messages = request["messages"]
    final_content = messages[-1].get("content")
    if isinstance(final_content, str) and final_content:
        metadata["prompt_text"] = final_content
    explicit_tool_schema_text = request.get("tool_schema_text")
    if isinstance(explicit_tool_schema_text, str):
        metadata["tool_schema_text"] = explicit_tool_schema_text
    else:
        tools = request.get("tools")
        if isinstance(tools, list) and tools:
            # Gateways forward the OpenAI `tools` array without a rendered
            # schema text (clients never send `ltr_tool_schema`). Derive a
            # deterministic serialization so real traffic is scored instead
            # of crashing into fail-open.
            metadata["tool_schema_text"] = json.dumps(
                tools, sort_keys=True, separators=(",", ":")
            )
        else:
            system_contents = [
                message.get("content")
                for message in messages
                if message.get("role") == "system"
                and isinstance(message.get("content"), str)
                and message.get("content")
            ]
            if len(system_contents) == 1:
                metadata["tool_schema_text"] = system_contents[0]
    return PredictorInput(
        request_id=str(request["request_id"]),
        prompt_token_ids=tuple(serialized.encode("utf-8")),
        metadata=metadata,
    )


def _has_missing_optional_features(
    request: Mapping[str, object], feature_variant: str
) -> bool:
    required_optional = FEATURE_VARIANTS[feature_variant]
    return any(
        _optional_feature_missing(request, field) for field in required_optional
    )


def _optional_feature_missing(request: Mapping[str, object], field: str) -> bool:
    if field == "tools":
        # "tools" stands for THE SCHEMA, which has three transports: explicit
        # tool_schema_text, the OpenAI tools array, or the single-system-message
        # fallback used by ToolACE replay traffic. Any of them satisfies it.
        if isinstance(request.get("tool_schema_text"), str):
            return False
        if request.get("tools") is not None:
            return False
        messages = request.get("messages")
        if isinstance(messages, list):
            system_contents = [
                message.get("content")
                for message in messages
                if isinstance(message, Mapping)
                and message.get("role") == "system"
                and isinstance(message.get("content"), str)
                and message.get("content")
            ]
            if len(system_contents) == 1:
                return False
        return True
    return request.get(field) is None


def _reason_code(
    *, ood: bool, confidence: float, missing_optional: bool, threshold: float
) -> str:
    if ood:
        return "ood_rejected"
    if confidence < threshold:
        return "low_reliability"
    if missing_optional:
        return "missing_optional_features"
    return "prediction_reliable"


def _score_to_estimated_tokens(score: float) -> int:
    return max(1, min(MAX_ESTIMATED_TOKENS, round(score * MAX_ESTIMATED_TOKENS)))


class DecisionHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        application: DecisionApplication,
        max_body_bytes: int,
    ) -> None:
        if max_body_bytes < 1:
            raise ValueError("max_body_bytes must be positive")
        self.application = application
        self.max_body_bytes = max_body_bytes
        super().__init__(server_address, DecisionRequestHandler)


class DecisionRequestHandler(BaseHTTPRequestHandler):
    def handle_one_request(self):  # noqa: D401
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    server: DecisionHTTPServer

    def do_GET(self) -> None:
        if self.path != "/healthz":
            self._write_error(DecisionError(404, "invalid_request"))
            return
        if not self.server.application.is_ready:
            self._write_error(DecisionError(503, "not_ready", retryable=True))
            return
        self._write_json(200, {"schema_version": SCHEMA_VERSION, "ready": True})

    def do_POST(self) -> None:
        if self.path != "/v1/decision":
            self._write_error(DecisionError(404, "invalid_request"))
            return
        try:
            content_length = self._content_length()
            if content_length > self.server.max_body_bytes:
                raise DecisionError(413, "body_too_large")
            body = self.rfile.read(content_length)
            if len(body) != content_length:
                raise DecisionError(422, "invalid_request")
            try:
                request = json.loads(body)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DecisionError(422, "invalid_request") from exc
            response = self.server.application.decide(request)
        except DecisionError as exc:
            self._write_error(exc)
            return
        except Exception as exc:
            self._write_error(
                DecisionError(503, "internal_error", retryable=True)
            )
            return
        self._write_json(200, response)

    def _content_length(self) -> int:
        raw_length = self.headers.get("Content-Length")
        try:
            content_length = int(raw_length) if raw_length is not None else -1
        except ValueError as exc:
            raise DecisionError(422, "invalid_request") from exc
        if content_length < 0:
            raise DecisionError(422, "invalid_request")
        return content_length

    def _write_error(self, error: DecisionError) -> None:
        self._write_json(error.status, error.body)

    def _write_json(self, status: int, body: Mapping[str, object]) -> None:
        encoded = (json.dumps(body, separators=(",", ":")) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        try:
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def log_message(self, format: str, *args: object) -> None:
        del format, args


def create_decision_server(
    application: DecisionApplication,
    *,
    host: str = "127.0.0.1",
    port: int = 8081,
    max_body_bytes: int = 2 * 1024 * 1024,
) -> DecisionHTTPServer:
    return DecisionHTTPServer((host, port), application, max_body_bytes)
