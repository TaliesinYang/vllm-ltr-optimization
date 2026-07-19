"""VeloxMesh-side validation and vLLM metadata transport helpers."""

from __future__ import annotations

import copy
import json
import math
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Mapping

from scheduler_benchmark.contracts import (
    MAX_ESTIMATED_TOKENS,
    RELIABLE,
    UNRELIABLE,
)
from scheduler_benchmark.decision_service import FEATURE_VARIANTS, SCHEMA_VERSION
from scheduler_benchmark.rank_quantiles import APPROXIMATION_NOTICE, MAPPING_VERSION

OPTIMIZER_XARGS = {
    "workflow_estimated_tokens",
    "prediction_reliable",
    "workflow_id",
    "step_id",
    "decision_id",
}
REASON_CODES = {
    "prediction_reliable",
    "low_reliability",
    "ood_rejected",
    "missing_optional_features",
}
SERVICE_ERROR_CODES = {
    "invalid_schema",
    "body_too_large",
    "invalid_request",
    "unsupported_controls",
    "rate_limited",
    "not_ready",
    "internal_error",
}


@dataclass(frozen=True)
class DecisionRPCResult:
    bundle: Mapping[str, object] | None
    error_code: str | None = None


@dataclass(frozen=True)
class GatewayDecisionAudit:
    decision_id: str
    fallback_source: str | None
    reason_code: str | None = None
    error_code: str | None = None


def call_decision_service(
    endpoint: str,
    request: Mapping[str, object],
    *,
    timeout_s: float,
) -> DecisionRPCResult:
    """Perform one decision RPC without request-path retries."""

    if timeout_s <= 0.0:
        raise ValueError("timeout_s must be positive")
    body = json.dumps(request, separators=(",", ":")).encode("utf-8")
    http_request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(http_request, timeout=timeout_s) as response:
            response_body = response.read()
    except urllib.error.HTTPError as exc:
        error_code = _service_error_code(exc.read())
        return DecisionRPCResult(bundle=None, error_code=error_code)
    except (TimeoutError, socket.timeout):
        return DecisionRPCResult(bundle=None, error_code="timeout")
    except urllib.error.URLError:
        return DecisionRPCResult(bundle=None, error_code="unavailable")
    try:
        parsed = json.loads(response_body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return DecisionRPCResult(bundle=None, error_code="malformed_response")
    if not isinstance(parsed, Mapping):
        return DecisionRPCResult(bundle=None, error_code="malformed_response")
    return DecisionRPCResult(bundle=dict(parsed))


def apply_decision_to_payload(
    payload: Mapping[str, object],
    rpc_result: DecisionRPCResult,
    *,
    expected_decision_id: str,
    workflow_id: str | None,
    step_id: str | None,
) -> tuple[dict[str, object], GatewayDecisionAudit]:
    forwarded = copy.deepcopy(dict(payload))
    xargs_value = forwarded.get("vllm_xargs", {})
    xargs = dict(xargs_value) if isinstance(xargs_value, Mapping) else {}
    for key in OPTIMIZER_XARGS:
        xargs.pop(key, None)
    if xargs:
        forwarded["vllm_xargs"] = xargs
    else:
        forwarded.pop("vllm_xargs", None)

    if rpc_result.bundle is None or rpc_result.error_code is not None:
        return forwarded, GatewayDecisionAudit(
            decision_id=expected_decision_id,
            fallback_source="fallback_native",
            error_code=rpc_result.error_code or "malformed_response",
        )

    try:
        bundle = _validate_bundle(
            rpc_result.bundle, expected_decision_id=expected_decision_id
        )
    except (TypeError, ValueError, KeyError):
        return forwarded, GatewayDecisionAudit(
            decision_id=expected_decision_id,
            fallback_source="fallback_native",
            error_code="malformed_response",
        )

    is_reliable = bool(bundle["prediction_reliable"])
    transported = dict(xargs)
    transported.update(
        {
            "prediction_reliable": RELIABLE if is_reliable else UNRELIABLE,
            "decision_id": expected_decision_id,
        }
    )
    if workflow_id is not None:
        transported["workflow_id"] = workflow_id
    if step_id is not None:
        transported["step_id"] = step_id
    if is_reliable:
        transported["workflow_estimated_tokens"] = bundle["estimated_tokens"]
    forwarded["vllm_xargs"] = transported
    return forwarded, GatewayDecisionAudit(
        decision_id=expected_decision_id,
        fallback_source=None if is_reliable else "fallback_native",
        reason_code=str(bundle["reason_code"]),
    )


def _validate_bundle(
    bundle: Mapping[str, object], *, expected_decision_id: str
) -> dict[str, object]:
    if not isinstance(bundle, Mapping):
        raise TypeError("decision response must be an object")
    required = {
        "schema_version",
        "decision_id",
        "reliability_probability",
        "ood_score",
        "prediction_reliable",
        "predictor_revision",
        "feature_variant",
        "reason_code",
        "mapping_version",
        "approximation_notice",
        "quantile_manifest_sha256",
    }
    if required - set(bundle):
        raise ValueError("decision response omitted required fields")
    if bundle["schema_version"] != SCHEMA_VERSION:
        raise ValueError("decision schema mismatch")
    if bundle["decision_id"] != expected_decision_id:
        raise ValueError("decision_id mismatch")
    is_reliable = bundle["prediction_reliable"]
    if not isinstance(is_reliable, bool):
        raise TypeError("prediction_reliable must be boolean")
    reliability = _bounded_float(bundle["reliability_probability"], 0.0, 1.0)
    ood_score = _bounded_float(bundle["ood_score"], 0.0, math.inf)
    if not isinstance(bundle["predictor_revision"], str) or not bundle[
        "predictor_revision"
    ]:
        raise TypeError("predictor_revision must be non-empty")
    if bundle["feature_variant"] not in FEATURE_VARIANTS:
        raise ValueError("unknown feature_variant")
    reason_code = bundle["reason_code"]
    if reason_code not in REASON_CODES:
        raise ValueError("unknown reason_code")
    if bundle["mapping_version"] != MAPPING_VERSION:
        raise ValueError("unknown mapping_version")
    if bundle["approximation_notice"] != APPROXIMATION_NOTICE:
        raise ValueError("invalid approximation_notice")
    manifest_sha256 = bundle["quantile_manifest_sha256"]
    if not (
        isinstance(manifest_sha256, str)
        and len(manifest_sha256) == 64
        and all(
            character in "0123456789abcdefABCDEF"
            for character in manifest_sha256
        )
    ):
        raise ValueError("invalid quantile_manifest_sha256")
    if is_reliable:
        estimated_tokens = bundle.get("estimated_tokens")
        if (
            isinstance(estimated_tokens, bool)
            or not isinstance(estimated_tokens, int)
            or not 1 <= estimated_tokens <= MAX_ESTIMATED_TOKENS
            or reason_code != "prediction_reliable"
        ):
            raise ValueError("reliable response has invalid estimate or reason")
    elif "estimated_tokens" in bundle or reason_code == "prediction_reliable":
        raise ValueError("unreliable response contains reliable-only fields")
    normalized = dict(bundle)
    normalized["reliability_probability"] = reliability
    normalized["ood_score"] = ood_score
    return normalized


def _bounded_float(value: object, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("value must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or not lower <= normalized <= upper:
        raise ValueError("numeric value outside contract")
    return normalized


def _service_error_code(body: bytes) -> str:
    try:
        parsed = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "malformed_response"
    if not isinstance(parsed, Mapping):
        return "malformed_response"
    error_code = parsed.get("error_code")
    return str(error_code) if error_code in SERVICE_ERROR_CODES else "malformed_response"
