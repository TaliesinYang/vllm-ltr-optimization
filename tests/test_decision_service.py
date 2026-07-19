import http.client
import json
import threading

import pytest

from scheduler_benchmark.decision_service import (
    DecisionApplication,
    DecisionError,
    create_decision_server,
)
from scheduler_benchmark.predictor import ConstantPredictor, Prediction
from scheduler_benchmark.rank_quantiles import (
    APPROXIMATION_NOTICE,
    MAPPING_VERSION,
    RankQuantileMapper,
)


QUANTILE_MANIFEST_SHA256 = "test-sha"


def minimal_quantile_manifest() -> dict[str, object]:
    return {
        "mapping_version": MAPPING_VERSION,
        "model_version": "test-model",
        "approximation_notice": APPROXIMATION_NOTICE,
        "sample_count": 6000,
        "percentiles": {
            str(percentile): float(10 + 5 * percentile)
            for percentile in range(10, 100)
        },
        "global_quantiles": {"50": 260.0, "70": 360.0, "90": 460.0},
    }


def valid_request(*, with_optional: bool = True) -> dict[str, object]:
    request: dict[str, object] = {
        "schema_version": "1.0",
        "request_id": "request-1",
        "decision_id": "decision-1",
        "model_id": "Qwen/Qwen3.5-9B",
        "request_age_ms": 0,
        "messages": [{"role": "user", "content": "Book a flight"}],
        "generation_controls": {
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
            "max_tokens": 4096,
        },
    }
    if with_optional:
        request.update(
            {
                "workflow_id": "workflow-1",
                "step_id": "step-1",
                "conversation_id": "conversation-1",
                "previous_tool_gap_ms": 25,
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "search", "parameters": {}},
                    }
                ],
            }
        )
    return request


def make_app(
    *,
    score: float = 0.25,
    confidence: float = 0.9,
    ood: bool = False,
    ready: bool = True,
    feature_variant: str = "prompt_schema_history_workflow",
    quantile_mapper: RankQuantileMapper | None = None,
    quantile_manifest_sha256: str | None = None,
) -> DecisionApplication:
    return DecisionApplication(
        predictor=ConstantPredictor(score, confidence, ood),
        predictor_revision="stub-constant-v1",
        feature_variant=feature_variant,
        ready=ready,
        quantile_mapper=quantile_mapper,
        quantile_manifest_sha256=quantile_manifest_sha256,
    )


def test_reliable_prediction_echoes_decision_id_and_estimate() -> None:
    response = make_app().decide(valid_request())

    assert response == {
        "schema_version": "1.0",
        "decision_id": "decision-1",
        "estimated_tokens": 1024,
        "reliability_probability": 0.9,
        "ood_score": 0.0,
        "prediction_reliable": True,
        "predictor_revision": "stub-constant-v1",
        "feature_variant": "prompt_schema_history_workflow",
        "reason_code": "prediction_reliable",
    }


@pytest.mark.parametrize(
    ("confidence", "ood", "with_optional", "expected_reason"),
    [
        (0.1, True, False, "ood_rejected"),
        (0.1, False, False, "low_reliability"),
        (0.9, False, False, "missing_optional_features"),
    ],
)
def test_unreliable_reason_precedence_omits_estimated_tokens(
    confidence: float,
    ood: bool,
    with_optional: bool,
    expected_reason: str,
) -> None:
    response = make_app(confidence=confidence, ood=ood).decide(
        valid_request(with_optional=with_optional)
    )

    assert response["decision_id"] == "decision-1"
    assert response["prediction_reliable"] is False
    assert response["reason_code"] == expected_reason
    assert "estimated_tokens" not in response


def test_prompt_variant_does_not_require_optional_features() -> None:
    response = make_app(feature_variant="prompt").decide(
        valid_request(with_optional=False)
    )

    assert response["prediction_reliable"] is True
    assert response["reason_code"] == "prediction_reliable"


def test_decision_application_transports_exact_prompt_schema_training_text() -> None:
    class CapturingPredictor:
        def __init__(self) -> None:
            self.predictor_input = None

        def predict(self, predictor_input):
            self.predictor_input = predictor_input
            return Prediction(0.25, 0.9, False, 1.0)

    predictor = CapturingPredictor()
    app = DecisionApplication(
        predictor=predictor,
        predictor_revision="capture",
        feature_variant="prompt_schema",
    )
    request = valid_request()
    request["messages"] = [
        {"role": "system", "content": "raw ToolACE system\nwith spacing\n"},
        {"role": "user", "content": "current prompt"},
    ]

    response = app.decide(request)

    assert response["reason_code"] == "prediction_reliable"
    assert predictor.predictor_input.metadata["prompt_text"] == "current prompt"
    assert (
        predictor.predictor_input.metadata["tool_schema_text"]
        == "raw ToolACE system\nwith spacing\n"
    )


def test_explicit_tool_schema_text_takes_precedence_over_system_message() -> None:
    class CapturingPredictor:
        def __init__(self) -> None:
            self.predictor_input = None

        def predict(self, predictor_input):
            self.predictor_input = predictor_input
            return Prediction(0.25, 0.9, False, 1.0)

    predictor = CapturingPredictor()
    app = DecisionApplication(
        predictor=predictor,
        predictor_revision="capture",
        feature_variant="prompt_schema",
    )
    request = valid_request()
    request["messages"] = [
        {"role": "system", "content": "system-message schema"},
        {"role": "user", "content": "current prompt"},
    ]
    request["tool_schema_text"] = "explicit schema\nwith exact spacing\n"

    app.decide(request)

    assert predictor.predictor_input.metadata["tool_schema_text"] == (
        "explicit schema\nwith exact spacing\n"
    )


def test_mapper_estimate_includes_quantile_provenance() -> None:
    response = make_app(
        quantile_mapper=RankQuantileMapper(minimal_quantile_manifest()),
        quantile_manifest_sha256=QUANTILE_MANIFEST_SHA256,
    ).decide(valid_request())

    assert response["estimated_tokens"] == 135
    assert response["mapping_version"] == MAPPING_VERSION
    assert response["approximation_notice"] == APPROXIMATION_NOTICE
    assert response["quantile_manifest_sha256"] == QUANTILE_MANIFEST_SHA256


def test_unreliable_mapper_response_retains_provenance_without_estimate() -> None:
    response = make_app(
        confidence=0.1,
        quantile_mapper=RankQuantileMapper(minimal_quantile_manifest()),
        quantile_manifest_sha256=QUANTILE_MANIFEST_SHA256,
    ).decide(valid_request())

    assert response["prediction_reliable"] is False
    assert "estimated_tokens" not in response
    assert response["mapping_version"] == MAPPING_VERSION
    assert response["approximation_notice"] == APPROXIMATION_NOTICE
    assert response["quantile_manifest_sha256"] == QUANTILE_MANIFEST_SHA256


def test_generation_controls_accept_max_tokens_4096() -> None:
    response = make_app().decide(valid_request())

    assert response["reason_code"] == "prediction_reliable"


def test_tool_schema_text_rejects_empty_string() -> None:
    request = valid_request()
    request["tool_schema_text"] = ""

    with pytest.raises(DecisionError, match="invalid_request"):
        make_app().decide(request)


def test_tool_schema_text_accepts_262144_utf8_bytes() -> None:
    request = valid_request()
    request["tool_schema_text"] = "é" * 131_072

    response = make_app().decide(request)

    assert response["reason_code"] == "prediction_reliable"


def test_tool_schema_text_rejects_262145_utf8_bytes() -> None:
    request = valid_request()
    request["tool_schema_text"] = "é" * 131_072 + "a"

    with pytest.raises(DecisionError, match="invalid_request"):
        make_app().decide(request)


@pytest.mark.parametrize(
    ("mutate", "status", "error_code"),
    [
        (lambda row: row.update(schema_version="2.0"), 400, "invalid_schema"),
        (lambda row: row.pop("request_id"), 422, "invalid_request"),
        (
            lambda row: row["generation_controls"].update(temperature=0.7),
            422,
            "unsupported_controls",
        ),
    ],
)
def test_request_validation_returns_typed_errors(
    mutate, status: int, error_code: str
) -> None:
    request = valid_request()
    mutate(request)

    with pytest.raises(DecisionError) as raised:
        make_app().decide(request)

    assert raised.value.status == status
    assert raised.value.error_code == error_code
    assert raised.value.body == {
        "schema_version": "1.0",
        "error_code": error_code,
        "retryable": False,
    }


def test_not_ready_is_typed_503() -> None:
    with pytest.raises(DecisionError) as raised:
        make_app(ready=False).decide(valid_request())

    assert raised.value.status == 503
    assert raised.value.error_code == "not_ready"
    assert raised.value.body["retryable"] is True


class RaisingPredictor:
    def predict(self, predictor_input):
        del predictor_input
        raise RuntimeError("checkpoint failure")


def test_predictor_exception_is_typed_internal_error() -> None:
    app = DecisionApplication(
        predictor=RaisingPredictor(),
        predictor_revision="broken",
        feature_variant="prompt",
    )

    with pytest.raises(DecisionError) as raised:
        app.decide(valid_request(with_optional=False))

    assert raised.value.status == 503
    assert raised.value.error_code == "internal_error"
    assert raised.value.body["retryable"] is True


class BlockingPredictor:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def predict(self, predictor_input):
        del predictor_input
        self.entered.set()
        self.release.wait(timeout=2.0)
        return ConstantPredictor(0.5, 1.0, False).predict(
            type("Input", (), {"request_id": "unused"})()
        )


def test_saturated_concurrency_is_typed_rate_limit() -> None:
    predictor = BlockingPredictor()
    app = DecisionApplication(
        predictor=predictor,
        predictor_revision="blocking",
        feature_variant="prompt",
        max_concurrency=1,
    )
    first_error: list[Exception] = []

    def first_request() -> None:
        try:
            app.decide(valid_request(with_optional=False))
        except Exception as exc:  # pragma: no cover - assertion reports thread failure
            first_error.append(exc)

    worker = threading.Thread(target=first_request)
    worker.start()
    assert predictor.entered.wait(timeout=1.0)

    try:
        with pytest.raises(DecisionError) as raised:
            app.decide(valid_request(with_optional=False))
        assert raised.value.status == 429
        assert raised.value.error_code == "rate_limited"
        assert raised.value.body["retryable"] is True
    finally:
        predictor.release.set()
        worker.join(timeout=2.0)

    assert not first_error


def request_json(
    server, method: str, path: str, body: bytes | None = None
) -> tuple[int, dict[str, object]]:
    connection = http.client.HTTPConnection(*server.server_address, timeout=2.0)
    headers = {"Content-Type": "application/json"} if body is not None else {}
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    payload = json.loads(response.read())
    connection.close()
    return response.status, payload


def start_server(app: DecisionApplication, *, max_body_bytes: int = 2 * 1024 * 1024):
    server = create_decision_server(
        app, host="127.0.0.1", port=0, max_body_bytes=max_body_bytes
    )
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    return server, worker


def stop_server(server, worker: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    worker.join(timeout=2.0)


def test_http_post_decision_returns_contract_json() -> None:
    server, worker = start_server(make_app())
    try:
        status, response = request_json(
            server,
            "POST",
            "/v1/decision",
            json.dumps(valid_request()).encode("utf-8"),
        )
    finally:
        stop_server(server, worker)

    assert status == 200
    assert response["decision_id"] == "decision-1"
    assert response["prediction_reliable"] is True


def test_http_invalid_json_is_typed_invalid_request() -> None:
    server, worker = start_server(make_app())
    try:
        status, response = request_json(
            server, "POST", "/v1/decision", b"not-json"
        )
    finally:
        stop_server(server, worker)

    assert status == 422
    assert response == {
        "schema_version": "1.0",
        "error_code": "invalid_request",
        "retryable": False,
    }


def test_http_oversized_body_is_typed_413() -> None:
    server, worker = start_server(make_app(), max_body_bytes=32)
    try:
        status, response = request_json(
            server, "POST", "/v1/decision", b"x" * 33
        )
    finally:
        stop_server(server, worker)

    assert status == 413
    assert response == {
        "schema_version": "1.0",
        "error_code": "body_too_large",
        "retryable": False,
    }


@pytest.mark.parametrize(("ready", "expected_status"), [(True, 200), (False, 503)])
def test_health_endpoint_reflects_readiness(
    ready: bool, expected_status: int
) -> None:
    server, worker = start_server(make_app(ready=ready))
    try:
        status, response = request_json(server, "GET", "/healthz")
    finally:
        stop_server(server, worker)

    assert status == expected_status
    if ready:
        assert response == {"schema_version": "1.0", "ready": True}
    else:
        assert response["error_code"] == "not_ready"
