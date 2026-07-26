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


class ExplodingPredictor:
    """Fails the test if the gate lets an abstained request reach the model."""

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, predictor_input):  # noqa: ANN001 - protocol shape
        self.calls += 1
        raise AssertionError("predictor must not run for an abstained stratum")


def gate_vocabulary(tmp_path):
    from scheduler_benchmark.tool_vocabulary import GateVocabulary, toolset_fingerprint

    path = tmp_path / "gate_confidence.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "gate-confidence-v1",
                "confidence_by_stratum": {"S1": 0.0, "S2": 0.0, "S3": 0.25, "S4": 0.5},
                "unknown_confidence": 0.0,
                "fingerprint_prefix_length": 32,
                "train_fingerprints": [
                    toolset_fingerprint(tool_schema("alpha", "beta"))[:32]
                ],
                "train_tool_names": ["alpha", "beta"],
            }
        )
    )
    return GateVocabulary.from_artifact(path)


def tool_schema(*names: str) -> str:
    tools = ", ".join(f'{{"name": "{name}", "description": "d"}}' for name in names)
    return (
        "You are an expert in composing functions.\n"
        "Here is a list of functions in JSON format that you can invoke:\n"
        f"[{tools}]. \n"
    )


def gated_request(schema: str) -> dict[str, object]:
    request = valid_request()
    request["tool_schema_text"] = schema
    return request


def test_abstained_stratum_short_circuits_without_running_the_predictor(
    tmp_path,
) -> None:
    predictor = ExplodingPredictor()
    app = DecisionApplication(
        predictor=predictor,
        predictor_revision="stub-exploding-v1",
        feature_variant="prompt_schema_history_workflow",
        gate_vocabulary=gate_vocabulary(tmp_path),
    )

    # S1: the tool-set combination is in the training vocabulary -> confidence 0.0
    response = app.decide(gated_request(tool_schema("alpha", "beta")))

    assert predictor.calls == 0
    assert response["prediction_reliable"] is False
    assert response["reason_code"] == "low_reliability"
    assert response["reliability_probability"] == 0.0
    assert "estimated_tokens" not in response


@pytest.mark.parametrize(
    "schema",
    (
        pytest.param(tool_schema("alpha"), id="S2_new_combination"),
        pytest.param(tool_schema(), id="unknown_empty_tool_list"),
        pytest.param("raw unparseable schema", id="unknown_unparseable"),
    ),
)
def test_every_zero_confidence_stratum_short_circuits(tmp_path, schema) -> None:
    predictor = ExplodingPredictor()
    app = DecisionApplication(
        predictor=predictor,
        predictor_revision="stub-exploding-v1",
        feature_variant="prompt_schema_history_workflow",
        gate_vocabulary=gate_vocabulary(tmp_path),
    )

    response = app.decide(gated_request(schema))

    assert predictor.calls == 0
    assert response["reliability_probability"] == 0.0
    assert response["prediction_reliable"] is False


@pytest.mark.parametrize(
    ("schema", "confidence"),
    (
        pytest.param(tool_schema("alpha", "gamma"), 0.25, id="S3_partial_new"),
        pytest.param(tool_schema("gamma", "delta"), 0.5, id="S4_all_new"),
    ),
)
def test_trusted_strata_still_run_the_predictor(tmp_path, schema, confidence) -> None:
    app = DecisionApplication(
        predictor=ConstantPredictor(0.25, confidence, False),
        predictor_revision="stub-constant-v1",
        feature_variant="prompt_schema_history_workflow",
        gate_vocabulary=gate_vocabulary(tmp_path),
        reliability_threshold=0.2,
    )

    response = app.decide(gated_request(schema))

    assert response["reliability_probability"] == confidence
    assert response["prediction_reliable"] is True
    assert response["estimated_tokens"] >= 1


def test_default_threshold_marks_every_rule_c_confidence_unreliable(tmp_path) -> None:
    """Operational tripwire: Rule C tops out at 0.6233, the default gate is 0.8.

    A deployment that keeps the default threshold runs BERT on S3/S4 and then
    discards the answer. Serving must lower --reliability-threshold; this test
    exists so that requirement cannot be forgotten silently.
    """
    app = DecisionApplication(
        predictor=ConstantPredictor(0.25, 0.6233, False),
        predictor_revision="stub-constant-v1",
        feature_variant="prompt_schema_history_workflow",
        gate_vocabulary=gate_vocabulary(tmp_path),
    )

    response = app.decide(gated_request(tool_schema("gamma", "delta")))

    assert response["reason_code"] == "low_reliability"
    assert response["prediction_reliable"] is False


def test_short_circuit_response_is_contract_identical_to_the_predictor_path(
    tmp_path,
) -> None:
    """The abstain path must not invent or drop a single field."""
    schema = tool_schema("alpha", "beta")
    gated = DecisionApplication(
        predictor=ExplodingPredictor(),
        predictor_revision="stub-v1",
        feature_variant="prompt_schema_history_workflow",
        gate_vocabulary=gate_vocabulary(tmp_path),
    )
    ungated = DecisionApplication(
        predictor=ConstantPredictor(0.25, 0.0, False),
        predictor_revision="stub-v1",
        feature_variant="prompt_schema_history_workflow",
    )

    assert gated.decide(gated_request(schema)) == ungated.decide(gated_request(schema))


def test_gate_vocabulary_is_auto_wired_from_the_predictor(tmp_path) -> None:
    """BertPredictor exposes .gate_vocabulary; serving must pick it up unasked.

    Without this the short-circuit would silently never fire in production,
    because run_decision_service does not pass gate_vocabulary explicitly.
    """

    class PredictorWithVocabulary(ExplodingPredictor):
        gate_vocabulary = None  # replaced per instance below

    predictor = PredictorWithVocabulary()
    predictor.gate_vocabulary = gate_vocabulary(tmp_path)
    app = DecisionApplication(
        predictor=predictor,
        predictor_revision="stub-v1",
        feature_variant="prompt_schema_history_workflow",
    )

    response = app.decide(gated_request(tool_schema("alpha", "beta")))

    assert predictor.calls == 0
    assert response["prediction_reliable"] is False


def test_without_a_gate_vocabulary_behaviour_is_unchanged() -> None:
    app = make_app(confidence=0.0)

    response = app.decide(valid_request())

    assert response["reliability_probability"] == 0.0
    assert response["prediction_reliable"] is False


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
    # The replay client (ltr_training/tier2.py) sends EITHER a system-message
    # schema OR a tools array, never both. The system-message fallback only
    # applies when no tools array is present; with both, tools wins (real
    # gateway traffic carries agent instructions in system, not a schema).
    request.pop("tools", None)
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


def test_tools_array_derives_schema_text_when_explicit_text_absent() -> None:
    # Gateways forward the OpenAI `tools` array without ltr_tool_schema; the
    # service must score that traffic instead of crashing into fail-open.
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
    request.pop("tool_schema_text", None)
    request["messages"] = [{"role": "user", "content": "current prompt"}]
    tools = [
        {
            "type": "function",
            "function": {"name": "glob", "parameters": {"type": "object"}},
        }
    ]
    request["tools"] = tools

    app.decide(request)

    derived = predictor.predictor_input.metadata["tool_schema_text"]
    assert json.loads(derived) == tools
    assert derived == json.dumps(tools, sort_keys=True, separators=(",", ":"))


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
