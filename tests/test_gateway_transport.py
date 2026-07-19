import threading

import pytest

from scheduler_benchmark.contracts import RELIABLE, UNRELIABLE
from scheduler_benchmark.gateway_transport import (
    DecisionRPCResult,
    apply_decision_to_payload,
    call_decision_service,
)
from scheduler_benchmark.decision_service import (
    DecisionApplication,
    create_decision_server,
)
from scheduler_benchmark.predictor import ConstantPredictor
from scheduler_benchmark.rank_quantiles import (
    APPROXIMATION_NOTICE,
    MAPPING_VERSION,
    RankQuantileMapper,
)


QUANTILE_MANIFEST_SHA256 = "b" * 64


def quantile_mapper() -> RankQuantileMapper:
    return RankQuantileMapper(
        {
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
    )


def base_payload() -> dict[str, object]:
    return {
        "model": "model-path",
        "prompt": "hello",
        "vllm_xargs": {"ltr_kind": "tool", "ltr_category": "multi_turn"},
    }


def reliable_bundle() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "decision_id": "decision-1",
        "estimated_tokens": 321,
        "reliability_probability": 0.95,
        "ood_score": 0.1,
        "prediction_reliable": True,
        "predictor_revision": "stub-v1",
        "feature_variant": "prompt",
        "reason_code": "prediction_reliable",
        "mapping_version": MAPPING_VERSION,
        "approximation_notice": APPROXIMATION_NOTICE,
        "quantile_manifest_sha256": QUANTILE_MANIFEST_SHA256,
    }


def unreliable_bundle() -> dict[str, object]:
    bundle = reliable_bundle()
    bundle.pop("estimated_tokens")
    bundle.update(
        {
            "reliability_probability": 0.4,
            "prediction_reliable": False,
            "reason_code": "low_reliability",
        }
    )
    return bundle


def test_reliable_decision_injects_namespaced_vllm_xargs() -> None:
    payload, audit = apply_decision_to_payload(
        base_payload(),
        DecisionRPCResult(bundle=reliable_bundle()),
        expected_decision_id="decision-1",
        workflow_id="workflow-1",
        step_id="step-1",
    )

    assert payload["vllm_xargs"] == {
        "ltr_kind": "tool",
        "ltr_category": "multi_turn",
        "workflow_estimated_tokens": 321,
        "prediction_reliable": RELIABLE,
        "workflow_id": "workflow-1",
        "step_id": "step-1",
        "decision_id": "decision-1",
    }
    assert audit.fallback_source is None
    assert audit.reason_code == "prediction_reliable"


def test_unreliable_decision_uses_native_fallback_without_estimate() -> None:
    payload, audit = apply_decision_to_payload(
        base_payload(),
        DecisionRPCResult(bundle=unreliable_bundle()),
        expected_decision_id="decision-1",
        workflow_id="workflow-1",
        step_id="step-1",
    )

    assert payload["vllm_xargs"] == {
        "ltr_kind": "tool",
        "ltr_category": "multi_turn",
        "prediction_reliable": UNRELIABLE,
        "workflow_id": "workflow-1",
        "step_id": "step-1",
        "decision_id": "decision-1",
    }
    assert audit.fallback_source == "fallback_native"
    assert audit.reason_code == "low_reliability"


@pytest.mark.parametrize("error_code", ["not_ready", "timeout", "malformed_response"])
def test_rpc_failure_omits_all_optimizer_metadata(error_code: str) -> None:
    stale_payload = base_payload()
    stale_payload["vllm_xargs"].update(
        {
            "workflow_estimated_tokens": 999,
            "prediction_reliable": True,
            "decision_id": "stale",
        }
    )

    payload, audit = apply_decision_to_payload(
        stale_payload,
        DecisionRPCResult(bundle=None, error_code=error_code),
        expected_decision_id="decision-1",
        workflow_id="workflow-1",
        step_id="step-1",
    )

    assert payload["vllm_xargs"] == {
        "ltr_kind": "tool",
        "ltr_category": "multi_turn",
    }
    assert audit.fallback_source == "fallback_native"
    assert audit.error_code == error_code


def test_transport_does_not_mutate_caller_payload() -> None:
    original = base_payload()

    apply_decision_to_payload(
        original,
        DecisionRPCResult(bundle=reliable_bundle()),
        expected_decision_id="decision-1",
        workflow_id="workflow-1",
        step_id="step-1",
    )

    assert original == base_payload()


def test_transport_xargs_contains_no_boolean_values() -> None:
    payload, _ = apply_decision_to_payload(
        base_payload(),
        DecisionRPCResult(bundle=reliable_bundle()),
        expected_decision_id="decision-1",
        workflow_id="workflow-1",
        step_id="step-1",
    )

    assert not any(
        isinstance(value, bool) for value in payload["vllm_xargs"].values()
    )


def test_mismatched_decision_id_falls_back_as_malformed_response() -> None:
    payload, audit = apply_decision_to_payload(
        base_payload(),
        DecisionRPCResult(bundle=reliable_bundle()),
        expected_decision_id="different-decision",
        workflow_id="workflow-1",
        step_id="step-1",
    )

    assert "workflow_estimated_tokens" not in payload["vllm_xargs"]
    assert audit.fallback_source == "fallback_native"
    assert audit.error_code == "malformed_response"


def test_decision_rpc_calls_real_service_without_retry_layer() -> None:
    application = DecisionApplication(
        predictor=ConstantPredictor(score=0.25, confidence=0.95, ood=False),
        predictor_revision="stub-v1",
        feature_variant="prompt",
        quantile_mapper=quantile_mapper(),
        quantile_manifest_sha256=QUANTILE_MANIFEST_SHA256,
    )
    server = create_decision_server(application, host="127.0.0.1", port=0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    host, port = server.server_address
    request = {
        "schema_version": "1.0",
        "request_id": "request-1",
        "decision_id": "decision-1",
        "model_id": "model-path",
        "request_age_ms": 0,
        "messages": [{"role": "user", "content": "hello"}],
        "generation_controls": {
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
            "max_tokens": 4096,
        },
    }
    try:
        result = call_decision_service(
            f"http://{host}:{port}/v1/decision", request, timeout_s=1.0
        )
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2.0)

    assert result.error_code is None
    assert result.bundle["decision_id"] == "decision-1"
    assert result.bundle["estimated_tokens"] == 135
