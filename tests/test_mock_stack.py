import asyncio

import pytest

import scheduler_benchmark.mock_stack as mock_stack_module
from scheduler_benchmark.decision_service import DecisionApplication
from scheduler_benchmark.contracts import RELIABLE
from scheduler_benchmark.mock_stack import MockGatewayStack
from scheduler_benchmark.predictor import ConstantPredictor
from scheduler_benchmark.rank_quantiles import (
    APPROXIMATION_NOTICE,
    MAPPING_VERSION,
    RankQuantileMapper,
)
from scheduler_benchmark.runner import WorkloadRequest, stream_completion


def test_decision_request_extracts_exact_tool_schema_from_vllm_xargs() -> None:
    tool_schema = '[{"type":"function","function":{"name":"lookup"}}]'
    payload = {
        "model": "model-path",
        "messages": [{"role": "user", "content": "hello"}],
        "vllm_xargs": {
            "ltr_kind": "tool",
            "ltr_category": "id:toolace",
            "ltr_tool_schema": tool_schema,
        },
    }

    request = mock_stack_module._build_decision_request(
        payload,
        request_id="request-1",
        decision_id="decision-request-1",
        workflow_id="request-1",
        step_id="0",
        conversation_id="request-1",
        previous_tool_gap_ms=0,
    )

    assert request["tool_schema_text"] == tool_schema


def test_strip_decision_only_xargs_preserves_engine_metadata_and_input() -> None:
    tool_schema = '[{"type":"function","function":{"name":"lookup"}}]'
    payload = {
        "model": "model-path",
        "vllm_xargs": {
            "ltr_kind": "tool",
            "ltr_category": "id:toolace",
            "ltr_tool_schema": tool_schema,
        },
    }

    stripped = mock_stack_module._strip_decision_only_xargs(payload)

    assert stripped["vllm_xargs"] == {
        "ltr_kind": "tool",
        "ltr_category": "id:toolace",
    }
    assert payload["vllm_xargs"]["ltr_tool_schema"] == tool_schema


def test_cpu_stack_runs_client_gateway_decision_engine_chain() -> None:
    mapper = RankQuantileMapper(
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
    application = DecisionApplication(
        predictor=ConstantPredictor(score=0.25, confidence=0.95, ood=False),
        predictor_revision="stub-v1",
        feature_variant="prompt",
        quantile_mapper=mapper,
        quantile_manifest_sha256="c" * 64,
    )
    request = WorkloadRequest(
        request_id="request-1",
        prompt="final",
        tool_schema="[]",
        history=[["human", "prior"]],
        baseline_service_ms=10.0,
        max_tokens=4096,
        kind="tool",
        category="single_turn",
    )

    try:
        stack = MockGatewayStack(application)
    except PermissionError as exc:
        pytest.skip(f"sandbox does not permit localhost socket bind: {exc}")

    with stack:
        async def run_request():
            import aiohttp

            async with aiohttp.ClientSession() as session:
                return await stream_completion(
                    session,
                    stack.gateway_endpoint,
                    "model-path",
                    request,
                    api_key=None,
                )

        sample = asyncio.run(run_request())
        forwarded = stack.last_engine_payload
        audit = stack.last_gateway_audit

    assert sample.output_tokens == 3
    assert sample.first_token_at_unix_s is not None
    assert sample.error is None
    assert stack.gateway_endpoint.endswith("/v1/chat/completions")
    assert stack.engine_endpoint.endswith("/v1/chat/completions")
    assert forwarded["messages"][0] == {"role": "user", "content": "prior"}
    assert forwarded["messages"][-1] == {"role": "user", "content": "final"}
    assert forwarded["stream_options"] == {"include_usage": True}
    assert forwarded["vllm_xargs"] == {
        "ltr_kind": "tool",
        "ltr_category": "single_turn",
        "workflow_estimated_tokens": 135,
        "prediction_reliable": RELIABLE,
        "workflow_id": "request-1",
        "step_id": "0",
        "decision_id": "decision-request-1",
    }
    assert audit.fallback_source is None
    assert stack.decision_request_count == 1
    assert stack.engine_request_count == 1
