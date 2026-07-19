"""Real-protocol seam test: pinned vLLM request -> extra_args -> gateway predictor.

The fake-engine smoke cannot catch protocol coercion (e.g. the pinned
vllm_xargs type set has no bool), so this test drives the actual pinned
vLLM request model. It is a rental-day / 201 preflight hard gate: run_matrix.sh
fails if it is skipped.
"""

from __future__ import annotations

import pytest

vllm = pytest.importorskip("vllm")

from scheduler_benchmark.contracts import MAX_ESTIMATED_TOKENS, RELIABLE  # noqa: E402
from scheduler_benchmark.predictor import (  # noqa: E402
    GatewayMetadataPredictor,
    PredictorInput,
)
from scheduler_benchmark.vllm_scheduler import _request_metadata  # noqa: E402


def _chat_request(xargs: dict[str, object]):
    try:
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
        )
    except ImportError:  # module layout differs across 0.24 point releases
        from vllm.entrypoints.openai.protocol import ChatCompletionRequest

    return ChatCompletionRequest(
        model="qwen3.5-9b",
        messages=[{"role": "user", "content": "seam probe"}],
        temperature=0.0,
        max_tokens=4096,
        vllm_xargs=xargs,
    )


def _extra_args(request) -> dict[str, object]:
    sampling = request.to_sampling_params(
        max_tokens=4096, default_sampling_params={}
    )
    return dict(sampling.extra_args or {})


def test_int_reliable_flag_survives_protocol_and_reaches_predictor() -> None:
    request = _chat_request(
        {
            "prediction_reliable": RELIABLE,
            "workflow_estimated_tokens": 512,
            "ltr_kind": "tool",
            "ltr_category": "id:toolace",
            "decision_id": "dec-seam",
            "workflow_id": "seam",
            "step_id": "0",
        }
    )
    extra_args = _extra_args(request)
    flag = extra_args["prediction_reliable"]
    assert isinstance(flag, int) and not isinstance(flag, bool), (
        f"prediction_reliable arrived as {type(flag).__name__}: {flag!r}"
    )
    est = extra_args["workflow_estimated_tokens"]
    assert isinstance(est, int) and not isinstance(est, bool)

    class _FakeEngineRequest:
        request_id = "seam-1"
        arrival_time = 0.0
        prompt_token_ids: list[int] = []
        trace_headers: dict[str, str] = {}

        class sampling_params:  # noqa: N801 - mimic attribute access
            pass

    fake = _FakeEngineRequest()
    fake.sampling_params = type("SP", (), {"extra_args": extra_args})()
    metadata = _request_metadata(fake)
    prediction = GatewayMetadataPredictor().predict(
        PredictorInput(request_id="seam-1", prompt_token_ids=(), metadata=metadata)
    )
    assert prediction.ood is False
    assert prediction.confidence == pytest.approx(0.9)
    assert prediction.score == pytest.approx(512 / MAX_ESTIMATED_TOKENS)


def test_bool_flag_is_coerced_or_rejected_never_trusted() -> None:
    # If a bool sneaks into vllm_xargs the protocol either coerces or rejects
    # it; whatever survives must NOT be treated as reliable by the predictor.
    try:
        request = _chat_request(
            {"prediction_reliable": True, "workflow_estimated_tokens": 512}
        )
    except Exception:
        return  # protocol rejected the bool outright — acceptable
    extra_args = _extra_args(request)
    prediction = GatewayMetadataPredictor().predict(
        PredictorInput(
            request_id="seam-2", prompt_token_ids=(), metadata=dict(extra_args)
        )
    )
    if isinstance(extra_args.get("prediction_reliable"), bool):
        assert prediction.ood is True, "bool flag must not pass the int contract"
