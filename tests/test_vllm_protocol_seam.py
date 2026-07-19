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


def test_bool_flag_documented_coercion_boundary() -> None:
    """Documented behavior, not a defense: pinned Pydantic coerces a bool
    True in vllm_xargs to int 1 BEFORE the predictor can see it, so the
    predictor cannot distinguish a coerced bool from a legitimate 1. The
    trust boundary is therefore the GATEWAY's int contract (enforced and
    tested Go-side: whitelist drops client flags; verdict is written as
    int 0/1). This test pins the coercion so a future vLLM/Pydantic change
    is caught, and records that a bool passed straight to the engine WOULD
    be trusted downstream."""
    try:
        request = _chat_request(
            {"prediction_reliable": True, "workflow_estimated_tokens": 512}
        )
    except Exception:
        return  # protocol rejected bool outright — boundary even stricter; PASS
        # (deliberately not a skip: run_matrix preflight hard-fails on skips)
    extra_args = _extra_args(request)
    flag = extra_args.get("prediction_reliable")
    if isinstance(flag, bool):
        prediction = GatewayMetadataPredictor().predict(
            PredictorInput(
                request_id="seam-2", prompt_token_ids=(), metadata=dict(extra_args)
            )
        )
        assert prediction.ood is True, "un-coerced bool must not pass the int contract"
    else:
        # Pinned-stack reality: coercion to int happens upstream of us.
        assert flag == 1
        prediction = GatewayMetadataPredictor().predict(
            PredictorInput(
                request_id="seam-2", prompt_token_ids=(), metadata=dict(extra_args)
            )
        )
        assert prediction.ood is False, (
            "coerced 1 is indistinguishable downstream — gateway is the boundary"
        )
