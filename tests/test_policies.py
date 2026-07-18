import pytest

from scheduler_benchmark.policies import (
    PolicyConfig,
    RequestContext,
    order_waiting_requests,
)
from scheduler_benchmark.predictor import Prediction


def prediction(
    score: float, *, confidence: float = 0.9, ood: bool = False, latency_ms: float = 0.1
) -> Prediction:
    return Prediction(score, confidence, ood, latency_ms)


def request(
    request_id: str,
    arrival_time_s: float,
    score: float,
    *,
    confidence: float = 0.9,
    ood: bool = False,
    kind: str = "chat",
    category: str = "",
    prompt_token_count: int = 1,
) -> RequestContext:
    return RequestContext(
        request_id=request_id,
        arrival_time_s=arrival_time_s,
        prediction=prediction(score, confidence=confidence, ood=ood),
        kind=kind,
        category=category,
        prompt_token_count=prompt_token_count,
    )


def test_fcfs_preserves_arrival_order() -> None:
    waiting = [request("late", 2.0, 0.1), request("early", 1.0, 0.99)]

    ordered = order_waiting_requests(waiting, "fcfs", now_s=3.0)

    assert [item.request_id for item in ordered] == ["early", "late"]


def test_pure_ltr_orders_by_predictor_score() -> None:
    waiting = [request("long", 1.0, 0.9), request("short", 2.0, 0.1)]

    ordered = order_waiting_requests(waiting, "pure_ltr", now_s=3.0)

    assert [item.request_id for item in ordered] == ["short", "long"]


def test_tail_safe_uses_visible_category_risk_not_prediction() -> None:
    waiting = [
        request("risky", 1.0, 0.0, kind="tool", category="multi_turn"),
        request("simple", 2.0, 0.999, kind="tool", category="simple"),
    ]

    ordered = order_waiting_requests(waiting, "tail_safe", now_s=3.0)

    assert [item.request_id for item in ordered] == ["simple", "risky"]


def test_tail_safe_aging_eventually_promotes_old_request() -> None:
    waiting = [
        request("old-risky", 0.0, 0.0, kind="tool", category="multi_turn"),
        request("new-simple", 19.9, 0.999, kind="tool", category="simple"),
    ]

    ordered = order_waiting_requests(waiting, "tail_safe", now_s=20.0)

    assert ordered[0].request_id == "old-risky"


def test_gated_hybrid_uses_ltr_for_reliable_opportunity() -> None:
    waiting = [
        request("predicted-long", 1.0, 0.9),
        request("predicted-short", 2.0, 0.1),
    ]

    ordered = order_waiting_requests(waiting, "gated_hybrid", now_s=3.0)

    assert [item.request_id for item in ordered] == [
        "predicted-short",
        "predicted-long",
    ]


def test_gated_hybrid_falls_back_for_ood_prediction() -> None:
    waiting = [
        request("risky", 1.0, 0.0, ood=True, kind="chat"),
        request("simple", 2.0, 0.999, ood=True, kind="tool", category="simple"),
    ]

    ordered = order_waiting_requests(waiting, "gated_hybrid", now_s=3.0)

    assert [item.request_id for item in ordered] == ["simple", "risky"]


def test_gated_hybrid_can_use_reliable_tool_prediction() -> None:
    waiting = [
        request("predicted-long", 1.0, 0.9, kind="tool", category="simple"),
        request("predicted-short", 2.0, 0.1, kind="tool", category="multi_turn"),
    ]

    ordered = order_waiting_requests(waiting, "gated_hybrid", now_s=3.0)

    assert [item.request_id for item in ordered] == [
        "predicted-short",
        "predicted-long",
    ]


def test_gated_hybrid_falls_back_for_low_confidence_chat() -> None:
    waiting = [
        request("older", 0.0, 0.0, confidence=0.1),
        request("newer", 19.9, 0.999, confidence=0.1),
    ]

    ordered = order_waiting_requests(waiting, "gated_hybrid", now_s=20.0)

    assert ordered[0].request_id == "older"


def test_prompt_sjf_orders_by_prompt_token_count_without_prediction() -> None:
    waiting = [
        request("long", 1.0, 0.01, prompt_token_count=100),
        request("short", 2.0, 0.99, prompt_token_count=10),
    ]

    ordered = order_waiting_requests(waiting, "prompt_sjf", now_s=3.0)

    assert [item.request_id for item in ordered] == ["short", "long"]


def test_prompt_sjf_falls_back_to_fcfs_when_prompt_is_empty() -> None:
    waiting = [
        request("new-empty", 2.0, 0.01, prompt_token_count=0),
        request("old", 1.0, 0.99, prompt_token_count=100),
    ]

    ordered = order_waiting_requests(waiting, "prompt_sjf", now_s=3.0)

    assert [item.request_id for item in ordered] == ["old", "new-empty"]


def test_ltr_aging_promotes_sufficiently_old_high_score_request() -> None:
    config = PolicyConfig(deadline_ms=1000.0, ltr_scale=400.0)
    waiting = [
        request("old-long", 0.0, 0.9),
        request("new-short", 9.9, 0.1),
    ]

    ordered = order_waiting_requests(waiting, "ltr_aging", now_s=10.0, config=config)

    assert [item.request_id for item in ordered] == ["old-long", "new-short"]


def test_policy_name_is_closed_to_six_benchmark_policies() -> None:
    with pytest.raises(ValueError, match="unknown policy"):
        order_waiting_requests([], "oracle", now_s=0.0, config=PolicyConfig())
