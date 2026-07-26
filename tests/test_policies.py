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


# --- gated_rule_c: slot-preserving gating (ticket #15) ---------------------


def abstained(request_id: str, arrival_time_s: float, score: float) -> RequestContext:
    """Rule C assigns confidence 0.0 to S1/S2/unknown - the Ranker abstains."""
    return request(request_id, arrival_time_s, score, confidence=0.0)


def trusted(request_id: str, arrival_time_s: float, score: float) -> RequestContext:
    """Rule C assigns S3 0.5787 / S4 0.6233 - above the 0.5 gate."""
    return request(request_id, arrival_time_s, score, confidence=0.6233)


def order_ids(waiting, policy, **kwargs) -> list[str]:
    return [
        item.request_id
        for item in order_waiting_requests(waiting, policy, now_s=10.0, **kwargs)
    ]


def test_gated_rule_c_with_no_trusted_requests_is_pure_fcfs() -> None:
    waiting = [
        abstained("c", 3.0, 0.1),
        abstained("a", 1.0, 0.9),
        abstained("b", 2.0, 0.5),
    ]

    assert order_ids(waiting, "gated_rule_c") == order_ids(waiting, "fcfs")


def test_gated_rule_c_with_all_trusted_requests_is_pure_ltr() -> None:
    waiting = [
        trusted("c", 3.0, 0.1),
        trusted("a", 1.0, 0.9),
        trusted("b", 2.0, 0.5),
    ]

    assert order_ids(waiting, "gated_rule_c") == order_ids(waiting, "pure_ltr")


def test_gated_rule_c_keeps_abstained_requests_in_their_arrival_slots() -> None:
    """The whole point: uncertainty must not cost a request its place."""
    waiting = [
        trusted("t1", 1.0, 0.9),
        abstained("a1", 2.0, 0.1),
        trusted("t2", 3.0, 0.2),
        abstained("a2", 4.0, 0.4),
        trusted("t3", 5.0, 0.5),
    ]

    ordered = order_ids(waiting, "gated_rule_c")

    # Abstained requests hold slots 1 and 3 (0-based), exactly where FCFS put them.
    assert ordered[1] == "a1"
    assert ordered[3] == "a2"
    # Trusted requests fill the remaining slots, re-sorted by predicted score.
    assert [ordered[0], ordered[2], ordered[4]] == ["t2", "t3", "t1"]


def test_gated_rule_c_only_permutes_within_the_trusted_slots() -> None:
    waiting = [
        abstained("a1", 1.0, 0.9),
        trusted("t1", 2.0, 0.8),
        trusted("t2", 3.0, 0.1),
        abstained("a2", 4.0, 0.2),
    ]

    ordered = order_ids(waiting, "gated_rule_c")

    assert ordered[0] == "a1"
    assert ordered[3] == "a2"
    assert sorted(ordered[1:3]) == ["t1", "t2"]
    # t2 has the lower predicted cost, so it takes the earlier trusted slot.
    assert ordered[1] == "t2"


def test_gated_rule_c_does_not_let_a_trusted_request_leapfrog_an_older_abstained_one() -> None:
    """No starvation via uncertainty - the failure mode this design prevents."""
    waiting = [
        abstained("old_abstained", 1.0, 0.99),
        trusted("new_trusted", 9.0, 0.01),
    ]

    assert order_ids(waiting, "gated_rule_c") == ["old_abstained", "new_trusted"]


def test_gated_rule_c_has_no_opportunity_condition() -> None:
    """Unlike gated_hybrid, a shallow queue does not silently disable the Ranker."""
    waiting = [trusted("t1", 1.0, 0.9), trusted("t2", 2.0, 0.1)]

    # queue_depth == 2 with a tiny opportunity would fail _has_ltr_opportunity;
    # gated_rule_c must still apply LTR order to trusted requests.
    assert order_ids(waiting, "gated_rule_c") == ["t2", "t1"]


def test_gated_rule_c_treats_ood_requests_as_abstained() -> None:
    waiting = [
        request("ood_first", 1.0, 0.99, confidence=0.9, ood=True),
        trusted("trusted_second", 2.0, 0.01),
    ]

    assert order_ids(waiting, "gated_rule_c") == ["ood_first", "trusted_second"]


def test_gated_rule_c_is_a_permutation_of_its_input() -> None:
    waiting = [
        trusted("t1", 1.0, 0.5),
        abstained("a1", 2.0, 0.5),
        trusted("t2", 3.0, 0.5),
    ]

    assert sorted(order_ids(waiting, "gated_rule_c")) == ["a1", "t1", "t2"]


# --- policy_fcfs: FCFS on the custom scheduler base (ticket #15) -----------


def test_policy_fcfs_orders_by_arrival_regardless_of_predictions() -> None:
    waiting = [
        request("late_but_cheap", 3.0, 0.01, confidence=0.99),
        request("early_but_costly", 1.0, 0.99, confidence=0.99),
        request("middle", 2.0, 0.5, confidence=0.0),
    ]

    assert order_ids(waiting, "policy_fcfs") == [
        "early_but_costly",
        "middle",
        "late_but_cheap",
    ]


def test_policy_fcfs_matches_fcfs_ordering_exactly() -> None:
    waiting = [
        request("c", 3.0, 0.1),
        request("a", 1.0, 0.9),
        request("b", 2.0, 0.5, confidence=0.0, ood=True),
    ]

    assert order_ids(waiting, "policy_fcfs") == order_ids(waiting, "fcfs")


def test_new_policies_are_registered() -> None:
    from scheduler_benchmark.policies import POLICIES

    assert "gated_rule_c" in POLICIES
    assert "policy_fcfs" in POLICIES


def test_new_schedulers_are_registered_on_the_custom_policy_base() -> None:
    """PolicyFCFS must share _PolicyScheduler with the other custom policies.

    That is the whole reason it exists: StockFCFSShim inherits vLLM's stock
    Scheduler, so it cannot separate "better ordering" from "different
    scheduler code path".
    """
    from scheduler_benchmark.vllm_scheduler import (
        SCHEDULER_CLASSES,
        GatedRuleCScheduler,
        PolicyFCFS,
        PureLTRScheduler,
        StockFCFSShim,
    )

    assert SCHEDULER_CLASSES["policy_fcfs"] is PolicyFCFS
    assert SCHEDULER_CLASSES["gated_rule_c"] is GatedRuleCScheduler
    assert PolicyFCFS.policy_name == "policy_fcfs"
    assert PolicyFCFS.uses_predictor is False
    assert GatedRuleCScheduler.policy_name == "gated_rule_c"
    assert GatedRuleCScheduler.uses_predictor is True

    base = PureLTRScheduler.__mro__[1]
    assert base in PolicyFCFS.__mro__
    assert base in GatedRuleCScheduler.__mro__
    assert base not in StockFCFSShim.__mro__


def test_every_policy_name_has_a_scheduler_class() -> None:
    from scheduler_benchmark.policies import POLICIES
    from scheduler_benchmark.vllm_scheduler import SCHEDULER_CLASSES

    assert set(POLICIES) == set(SCHEDULER_CLASSES)


def test_runner_scheduler_mapping_covers_every_policy() -> None:
    # The argparse choices for --scheduler-cls derive from this mapping; a
    # policy missing here is invisible to the benchmark runner (bit us on
    # rental day: PolicyFCFS/GatedRuleC existed but the runner rejected them).
    from scheduler_benchmark.policies import POLICIES
    from scheduler_benchmark.runner import SCHEDULER_CLASS_TO_POLICY

    assert set(SCHEDULER_CLASS_TO_POLICY.values()) >= set(POLICIES)
