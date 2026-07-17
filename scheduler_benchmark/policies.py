"""Pure queue-ordering policies shared by tests and vLLM adapter."""

from __future__ import annotations

from dataclasses import dataclass

from .predictor import Prediction

POLICIES = ("fcfs", "pure_ltr", "tail_safe", "gated_hybrid")


@dataclass(frozen=True)
class PolicyConfig:
    deadline_ms: float = 2000.0
    tail_weight: float = 0.35
    confidence_threshold: float = 0.5
    ltr_scale: float = 400.0


@dataclass(frozen=True)
class RequestContext:
    request_id: str
    arrival_time_s: float
    prediction: Prediction
    kind: str = "chat"
    category: str = ""


def _age_ms(item: RequestContext, now_s: float) -> float:
    return max(0.0, (now_s - item.arrival_time_s) * 1000.0)


def _category_cost(item: RequestContext) -> float:
    if item.kind == "chat":
        return 250.0
    if "multi_turn" in item.category:
        return 420.0
    if "live_multiple" in item.category:
        return 180.0
    return 90.0


def _tail_risk_multiplier(item: RequestContext) -> float:
    if item.kind == "chat":
        return 0.65
    if "multi_turn" in item.category:
        return 1.25
    if "live_multiple" in item.category:
        return 0.90
    return 0.45


def _tail_safe_score(item: RequestContext, now_s: float, config: PolicyConfig) -> float:
    age_discount = 1.0 + _age_ms(item, now_s) / config.deadline_ms
    risk_adjusted_cost = _category_cost(item) * (
        1.0 + config.tail_weight * _tail_risk_multiplier(item)
    )
    return risk_adjusted_cost / age_discount


def _ltr_score(item: RequestContext, now_s: float, config: PolicyConfig) -> float:
    age_discount = 1.0 + _age_ms(item, now_s) / config.deadline_ms
    return item.prediction.score * config.ltr_scale / age_discount


def _has_ltr_opportunity(
    item: RequestContext, queue_depth: int, config: PolicyConfig
) -> bool:
    if queue_depth < 2:
        return False
    predicted_cost = item.prediction.score * config.ltr_scale
    opportunity_ms = abs(_category_cost(item) - predicted_cost) * (queue_depth - 1)
    return opportunity_ms > item.prediction.latency_ms


def _gated_score(
    item: RequestContext,
    now_s: float,
    queue_depth: int,
    config: PolicyConfig,
) -> float:
    is_reliable = (
        not item.prediction.ood
        and item.prediction.confidence >= config.confidence_threshold
    )
    if is_reliable and _has_ltr_opportunity(item, queue_depth, config):
        return _ltr_score(item, now_s, config)
    return _tail_safe_score(item, now_s, config)


def order_waiting_requests(
    waiting: list[RequestContext],
    policy: str,
    *,
    now_s: float,
    config: PolicyConfig | None = None,
) -> list[RequestContext]:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    resolved_config = config or PolicyConfig()
    queue_depth = len(waiting)

    def key(item: RequestContext) -> tuple[float, float, str]:
        if policy == "fcfs":
            score = item.arrival_time_s
        elif policy == "pure_ltr":
            score = item.prediction.score
        elif policy == "tail_safe":
            score = _tail_safe_score(item, now_s, resolved_config)
        else:
            score = _gated_score(item, now_s, queue_depth, resolved_config)
        return (score, item.arrival_time_s, item.request_id)

    return sorted(waiting, key=key)
