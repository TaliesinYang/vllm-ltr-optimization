"""vLLM v0.24 custom scheduler adapter loaded by ``--scheduler-cls``."""

from __future__ import annotations

import importlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from .policies import RequestContext, order_waiting_requests
from .predictor import (
    ConstantPredictor,
    GatewayMetadataPredictor,
    OracleFromFilePredictor,
    Prediction,
    Predictor,
    PredictorInput,
    RandomPredictor,
)

try:
    from vllm.v1.core.sched.scheduler import Scheduler as _StockSchedulerBase
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler as _SchedulerBase

    _VLLM_AVAILABLE = True
except ImportError:
    _StockSchedulerBase = object
    _SchedulerBase = object
    _VLLM_AVAILABLE = False

LOGGER = logging.getLogger(__name__)


class StockFCFSShim(_StockSchedulerBase):
    pass


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"invalid boolean: {value}")


def build_predictor_from_env() -> Predictor:
    kind = os.environ.get("LTR_PREDICTOR", "constant")
    if kind == "constant":
        return ConstantPredictor(
            score=float(os.environ.get("LTR_CONSTANT_SCORE", "1.0")),
            confidence=float(os.environ.get("LTR_CONSTANT_CONFIDENCE", "0.9")),
            ood=_parse_bool(os.environ.get("LTR_CONSTANT_OOD", "false")),
        )
    if kind == "random":
        return RandomPredictor(seed=int(os.environ.get("LTR_RANDOM_SEED", "17")))
    if kind == "gateway":
        return GatewayMetadataPredictor()
    if kind == "oracle":
        try:
            path = Path(os.environ["LTR_ORACLE_FILE"])
        except KeyError as exc:
            raise ValueError(
                "LTR_ORACLE_FILE is required for oracle predictor"
            ) from exc
        return OracleFromFilePredictor(path)
    if ":" in kind:
        module_name, factory_name = kind.split(":", 1)
        factory = getattr(importlib.import_module(module_name), factory_name)
        predictor = factory()
        if not callable(getattr(predictor, "predict", None)):
            raise TypeError("predictor factory must return an object with predict()")
        return predictor
    raise ValueError(f"unknown LTR_PREDICTOR: {kind}")


def _request_metadata(request) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for key, value in (getattr(request, "trace_headers", None) or {}).items():
        normalized = key.lower().removeprefix("x-ltr-").replace("-", "_")
        metadata[normalized] = value
    sampling_params = getattr(request, "sampling_params", None)
    for key, value in (getattr(sampling_params, "extra_args", None) or {}).items():
        normalized = key.removeprefix("ltr_")
        metadata[normalized] = value
    return metadata


def _request_context(request, prediction: Prediction) -> RequestContext:
    metadata = _request_metadata(request)
    return RequestContext(
        request_id=request.request_id,
        arrival_time_s=float(request.arrival_time),
        prediction=prediction,
        kind=str(metadata.get("kind", "chat")),
        category=str(metadata.get("category", "")),
        prompt_token_count=len(request.prompt_token_ids or ()),
    )


def reorder_request_queue(queue, policy, predictions, *, now_s):
    requests = list(queue)
    if len(requests) < 2:
        _write_order_log(requests, policy, predictions)
        return
    contexts = [
        _request_context(request, predictions[request.request_id])
        for request in requests
    ]
    ordered_contexts = order_waiting_requests(contexts, policy, now_s=now_s)
    by_request_id = {request.request_id: request for request in requests}
    queue.remove_requests(requests)
    for context in ordered_contexts:
        queue.add_request(by_request_id[context.request_id])
    _write_order_log(list(queue), policy, predictions)


def _write_order_log(requests, policy, predictions) -> None:
    path = os.environ.get("LTR_ORDER_LOG")
    if not path:
        return
    entry = {
        "policy": str(policy),
        "order": [request.request_id for request in requests],
        "predictions": {
            request.request_id: {
                "score": predictions[request.request_id].score,
                "ood": predictions[request.request_id].ood,
            }
            for request in requests
            if request.request_id in predictions
        },
    }
    with Path(path).open("a", encoding="utf-8") as order_log:
        order_log.write(json.dumps(entry) + "\n")


def predict_or_fallback(
    predictor: Predictor, predictor_input: PredictorInput
) -> tuple[Prediction, str | None]:
    started = time.perf_counter()
    try:
        return predictor.predict(predictor_input), None
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return (
            Prediction(
                score=1.0,
                confidence=0.0,
                ood=True,
                latency_ms=latency_ms,
            ),
            str(exc),
        )


class _PolicyScheduler(_SchedulerBase):
    policy_name = ""
    uses_predictor = True

    def __init__(self, *args, predictor: Predictor | None = None, **kwargs) -> None:
        if not _VLLM_AVAILABLE:
            raise RuntimeError("vLLM v0.24 is required to instantiate scheduler")
        super().__init__(*args, **kwargs)
        self._ltr_predictor = (
            predictor or build_predictor_from_env() if self.uses_predictor else None
        )
        self._ltr_predictions: dict[str, Prediction] = {}
        self._ltr_prediction_errors: dict[str, str] = {}

    def add_request(self, request) -> None:
        if request.request_id not in self._ltr_predictions:
            if self._ltr_predictor is None:
                prediction = Prediction(1.0, 0.0, True, 0.0)
                error = None
            else:
                predictor_input = PredictorInput(
                    request_id=request.request_id,
                    prompt_token_ids=tuple(request.prompt_token_ids or ()),
                    metadata=_request_metadata(request),
                )
                prediction, error = predict_or_fallback(
                    self._ltr_predictor, predictor_input
                )
            self._ltr_predictions[request.request_id] = prediction
            if error is not None:
                self._ltr_prediction_errors[request.request_id] = error
                LOGGER.warning(
                    "predictor failed for request %s; using safe fallback: %s",
                    request.request_id,
                    error,
                )
        super().add_request(request)

    def schedule(self, throttle_prefills: bool = False):
        reorder_request_queue(
            self.waiting,
            self.policy_name,
            self._ltr_predictions,
            now_s=time.time(),
        )
        return super().schedule(throttle_prefills=throttle_prefills)


class PureLTRScheduler(_PolicyScheduler):
    policy_name = "pure_ltr"


class TailSafeScheduler(_PolicyScheduler):
    policy_name = "tail_safe"
    uses_predictor = False


class GatedHybridScheduler(_PolicyScheduler):
    policy_name = "gated_hybrid"


class GatedRuleCScheduler(_PolicyScheduler):
    """Slot-preserving gating: only Ranker-eligible requests are reordered.

    Abstained requests keep the slot arrival order gave them, so uncertainty
    about a request can never cost it its place in line.
    """

    policy_name = "gated_rule_c"


class PolicyFCFS(_PolicyScheduler):
    """FCFS on the custom scheduler base, as an algorithmic control.

    StockFCFSShim inherits vLLM's stock Scheduler, so comparing a custom policy
    against it confounds "this ordering is better" with "these are different
    scheduler code paths". This class runs arrival order through the same
    _PolicyScheduler machinery as every other custom policy, which isolates the
    ordering as the only difference.
    """

    policy_name = "policy_fcfs"
    uses_predictor = False


class PromptLengthSJFScheduler(_PolicyScheduler):
    """Prompt-length SJF with zero predictor inference overhead."""

    policy_name = "prompt_sjf"
    uses_predictor = False


class LTRAgingScheduler(_PolicyScheduler):
    policy_name = "ltr_aging"


SCHEDULER_CLASSES = {
    "fcfs": StockFCFSShim,
    "policy_fcfs": PolicyFCFS,
    "pure_ltr": PureLTRScheduler,
    "tail_safe": TailSafeScheduler,
    "gated_hybrid": GatedHybridScheduler,
    "gated_rule_c": GatedRuleCScheduler,
    "prompt_sjf": PromptLengthSJFScheduler,
    "ltr_aging": LTRAgingScheduler,
}

FCFS_PARITY_TOLERANCES = {
    "throughput_rps": 0.03,
    "mean_ttlt_ms": 0.05,
    "p95_ttlt_ms": 0.05,
    "p99_ttlt_ms": 0.05,
    "mean_ttft_ms": 0.05,
}


@dataclass(frozen=True)
class ParityCheck:
    relative_delta: float
    tolerance: float
    passed: bool


@dataclass(frozen=True)
class ParityResult:
    passed: bool
    checks: Mapping[str, ParityCheck]


def evaluate_fcfs_parity(stock_metrics, shim_metrics):
    checks: dict[str, ParityCheck] = {}
    for metric, tolerance in FCFS_PARITY_TOLERANCES.items():
        stock_value = float(stock_metrics[metric])
        if stock_value == 0.0:
            raise ValueError(f"stock metric must be non-zero: {metric}")
        relative_delta = round(
            abs(float(shim_metrics[metric]) - stock_value) / abs(stock_value), 6
        )
        checks[metric] = ParityCheck(
            relative_delta=relative_delta,
            tolerance=tolerance,
            passed=relative_delta <= tolerance,
        )
    return ParityResult(
        passed=all(check.passed for check in checks.values()),
        checks=checks,
    )


def evaluate_parity_report(stock_result, shim_result):
    if stock_result.get("valid") is not True or shim_result.get("valid") is not True:
        raise ValueError("parity requires complete valid benchmark runs")
    expected_schedulers = (
        "vllm.v1.core.sched.scheduler.Scheduler",
        "scheduler_benchmark.vllm_scheduler.StockFCFSShim",
    )
    actual_schedulers = (
        stock_result.get("scheduler_cls"),
        shim_result.get("scheduler_cls"),
    )
    if actual_schedulers != expected_schedulers:
        raise ValueError("parity requires stock Scheduler and StockFCFSShim")
    if (stock_result.get("policy"), shim_result.get("policy")) != (
        "stock_fcfs",
        "fcfs",
    ):
        raise ValueError("parity policy labels do not match scheduler classes")
    identity_fields = (
        "model",
        "workload_sha256",
        "capacity_rps",
        "vllm_version",
        "repeats",
    )
    optional_identity_fields = ("seed", "seed_derivation", "warmup_requested")
    for field in identity_fields:
        if stock_result.get(field) != shim_result.get(field):
            raise ValueError(f"parity inputs disagree on {field}")
    for field in optional_identity_fields:
        if field in stock_result or field in shim_result:
            if stock_result.get(field) != shim_result.get(field):
                raise ValueError(f"parity inputs disagree on {field}")
    for label, result in (("stock", stock_result), ("shim", shim_result)):
        if any(
            row.get("completeness", {}).get("valid") is not True
            for row in result["scenarios"]
        ):
            raise ValueError(f"{label} contains an incomplete scenario")

    def scenarios_by_name(result):
        return {
            (row["scenario"]["name"], row.get("profile", "mixed")): row["aggregate"][
                "metrics"
            ]
            for row in result["scenarios"]
        }

    stock_scenarios = scenarios_by_name(stock_result)
    shim_scenarios = scenarios_by_name(shim_result)
    if set(stock_scenarios) != set(shim_scenarios):
        raise ValueError("stock and shim scenario sets must match")
    scenario_reports: dict[str, object] = {}
    all_passed = True
    for name, profile in sorted(stock_scenarios):
        key = (name, profile)
        stock_metrics = {
            metric: stock_scenarios[key][metric]["mean"]
            for metric in FCFS_PARITY_TOLERANCES
        }
        shim_metrics = {
            metric: shim_scenarios[key][metric]["mean"]
            for metric in FCFS_PARITY_TOLERANCES
        }
        result = evaluate_fcfs_parity(stock_metrics, shim_metrics)
        all_passed = all_passed and result.passed
        report_name = name if profile == "mixed" else f"{name}/{profile}"
        scenario_reports[report_name] = {
            "passed": result.passed,
            "checks": {
                metric: asdict(check) for metric, check in result.checks.items()
            },
        }
    return {
        "passed": all_passed,
        "tolerances": FCFS_PARITY_TOLERANCES,
        "scenarios": scenario_reports,
    }
