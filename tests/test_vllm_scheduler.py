from dataclasses import dataclass

from scheduler_benchmark.predictor import Prediction
from scheduler_benchmark.vllm_scheduler import (
    FCFS_PARITY_TOLERANCES,
    SCHEDULER_CLASSES,
    StockFCFSShim,
    TailSafeScheduler,
    evaluate_fcfs_parity,
    evaluate_parity_report,
    predict_or_fallback,
    reorder_request_queue,
)
from scheduler_benchmark.predictor import PredictorInput


@dataclass
class FakeRequest:
    request_id: str
    arrival_time: float
    prompt_token_ids: list[int]
    trace_headers: dict[str, str]
    sampling_params: object | None = None


class FakeQueue(list[FakeRequest]):
    def remove_requests(self, requests):
        for request in list(requests):
            self.remove(request)

    def add_request(self, request):
        self.append(request)


def test_stock_fcfs_shim_does_not_override_scheduler_behavior() -> None:
    assert "schedule" not in StockFCFSShim.__dict__
    assert "add_request" not in StockFCFSShim.__dict__


def test_scheduler_class_roster_matches_four_benchmark_policies() -> None:
    assert set(SCHEDULER_CLASSES) == {
        "fcfs",
        "pure_ltr",
        "tail_safe",
        "gated_hybrid",
    }


def test_tail_safe_scheduler_does_not_invoke_predictor() -> None:
    assert TailSafeScheduler.uses_predictor is False


def test_predictor_error_returns_ood_fallback_instead_of_aborting() -> None:
    class BrokenPredictor:
        def predict(self, predictor_input):
            raise RuntimeError("checkpoint unavailable")

    result, error = predict_or_fallback(
        BrokenPredictor(),
        PredictorInput("req", (1, 2), {}),
    )

    assert result.score == 1.0
    assert result.confidence == 0.0
    assert result.ood is True
    assert result.latency_ms >= 0.0
    assert error == "checkpoint unavailable"


def test_mock_engine_queue_uses_policy_order() -> None:
    queue = FakeQueue(
        [
            FakeRequest("long", 1.0, [1], {"x-ltr-kind": "chat"}),
            FakeRequest("short", 2.0, [2], {"x-ltr-kind": "chat"}),
        ]
    )
    predictions = {
        "long": Prediction(0.9, 0.9, False, 0.1),
        "short": Prediction(0.1, 0.9, False, 0.1),
    }

    reorder_request_queue(queue, "pure_ltr", predictions, now_s=3.0)

    assert [request.request_id for request in queue] == ["short", "long"]


def test_parity_tolerances_are_predefined_before_live_run() -> None:
    assert FCFS_PARITY_TOLERANCES == {
        "throughput_rps": 0.03,
        "mean_ttlt_ms": 0.05,
        "p95_ttlt_ms": 0.05,
        "p99_ttlt_ms": 0.05,
        "mean_ttft_ms": 0.05,
    }


def test_fcfs_parity_accepts_metrics_inside_tolerance() -> None:
    stock = {
        "throughput_rps": 10.0,
        "mean_ttlt_ms": 100.0,
        "p95_ttlt_ms": 200.0,
        "p99_ttlt_ms": 300.0,
        "mean_ttft_ms": 50.0,
    }
    shim = {
        "throughput_rps": 9.8,
        "mean_ttlt_ms": 104.0,
        "p95_ttlt_ms": 209.0,
        "p99_ttlt_ms": 314.0,
        "mean_ttft_ms": 52.0,
    }

    result = evaluate_fcfs_parity(stock, shim)

    assert result.passed is True
    assert all(check.passed for check in result.checks.values())


def test_fcfs_parity_rejects_metric_outside_tolerance() -> None:
    stock = {name: 100.0 for name in FCFS_PARITY_TOLERANCES}
    shim = dict(stock, p99_ttlt_ms=106.0)

    result = evaluate_fcfs_parity(stock, shim)

    assert result.passed is False
    assert result.checks["p99_ttlt_ms"].relative_delta == 0.06


def test_parity_report_compares_matching_runner_scenarios() -> None:
    def runner_result(p99_ttlt_ms: float, *, is_stock: bool):
        metrics = {name: {"mean": 100.0} for name in FCFS_PARITY_TOLERANCES}
        metrics["p99_ttlt_ms"] = {"mean": p99_ttlt_ms}
        return {
            "valid": True,
            "policy": "stock_fcfs" if is_stock else "fcfs",
            "scheduler_cls": (
                "vllm.v1.core.sched.scheduler.Scheduler"
                if is_stock
                else "scheduler_benchmark.vllm_scheduler.StockFCFSShim"
            ),
            "model": "model",
            "workload_sha256": "abc",
            "capacity_rps": 10.0,
            "seed": 17,
            "vllm_version": "0.24.0",
            "repeats": 3,
            "scenarios": [
                {
                    "scenario": {"name": "saturation-90"},
                    "aggregate": {"metrics": metrics},
                    "completeness": {"valid": True},
                }
            ],
        }

    report = evaluate_parity_report(
        runner_result(100.0, is_stock=True),
        runner_result(104.0, is_stock=False),
    )

    assert report["passed"] is True
    assert report["scenarios"]["saturation-90"]["passed"] is True


def test_parity_report_rejects_mismatched_run_identity() -> None:
    base = {
        "valid": True,
        "policy": "stock_fcfs",
        "scheduler_cls": "vllm.v1.core.sched.scheduler.Scheduler",
        "model": "model",
        "workload_sha256": "abc",
        "capacity_rps": 10.0,
        "seed": 17,
        "vllm_version": "0.24.0",
        "repeats": 3,
        "scenarios": [],
    }
    shim = dict(
        base,
        policy="fcfs",
        scheduler_cls="scheduler_benchmark.vllm_scheduler.StockFCFSShim",
        workload_sha256="different",
    )

    try:
        evaluate_parity_report(base, shim)
    except ValueError as exc:
        assert "workload_sha256" in str(exc)
    else:
        raise AssertionError("mismatched workload must be rejected")
