import asyncio

from scheduler_benchmark.gateway_overhead import (
    absolute_overhead,
    run_overhead_pair,
)
from scheduler_benchmark.runner import ResponseSample, WorkloadRequest


def workload() -> list[WorkloadRequest]:
    return [
        WorkloadRequest("a", "one", baseline_service_ms=10.0),
        WorkloadRequest("b", "two", baseline_service_ms=20.0),
    ]


def recording_sender(seen: list[str], *, added_ms: float):
    async def sender(request: WorkloadRequest) -> ResponseSample:
        seen.append(request.request_id)
        return ResponseSample(
            request_id=request.request_id,
            ttft_ms=request.baseline_service_ms + added_ms,
            ttlt_ms=request.baseline_service_ms * 2.0 + added_ms,
            output_tokens=3,
            baseline_service_ms=request.baseline_service_ms,
        )

    return sender


def test_gateway_overhead_uses_same_requests_and_arrivals_once() -> None:
    direct_seen: list[str] = []
    gateway_seen: list[str] = []

    result = asyncio.run(
        run_overhead_pair(
            workload=workload(),
            offsets=[0.0, 0.0],
            direct_sender=recording_sender(direct_seen, added_ms=0.0),
            gateway_sender=recording_sender(gateway_seen, added_ms=2.0),
        )
    )

    assert result["mode"] == "gateway_overhead_fcfs"
    assert result["repeats"] == 1
    assert result["valid"] is True
    assert direct_seen == ["a", "b"]
    assert gateway_seen == direct_seen
    assert result["direct"]["metrics"]["completed"] == 2
    assert result["gateway"]["metrics"]["completed"] == 2


def test_absolute_overhead_is_gateway_minus_direct() -> None:
    direct = {
        "mean_ttlt_ms": 10.0,
        "p95_ttlt_ms": 12.0,
        "p99_ttlt_ms": 13.0,
        "mean_ttft_ms": 2.0,
        "p95_ttft_ms": 3.0,
        "p99_ttft_ms": 4.0,
        "throughput_rps": 5.0,
        "output_tokens_per_s": 15.0,
    }
    gateway = {
        "mean_ttlt_ms": 12.0,
        "p95_ttlt_ms": 15.5,
        "p99_ttlt_ms": 17.0,
        "mean_ttft_ms": 2.5,
        "p95_ttft_ms": 4.0,
        "p99_ttft_ms": 5.5,
        "throughput_rps": 4.5,
        "output_tokens_per_s": 13.5,
    }

    report = absolute_overhead(direct, gateway)

    assert report["p95_ttlt_ms"] == 3.5
    assert report["mean_ttft_ms"] == 0.5
    assert report["throughput_rps"] == -0.5


def test_gateway_error_makes_overhead_pair_invalid() -> None:
    direct_seen: list[str] = []

    async def failed_gateway(request: WorkloadRequest) -> ResponseSample:
        return ResponseSample(
            request_id=request.request_id,
            ttft_ms=0.0,
            ttlt_ms=0.0,
            output_tokens=0,
            baseline_service_ms=request.baseline_service_ms,
            error="gateway failed",
        )

    result = asyncio.run(
        run_overhead_pair(
            workload=workload(),
            offsets=[0.0, 0.0],
            direct_sender=recording_sender(direct_seen, added_ms=0.0),
            gateway_sender=failed_gateway,
        )
    )

    assert result["valid"] is False
    assert result["gateway"]["metrics"]["errors"] == 2
