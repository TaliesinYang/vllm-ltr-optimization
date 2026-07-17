import json
import subprocess
import sys
from pathlib import Path

import pytest

from scheduler_benchmark.runner import (
    ResponseSample,
    WorkloadRequest,
    aggregate_repeats,
    assess_completeness,
    build_arrival_offsets,
    benchmark_scenarios,
    gateway_manifest,
    gateway_request_headers,
    load_workload,
    make_completion_payload,
    policy_for_scheduler_cls,
    summarize_samples,
    validate_vllm_version,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_matrix_has_three_saturations_and_burst() -> None:
    scenarios = benchmark_scenarios()

    assert [scenario.name for scenario in scenarios] == [
        "saturation-40",
        "saturation-70",
        "saturation-90",
        "burst-90",
    ]
    assert [scenario.saturation for scenario in scenarios] == [0.4, 0.7, 0.9, 0.9]
    assert scenarios[-1].burst_multiplier > 1.0


def test_burst_schedule_compresses_arrivals() -> None:
    steady, burst = benchmark_scenarios()[2:]

    steady_offsets = build_arrival_offsets(
        100, capacity_rps=100.0, scenario=steady, seed=17
    )
    burst_offsets = build_arrival_offsets(
        100, capacity_rps=100.0, scenario=burst, seed=17
    )

    assert burst_offsets[-1] < steady_offsets[-1]


def test_metrics_include_ttlt_slowdown_ttft_tail_and_throughput() -> None:
    samples = [
        ResponseSample(
            "a", ttft_ms=10.0, ttlt_ms=100.0, output_tokens=10, baseline_service_ms=50.0
        ),
        ResponseSample(
            "b",
            ttft_ms=20.0,
            ttlt_ms=200.0,
            output_tokens=20,
            baseline_service_ms=100.0,
        ),
    ]

    metrics = summarize_samples(samples, wall_time_s=1.0)

    assert metrics["mean_ttlt_ms"] == 150.0
    assert metrics["p95_ttlt_ms"] == 195.0
    assert metrics["p99_ttlt_ms"] == 199.0
    assert metrics["mean_normalized_slowdown"] == 2.0
    assert metrics["p95_normalized_slowdown"] == 2.0
    assert metrics["p99_normalized_slowdown"] == 2.0
    assert metrics["mean_ttft_ms"] == 15.0
    assert metrics["throughput_rps"] == 2.0
    assert metrics["output_tokens_per_s"] == 30.0


def test_three_repeat_aggregate_reports_student_t_ci() -> None:
    repeats = [
        {"mean_ttlt_ms": 90.0},
        {"mean_ttlt_ms": 100.0},
        {"mean_ttlt_ms": 110.0},
    ]

    aggregate = aggregate_repeats(repeats)

    assert aggregate["repeats"] == 3
    interval = aggregate["metrics"]["mean_ttlt_ms"]
    assert interval["mean"] == 100.0
    assert interval["ci95_low"] < 100.0 < interval["ci95_high"]


def test_completeness_gate_rejects_dropped_requests() -> None:
    repeats = [
        {"completed": 10, "errors": 0},
        {"completed": 9, "errors": 1},
        {"completed": 10, "errors": 0},
    ]

    result = assess_completeness(repeats, expected_requests=10)

    assert result["valid"] is False
    assert result["expected_requests_per_repeat"] == 10


def test_policy_is_derived_from_exact_scheduler_class() -> None:
    assert (
        policy_for_scheduler_cls("scheduler_benchmark.vllm_scheduler.PureLTRScheduler")
        == "pure_ltr"
    )
    with pytest.raises(ValueError, match="scheduler class"):
        policy_for_scheduler_cls("unknown.Scheduler")


def test_live_runner_requires_vllm_v024() -> None:
    assert validate_vllm_version("0.24.0") == "0.24.0"
    with pytest.raises(ValueError, match="vLLM 0.24"):
        validate_vllm_version("0.23.1")


def test_workload_loader_requires_isolated_baseline_service(tmp_path) -> None:
    workload_path = tmp_path / "workload.jsonl"
    workload_path.write_text(json.dumps({"request_id": "a", "prompt": "hello"}) + "\n")

    with pytest.raises(ValueError, match="baseline_service_ms"):
        load_workload(workload_path)


def test_completion_payload_carries_scheduler_visible_metadata() -> None:
    request = WorkloadRequest(
        request_id="req-1",
        prompt="hello",
        baseline_service_ms=100.0,
        max_tokens=32,
        kind="tool",
        category="multi_turn",
    )

    payload = make_completion_payload(request, model="model-path")

    assert payload["stream"] is True
    assert payload["vllm_xargs"] == {
        "ltr_kind": "tool",
        "ltr_category": "multi_turn",
    }


def test_runner_allocates_gateway_workflow_headers() -> None:
    request = WorkloadRequest(
        request_id="req-1",
        prompt="hello",
        baseline_service_ms=100.0,
    )

    headers = gateway_request_headers(request, api_key="secret")

    assert headers == {
        "X-Request-Id": "req-1",
        "X-Workflow-Id": "req-1",
        "X-Step-Id": "0",
        "X-Conversation-Id": "req-1",
        "X-Previous-Tool-Gap-Ms": "0",
        "Authorization": "Bearer secret",
    }


def test_runner_manifest_declares_gateway_main_path() -> None:
    assert gateway_manifest("http://gateway/v1/completions") == {
        "request_path": "client->gateway->decision->vllm",
        "gateway_endpoint": "http://gateway/v1/completions",
    }


def test_runner_script_is_directly_executable() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_scheduler_benchmark.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "VeloxMesh" in result.stdout
