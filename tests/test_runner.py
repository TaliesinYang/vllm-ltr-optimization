import json
import asyncio
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
    derive_run_seed,
    gateway_manifest,
    gateway_request_headers,
    load_completed_subruns,
    load_workload,
    make_completion_payload,
    measurement_window,
    parse_args,
    policy_for_scheduler_cls,
    resolve_warmup_requests,
    resolve_scenario_matrix,
    run_replay,
    run_benchmark,
    select_workload_profile,
    subrun_fingerprint,
    summarize_samples,
    validate_vllm_version,
    write_subrun_artifacts,
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


def test_arbitrary_repeat_aggregate_reports_scatter_range() -> None:
    repeats = [
        {"mean_ttlt_ms": 90.0},
        {"mean_ttlt_ms": 100.0},
        {"mean_ttlt_ms": 110.0},
    ]

    aggregate = aggregate_repeats(repeats)

    assert aggregate["repeats"] == 3
    interval = aggregate["metrics"]["mean_ttlt_ms"]
    assert interval == {
        "values": [90.0, 100.0, 110.0],
        "mean": 100.0,
        "min": 90.0,
        "max": 110.0,
    }


def test_replay_promotes_scheduled_latency_and_preserves_send_latency() -> None:
    request = WorkloadRequest("a", "hello", baseline_service_ms=10.0)

    async def sender(item: WorkloadRequest) -> ResponseSample:
        return ResponseSample(
            request_id=item.request_id,
            ttft_ms=5.0,
            ttlt_ms=10.0,
            send_ttft_ms=5.0,
            send_ttlt_ms=10.0,
            output_tokens=1,
            baseline_service_ms=item.baseline_service_ms,
        )

    samples, _ = asyncio.run(
        run_replay([request], [-0.02], sender, policy="fcfs", profile="mixed")
    )

    sample = samples[0]
    assert sample.dispatch_lag_ms >= 15.0
    assert sample.send_ttft_ms == 5.0
    assert sample.send_ttlt_ms == 10.0
    assert sample.ttft_ms == pytest.approx(sample.dispatch_lag_ms + 5.0)
    assert sample.ttlt_ms == pytest.approx(sample.dispatch_lag_ms + 10.0)
    assert sample.scheduled_at_unix_s < sample.dispatched_at_unix_s
    assert sample.category == ""
    assert sample.policy == "fcfs"
    assert sample.profile == "mixed"


def test_warmup_count_and_ratio_resolve_to_requests() -> None:
    assert resolve_warmup_requests(10, requested_count=3) == 3
    assert resolve_warmup_requests(10, requested_ratio=0.25) == 2
    assert resolve_warmup_requests(10) == 0
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_warmup_requests(10, requested_count=1, requested_ratio=0.1)
    with pytest.raises(ValueError, match="measurement request"):
        resolve_warmup_requests(10, requested_count=10)


def test_measurement_window_discards_warmup_and_truncates_duration() -> None:
    samples = [
        ResponseSample(
            request_id="warmup",
            ttft_ms=10.0,
            ttlt_ms=20.0,
            output_tokens=1,
            baseline_service_ms=10.0,
            scheduled_at_unix_s=100.0,
            completed_at_unix_s=100.02,
        ),
        ResponseSample(
            request_id="measure-a",
            ttft_ms=10.0,
            ttlt_ms=20.0,
            output_tokens=2,
            baseline_service_ms=10.0,
            scheduled_at_unix_s=101.0,
            completed_at_unix_s=101.02,
        ),
        ResponseSample(
            request_id="measure-b",
            ttft_ms=10.0,
            ttlt_ms=40.0,
            output_tokens=3,
            baseline_service_ms=20.0,
            scheduled_at_unix_s=101.01,
            completed_at_unix_s=101.05,
        ),
    ]

    measured, duration_s = measurement_window(samples, warmup_requests=1)
    metrics = summarize_samples(measured, wall_time_s=duration_s)

    assert [sample.request_id for sample in measured] == ["measure-a", "measure-b"]
    assert duration_s == pytest.approx(0.05)
    assert metrics["completed"] == 2
    assert metrics["throughput_rps"] == 40.0
    assert metrics["output_tokens_per_s"] == 100.0


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


def runner_argv(output: Path) -> list[str]:
    return [
        "--endpoint",
        "http://gateway/v1/completions",
        "--model",
        "model-path",
        "--workload",
        "workload.jsonl",
        "--capacity-rps",
        "10",
        "--scheduler-cls",
        "scheduler_benchmark.vllm_scheduler.GatedHybridScheduler",
        "--output",
        str(output),
    ]


def test_cli_selects_scenario_load_profile_repeats_warmup_and_resume(tmp_path) -> None:
    args = parse_args(
        runner_argv(tmp_path / "result.json")
        + [
            "--scenario",
            "steady",
            "--load",
            "90",
            "--profile",
            "mixed",
            "--repeats",
            "5",
            "--warmup-requests",
            "100",
            "--resume",
        ]
    )

    assert args.scenario == ["steady"]
    assert args.load == [90]
    assert args.profile == ["mixed"]
    assert args.repeats == 5
    assert args.warmup_requests == 100
    assert args.resume is True


def test_scenario_matrix_preserves_legacy_defaults_and_supports_cross_product() -> None:
    defaults = resolve_scenario_matrix(None, None)
    assert [scenario.name for scenario in defaults] == [
        "saturation-40",
        "saturation-70",
        "saturation-90",
        "burst-90",
    ]

    selected = resolve_scenario_matrix(["steady", "burst"], [70, 90])
    assert [scenario.name for scenario in selected] == [
        "saturation-70",
        "saturation-90",
        "burst-70",
        "burst-90",
    ]


def test_seed_depends_on_profile_load_repeat_not_selection_order() -> None:
    expected = derive_run_seed(profile="mixed", load_pct=90, repeat=3)
    reordered = [
        derive_run_seed(profile=profile, load_pct=load, repeat=repeat)
        for profile, load, repeat in [
            ("ood", 40, 1),
            ("mixed", 90, 3),
            ("id", 70, 2),
        ]
    ]

    assert reordered[1] == expected
    assert expected == derive_run_seed(profile="mixed", load_pct=90, repeat=3)


def test_profile_filter_uses_category_prefixes() -> None:
    workload = [
        WorkloadRequest("id", "a", 1.0, category="id:toolace"),
        WorkloadRequest("ood", "b", 1.0, category="ood:bfcl"),
    ]

    assert [row.request_id for row in select_workload_profile(workload, "id")] == ["id"]
    assert [row.request_id for row in select_workload_profile(workload, "ood")] == [
        "ood"
    ]
    assert [row.request_id for row in select_workload_profile(workload, "mixed")] == [
        "id",
        "ood",
    ]


def test_subrun_artifacts_are_fingerprinted_and_resumable(tmp_path) -> None:
    sample = ResponseSample(
        request_id="req",
        ttft_ms=20.0,
        ttlt_ms=30.0,
        send_ttft_ms=10.0,
        send_ttlt_ms=20.0,
        dispatch_lag_ms=10.0,
        output_tokens=2,
        baseline_service_ms=15.0,
        category="ood:bfcl",
        policy="gated_hybrid",
        profile="ood",
        scheduled_at_unix_s=100.0,
        dispatched_at_unix_s=100.01,
        first_token_at_unix_s=100.02,
        completed_at_unix_s=100.03,
    )
    record = {
        "schema_version": 2,
        "status": "complete",
        "workload_sha256": "a" * 64,
        "policy": "gated_hybrid",
        "scenario": {"name": "saturation-90", "saturation": 0.9},
        "load_pct": 90,
        "profile": "ood",
        "repeat": 1,
        "seed": 7,
        "warmup": {"requested": {"count": 0, "ratio": None}, "resolved": 0},
        "completed": 1,
        "errors": 0,
        "metrics": {"completed": 1, "errors": 0},
    }
    record["fingerprint"] = subrun_fingerprint(record)

    json_path, csv_path = write_subrun_artifacts(tmp_path, record, [sample])
    completed = load_completed_subruns(tmp_path)

    assert json_path.name == f"{record['fingerprint']}.json"
    assert csv_path.name == f"{record['fingerprint']}.samples.csv"
    assert completed[0]["fingerprint"] == record["fingerprint"]
    csv_text = csv_path.read_text()
    for field in (
        "category",
        "policy",
        "profile",
        "scheduled_at_unix_s",
        "dispatched_at_unix_s",
        "first_token_at_unix_s",
        "completed_at_unix_s",
    ):
        assert field in csv_text.splitlines()[0]


def test_live_orchestrator_writes_each_subrun_and_resume_skips_completed(
    tmp_path, monkeypatch
) -> None:
    workload_path = tmp_path / "workload.jsonl"
    workload_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "request_id": f"req-{index}",
                    "prompt": "hello",
                    "baseline_service_ms": 10.0,
                    "category": "id:toolace",
                }
            )
            for index in range(3)
        )
        + "\n"
    )
    output = tmp_path / "result.json"
    args = parse_args(
        [
            *runner_argv(output),
            "--workload",
            str(workload_path),
            "--capacity-rps",
            "1000000000",
            "--scenario",
            "steady",
            "--load",
            "40",
            "--profile",
            "id",
            "--repeats",
            "2",
            "--warmup-requests",
            "1",
        ]
    )
    monkeypatch.setattr(
        "scheduler_benchmark.runner.distribution_version", lambda _name: "0.24.0"
    )

    async def fake_stream(_session, _endpoint, _model, request, _api_key):
        return ResponseSample(
            request_id=request.request_id,
            ttft_ms=1.0,
            ttlt_ms=2.0,
            output_tokens=1,
            baseline_service_ms=request.baseline_service_ms,
        )

    monkeypatch.setattr("scheduler_benchmark.runner.stream_completion", fake_stream)
    first = asyncio.run(run_benchmark(args))

    assert first["valid"] is True
    assert first["schema_version"] == 2
    assert first["scenarios"][0]["runs"][0]["warmup"] == {
        "requested": {"count": 1, "ratio": None},
        "resolved": 1,
        "measured": 2,
        "discarded": 1,
    }
    assert first["scenarios"][0]["runs"][0]["metrics"]["completed"] == 2
    runs_dir = tmp_path / "result.runs"
    assert len(list(runs_dir.glob("*.json"))) == 2
    assert len(list(runs_dir.glob("*.samples.csv"))) == 2

    async def forbidden_stream(*_args, **_kwargs):
        raise AssertionError("resume should not replay completed subruns")

    monkeypatch.setattr(
        "scheduler_benchmark.runner.stream_completion", forbidden_stream
    )
    args.resume = True
    resumed = asyncio.run(run_benchmark(args))

    assert resumed["valid"] is True
    assert [run["fingerprint"] for run in resumed["scenarios"][0]["runs"]] == [
        run["fingerprint"] for run in first["scenarios"][0]["runs"]
    ]


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
