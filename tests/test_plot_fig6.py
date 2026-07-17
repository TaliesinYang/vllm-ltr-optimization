import json

from scripts.plot_fig6 import load_policy_result, plot_figure


def test_load_policy_result_extracts_load_sweep_means(tmp_path) -> None:
    result_path = tmp_path / "fcfs.json"
    result_path.write_text(
        json.dumps(
            {
                "valid": True,
                "policy": "fcfs",
                "scheduler_cls": "scheduler_benchmark.vllm_scheduler.StockFCFSShim",
                "model": "model",
                "workload_sha256": "abc",
                "capacity_rps": 10.0,
                "seed": 17,
                "vllm_version": "0.24.0",
                "repeats": 3,
                "scenarios": [
                    {
                        "scenario": {"name": "saturation-40", "saturation": 0.4},
                        "aggregate": {
                            "metrics": {
                                "p95_ttlt_ms": {"mean": 100.0},
                                "p99_ttlt_ms": {"mean": 120.0},
                            }
                        },
                        "completeness": {"valid": True},
                    },
                    {
                        "scenario": {"name": "burst-90", "saturation": 0.9},
                        "aggregate": {
                            "metrics": {
                                "p95_ttlt_ms": {"mean": 999.0},
                                "p99_ttlt_ms": {"mean": 999.0},
                            }
                        },
                        "completeness": {"valid": True},
                    },
                ],
            }
        )
    )

    rows = load_policy_result(result_path)

    assert rows == [
        {
            "policy": "fcfs",
            "saturation_pct": 40.0,
            "p95_ttlt_ms": 100.0,
            "p99_ttlt_ms": 120.0,
            "model": "model",
            "workload_sha256": "abc",
            "capacity_rps": 10.0,
            "seed": 17,
            "vllm_version": "0.24.0",
            "repeats": 3,
            "scheduler_cls": "scheduler_benchmark.vllm_scheduler.StockFCFSShim",
        }
    ]


def test_load_policy_result_rejects_incomplete_run(tmp_path) -> None:
    result_path = tmp_path / "incomplete.json"
    result_path.write_text(json.dumps({"valid": False}))

    try:
        load_policy_result(result_path)
    except ValueError as exc:
        assert "incomplete" in str(exc)
    else:
        raise AssertionError("incomplete run must be rejected")


def test_placeholder_figure_renders_without_fake_measurements(tmp_path) -> None:
    output = tmp_path / "fig6.pdf"

    plot_figure([], output, is_placeholder=True)

    assert output.exists()
    assert output.stat().st_size > 0
