import subprocess
import sys
import unittest
from pathlib import Path

from ltr_training.fcfs_parity import (
    FCFS_PARITY_TOLERANCES,
    StockFCFSShim,
    compare_benchmark_results,
)
from ltr_training.fcfs_replay import (
    REPEAT_COUNT,
    ResponseSample,
    benchmark_scenarios,
    summarize_samples,
)


ROOT = Path(__file__).resolve().parents[1]


def benchmark_result(*, throughput: float, p99: float) -> dict[str, object]:
    metrics = {
        "throughput_rps": {"mean": throughput},
        "mean_ttlt_ms": {"mean": 100.0},
        "p95_ttlt_ms": {"mean": 200.0},
        "p99_ttlt_ms": {"mean": p99},
        "mean_ttft_ms": {"mean": 50.0},
    }
    return {
        "scenarios": [
            {
                "scenario": {"name": name},
                "aggregate": {"metrics": metrics},
            }
            for name in ("saturation-40", "saturation-70", "saturation-90", "burst-90")
        ]
    }


class StockFcfsParityTest(unittest.TestCase):
    def test_live_harness_uses_four_loads_and_three_repeats(self) -> None:
        self.assertEqual(REPEAT_COUNT, 3)
        self.assertEqual(
            [scenario.name for scenario in benchmark_scenarios()],
            ["saturation-40", "saturation-70", "saturation-90", "burst-90"],
        )

    def test_live_harness_summarizes_parity_metrics(self) -> None:
        metrics = summarize_samples(
            [
                ResponseSample(ttft_ms=10.0, ttlt_ms=100.0, output_tokens=10),
                ResponseSample(ttft_ms=20.0, ttlt_ms=200.0, output_tokens=20),
            ],
            wall_time_s=1.0,
        )

        self.assertEqual(metrics["throughput_rps"], 2.0)
        self.assertEqual(metrics["mean_ttlt_ms"], 150.0)
        self.assertEqual(metrics["p95_ttlt_ms"], 195.0)
        self.assertEqual(metrics["p99_ttlt_ms"], 199.0)
        self.assertEqual(metrics["mean_ttft_ms"], 15.0)

    def test_custom_fcfs_shim_does_not_override_stock_behavior(self) -> None:
        self.assertNotIn("schedule", StockFCFSShim.__dict__)
        self.assertNotIn("add_request", StockFCFSShim.__dict__)

    def test_tolerances_are_predefined_as_three_to_five_percent(self) -> None:
        self.assertEqual(
            FCFS_PARITY_TOLERANCES,
            {
                "throughput_rps": 0.03,
                "mean_ttlt_ms": 0.05,
                "p95_ttlt_ms": 0.05,
                "p99_ttlt_ms": 0.05,
                "mean_ttft_ms": 0.05,
            },
        )

    def test_report_compares_every_scenario_and_accepts_within_tolerance(self) -> None:
        report = compare_benchmark_results(
            benchmark_result(throughput=10.0, p99=300.0),
            benchmark_result(throughput=9.8, p99=314.0),
        )

        self.assertTrue(report["within_tolerance"])
        self.assertEqual(len(report["scenarios"]), 4)

    def test_report_flags_metric_outside_tolerance_without_hiding_delta(self) -> None:
        report = compare_benchmark_results(
            benchmark_result(throughput=10.0, p99=300.0),
            benchmark_result(throughput=10.0, p99=318.0),
        )

        self.assertFalse(report["within_tolerance"])
        first = report["scenarios"][0]["metrics"]["p99_ttlt_ms"]
        self.assertEqual(first["relative_delta"], 0.06)
        self.assertEqual(first["tolerance"], 0.05)


class FcfsParityCliTest(unittest.TestCase):
    def test_help_accepts_stock_shim_and_output_paths(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "report_fcfs_parity.py"), "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--stock", result.stdout)
        self.assertIn("--shim", result.stdout)
        self.assertIn("--output", result.stdout)

    def test_live_runner_help_accepts_both_endpoints_and_shared_workload(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_fcfs_parity.py"), "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--stock-endpoint", result.stdout)
        self.assertIn("--shim-endpoint", result.stdout)
        self.assertIn("--workload", result.stdout)
        self.assertIn("--output-dir", result.stdout)


if __name__ == "__main__":
    unittest.main()
