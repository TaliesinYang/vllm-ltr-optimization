from __future__ import annotations

from typing import Mapping


try:
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler as _SchedulerBase
except ImportError:
    _SchedulerBase = object


class StockFCFSShim(_SchedulerBase):
    """Custom scheduler entry point that intentionally preserves stock FCFS behavior."""


FCFS_PARITY_TOLERANCES = {
    "throughput_rps": 0.03,
    "mean_ttlt_ms": 0.05,
    "p95_ttlt_ms": 0.05,
    "p99_ttlt_ms": 0.05,
    "mean_ttft_ms": 0.05,
}


def _scenario_metrics(result: Mapping[str, object]) -> dict[str, dict[str, float]]:
    scenarios = result.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError("benchmark result has no scenarios list")
    extracted: dict[str, dict[str, float]] = {}
    for item in scenarios:
        if not isinstance(item, Mapping):
            raise ValueError("scenario entry is not an object")
        scenario = item.get("scenario")
        aggregate = item.get("aggregate")
        if not isinstance(scenario, Mapping) or not isinstance(aggregate, Mapping):
            raise ValueError("scenario entry lacks scenario/aggregate")
        name = scenario.get("name")
        profile = str(item.get("profile", "mixed"))
        metrics = aggregate.get("metrics")
        if not isinstance(name, str) or not isinstance(metrics, Mapping):
            raise ValueError("scenario entry lacks name/metrics")
        values: dict[str, float] = {}
        for metric in FCFS_PARITY_TOLERANCES:
            summary = metrics.get(metric)
            if not isinstance(summary, Mapping) or "mean" not in summary:
                raise ValueError(f"scenario {name} lacks aggregate metric {metric}")
            values[metric] = float(summary["mean"])
        report_name = name if profile == "mixed" else f"{name}/{profile}"
        if report_name in extracted:
            raise ValueError(f"duplicate scenario/profile: {report_name}")
        extracted[report_name] = values
    return extracted


def compare_benchmark_results(
    stock_result: Mapping[str, object], shim_result: Mapping[str, object]
) -> dict[str, object]:
    stock_scenarios = _scenario_metrics(stock_result)
    shim_scenarios = _scenario_metrics(shim_result)
    if stock_scenarios.keys() != shim_scenarios.keys():
        raise ValueError("stock and shim scenario sets differ")

    scenario_reports: list[dict[str, object]] = []
    all_within_tolerance = True
    for scenario_name in stock_scenarios:
        metric_reports: dict[str, dict[str, float | bool]] = {}
        for metric, tolerance in FCFS_PARITY_TOLERANCES.items():
            stock_value = stock_scenarios[scenario_name][metric]
            shim_value = shim_scenarios[scenario_name][metric]
            if stock_value == 0.0:
                raise ValueError(
                    f"stock metric must be non-zero: {scenario_name}/{metric}"
                )
            relative_delta = round(abs(shim_value - stock_value) / abs(stock_value), 6)
            is_within_tolerance = relative_delta <= tolerance
            all_within_tolerance = all_within_tolerance and is_within_tolerance
            metric_reports[metric] = {
                "stock": stock_value,
                "shim": shim_value,
                "relative_delta": relative_delta,
                "tolerance": tolerance,
                "within_tolerance": is_within_tolerance,
            }
        scenario_reports.append({"name": scenario_name, "metrics": metric_reports})
    return {
        "mode": "report_only",
        "within_tolerance": all_within_tolerance,
        "scenarios": scenario_reports,
    }
