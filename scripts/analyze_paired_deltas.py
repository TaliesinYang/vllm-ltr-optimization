#!/usr/bin/env python3
"""Compute paired per-repeat benchmark deltas outside the live runner."""

from __future__ import annotations

import argparse
import json
import statistics
import uuid
from pathlib import Path
from typing import Mapping


IDENTITY_FIELDS = (
    "model",
    "workload_sha256",
    "capacity_rps",
    "seed_derivation",
    "warmup_requested",
)


def _scenario_groups(result: Mapping[str, object]) -> dict[tuple[str, int, str], dict]:
    groups: dict[tuple[str, int, str], dict] = {}
    for group in result.get("scenarios", []):
        scenario = group["scenario"]
        key = (
            str(scenario["name"]),
            int(group.get("load_pct", round(float(scenario["saturation"]) * 100))),
            str(group.get("profile", "mixed")),
        )
        if key in groups:
            raise ValueError(f"duplicate scenario/profile group: {key}")
        groups[key] = group
    return groups


def _runs_by_pair(group: Mapping[str, object]) -> dict[tuple[int, int], Mapping]:
    runs = {}
    for run in group["runs"]:
        key = (int(run["repeat"]), int(run["seed"]))
        if key in runs:
            raise ValueError(f"duplicate repeat/seed pair: {key}")
        runs[key] = run
    return runs


def _scatter(values: list[float]) -> dict[str, object]:
    return {
        "values": values,
        "mean": round(statistics.mean(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def compute_paired_differences(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    metrics: list[str] | None = None,
) -> dict[str, object]:
    if baseline.get("valid") is not True or candidate.get("valid") is not True:
        raise ValueError("paired analysis requires valid benchmark results")
    for field in IDENTITY_FIELDS:
        if baseline.get(field) != candidate.get(field):
            raise ValueError(f"paired inputs disagree on {field}")
    baseline_groups = _scenario_groups(baseline)
    candidate_groups = _scenario_groups(candidate)
    if baseline_groups.keys() != candidate_groups.keys():
        raise ValueError("paired inputs have different scenario/profile groups")

    output_groups: list[dict[str, object]] = []
    for scenario_name, load_pct, profile in sorted(baseline_groups):
        baseline_runs = _runs_by_pair(
            baseline_groups[(scenario_name, load_pct, profile)]
        )
        candidate_runs = _runs_by_pair(
            candidate_groups[(scenario_name, load_pct, profile)]
        )
        if baseline_runs.keys() != candidate_runs.keys():
            raise ValueError(f"paired runs disagree for {scenario_name}/{profile}")
        resolved_metrics = metrics
        if resolved_metrics is None:
            first_key = next(iter(baseline_runs))
            baseline_metrics = baseline_runs[first_key]["metrics"]
            candidate_metrics = candidate_runs[first_key]["metrics"]
            resolved_metrics = sorted(
                name
                for name in baseline_metrics.keys() & candidate_metrics.keys()
                if isinstance(baseline_metrics[name], (int, float))
                and isinstance(candidate_metrics[name], (int, float))
            )
        metric_results: dict[str, object] = {}
        for metric in resolved_metrics:
            deltas = [
                round(
                    float(candidate_runs[key]["metrics"][metric])
                    - float(baseline_runs[key]["metrics"][metric]),
                    6,
                )
                for key in sorted(baseline_runs)
            ]
            metric_results[metric] = _scatter(deltas)
        output_groups.append(
            {
                "scenario": scenario_name,
                "load_pct": load_pct,
                "profile": profile,
                "pair_count": len(baseline_runs),
                "pairs": [
                    {"repeat": repeat, "seed": seed}
                    for repeat, seed in sorted(baseline_runs)
                ],
                "metrics": metric_results,
            }
        )
    return {
        "schema_version": 1,
        "direction": "candidate_minus_baseline",
        "baseline_policy": baseline["policy"],
        "candidate_policy": candidate["policy"],
        "identity": {field: baseline[field] for field in IDENTITY_FIELDS},
        "groups": output_groups,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--metric", action="append")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = compute_paired_differences(
        json.loads(args.baseline.read_text()),
        json.loads(args.candidate.read_text()),
        metrics=args.metric,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
