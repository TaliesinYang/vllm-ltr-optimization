#!/usr/bin/env python3
"""Plot Figure 6: live load sweep P95/P99 scheduled-origin TTLT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

POLICY_ORDER = (
    "fcfs",
    "pure_ltr",
    "tail_safe",
    "gated_hybrid",
    "prompt_sjf",
    "ltr_aging",
)
POLICY_LABELS = {
    "fcfs": "FCFS",
    "pure_ltr": "Pure LTR",
    "tail_safe": "Tail-safe",
    "gated_hybrid": "Gated hybrid",
    "prompt_sjf": "Prompt SJF",
    "ltr_aging": "LTR + aging",
}
POLICY_STYLES = {
    "fcfs": {"color": "#555555", "marker": "o"},
    "pure_ltr": {"color": "#d95f02", "marker": "s"},
    "tail_safe": {"color": "#1b9e77", "marker": "^"},
    "gated_hybrid": {"color": "#3f51b5", "marker": "D"},
    "prompt_sjf": {"color": "#7570b3", "marker": "v"},
    "ltr_aging": {"color": "#e7298a", "marker": "P"},
}
EXPECTED_SCHEDULER_CLASSES = {
    "fcfs": "scheduler_benchmark.vllm_scheduler.StockFCFSShim",
    "pure_ltr": "scheduler_benchmark.vllm_scheduler.PureLTRScheduler",
    "tail_safe": "scheduler_benchmark.vllm_scheduler.TailSafeScheduler",
    "gated_hybrid": "scheduler_benchmark.vllm_scheduler.GatedHybridScheduler",
    "prompt_sjf": "scheduler_benchmark.vllm_scheduler.PromptLengthSJFScheduler",
    "ltr_aging": "scheduler_benchmark.vllm_scheduler.LTRAgingScheduler",
}
RUN_IDENTITY_FIELDS = (
    "model",
    "workload_sha256",
    "capacity_rps",
    "seed_derivation",
    "vllm_version",
    "repeats",
)


def load_policy_result(path: Path) -> list[dict[str, float | str | int]]:
    result = json.loads(path.read_text())
    if result.get("valid") is not True:
        raise ValueError(f"incomplete benchmark result: {path}")
    if int(result.get("repeats", 0)) < 1:
        raise ValueError(f"Figure 6 requires at least one repeat per policy: {path}")
    if any(
        row.get("completeness", {}).get("valid") is not True
        for row in result["scenarios"]
    ):
        raise ValueError(f"incomplete benchmark scenario: {path}")
    policy = str(result["policy"])
    expected_scheduler_cls = EXPECTED_SCHEDULER_CLASSES.get(policy)
    if result.get("scheduler_cls") != expected_scheduler_cls:
        raise ValueError(f"scheduler class does not match policy {policy}")
    identity = {field: result[field] for field in RUN_IDENTITY_FIELDS}
    rows: list[dict[str, float | str | int]] = []
    for scenario_result in result["scenarios"]:
        scenario = scenario_result["scenario"]
        if not str(scenario["name"]).startswith("saturation-"):
            continue
        metrics = scenario_result["aggregate"]["metrics"]
        rows.append(
            {
                "policy": policy,
                "saturation_pct": float(scenario["saturation"]) * 100.0,
                "p95_ttlt_ms": float(metrics["p95_ttlt_ms"]["mean"]),
                "p99_ttlt_ms": float(metrics["p99_ttlt_ms"]["mean"]),
                "scheduler_cls": str(result["scheduler_cls"]),
                "profile": str(scenario_result.get("profile", "mixed")),
                **identity,
            }
        )
    return sorted(rows, key=lambda row: float(row["saturation_pct"]))


def _validate_live_rows(rows: list[dict[str, float | str | int]]) -> None:
    if not rows:
        raise ValueError("live Figure 6 requires at least one row")
    series = {(str(row["policy"]), str(row["profile"])) for row in rows}
    saturation_sets = {
        tuple(
            sorted(
                float(row["saturation_pct"])
                for row in rows
                if (str(row["policy"]), str(row["profile"])) == series_key
            )
        )
        for series_key in series
    }
    if len(saturation_sets) != 1:
        raise ValueError("Figure 6 series must use the same load points")
    for field in RUN_IDENTITY_FIELDS:
        if len({row[field] for row in rows}) != 1:
            raise ValueError(f"Figure 6 inputs disagree on {field}")


def plot_figure(
    rows: list[dict[str, float | str | int]],
    output: Path,
    *,
    is_placeholder: bool = False,
) -> None:
    if not is_placeholder:
        _validate_live_rows(rows)
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), sharex=True)
    series = sorted(
        {(str(row["policy"]), str(row.get("profile", "mixed"))) for row in rows},
        key=lambda key: (
            POLICY_ORDER.index(key[0]) if key[0] in POLICY_ORDER else len(POLICY_ORDER),
            key,
        ),
    )
    for axis, metric, title in zip(
        axes,
        ("p95_ttlt_ms", "p99_ttlt_ms"),
        ("P95 TTLT", "P99 TTLT"),
    ):
        for index, (policy, profile) in enumerate(series):
            policy_rows = sorted(
                (
                    row
                    for row in rows
                    if row["policy"] == policy
                    and row.get("profile", "mixed") == profile
                ),
                key=lambda row: float(row["saturation_pct"]),
            )
            style = POLICY_STYLES.get(
                policy,
                {
                    "color": plt.get_cmap("tab10")(index % 10),
                    "marker": ("o", "s", "^", "D", "v", "P", "X")[index % 7],
                },
            )
            label = POLICY_LABELS.get(policy, policy.replace("_", " ").title())
            if len({str(row.get("profile", "mixed")) for row in rows}) > 1:
                label = f"{label} ({profile})"
            axis.plot(
                [float(row["saturation_pct"]) for row in policy_rows],
                [float(row[metric]) for row in policy_rows],
                linewidth=1.8,
                markersize=5,
                label=label,
                **style,
            )
        axis.set_title(title)
        axis.set_xlabel("Offered load (% saturation)")
        axis.set_xticks(
            sorted({float(row["saturation_pct"]) for row in rows}) or (40, 70, 90)
        )
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Time to last token (ms)")
    if series:
        axes[1].legend(frameon=False, fontsize=8)
    if is_placeholder:
        for axis in axes:
            axis.text(
                0.5,
                0.5,
                "Awaiting live GPU measurements",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#666666",
            )
    figure.suptitle("Figure 6. Live scheduler load sweep")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        default=[],
        help="Runner result JSON; repeat once per policy",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--placeholder-data",
        type=Path,
        default=Path(__file__).with_name("fig6-placeholder.json"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.input:
        rows = [row for path in args.input for row in load_policy_result(path)]
        plot_figure(rows, args.output)
    else:
        placeholder = json.loads(args.placeholder_data.read_text())
        if placeholder.get("placeholder") is not True:
            raise ValueError("placeholder data must be explicitly marked placeholder")
        plot_figure([], args.output, is_placeholder=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
