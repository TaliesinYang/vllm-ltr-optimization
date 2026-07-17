#!/usr/bin/env python3
"""Parameter sweep for tail-risk and aging policy variants."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import offline_policy_simulator as sim


@dataclass(frozen=True)
class SweepCandidate:
    tail_weight: float
    aging_weight: float
    deadline_ms: float
    deadline_guard: bool

    @property
    def name(self) -> str:
        guard = "guard" if self.deadline_guard else "noguard"
        return (
            f"tail{self.tail_weight:g}_age{self.aging_weight:g}_"
            f"deadline{self.deadline_ms:g}_{guard}"
        )


def candidate_score(
    item: sim.WorkItem,
    age: float,
    candidate: SweepCandidate,
    arrival_time: float,
) -> float:
    if candidate.deadline_guard and age >= candidate.deadline_ms:
        return -1_000_000_000.0 + arrival_time
    age_discount = 1.0 + candidate.aging_weight * age / max(candidate.deadline_ms, 1.0)
    return sim.tail_risk_cost_score(item, candidate.tail_weight) / age_discount


def simulate_candidate(
    arrivals: list[tuple[float, sim.WorkItem]],
    candidate: SweepCandidate,
    *,
    qps: float,
    seed: int,
    tool_ratio: float,
    starvation_ms: float,
) -> sim.QueueResult:
    pending: list[tuple[float, sim.WorkItem]] = []
    now = 0.0
    index = 0
    waits: list[float] = []
    slow_waits: list[float] = []
    services: list[float] = []

    while index < len(arrivals) or pending:
        if not pending and index < len(arrivals) and now < arrivals[index][0]:
            now = arrivals[index][0]

        while index < len(arrivals) and arrivals[index][0] <= now:
            pending.append(arrivals[index])
            index += 1

        if not pending:
            continue

        selected_index = min(
            range(len(pending)),
            key=lambda pos: (
                candidate_score(
                    pending[pos][1],
                    max(0.0, now - pending[pos][0]),
                    candidate,
                    pending[pos][0],
                ),
                pending[pos][0],
            ),
        )

        arrival_time, item = pending.pop(selected_index)
        wait = max(0.0, now - arrival_time)
        cost = sim.service_time(item)
        waits.append(wait)
        services.append(cost)
        if cost >= 250.0:
            slow_waits.append(wait)
        now += cost

    return sim.QueueResult(
        policy="tail_sweep",
        qps=qps,
        seed=seed,
        tool_ratio=tool_ratio,
        count=len(waits),
        mean_wait=statistics.mean(waits) if waits else 0.0,
        p50_wait=sim.percentile(waits, 50),
        p95_wait=sim.percentile(waits, 95),
        p99_wait=sim.percentile(waits, 99),
        max_wait=max(waits) if waits else 0.0,
        slow_mean_wait=statistics.mean(slow_waits) if slow_waits else 0.0,
        starvation_count=sum(1 for wait in waits if wait >= starvation_ms),
        mean_service=statistics.mean(services) if services else 0.0,
    )


def evaluate_case(
    arrivals: list[tuple[float, sim.WorkItem]],
    fcfs: sim.QueueResult,
    candidate: SweepCandidate,
    *,
    qps: float,
    seed: int,
    tool_ratio: float,
    starvation_ms: float,
) -> dict[str, object]:
    result = simulate_candidate(
        arrivals,
        candidate,
        qps=qps,
        seed=seed,
        tool_ratio=tool_ratio,
        starvation_ms=starvation_ms,
    )
    p95_speedup = fcfs.p95_wait / result.p95_wait if result.p95_wait else 0.0
    p99_ratio = result.p99_wait / fcfs.p99_wait if fcfs.p99_wait else 0.0
    return {
        "policy": result.policy,
        "candidate": candidate.name,
        "tail_weight": candidate.tail_weight,
        "aging_weight": candidate.aging_weight,
        "deadline_ms": candidate.deadline_ms,
        "deadline_guard": int(candidate.deadline_guard),
        "tool_ratio": tool_ratio,
        "qps": qps,
        "seed": seed,
        "count": result.count,
        "mean_wait": round(result.mean_wait, 3),
        "p95_wait": round(result.p95_wait, 3),
        "p99_wait": round(result.p99_wait, 3),
        "max_wait": round(result.max_wait, 3),
        "slow_mean_wait": round(result.slow_mean_wait, 3),
        "starvation_count": result.starvation_count,
        "p95_speedup_vs_fcfs": round(p95_speedup, 3),
        "p99_ratio_vs_fcfs": round(p99_ratio, 3),
    }


def build_candidates(
    tail_weights: Iterable[float],
    aging_weights: Iterable[float],
    deadline_values: Iterable[float],
    guard_values: Iterable[bool],
) -> list[SweepCandidate]:
    return [
        SweepCandidate(tail, aging, deadline, guard)
        for tail in tail_weights
        for aging in aging_weights
        for deadline in deadline_values
        for guard in guard_values
    ]


def run_sweep(args: argparse.Namespace) -> list[dict[str, object]]:
    chat_items = sim.load_chat_items(args.chat_archive, args.chat_json_pattern, args.chat_limit)
    tool_items = sim.load_bfcl_items(args.bfcl_lengths, args.tool_limit)
    candidates = build_candidates(
        args.tail_weights,
        args.aging_weights,
        args.deadline_values,
        args.guard_values,
    )

    rows: list[dict[str, object]] = []
    for tool_ratio in args.tool_ratios:
        for seed in args.seeds:
            mixed = sim.build_mixed_items(
                chat_items,
                tool_items,
                total=args.total_requests,
                tool_ratio=tool_ratio,
                seed=seed,
            )
            for qps in args.qps_values:
                arrivals = sim.make_arrivals(mixed, qps, seed + int(qps * 100))
                fcfs = sim.simulate_queue(
                    arrivals,
                    "fcfs",
                    qps=qps,
                    seed=seed,
                    tool_ratio=tool_ratio,
                    starvation_ms=args.starvation_ms,
                )
                for candidate in candidates:
                    rows.append(
                        evaluate_case(
                            arrivals,
                            fcfs,
                            candidate,
                            qps=qps,
                            seed=seed,
                            tool_ratio=tool_ratio,
                            starvation_ms=args.starvation_ms,
                        )
                    )
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["candidate"])].append(row)

    summary: list[dict[str, object]] = []
    for candidate, candidate_rows in grouped.items():
        safe = [
            row
            for row in candidate_rows
            if float(row["p95_speedup_vs_fcfs"]) > 1.1
            and float(row["p99_ratio_vs_fcfs"]) <= 1.2
        ]
        summary.append(
            {
                "candidate": candidate,
                "safe_cases": len(safe),
                "median_p95_speedup": round(
                    statistics.median(
                        float(row["p95_speedup_vs_fcfs"]) for row in candidate_rows
                    ),
                    3,
                ),
                "median_p99_ratio": round(
                    statistics.median(
                        float(row["p99_ratio_vs_fcfs"]) for row in candidate_rows
                    ),
                    3,
                ),
                "total_starvation": sum(
                    int(row["starvation_count"]) for row in candidate_rows
                ),
            }
        )
    return sorted(
        summary,
        key=lambda row: (
            -int(row["safe_cases"]),
            float(row["median_p99_ratio"]),
            int(row["total_starvation"]),
        ),
    )


def write_readme(path: Path, rows: list[dict[str, object]], summary: list[dict[str, object]]) -> None:
    guard_counts: Counter[str] = Counter()
    for row in summary:
        guard = "guard" if str(row["candidate"]).endswith("_guard") else "noguard"
        guard_counts[guard] += int(row["safe_cases"])

    lines = [
        "# Tail-Risk Aging Sweep",
        "",
        "Generated by `tail_risk_aging_sweep.py`.",
        "",
        "Safe case criterion: `p95_speedup_vs_fcfs > 1.1` and `p99_ratio_vs_fcfs <= 1.2`.",
        "",
        "## Top Candidates",
        "",
        "| candidate | safe_cases | median_p95_speedup | median_p99_ratio | total_starvation |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary[:20]:
        lines.append(
            "| {candidate} | {safe_cases} | {median_p95_speedup} | {median_p99_ratio} | {total_starvation} |".format(
                **row
            )
        )

    if summary:
        best = summary[0]
        lines.extend(
            [
                "",
                "## Current Interpretation",
                "",
                f"- Best candidate: `{best['candidate']}` with {best['safe_cases']} safe cases.",
                "- Compare this against the fixed simulator baselines before porting anything to VeloxMesh.",
                "- If guarded candidates do not rank near the top, deadline guard should remain a fallback rather than the main score.",
            ]
        )

    path.write_text("\n".join(lines) + "\n")


def parse_csv_floats(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_guard_values(raw: str) -> list[bool]:
    values: list[bool] = []
    for part in raw.split(","):
        value = part.strip().lower()
        if not value:
            continue
        if value in {"1", "true", "on", "guard"}:
            values.append(True)
        elif value in {"0", "false", "off", "noguard"}:
            values.append(False)
        else:
            raise argparse.ArgumentTypeError(f"unknown guard value {part!r}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chat-archive",
        type=Path,
        default=Path("deliverables/04-evaluation/baseline-2026-06-22/baseline-results.tgz"),
    )
    parser.add_argument("--chat-json-pattern", default="RESULTS/vllm-16.0qps-*fcfs*.json")
    parser.add_argument(
        "--bfcl-lengths",
        type=Path,
        default=Path("project/bfcl_probe/results/bfcl_v3_lengths.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("project/gateway_policy_probe/results"))
    parser.add_argument("--chat-limit", type=int, default=1200)
    parser.add_argument("--tool-limit", type=int, default=3600)
    parser.add_argument("--total-requests", type=int, default=1000)
    parser.add_argument("--tool-ratios", type=parse_csv_floats, default="0.1,0.3,0.5,0.7")
    parser.add_argument("--qps-values", type=parse_csv_floats, default="1.5,2,3,4,5,6")
    parser.add_argument("--seeds", type=parse_csv_ints, default="42,6806,20260709")
    parser.add_argument("--tail-weights", type=parse_csv_floats, default="0,0.2,0.35,0.6,0.9")
    parser.add_argument("--aging-weights", type=parse_csv_floats, default="0.5,1,2")
    parser.add_argument("--deadline-values", type=parse_csv_floats, default="1000,2000")
    parser.add_argument("--guard-values", type=parse_guard_values, default="off,on")
    parser.add_argument("--starvation-ms", type=float, default=5000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_sweep(args)
    summary = summarize_rows(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "tail_risk_aging_sweep.csv", rows)
    write_csv(args.out_dir / "tail_risk_aging_sweep_summary.csv", summary)
    write_readme(args.out_dir / "tail_risk_aging_sweep.md", rows, summary)
    print(f"wrote {len(rows)} sweep rows to {args.out_dir / 'tail_risk_aging_sweep.csv'}")
    print(f"wrote {len(summary)} summary rows to {args.out_dir / 'tail_risk_aging_sweep_summary.csv'}")


if __name__ == "__main__":
    main()
