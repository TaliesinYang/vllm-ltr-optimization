#!/usr/bin/env python3
"""Offline policy simulator for gateway-side latency policy experiments.

The simulator intentionally stays CPU-only and stdlib-only. It combines prior
chat serving outputs with BFCL tool-call lengths, then compares queue policies
under mixed workloads.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import tarfile
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import gateway_policy_optimizer


@dataclass(frozen=True)
class WorkItem:
    request_id: str
    kind: str
    output_tokens: int
    prompt_proxy: int
    category: str


@dataclass(frozen=True)
class QueueResult:
    policy: str
    qps: float
    seed: int
    tool_ratio: float
    count: int
    mean_wait: float
    p50_wait: float
    p95_wait: float
    p99_wait: float
    max_wait: float
    slow_mean_wait: float
    starvation_count: int
    mean_service: float


def service_time(item: WorkItem) -> float:
    """Abstract service cost calibrated for relative queueing behavior."""
    if item.kind == "chat":
        return 25.0 + 0.85 * item.output_tokens + 0.03 * item.prompt_proxy

    overhead = 35.0
    if "multi_turn" in item.category:
        overhead += 90.0
    if "live_multiple" in item.category:
        overhead += 25.0
    return overhead + 1.10 * item.output_tokens + 0.01 * item.prompt_proxy


def prompt_length_score(item: WorkItem) -> float:
    return float(item.prompt_proxy)


def toolaware_rule_score(item: WorkItem) -> float:
    score = float(item.prompt_proxy)
    if item.kind == "tool":
        score += 80.0
    if "multi_turn" in item.category:
        score += 180.0
    return score


def category_cost_score(item: WorkItem) -> float:
    if item.kind == "chat":
        return 250.0
    if "multi_turn" in item.category:
        return 420.0
    if "live_multiple" in item.category:
        return 180.0
    return 90.0


def tail_risk_multiplier(item: WorkItem) -> float:
    """Category-level uncertainty proxy for a TIE-style baseline.

    This is deliberately not a fitted log-t model. It uses only gateway-visible
    workload/category signals, giving us a no-GPU competitor that asks whether
    tail-risk inflation alone is enough without the explicit reliability gate.
    """
    if item.kind == "chat":
        return 0.65
    if "multi_turn" in item.category:
        return 1.25
    if "live_multiple" in item.category:
        return 0.90
    return 0.45


def tail_risk_cost_score(item: WorkItem, tail_weight: float = 0.35) -> float:
    expected = service_time(item) if item.kind == "chat" else category_cost_score(item)
    return expected * (1.0 + tail_weight * tail_risk_multiplier(item))


def gateway_gate_score(
    item: WorkItem,
    age: float,
    deadline_ms: float,
    arrival_time: float,
) -> float:
    workload_class = "tool_call" if item.kind == "tool" else "chat"
    predictor_tau = -0.015 if workload_class == "tool_call" else 0.596
    decision = gateway_policy_optimizer.choose_policy(
        workload_class=workload_class,
        predicted_cost=service_time(item),
        fallback_cost=category_cost_score(item),
        age_ms=age,
        deadline_ms=deadline_ms,
        predictor_tau=predictor_tau,
        predictor_confidence=0.8,
        arrival_time_ms=arrival_time,
    )
    return decision.score


def gateway_tail_risk_score(
    item: WorkItem,
    age: float,
    deadline_ms: float,
    arrival_time: float,
) -> float:
    workload_class = "tool_call" if item.kind == "tool" else "chat"
    predictor_tau = -0.015 if workload_class == "tool_call" else 0.596
    decision = gateway_policy_optimizer.choose_policy(
        workload_class=workload_class,
        predicted_cost=tail_risk_cost_score(item),
        fallback_cost=tail_risk_cost_score(item),
        age_ms=age,
        deadline_ms=deadline_ms,
        predictor_tau=predictor_tau,
        predictor_confidence=0.8,
        arrival_time_ms=arrival_time,
    )
    return decision.score


SCORE_FUNCTIONS: dict[str, Callable[[WorkItem], float]] = {
    "prompt_len": prompt_length_score,
    "toolaware_rule": toolaware_rule_score,
    "category_cost": category_cost_score,
    "tail_risk": tail_risk_cost_score,
}


POLICIES = (
    "fcfs",
    "oracle_sjf",
    "oracle_aging",
    "prompt_len",
    "toolaware_rule",
    "category_cost",
    "category_aging",
    "tail_risk_gate",
    "gateway_tail_risk_gate",
    "deadline_guard",
    "gateway_gate",
)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def pairwise_rank_accuracy(
    items: list[WorkItem], score_fn: Callable[[WorkItem], float]
) -> float:
    correct = 0
    total = 0
    for i, left in enumerate(items):
        for right in items[i + 1 :]:
            true_delta = service_time(left) - service_time(right)
            score_delta = score_fn(left) - score_fn(right)
            if true_delta == 0 or score_delta == 0:
                continue
            total += 1
            correct += (true_delta > 0) == (score_delta > 0)
    return correct / total if total else 0.0


def policy_key(
    item: WorkItem,
    policy: str,
    age: float,
    deadline_ms: float,
    arrival_time: float,
) -> tuple[float, float]:
    if policy == "oracle_sjf":
        return (service_time(item), arrival_time)
    if policy == "oracle_aging":
        return (service_time(item) / (1.0 + age / max(deadline_ms, 1.0)), arrival_time)
    if policy == "prompt_len":
        return (prompt_length_score(item), arrival_time)
    if policy == "toolaware_rule":
        return (toolaware_rule_score(item), arrival_time)
    if policy == "category_cost":
        return (category_cost_score(item), arrival_time)
    if policy == "category_aging":
        return (
            category_cost_score(item) / (1.0 + age / max(deadline_ms, 1.0)),
            arrival_time,
        )
    if policy == "tail_risk_gate":
        return (
            tail_risk_cost_score(item) / (1.0 + age / max(deadline_ms, 1.0)),
            arrival_time,
        )
    if policy == "gateway_tail_risk_gate":
        return (
            gateway_tail_risk_score(item, age, deadline_ms, arrival_time),
            arrival_time,
        )
    if policy == "deadline_guard":
        if age >= deadline_ms:
            return (-1_000_000_000.0 + arrival_time, arrival_time)
        return (category_cost_score(item), arrival_time)
    if policy == "gateway_gate":
        return (
            gateway_gate_score(item, age, deadline_ms, arrival_time),
            arrival_time,
        )
    return (arrival_time, arrival_time)


def simulate_queue(
    arrivals: list[tuple[float, WorkItem]],
    policy: str,
    *,
    qps: float = 0.0,
    seed: int = 0,
    tool_ratio: float = 0.0,
    deadline_ms: float = 2000.0,
    starvation_ms: float = 5000.0,
    key_fn: Callable[[WorkItem, float, float], tuple] | None = None,
) -> QueueResult:
    pending: list[tuple[float, WorkItem]] = []
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

        if key_fn is not None:
            selected_index = min(
                range(len(pending)),
                key=lambda pos: key_fn(
                    pending[pos][1],
                    max(0.0, now - pending[pos][0]),
                    pending[pos][0],
                ),
            )
        elif policy == "fcfs":
            selected_index = 0
        else:
            selected_index = min(
                range(len(pending)),
                key=lambda pos: policy_key(
                    pending[pos][1],
                    policy,
                    max(0.0, now - pending[pos][0]),
                    deadline_ms,
                    pending[pos][0],
                ),
            )

        arrival_time, item = pending.pop(selected_index)
        wait = max(0.0, now - arrival_time)
        cost = service_time(item)
        waits.append(wait)
        services.append(cost)
        if cost >= 250.0:
            slow_waits.append(wait)
        now += cost

    return QueueResult(
        policy=policy,
        qps=qps,
        seed=seed,
        tool_ratio=tool_ratio,
        count=len(waits),
        mean_wait=statistics.mean(waits) if waits else 0.0,
        p50_wait=percentile(waits, 50),
        p95_wait=percentile(waits, 95),
        p99_wait=percentile(waits, 99),
        max_wait=max(waits) if waits else 0.0,
        slow_mean_wait=statistics.mean(slow_waits) if slow_waits else 0.0,
        starvation_count=sum(1 for wait in waits if wait >= starvation_ms),
        mean_service=statistics.mean(services) if services else 0.0,
    )


def load_chat_items(archive_path: Path, pattern: str, limit: int) -> list[WorkItem]:
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(archive_path, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.name.endswith(".json")]
            tar.extractall(tmp, members=members)

        matches = sorted(Path(tmp).glob(pattern))
        if not matches:
            raise FileNotFoundError(f"no chat JSON matched pattern {pattern!r}")
        data = json.loads(matches[0].read_text())

    items: list[WorkItem] = []
    for idx, (out_len, prompt_len) in enumerate(
        zip(data["output_lens"], data["input_lens"])
    ):
        out = int(out_len)
        if out <= 0:
            continue
        items.append(WorkItem(f"chat-{idx}", "chat", out, int(prompt_len), "chat"))
        if len(items) >= limit:
            break
    return items


def load_bfcl_items(csv_path: Path, limit: int) -> list[WorkItem]:
    items: list[WorkItem] = []
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            output_tokens = int(float(row["answer_tokens"]))
            prompt_proxy = max(1, len(row["prompt"]) // 20)
            items.append(
                WorkItem(
                    row["id"],
                    "tool",
                    output_tokens,
                    prompt_proxy,
                    row["category"],
                )
            )
            if len(items) >= limit:
                break
    return items


def build_mixed_items(
    chat_items: list[WorkItem],
    tool_items: list[WorkItem],
    *,
    total: int,
    tool_ratio: float,
    seed: int,
) -> list[WorkItem]:
    rng = random.Random(seed)
    tool_count = round(total * tool_ratio)
    chat_count = total - tool_count
    if chat_count > len(chat_items) or tool_count > len(tool_items):
        raise ValueError("not enough chat or tool items for requested mix")
    mixed = rng.sample(chat_items, chat_count) + rng.sample(tool_items, tool_count)
    rng.shuffle(mixed)
    return mixed


def make_arrivals(items: list[WorkItem], qps: float, seed: int) -> list[tuple[float, WorkItem]]:
    rng = random.Random(seed)
    now = 0.0
    arrivals: list[tuple[float, WorkItem]] = []
    for item in items:
        now += rng.expovariate(qps / 1000.0)
        arrivals.append((now, item))
    return arrivals


def summarize_rank_accuracy(items: list[WorkItem], seed: int, sample_size: int) -> dict[str, float]:
    rng = random.Random(seed)
    sample = rng.sample(items, min(sample_size, len(items)))
    return {
        name: pairwise_rank_accuracy(sample, fn)
        for name, fn in SCORE_FUNCTIONS.items()
    }


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_matrix(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    chat_items = load_chat_items(args.chat_archive, args.chat_json_pattern, args.chat_limit)
    tool_items = load_bfcl_items(args.bfcl_lengths, args.tool_limit)

    matrix_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []

    for tool_ratio in args.tool_ratios:
        for seed in args.seeds:
            mixed = build_mixed_items(
                chat_items,
                tool_items,
                total=args.total_requests,
                tool_ratio=tool_ratio,
                seed=seed,
            )
            rank_acc = summarize_rank_accuracy(mixed, seed, args.rank_sample)
            rank_rows.append(
                {
                    "tool_ratio": tool_ratio,
                    "seed": seed,
                    **{f"{name}_pair_acc": value for name, value in rank_acc.items()},
                }
            )
            for qps in args.qps_values:
                arrivals = make_arrivals(mixed, qps, seed + int(qps * 100))
                fcfs = simulate_queue(
                    arrivals,
                    "fcfs",
                    qps=qps,
                    seed=seed,
                    tool_ratio=tool_ratio,
                    deadline_ms=args.deadline_ms,
                    starvation_ms=args.starvation_ms,
                )
                for policy in POLICIES:
                    result = simulate_queue(
                        arrivals,
                        policy,
                        qps=qps,
                        seed=seed,
                        tool_ratio=tool_ratio,
                        deadline_ms=args.deadline_ms,
                        starvation_ms=args.starvation_ms,
                    )
                    p95_speedup = (
                        fcfs.p95_wait / result.p95_wait if result.p95_wait else 0.0
                    )
                    matrix_rows.append(
                        {
                            "tool_ratio": tool_ratio,
                            "qps": qps,
                            "seed": seed,
                            "policy": policy,
                            "count": result.count,
                            "mean_service": round(result.mean_service, 3),
                            "mean_wait": round(result.mean_wait, 3),
                            "p50_wait": round(result.p50_wait, 3),
                            "p95_wait": round(result.p95_wait, 3),
                            "p99_wait": round(result.p99_wait, 3),
                            "max_wait": round(result.max_wait, 3),
                            "slow_mean_wait": round(result.slow_mean_wait, 3),
                            "starvation_count": result.starvation_count,
                            "p95_speedup_vs_fcfs": round(p95_speedup, 3),
                        }
                    )

    return matrix_rows, rank_rows


def write_readme(
    path: Path, matrix_rows: list[dict[str, object]], rank_rows: list[dict[str, object]]
) -> None:
    by_case: dict[tuple[float, float, int], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in matrix_rows:
        key = (float(row["tool_ratio"]), float(row["qps"]), int(row["seed"]))
        by_case[key][str(row["policy"])] = row

    safe_counts: Counter[str] = Counter()
    for policies in by_case.values():
        fcfs = policies.get("fcfs")
        if not fcfs:
            continue
        fcfs_p99 = float(fcfs["p99_wait"])
        for policy, row in policies.items():
            if policy == "fcfs":
                continue
            if (
                float(row["p95_speedup_vs_fcfs"]) > 1.1
                and float(row["p99_wait"]) <= 1.1 * fcfs_p99
            ):
                safe_counts[policy] += 1

    rank_by_ratio: dict[float, list[dict[str, object]]] = defaultdict(list)
    for row in rank_rows:
        rank_by_ratio[float(row["tool_ratio"])].append(row)

    best_rows = [
        row
        for row in matrix_rows
        if row["policy"]
        in {
            "fcfs",
            "oracle_aging",
            "gateway_gate",
            "tail_risk_gate",
            "gateway_tail_risk_gate",
            "deadline_guard",
            "category_aging",
        }
    ]
    best_rows = sorted(
        best_rows,
        key=lambda row: (
            float(row["tool_ratio"]),
            float(row["qps"]),
            str(row["policy"]),
            int(row["seed"]),
        ),
    )

    lines = [
        "# Gateway Policy Probe Results",
        "",
        "Generated by `offline_policy_simulator.py`.",
        "",
        "## Interpretation",
        "",
        "- `oracle_sjf` shows the upper bound of shortest-job-first style ordering, but it can hurt p99/max wait under overload.",
        "- `category_aging` and `deadline_guard` are conservative policies intended to protect tail latency.",
        "- Pairwise accuracy compares cheap request features against the simulator service-time proxy.",
        "",
        "## Key Findings",
        "",
        "- Pure shortest-job-first is useful as an upper bound, not as the deployable policy.",
        "- `tail_risk_gate` is the strongest current no-GPU policy under the p95/p99 safe-case criterion below.",
        "- `oracle_aging` remains the strongest oracle-style algorithmic target because it is close to `tail_risk_gate` while using true service time.",
        "- `gateway_gate` is the first deployable-policy sketch: it trusts chat prediction when reliable and falls back to category aging on tool-call distribution shift.",
        "- `tail_risk_gate` is a TIE-style lightweight competitor: it inflates expected cost by category-level tail risk without fitting a full output-length distribution.",
        "- `gateway_tail_risk_gate` is a naive composition of reliability gate plus tail-risk inflation; it does not improve safe cases over `gateway_gate` in this matrix.",
        "- `category_cost` is the strongest cheap gateway feature; its ranking accuracy rises as the tool-call share grows.",
        "- `deadline_guard` protects long requests but is too close to FCFS in many cases, so it is better as a fallback guard than the main optimizer.",
        "",
        "## Safe Candidate Count",
        "",
        "Safe means `p95_speedup_vs_fcfs > 1.1` and `policy_p99_wait <= 1.1 * fcfs_p99_wait` for the same tool ratio, qps, and seed (unified with the envelope tau* criterion).",
        "",
        "| policy | safe_cases |",
        "|---|---:|",
    ]
    for policy, count in safe_counts.most_common():
        lines.append(f"| {policy} | {count} |")

    if rank_by_ratio:
        lines.extend(
            [
                "",
                "## Median Pairwise Feature Accuracy By Tool Ratio",
                "",
                "| tool_ratio | prompt_len | toolaware_rule | category_cost | tail_risk |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for ratio in sorted(rank_by_ratio):
            rows = rank_by_ratio[ratio]
            lines.append(
                "| {:.1f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                    ratio,
                    statistics.median(
                        float(row["prompt_len_pair_acc"]) for row in rows
                    ),
                    statistics.median(
                        float(row["toolaware_rule_pair_acc"]) for row in rows
                    ),
                    statistics.median(
                        float(row["category_cost_pair_acc"]) for row in rows
                    ),
                    statistics.median(
                        float(row["tail_risk_pair_acc"]) for row in rows
                    ),
                )
            )

    lines.extend(
        [
            "",
        "## Selected Rows",
        "",
        "| tool_ratio | qps | seed | policy | p95_wait | p99_wait | max_wait | p95_speedup_vs_fcfs | starvation |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in best_rows[:96]:
        lines.append(
            "| {tool_ratio} | {qps} | {seed} | {policy} | {p95_wait} | {p99_wait} | {max_wait} | {p95_speedup_vs_fcfs} | {starvation_count} |".format(
                **row
            )
        )

    if rank_rows:
        lines.extend(
            [
                "",
                "## Rank Accuracy Sample",
                "",
                "| tool_ratio | seed | prompt_len | toolaware_rule | category_cost | tail_risk |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rank_rows[:40]:
            lines.append(
                "| {tool_ratio} | {seed} | {prompt_len_pair_acc:.3f} | {toolaware_rule_pair_acc:.3f} | {category_cost_pair_acc:.3f} | {tail_risk_pair_acc:.3f} |".format(
                    **row
                )
            )

    path.write_text("\n".join(lines) + "\n")


def parse_csv_floats(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chat-archive",
        type=Path,
        default=Path("deliverables/04-evaluation/baseline-2026-06-22/baseline-results.tgz"),
    )
    parser.add_argument(
        "--chat-json-pattern",
        default="RESULTS/vllm-16.0qps-*fcfs*.json",
    )
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
    parser.add_argument("--deadline-ms", type=float, default=2000.0)
    parser.add_argument("--starvation-ms", type=float, default=5000.0)
    parser.add_argument("--rank-sample", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix_rows, rank_rows = run_matrix(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "first_policy_matrix.csv", matrix_rows)
    write_csv(args.out_dir / "feature_rank_accuracy.csv", rank_rows)
    write_readme(args.out_dir / "README.md", matrix_rows, rank_rows)
    print(f"wrote {len(matrix_rows)} policy rows to {args.out_dir / 'first_policy_matrix.csv'}")
    print(f"wrote {len(rank_rows)} rank rows to {args.out_dir / 'feature_rank_accuracy.csv'}")


if __name__ == "__main__":
    main()
