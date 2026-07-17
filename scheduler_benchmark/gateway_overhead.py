"""Isolated FCFS direct-engine versus gateway overhead measurement."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path
from typing import Awaitable, Callable, Mapping

from scheduler_benchmark.runner import (
    ResponseSample,
    ReplayScenario,
    STOCK_SCHEDULER_CLS,
    WorkloadRequest,
    build_arrival_offsets,
    load_workload,
    run_replay,
    stream_completion,
    summarize_samples,
    validate_vllm_version,
)

OVERHEAD_METRICS = (
    "mean_ttlt_ms",
    "p95_ttlt_ms",
    "p99_ttlt_ms",
    "mean_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "throughput_rps",
    "output_tokens_per_s",
)
Sender = Callable[[WorkloadRequest], Awaitable[ResponseSample]]
FCFS_SCHEDULER_CLASSES = (
    STOCK_SCHEDULER_CLS,
    "scheduler_benchmark.vllm_scheduler.StockFCFSShim",
)


def absolute_overhead(
    direct_metrics: Mapping[str, float | int],
    gateway_metrics: Mapping[str, float | int],
) -> dict[str, float]:
    return {
        name: round(float(gateway_metrics[name]) - float(direct_metrics[name]), 6)
        for name in OVERHEAD_METRICS
    }


async def run_overhead_pair(
    *,
    workload: list[WorkloadRequest],
    offsets: list[float],
    direct_sender: Sender,
    gateway_sender: Sender,
) -> dict[str, object]:
    """Replay the exact same FCFS arrivals once on each request path."""

    direct_samples, direct_wall_time_s = await run_replay(
        workload, offsets, direct_sender
    )
    gateway_samples, gateway_wall_time_s = await run_replay(
        workload, offsets, gateway_sender
    )
    direct_metrics = summarize_samples(
        direct_samples, wall_time_s=direct_wall_time_s
    )
    gateway_metrics = summarize_samples(
        gateway_samples, wall_time_s=gateway_wall_time_s
    )
    expected = len(workload)
    direct_valid = (
        int(direct_metrics["completed"]) == expected
        and int(direct_metrics["errors"]) == 0
    )
    gateway_valid = (
        int(gateway_metrics["completed"]) == expected
        and int(gateway_metrics["errors"]) == 0
    )
    return {
        "schema_version": 1,
        "mode": "gateway_overhead_fcfs",
        "repeats": 1,
        "valid": direct_valid and gateway_valid,
        "expected_requests": expected,
        "direct": {
            "wall_time_s": direct_wall_time_s,
            "metrics": direct_metrics,
            "samples": [asdict(sample) for sample in direct_samples],
        },
        "gateway": {
            "wall_time_s": gateway_wall_time_s,
            "metrics": gateway_metrics,
            "samples": [asdict(sample) for sample in gateway_samples],
        },
        "absolute_gateway_minus_direct": absolute_overhead(
            direct_metrics, gateway_metrics
        ),
    }


async def run_live_overhead(args: argparse.Namespace) -> dict[str, object]:
    try:
        import aiohttp
    except ImportError as exc:
        raise RuntimeError("aiohttp is required for live replay") from exc
    try:
        vllm_version = validate_vllm_version(distribution_version("vllm"))
    except PackageNotFoundError as exc:
        raise RuntimeError("vLLM 0.24.x must be installed for live replay") from exc

    requests = load_workload(args.workload)
    scenario = ReplayScenario("gateway-overhead", args.saturation)
    offsets = build_arrival_offsets(
        len(requests),
        capacity_rps=args.capacity_rps,
        scenario=scenario,
        seed=args.seed,
    )
    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def direct_sender(request: WorkloadRequest) -> ResponseSample:
            return await stream_completion(
                session,
                args.direct_endpoint,
                args.model,
                request,
                args.api_key,
            )

        async def gateway_sender(request: WorkloadRequest) -> ResponseSample:
            return await stream_completion(
                session,
                args.gateway_endpoint,
                args.model,
                request,
                args.api_key,
            )

        report = await run_overhead_pair(
            workload=requests,
            offsets=offsets,
            direct_sender=direct_sender,
            gateway_sender=gateway_sender,
        )
    report.update(
        {
            "direct_endpoint": args.direct_endpoint,
            "gateway_endpoint": args.gateway_endpoint,
            "model": args.model,
            "scheduler_cls": args.scheduler_cls,
            "capacity_rps": args.capacity_rps,
            "saturation": args.saturation,
            "seed": args.seed,
            "workload_sha256": hashlib.sha256(args.workload.read_bytes()).hexdigest(),
            "vllm_version": vllm_version,
        }
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure one matched FCFS direct-vLLM versus gateway run."
    )
    parser.add_argument("--direct-endpoint", required=True)
    parser.add_argument("--gateway-endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--capacity-rps", required=True, type=float)
    parser.add_argument(
        "--scheduler-cls",
        required=True,
        choices=FCFS_SCHEDULER_CLASSES,
        help="FCFS scheduler used by both direct and gateway paths",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--saturation", type=float, default=0.4)
    parser.add_argument("--api-key")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = asyncio.run(run_live_overhead(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["valid"] else 2
