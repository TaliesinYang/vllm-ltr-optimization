"""OpenAI-compatible workload replay runner for live scheduler benchmarks."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path
from typing import Awaitable, Callable

REPEAT_COUNT = 3
T_CRITICAL_95_DF2 = 4.303
STOCK_SCHEDULER_CLS = "vllm.v1.core.sched.scheduler.Scheduler"
SCHEDULER_CLASS_TO_POLICY = {
    STOCK_SCHEDULER_CLS: "stock_fcfs",
    "scheduler_benchmark.vllm_scheduler.StockFCFSShim": "fcfs",
    "scheduler_benchmark.vllm_scheduler.PureLTRScheduler": "pure_ltr",
    "scheduler_benchmark.vllm_scheduler.TailSafeScheduler": "tail_safe",
    "scheduler_benchmark.vllm_scheduler.GatedHybridScheduler": "gated_hybrid",
}


@dataclass(frozen=True)
class WorkloadRequest:
    request_id: str
    prompt: str
    baseline_service_ms: float
    max_tokens: int = 256
    kind: str = "chat"
    category: str = ""


@dataclass(frozen=True)
class ResponseSample:
    request_id: str
    ttft_ms: float
    ttlt_ms: float
    output_tokens: int
    baseline_service_ms: float
    error: str | None = None


@dataclass(frozen=True)
class ReplayScenario:
    name: str
    saturation: float
    burst_multiplier: float = 1.0
    burst_fraction: float = 0.0


def benchmark_scenarios() -> tuple[ReplayScenario, ...]:
    return (
        ReplayScenario("saturation-40", 0.4),
        ReplayScenario("saturation-70", 0.7),
        ReplayScenario("saturation-90", 0.9),
        ReplayScenario("burst-90", 0.9, burst_multiplier=2.0, burst_fraction=0.2),
    )


def policy_for_scheduler_cls(scheduler_cls: str) -> str:
    try:
        return SCHEDULER_CLASS_TO_POLICY[scheduler_cls]
    except KeyError as exc:
        raise ValueError(f"unknown scheduler class: {scheduler_cls}") from exc


def validate_vllm_version(version: str) -> str:
    if not version.startswith("0.24."):
        raise ValueError(f"live runner requires vLLM 0.24.x, found {version}")
    return version


def build_arrival_offsets(
    count: int,
    *,
    capacity_rps: float,
    scenario: ReplayScenario,
    seed: int,
) -> list[float]:
    if capacity_rps <= 0.0:
        raise ValueError("capacity_rps must be positive")
    rng = random.Random(seed)
    offsets: list[float] = []
    elapsed = 0.0
    burst_start = int(count * (0.5 - scenario.burst_fraction / 2.0))
    burst_end = int(count * (0.5 + scenario.burst_fraction / 2.0))
    for index in range(count):
        is_burst = burst_start <= index < burst_end
        multiplier = scenario.burst_multiplier if is_burst else 1.0
        request_rate = capacity_rps * scenario.saturation * multiplier
        elapsed += rng.expovariate(request_rate)
        offsets.append(elapsed)
    return offsets


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summarize_samples(
    samples: list[ResponseSample], *, wall_time_s: float
) -> dict[str, float | int]:
    completed = [sample for sample in samples if sample.error is None]
    ttlt = [sample.ttlt_ms for sample in completed]
    ttft = [sample.ttft_ms for sample in completed]
    slowdown = [sample.ttlt_ms / sample.baseline_service_ms for sample in completed]
    output_tokens = sum(sample.output_tokens for sample in completed)
    safe_wall_time = max(wall_time_s, 1e-9)
    return {
        "completed": len(completed),
        "errors": len(samples) - len(completed),
        "mean_ttlt_ms": round(statistics.mean(ttlt), 6) if ttlt else 0.0,
        "p95_ttlt_ms": round(_percentile(ttlt, 95), 6),
        "p99_ttlt_ms": round(_percentile(ttlt, 99), 6),
        "mean_normalized_slowdown": (
            round(statistics.mean(slowdown), 6) if slowdown else 0.0
        ),
        "p95_normalized_slowdown": round(_percentile(slowdown, 95), 6),
        "p99_normalized_slowdown": round(_percentile(slowdown, 99), 6),
        "mean_ttft_ms": round(statistics.mean(ttft), 6) if ttft else 0.0,
        "p95_ttft_ms": round(_percentile(ttft, 95), 6),
        "p99_ttft_ms": round(_percentile(ttft, 99), 6),
        "throughput_rps": round(len(completed) / safe_wall_time, 6),
        "output_tokens_per_s": round(output_tokens / safe_wall_time, 6),
    }


def aggregate_repeats(
    repeats: list[dict[str, float | int]],
) -> dict[str, object]:
    if len(repeats) != REPEAT_COUNT:
        raise ValueError(f"benchmark requires exactly {REPEAT_COUNT} repeats")
    metrics: dict[str, dict[str, float]] = {}
    for name in repeats[0]:
        values = [float(repeat[name]) for repeat in repeats]
        mean = statistics.mean(values)
        margin = T_CRITICAL_95_DF2 * statistics.stdev(values) / math.sqrt(REPEAT_COUNT)
        metrics[name] = {
            "mean": round(mean, 6),
            "ci95_low": round(mean - margin, 6),
            "ci95_high": round(mean + margin, 6),
        }
    return {"repeats": REPEAT_COUNT, "metrics": metrics}


def assess_completeness(repeats, *, expected_requests):
    valid = all(
        int(repeat["completed"]) == expected_requests and int(repeat["errors"]) == 0
        for repeat in repeats
    )
    return {
        "valid": valid,
        "expected_requests_per_repeat": expected_requests,
        "completed_per_repeat": [int(repeat["completed"]) for repeat in repeats],
        "errors_per_repeat": [int(repeat["errors"]) for repeat in repeats],
    }


def load_workload(path: Path) -> list[WorkloadRequest]:
    requests: list[WorkloadRequest] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "baseline_service_ms" not in row:
            raise ValueError(f"line {line_number}: baseline_service_ms is required")
        baseline_service_ms = float(row["baseline_service_ms"])
        if baseline_service_ms <= 0.0:
            raise ValueError(
                f"line {line_number}: baseline_service_ms must be positive"
            )
        request_id = str(row["request_id"])
        if request_id in seen_ids:
            raise ValueError(f"line {line_number}: duplicate request_id {request_id}")
        seen_ids.add(request_id)
        requests.append(
            WorkloadRequest(
                request_id=request_id,
                prompt=str(row["prompt"]),
                baseline_service_ms=baseline_service_ms,
                max_tokens=int(row.get("max_tokens", 256)),
                kind=str(row.get("kind", "chat")),
                category=str(row.get("category", "")),
            )
        )
    if not requests:
        raise ValueError("workload is empty")
    return requests


def make_completion_payload(
    request: WorkloadRequest, *, model: str
) -> dict[str, object]:
    return {
        "model": model,
        "prompt": request.prompt,
        "max_tokens": request.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "vllm_xargs": {
            "ltr_kind": request.kind,
            "ltr_category": request.category,
        },
    }


async def stream_completion(
    session,
    endpoint: str,
    model: str,
    request: WorkloadRequest,
    api_key: str | None,
) -> ResponseSample:
    headers = {"X-Request-Id": request.request_id}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    started = time.perf_counter()
    first_token_at: float | None = None
    output_tokens: int | None = None
    async with session.post(
        endpoint,
        json=make_completion_payload(request, model=model),
        headers=headers,
    ) as response:
        if response.status >= 400:
            body = await response.text()
            raise RuntimeError(f"HTTP {response.status}: {body[:500]}")
        while not response.content.at_eof():
            raw_line = await response.content.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            event = json.loads(data)
            usage = event.get("usage") or {}
            if "completion_tokens" in usage:
                output_tokens = int(usage["completion_tokens"])
            choices = event.get("choices") or []
            text = choices[0].get("text", "") if choices else ""
            if text:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
    finished = time.perf_counter()
    if output_tokens is None:
        raise RuntimeError("stream response omitted completion_tokens usage")
    ttlt_ms = (finished - started) * 1000.0
    ttft_ms = ((first_token_at or finished) - started) * 1000.0
    return ResponseSample(
        request_id=request.request_id,
        ttft_ms=ttft_ms,
        ttlt_ms=ttlt_ms,
        output_tokens=output_tokens,
        baseline_service_ms=request.baseline_service_ms,
    )


async def run_replay(
    workload: list[WorkloadRequest],
    offsets: list[float],
    sender: Callable[[WorkloadRequest], Awaitable[ResponseSample]],
) -> tuple[list[ResponseSample], float]:
    if len(workload) != len(offsets):
        raise ValueError("workload and arrival offsets must have equal length")
    loop = asyncio.get_running_loop()
    replay_started = loop.time()

    async def send_at_offset(request: WorkloadRequest, offset: float) -> ResponseSample:
        await asyncio.sleep(max(0.0, replay_started + offset - loop.time()))
        try:
            return await sender(request)
        except Exception as exc:
            return ResponseSample(
                request_id=request.request_id,
                ttft_ms=0.0,
                ttlt_ms=0.0,
                output_tokens=0,
                baseline_service_ms=request.baseline_service_ms,
                error=str(exc),
            )

    samples = await asyncio.gather(
        *(send_at_offset(request, offset) for request, offset in zip(workload, offsets))
    )
    return list(samples), loop.time() - replay_started


async def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    try:
        import aiohttp
    except ImportError as exc:
        raise RuntimeError("aiohttp is required for live replay") from exc

    try:
        vllm_version = validate_vllm_version(distribution_version("vllm"))
    except PackageNotFoundError as exc:
        raise RuntimeError("vLLM 0.24.x must be installed for live replay") from exc
    workload = load_workload(args.workload)
    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    scenario_results: list[dict[str, object]] = []
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for scenario_index, scenario in enumerate(benchmark_scenarios()):
            runs: list[dict[str, object]] = []
            repeat_metrics: list[dict[str, float | int]] = []
            for repeat in range(REPEAT_COUNT):
                offsets = build_arrival_offsets(
                    len(workload),
                    capacity_rps=args.capacity_rps,
                    scenario=scenario,
                    seed=args.seed + scenario_index * 100 + repeat,
                )

                async def sender(request: WorkloadRequest) -> ResponseSample:
                    return await stream_completion(
                        session, args.endpoint, args.model, request, args.api_key
                    )

                samples, wall_time_s = await run_replay(workload, offsets, sender)
                metrics = summarize_samples(samples, wall_time_s=wall_time_s)
                repeat_metrics.append(metrics)
                runs.append(
                    {
                        "repeat": repeat + 1,
                        "wall_time_s": wall_time_s,
                        "metrics": metrics,
                        "samples": [asdict(sample) for sample in samples],
                    }
                )
            scenario_results.append(
                {
                    "scenario": asdict(scenario),
                    "runs": runs,
                    "aggregate": aggregate_repeats(repeat_metrics),
                    "completeness": assess_completeness(
                        repeat_metrics, expected_requests=len(workload)
                    ),
                }
            )
    is_valid = all(bool(result["completeness"]["valid"]) for result in scenario_results)
    return {
        "schema_version": 1,
        "valid": is_valid,
        "policy": policy_for_scheduler_cls(args.scheduler_cls),
        "scheduler_cls": args.scheduler_cls,
        "model": args.model,
        "capacity_rps": args.capacity_rps,
        "workload_sha256": hashlib.sha256(args.workload.read_bytes()).hexdigest(),
        "seed": args.seed,
        "vllm_version": vllm_version,
        "repeats": REPEAT_COUNT,
        "scenarios": scenario_results,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--capacity-rps", required=True, type=float)
    parser.add_argument(
        "--scheduler-cls",
        required=True,
        choices=tuple(SCHEDULER_CLASS_TO_POLICY),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--api-key")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = asyncio.run(run_benchmark(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["valid"] else 2
