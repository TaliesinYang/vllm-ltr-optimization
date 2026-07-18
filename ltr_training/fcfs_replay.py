from __future__ import annotations

import asyncio
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Awaitable, Callable


REPEAT_COUNT = 3
T_CRITICAL_95_DF2 = 4.303


@dataclass(frozen=True)
class WorkloadRequest:
    request_id: str
    prompt: str
    max_tokens: int


@dataclass(frozen=True)
class ResponseSample:
    ttft_ms: float
    ttlt_ms: float
    output_tokens: int
    send_ttft_ms: float | None = None
    send_ttlt_ms: float | None = None
    dispatch_lag_ms: float = 0.0
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


def load_workload(path: Path) -> list[WorkloadRequest]:
    requests: list[WorkloadRequest] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        request_id = str(row["request_id"])
        if request_id in seen_ids:
            raise ValueError(f"line {line_number}: duplicate request_id {request_id}")
        seen_ids.add(request_id)
        requests.append(
            WorkloadRequest(
                request_id=request_id,
                prompt=str(row["prompt"]),
                max_tokens=int(row.get("max_tokens", 256)),
            )
        )
    if not requests:
        raise ValueError("workload is empty")
    return requests


def build_arrival_offsets(
    count: int,
    *,
    capacity_rps: float,
    scenario: ReplayScenario,
    seed: int,
) -> list[float]:
    if capacity_rps <= 0:
        raise ValueError("capacity_rps must be positive")
    rng = random.Random(seed)
    offsets: list[float] = []
    elapsed = 0.0
    burst_start = int(count * (0.5 - scenario.burst_fraction / 2))
    burst_end = int(count * (0.5 + scenario.burst_fraction / 2))
    for index in range(count):
        multiplier = (
            scenario.burst_multiplier if burst_start <= index < burst_end else 1.0
        )
        elapsed += rng.expovariate(capacity_rps * scenario.saturation * multiplier)
        offsets.append(elapsed)
    return offsets


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
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
    return {
        "completed": len(completed),
        "errors": len(samples) - len(completed),
        "throughput_rps": round(len(completed) / max(wall_time_s, 1e-9), 6),
        "mean_ttlt_ms": round(statistics.mean(ttlt), 6) if ttlt else 0.0,
        "p95_ttlt_ms": round(_percentile(ttlt, 95), 6),
        "p99_ttlt_ms": round(_percentile(ttlt, 99), 6),
        "mean_ttft_ms": round(statistics.mean(ttft), 6) if ttft else 0.0,
    }


def _aggregate(repeats: list[dict[str, float | int]]) -> dict[str, object]:
    if len(repeats) != REPEAT_COUNT:
        raise ValueError(f"expected {REPEAT_COUNT} repeats")
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


async def _stream_completion(
    session: object,
    *,
    endpoint: str,
    model: str,
    request: WorkloadRequest,
    api_key: str | None,
) -> ResponseSample:
    headers = {"X-Request-Id": request.request_id}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "prompt": request.prompt,
        "max_tokens": request.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first_token_at: float | None = None
    output_tokens = 0
    token_events = 0
    async with session.post(endpoint, json=payload, headers=headers) as response:
        if response.status >= 400:
            raise RuntimeError(
                f"HTTP {response.status}: {(await response.text())[:500]}"
            )
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
            output_tokens = int(usage.get("completion_tokens", output_tokens))
            choices = event.get("choices") or []
            text = choices[0].get("text", "") if choices else ""
            if text:
                token_events += 1
                first_token_at = first_token_at or time.perf_counter()
    finished = time.perf_counter()
    send_ttft_ms = ((first_token_at or finished) - started) * 1000
    send_ttlt_ms = (finished - started) * 1000
    return ResponseSample(
        ttft_ms=send_ttft_ms,
        ttlt_ms=send_ttlt_ms,
        output_tokens=output_tokens or token_events,
        send_ttft_ms=send_ttft_ms,
        send_ttlt_ms=send_ttlt_ms,
    )


async def _run_replay(
    workload: list[WorkloadRequest],
    offsets: list[float],
    sender: Callable[[WorkloadRequest], Awaitable[ResponseSample]],
) -> tuple[list[ResponseSample], float]:
    loop = asyncio.get_running_loop()
    started = loop.time()

    async def send_at(request: WorkloadRequest, offset: float) -> ResponseSample:
        scheduled_at = started + offset
        await asyncio.sleep(max(0.0, scheduled_at - loop.time()))
        dispatched_at = loop.time()
        dispatch_lag_ms = max(0.0, (dispatched_at - scheduled_at) * 1000.0)
        try:
            sample = await sender(request)
            send_ttft_ms = (
                sample.send_ttft_ms
                if sample.send_ttft_ms is not None
                else sample.ttft_ms
            )
            send_ttlt_ms = (
                sample.send_ttlt_ms
                if sample.send_ttlt_ms is not None
                else sample.ttlt_ms
            )
            return replace(
                sample,
                ttft_ms=dispatch_lag_ms + send_ttft_ms,
                ttlt_ms=dispatch_lag_ms + send_ttlt_ms,
                send_ttft_ms=send_ttft_ms,
                send_ttlt_ms=send_ttlt_ms,
                dispatch_lag_ms=dispatch_lag_ms,
            )
        except Exception as error:
            send_elapsed_ms = max(0.0, (loop.time() - dispatched_at) * 1000.0)
            return ResponseSample(
                dispatch_lag_ms + send_elapsed_ms,
                dispatch_lag_ms + send_elapsed_ms,
                0,
                send_ttft_ms=send_elapsed_ms,
                send_ttlt_ms=send_elapsed_ms,
                dispatch_lag_ms=dispatch_lag_ms,
                error=str(error),
            )

    samples = await asyncio.gather(
        *(send_at(request, offset) for request, offset in zip(workload, offsets))
    )
    return list(samples), loop.time() - started


async def benchmark_endpoint(
    *,
    endpoint: str,
    model: str,
    workload: list[WorkloadRequest],
    capacity_rps: float,
    seed: int,
    timeout_s: float,
    api_key: str | None,
) -> dict[str, object]:
    import aiohttp

    scenario_results: list[dict[str, object]] = []
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for scenario_index, scenario in enumerate(benchmark_scenarios()):
            repeat_metrics: list[dict[str, float | int]] = []
            raw_repeats: list[dict[str, object]] = []
            for repeat in range(REPEAT_COUNT):
                offsets = build_arrival_offsets(
                    len(workload),
                    capacity_rps=capacity_rps,
                    scenario=scenario,
                    seed=seed + scenario_index * 100 + repeat,
                )

                async def sender(request: WorkloadRequest) -> ResponseSample:
                    return await _stream_completion(
                        session,
                        endpoint=endpoint,
                        model=model,
                        request=request,
                        api_key=api_key,
                    )

                samples, wall_time_s = await _run_replay(workload, offsets, sender)
                metrics = summarize_samples(samples, wall_time_s=wall_time_s)
                repeat_metrics.append(metrics)
                raw_repeats.append(
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
                    "repeats": raw_repeats,
                    "aggregate": _aggregate(repeat_metrics),
                }
            )
    return {"model": model, "capacity_rps": capacity_rps, "scenarios": scenario_results}
