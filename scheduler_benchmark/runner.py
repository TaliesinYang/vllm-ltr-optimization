"""OpenAI-compatible workload replay runner for live scheduler benchmarks."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import io
import json
import math
import random
import statistics
import time
import uuid
from dataclasses import dataclass
from dataclasses import field
from dataclasses import asdict
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path
from typing import Awaitable, Callable

from ltr_training.tier2 import build_request
from scheduler_benchmark.contracts import MAX_ESTIMATED_TOKENS

REPEAT_COUNT = 3
SCHEMA_VERSION = 2
STOCK_SCHEDULER_CLS = "vllm.v1.core.sched.scheduler.Scheduler"
SCHEDULER_CLASS_TO_POLICY = {
    STOCK_SCHEDULER_CLS: "stock_fcfs",
    "scheduler_benchmark.vllm_scheduler.StockFCFSShim": "fcfs",
    "scheduler_benchmark.vllm_scheduler.PureLTRScheduler": "pure_ltr",
    "scheduler_benchmark.vllm_scheduler.TailSafeScheduler": "tail_safe",
    "scheduler_benchmark.vllm_scheduler.GatedHybridScheduler": "gated_hybrid",
    "scheduler_benchmark.vllm_scheduler.PromptLengthSJFScheduler": "prompt_sjf",
    "scheduler_benchmark.vllm_scheduler.LTRAgingScheduler": "ltr_aging",
}


@dataclass(frozen=True)
class WorkloadRequest:
    request_id: str
    prompt: str
    baseline_service_ms: float
    max_tokens: int = MAX_ESTIMATED_TOKENS
    kind: str = "chat"
    category: str = ""
    tool_schema: str = ""
    history: list[list[str]] = field(default_factory=list)


@dataclass(frozen=True)
class ResponseSample:
    request_id: str
    ttft_ms: float
    ttlt_ms: float
    output_tokens: int
    baseline_service_ms: float
    send_ttft_ms: float | None = None
    send_ttlt_ms: float | None = None
    dispatch_lag_ms: float = 0.0
    category: str = ""
    policy: str = ""
    profile: str = ""
    scheduled_at_unix_s: float | None = None
    dispatched_at_unix_s: float | None = None
    first_token_at_unix_s: float | None = None
    completed_at_unix_s: float | None = None
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


def resolve_scenario_matrix(
    requested_scenarios: list[str] | None,
    requested_loads: list[int] | None,
) -> list[ReplayScenario]:
    if requested_scenarios is None and requested_loads is None:
        return list(benchmark_scenarios())
    scenario_kinds = sorted(
        set(requested_scenarios or ["steady"]), key=("steady", "burst").index
    )
    loads = sorted(set(requested_loads or [40, 70, 90]))
    scenarios: list[ReplayScenario] = []
    for kind in scenario_kinds:
        for load in loads:
            saturation = load / 100.0
            if kind == "steady":
                scenarios.append(ReplayScenario(f"saturation-{load}", saturation))
            elif kind == "burst":
                scenarios.append(
                    ReplayScenario(
                        f"burst-{load}",
                        saturation,
                        burst_multiplier=2.0,
                        burst_fraction=0.2,
                    )
                )
            else:
                raise ValueError(f"unknown scenario kind: {kind}")
    return scenarios


def derive_run_seed(*, profile: str, load_pct: int, repeat: int) -> int:
    payload = json.dumps(
        [profile, int(load_pct), int(repeat)], separators=(",", ":")
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def select_workload_profile(
    workload: list[WorkloadRequest], profile: str
) -> list[WorkloadRequest]:
    if profile == "mixed":
        selected = list(workload)
    elif profile in {"id", "ood"}:
        prefix = f"{profile}:"
        selected = [
            request
            for request in workload
            if request.category.lower() == profile
            or request.category.lower().startswith(prefix)
        ]
    else:
        raise ValueError(f"unknown workload profile: {profile}")
    if not selected:
        raise ValueError(f"workload profile {profile} selected no requests")
    return selected


def subrun_fingerprint(record: dict[str, object]) -> str:
    scenario = record.get("scenario")
    if not isinstance(scenario, dict):
        raise ValueError("subrun record lacks scenario")
    identity = {
        "schema_version": record["schema_version"],
        "workload_sha256": record["workload_sha256"],
        "policy": record["policy"],
        "scenario": scenario.get("name"),
        "load_pct": record["load_pct"],
        "profile": record["profile"],
        "seed": record["seed"],
        "warmup": record["warmup"],
        "completed": record["completed"],
        "errors": record["errors"],
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_subrun_artifacts(
    runs_dir: Path,
    record: dict[str, object],
    samples: list[ResponseSample],
) -> tuple[Path, Path]:
    expected = subrun_fingerprint(record)
    if record.get("fingerprint") != expected:
        raise ValueError("subrun fingerprint does not match record")
    json_path = runs_dir / f"{expected}.json"
    csv_path = runs_dir / f"{expected}.samples.csv"
    fieldnames = (
        list(asdict(samples[0]).keys())
        if samples
        else list(ResponseSample.__dataclass_fields__)
    )
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for sample in samples:
        writer.writerow(asdict(sample))
    _atomic_write_text(csv_path, buffer.getvalue())
    persisted = dict(record)
    persisted["samples_csv"] = csv_path.name
    _atomic_write_text(
        json_path, json.dumps(persisted, indent=2, sort_keys=True) + "\n"
    )
    return json_path, csv_path


def load_completed_subruns(runs_dir: Path) -> list[dict[str, object]]:
    if not runs_dir.exists():
        return []
    completed: list[dict[str, object]] = []
    for path in sorted(runs_dir.glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("status") != "complete":
            continue
        expected = subrun_fingerprint(record)
        if record.get("fingerprint") != expected or path.stem != expected:
            raise ValueError(f"invalid subrun fingerprint: {path}")
        samples_csv = record.get("samples_csv")
        if not isinstance(samples_csv, str) or not (runs_dir / samples_csv).is_file():
            raise ValueError(f"subrun sample CSV is missing: {path}")
        completed.append(record)
    return completed


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
    send_ttlt = [
        sample.send_ttlt_ms if sample.send_ttlt_ms is not None else sample.ttlt_ms
        for sample in completed
    ]
    send_ttft = [
        sample.send_ttft_ms if sample.send_ttft_ms is not None else sample.ttft_ms
        for sample in completed
    ]
    dispatch_lag = [sample.dispatch_lag_ms for sample in completed]
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
        "mean_send_ttlt_ms": round(statistics.mean(send_ttlt), 6) if send_ttlt else 0.0,
        "p95_send_ttlt_ms": round(_percentile(send_ttlt, 95), 6),
        "p99_send_ttlt_ms": round(_percentile(send_ttlt, 99), 6),
        "mean_send_ttft_ms": round(statistics.mean(send_ttft), 6) if send_ttft else 0.0,
        "p95_send_ttft_ms": round(_percentile(send_ttft, 95), 6),
        "p99_send_ttft_ms": round(_percentile(send_ttft, 99), 6),
        "mean_dispatch_lag_ms": (
            round(statistics.mean(dispatch_lag), 6) if dispatch_lag else 0.0
        ),
        "p95_dispatch_lag_ms": round(_percentile(dispatch_lag, 95), 6),
        "p99_dispatch_lag_ms": round(_percentile(dispatch_lag, 99), 6),
        "throughput_rps": round(len(completed) / safe_wall_time, 6),
        "output_tokens_per_s": round(output_tokens / safe_wall_time, 6),
    }


def aggregate_repeats(
    repeats: list[dict[str, float | int]],
) -> dict[str, object]:
    if not repeats:
        raise ValueError("benchmark requires at least one repeat")
    metrics: dict[str, dict[str, float]] = {}
    for name in repeats[0]:
        values = [float(repeat[name]) for repeat in repeats]
        mean = statistics.mean(values)
        metrics[name] = {
            "values": values,
            "mean": round(mean, 6),
            "min": round(min(values), 6),
            "max": round(max(values), 6),
        }
    return {"repeats": len(repeats), "metrics": metrics}


def resolve_warmup_requests(
    total_requests: int,
    *,
    requested_count: int | None = None,
    requested_ratio: float | None = None,
) -> int:
    if requested_count is not None and requested_ratio is not None:
        raise ValueError("warmup count and ratio are mutually exclusive")
    if total_requests < 1:
        raise ValueError("total_requests must be positive")
    if requested_count is not None:
        if requested_count < 0:
            raise ValueError("warmup request count must be non-negative")
        resolved = requested_count
    elif requested_ratio is not None:
        if not 0.0 <= requested_ratio < 1.0:
            raise ValueError("warmup ratio must be between zero and one")
        resolved = math.floor(total_requests * requested_ratio)
    else:
        resolved = 0
    if resolved >= total_requests:
        raise ValueError("warmup must leave at least one measurement request")
    return resolved


def measurement_window(
    samples: list[ResponseSample], *, warmup_requests: int
) -> tuple[list[ResponseSample], float]:
    if not 0 <= warmup_requests < len(samples):
        raise ValueError("warmup_requests must leave a non-empty measurement window")
    measured = samples[warmup_requests:]
    starts = [
        sample.scheduled_at_unix_s
        for sample in measured
        if sample.scheduled_at_unix_s is not None
    ]
    finishes = [
        sample.completed_at_unix_s
        for sample in measured
        if sample.completed_at_unix_s is not None
    ]
    if len(starts) != len(measured) or len(finishes) != len(measured):
        raise ValueError("measurement window requires absolute sample timestamps")
    duration_s = max(max(finishes) - min(starts), 1e-9)
    return measured, duration_s


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
        if not isinstance(row, dict):
            raise ValueError(f"line {line_number}: workload row must be an object")
        if "baseline_service_ms" not in row:
            raise ValueError(f"line {line_number}: baseline_service_ms is required")
        for field_name in ("prompt", "tool_schema", "history"):
            if field_name not in row:
                raise ValueError(f"line {line_number}: {field_name} is required")
        if not isinstance(row["prompt"], str):
            raise ValueError(f"line {line_number}: prompt must be a string")
        if not isinstance(row["tool_schema"], str):
            raise ValueError(f"line {line_number}: tool_schema must be a string")
        raw_history = row["history"]
        if not isinstance(raw_history, list):
            raise ValueError(f"line {line_number}: history must be a list")
        history: list[list[str]] = []
        for history_index, item in enumerate(raw_history):
            if (
                not isinstance(item, list)
                or len(item) != 2
                or not all(isinstance(value, str) for value in item)
            ):
                raise ValueError(
                    f"line {line_number}: history item {history_index} "
                    "must be a two-string role/text pair"
                )
            history.append([item[0], item[1]])
        baseline_service_ms = float(row["baseline_service_ms"])
        if baseline_service_ms <= 0.0:
            raise ValueError(
                f"line {line_number}: baseline_service_ms must be positive"
            )
        request_id = str(row["request_id"])
        if request_id in seen_ids:
            raise ValueError(f"line {line_number}: duplicate request_id {request_id}")
        seen_ids.add(request_id)
        max_tokens = int(row.get("max_tokens", MAX_ESTIMATED_TOKENS))
        if max_tokens != MAX_ESTIMATED_TOKENS:
            raise ValueError(
                f"line {line_number}: max_tokens must be "
                f"{MAX_ESTIMATED_TOKENS}"
            )
        requests.append(
            WorkloadRequest(
                request_id=request_id,
                prompt=row["prompt"],
                tool_schema=row["tool_schema"],
                history=history,
                baseline_service_ms=baseline_service_ms,
                max_tokens=max_tokens,
                kind=str(row.get("kind", "chat")),
                category=str(row.get("category", "")),
            )
        )
    if not requests:
        raise ValueError("workload is empty")
    return requests


def make_chat_payload(request: WorkloadRequest, *, model: str) -> dict[str, object]:
    payload = build_request(
        {
            "prompt": request.prompt,
            "tool_schema": request.tool_schema,
            "history": request.history,
        },
        model=model,
    )
    return {
        **payload,
        "stream": True,
        "stream_options": {"include_usage": True},
        "vllm_xargs": {
            "ltr_kind": request.kind,
            "ltr_category": request.category,
            "ltr_tool_schema": request.tool_schema,
        },
    }


def gateway_request_headers(
    request: WorkloadRequest, api_key: str | None
) -> dict[str, str]:
    headers = {
        "X-Request-Id": request.request_id,
        "X-Workflow-Id": request.request_id,
        "X-Step-Id": "0",
        "X-Conversation-Id": request.request_id,
        "X-Previous-Tool-Gap-Ms": "0",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def gateway_manifest(endpoint: str) -> dict[str, str]:
    return {
        "request_path": "client->gateway->decision->vllm",
        "gateway_endpoint": endpoint,
    }


async def stream_completion(
    session,
    endpoint: str,
    model: str,
    request: WorkloadRequest,
    api_key: str | None,
) -> ResponseSample:
    headers = gateway_request_headers(request, api_key)
    started = time.perf_counter()
    dispatched_at_unix_s = time.time()
    first_token_at: float | None = None
    first_token_at_unix_s: float | None = None
    output_tokens: int | None = None
    async with session.post(
        endpoint,
        json=make_chat_payload(request, model=model),
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
            if not data:
                continue
            if data == "[DONE]":
                break
            event = json.loads(data)
            usage = event.get("usage") or {}
            if "completion_tokens" in usage:
                output_tokens = int(usage["completion_tokens"])
            choices = event.get("choices") or []
            delta = (choices[0].get("delta") or {}) if choices else {}
            if delta.get("content") or delta.get("tool_calls"):
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                    first_token_at_unix_s = time.time()
    finished = time.perf_counter()
    completed_at_unix_s = time.time()
    if output_tokens is None:
        raise RuntimeError("stream response omitted completion_tokens usage")
    send_ttlt_ms = (finished - started) * 1000.0
    send_ttft_ms = ((first_token_at or finished) - started) * 1000.0
    return ResponseSample(
        request_id=request.request_id,
        ttft_ms=send_ttft_ms,
        ttlt_ms=send_ttlt_ms,
        output_tokens=output_tokens,
        baseline_service_ms=request.baseline_service_ms,
        send_ttft_ms=send_ttft_ms,
        send_ttlt_ms=send_ttlt_ms,
        dispatched_at_unix_s=dispatched_at_unix_s,
        first_token_at_unix_s=first_token_at_unix_s,
        completed_at_unix_s=completed_at_unix_s,
    )


async def run_replay(
    workload: list[WorkloadRequest],
    offsets: list[float],
    sender: Callable[[WorkloadRequest], Awaitable[ResponseSample]],
    *,
    policy: str = "",
    profile: str = "mixed",
) -> tuple[list[ResponseSample], float]:
    if len(workload) != len(offsets):
        raise ValueError("workload and arrival offsets must have equal length")
    loop = asyncio.get_running_loop()
    replay_started = loop.time()
    replay_started_unix_s = time.time()

    async def send_at_offset(request: WorkloadRequest, offset: float) -> ResponseSample:
        scheduled_at = replay_started + offset
        scheduled_at_unix_s = replay_started_unix_s + offset
        await asyncio.sleep(max(0.0, scheduled_at - loop.time()))
        dispatched_at = loop.time()
        dispatched_at_unix_s = time.time()
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
                category=request.category,
                policy=policy,
                profile=profile,
                scheduled_at_unix_s=scheduled_at_unix_s,
                dispatched_at_unix_s=dispatched_at_unix_s,
                first_token_at_unix_s=(
                    sample.first_token_at_unix_s
                    if sample.first_token_at_unix_s is not None
                    else dispatched_at_unix_s + send_ttft_ms / 1000.0
                ),
                completed_at_unix_s=(
                    sample.completed_at_unix_s
                    if sample.completed_at_unix_s is not None
                    else dispatched_at_unix_s + send_ttlt_ms / 1000.0
                ),
            )
        except Exception as exc:
            failed_at = loop.time()
            failed_at_unix_s = time.time()
            send_elapsed_ms = max(0.0, (failed_at - dispatched_at) * 1000.0)
            return ResponseSample(
                request_id=request.request_id,
                ttft_ms=dispatch_lag_ms + send_elapsed_ms,
                ttlt_ms=dispatch_lag_ms + send_elapsed_ms,
                output_tokens=0,
                baseline_service_ms=request.baseline_service_ms,
                send_ttft_ms=send_elapsed_ms,
                send_ttlt_ms=send_elapsed_ms,
                dispatch_lag_ms=dispatch_lag_ms,
                category=request.category,
                policy=policy,
                profile=profile,
                scheduled_at_unix_s=scheduled_at_unix_s,
                dispatched_at_unix_s=dispatched_at_unix_s,
                first_token_at_unix_s=failed_at_unix_s,
                completed_at_unix_s=failed_at_unix_s,
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
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    workload = load_workload(args.workload)
    workload_sha256 = hashlib.sha256(args.workload.read_bytes()).hexdigest()
    policy = policy_for_scheduler_cls(args.scheduler_cls)
    profiles = list(dict.fromkeys(args.profile or ["mixed"]))
    scenarios = resolve_scenario_matrix(args.scenario, args.load)
    runs_dir = args.output.parent / f"{args.output.stem}.runs"
    completed_subruns = load_completed_subruns(runs_dir) if args.resume else []
    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    scenario_results: list[dict[str, object]] = []
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for profile in profiles:
            profile_workload = select_workload_profile(workload, profile)
            warmup_requests = resolve_warmup_requests(
                len(profile_workload),
                requested_count=args.warmup_requests,
                requested_ratio=args.warmup_ratio,
            )
            for scenario in scenarios:
                load_pct = int(round(scenario.saturation * 100))
                runs: list[dict[str, object]] = []
                repeat_metrics: list[dict[str, float | int]] = []
                full_counts: list[dict[str, int]] = []
                for repeat in range(1, args.repeats + 1):
                    seed = derive_run_seed(
                        profile=profile, load_pct=load_pct, repeat=repeat
                    )
                    warmup_manifest = {
                        "requested": {
                            "count": args.warmup_requests,
                            "ratio": args.warmup_ratio,
                        },
                        "resolved": warmup_requests,
                        "measured": len(profile_workload) - warmup_requests,
                        "discarded": warmup_requests,
                    }
                    expected_identity = {
                        "schema_version": SCHEMA_VERSION,
                        "workload_sha256": workload_sha256,
                        "policy": policy,
                        "scheduler_cls": args.scheduler_cls,
                        "model": args.model,
                        "capacity_rps": args.capacity_rps,
                        "scenario": asdict(scenario),
                        "load_pct": load_pct,
                        "profile": profile,
                        "repeat": repeat,
                        "seed": seed,
                        "warmup": warmup_manifest,
                    }
                    resumed = next(
                        (
                            record
                            for record in completed_subruns
                            if all(
                                record.get(key) == value
                                for key, value in expected_identity.items()
                            )
                        ),
                        None,
                    )
                    if resumed is not None:
                        runs.append(resumed)
                        repeat_metrics.append(resumed["metrics"])
                        full_counts.append(
                            {
                                "completed": int(resumed["completed"]),
                                "errors": int(resumed["errors"]),
                            }
                        )
                        continue

                    offsets = build_arrival_offsets(
                        len(profile_workload),
                        capacity_rps=args.capacity_rps,
                        scenario=scenario,
                        seed=seed,
                    )

                    async def sender(request: WorkloadRequest) -> ResponseSample:
                        return await stream_completion(
                            session, args.endpoint, args.model, request, args.api_key
                        )

                    started_at_unix_s = time.time()
                    samples, wall_time_s = await run_replay(
                        profile_workload,
                        offsets,
                        sender,
                        policy=policy,
                        profile=profile,
                    )
                    completed_at_unix_s = time.time()
                    measured_samples, measurement_wall_time_s = measurement_window(
                        samples, warmup_requests=warmup_requests
                    )
                    metrics = summarize_samples(
                        measured_samples, wall_time_s=measurement_wall_time_s
                    )
                    full_completed = sum(sample.error is None for sample in samples)
                    full_errors = len(samples) - full_completed
                    record: dict[str, object] = {
                        **expected_identity,
                        "status": "complete",
                        "completed": full_completed,
                        "errors": full_errors,
                        "started_at_unix_s": started_at_unix_s,
                        "completed_at_unix_s": completed_at_unix_s,
                        "wall_time_s": wall_time_s,
                        "measurement_wall_time_s": measurement_wall_time_s,
                        "metrics": metrics,
                    }
                    record["fingerprint"] = subrun_fingerprint(record)
                    write_subrun_artifacts(runs_dir, record, samples)
                    runs.append(record)
                    repeat_metrics.append(metrics)
                    full_counts.append(
                        {"completed": full_completed, "errors": full_errors}
                    )
                scenario_results.append(
                    {
                        "scenario": asdict(scenario),
                        "load_pct": load_pct,
                        "profile": profile,
                        "runs": runs,
                        "aggregate": aggregate_repeats(repeat_metrics),
                        "completeness": assess_completeness(
                            full_counts, expected_requests=len(profile_workload)
                        ),
                    }
                )
    is_valid = all(bool(result["completeness"]["valid"]) for result in scenario_results)
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": is_valid,
        **gateway_manifest(args.endpoint),
        "policy": policy,
        "scheduler_cls": args.scheduler_cls,
        "model": args.model,
        "capacity_rps": args.capacity_rps,
        "workload_sha256": workload_sha256,
        "seed_derivation": "sha256(profile,load_pct,repeat)",
        "vllm_version": vllm_version,
        "repeats": args.repeats,
        "profiles": profiles,
        "warmup_requested": {
            "count": args.warmup_requests,
            "ratio": args.warmup_ratio,
        },
        "runs_dir": str(runs_dir),
        "scenarios": scenario_results,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        required=True,
        help="VeloxMesh OpenAI-compatible gateway endpoint; never direct vLLM",
    )
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
    parser.add_argument(
        "--scenario",
        action="append",
        choices=("steady", "burst"),
        help="Scenario family; repeat to select multiple. Defaults to the legacy matrix.",
    )
    parser.add_argument(
        "--load",
        action="append",
        type=int,
        choices=(40, 70, 90),
        help="Offered load percentage; repeat to select multiple.",
    )
    parser.add_argument(
        "--profile",
        action="append",
        choices=("id", "ood", "mixed"),
        help="Workload category profile; repeat to select multiple. Defaults to mixed.",
    )
    parser.add_argument("--repeats", type=int, default=REPEAT_COUNT)
    warmup = parser.add_mutually_exclusive_group()
    warmup.add_argument("--warmup-requests", type=int)
    warmup.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=600.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = asyncio.run(run_benchmark(args))
    _atomic_write_text(args.output, json.dumps(result, indent=2) + "\n")
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
