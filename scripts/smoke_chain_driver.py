#!/usr/bin/env python3
"""Drive the runner internals through a live gateway without the vLLM guard."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import aiohttp


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scheduler_benchmark.runner import (  # noqa: E402
    load_workload,
    make_chat_payload,
    run_replay,
    stream_completion,
)


DEFAULT_WORKLOAD = REPO_ROOT / "runs/workloads-v2/workload-id.v2.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--arrival-interval-s", type=float, default=0.25)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    return parser.parse_args()


async def run(args: argparse.Namespace) -> tuple[dict[str, object], bool]:
    if args.count < 1:
        raise ValueError("--count must be positive")
    if args.arrival_interval_s < 0.0:
        raise ValueError("--arrival-interval-s must be non-negative")

    workload = load_workload(args.workload)[: args.count]
    if len(workload) != args.count:
        raise ValueError(
            f"workload contains {len(workload)} requests, expected {args.count}"
        )

    # Fail before network activity if the Tier-2-consistent payload cannot be built.
    for request in workload:
        make_chat_payload(request, model=args.model)

    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def sender(request):
            return await stream_completion(
                session,
                args.endpoint,
                args.model,
                request,
                args.api_key,
            )

        offsets = [
            index * args.arrival_interval_s for index in range(len(workload))
        ]
        samples, wall_time_s = await run_replay(
            workload,
            offsets,
            sender,
            policy="smoke_chain",
            profile="id",
        )

    failures = [
        {"request_id": sample.request_id, "error": sample.error}
        for sample in samples
        if sample.error is not None
    ]
    completed = sum(sample.error is None for sample in samples)
    usage_tokens_nonzero_count = sum(
        sample.error is None and sample.output_tokens > 0 for sample in samples
    )
    summary: dict[str, object] = {
        "requested": args.count,
        "completed": completed,
        "errors": failures,
        "error_count": len(failures),
        "usage_tokens_nonzero_count": usage_tokens_nonzero_count,
        "usage_tokens_total": sum(sample.output_tokens for sample in samples),
        "wall_time_s": round(wall_time_s, 6),
    }
    passed = (
        completed == args.count
        and not failures
        and usage_tokens_nonzero_count == args.count
    )
    return summary, passed


def main() -> int:
    args = parse_args()
    summary, passed = asyncio.run(run(args))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
