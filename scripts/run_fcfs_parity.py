#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ltr_training.fcfs_parity import compare_benchmark_results  # noqa: E402
from ltr_training.fcfs_replay import benchmark_endpoint, load_workload  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay one workload against stock FCFS and custom FCFS shim."
    )
    parser.add_argument("--stock-endpoint", required=True)
    parser.add_argument("--shim-endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--capacity-rps", required=True, type=float)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--api-key")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    workload = load_workload(args.workload)
    common = {
        "model": args.model,
        "workload": workload,
        "capacity_rps": args.capacity_rps,
        "seed": args.seed,
        "timeout_s": args.timeout_s,
        "api_key": args.api_key,
    }
    stock = await benchmark_endpoint(endpoint=args.stock_endpoint, **common)
    shim = await benchmark_endpoint(endpoint=args.shim_endpoint, **common)
    report = compare_benchmark_results(stock, shim)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "stock.json").write_text(json.dumps(stock, indent=2) + "\n")
    (args.output_dir / "shim.json").write_text(json.dumps(shim, indent=2) + "\n")
    (args.output_dir / "parity.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        f'within_tolerance={report["within_tolerance"]} output={args.output_dir}',
        flush=True,
    )


def main() -> int:
    asyncio.run(run(parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
