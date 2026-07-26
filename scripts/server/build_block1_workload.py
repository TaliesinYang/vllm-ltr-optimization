#!/usr/bin/env python3
"""Emit the Block-1 workload: trace-calibrated synthetic traffic + real traces.

Calibration is measured from the captured agent trace at build time; nothing
distributional is passed on the command line. What IS a command-line choice is
the shape of the synthetic tenancy (how many clients, how many requests), which
the trace cannot supply because it was a single client.

    build_block1_workload.py \
        --trace probes/agent-traces-2026-07-26/agent_trace_vanilla.jsonl.gz \
        --sample <tier2 sample>.jsonl --ledger <tier2 ledger>.jsonl \
        --output runs/block1/workload-block1.jsonl \
        --manifest runs/block1/workload-block1-manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ltr_training.block1_workload import (  # noqa: E402
    DEFAULT_PER_TOKEN_MS,
    build_clients,
    build_manifest,
    generate_requests,
    measure_trace,
    trace_rows,
)
from ltr_training.tier2_training import load_tier2_split_examples  # noqa: E402
from scheduler_benchmark.tool_vocabulary import tool_names  # noqa: E402

DEFAULT_TRACE = (
    REPO_ROOT / "probes" / "agent-traces-2026-07-26" / "agent_trace_vanilla.jsonl.gz"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clients-per-stratum", type=int, default=8)
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--per-token-ms",
        type=float,
        default=DEFAULT_PER_TOKEN_MS,
        help="Proxy service time per output token, matching the legacy "
        "workload builder so slowdown stays comparable across workloads.",
    )
    parser.add_argument(
        "--no-trace-rows",
        action="store_true",
        help="Emit synthetic traffic only, without the 75 real trace requests.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    calibration = measure_trace(args.trace)
    splits, _ = load_tier2_split_examples(
        sample_path=args.sample, ledger_path=args.ledger
    )
    clients = build_clients(
        train_examples=splits["train"],
        per_stratum=args.clients_per_stratum,
        seed=args.seed,
        tool_names_of=tool_names,
    )
    synthetic = generate_requests(
        calibration=calibration,
        clients=clients,
        request_count=args.requests,
        seed=args.seed,
        per_token_ms=args.per_token_ms,
    )
    traces = (
        []
        if args.no_trace_rows
        else trace_rows(calibration, per_token_ms=args.per_token_ms)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in [*synthetic, *traces]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    manifest = build_manifest(
        calibration=calibration,
        clients=clients,
        synthetic=synthetic,
        traces=traces,
        seed=args.seed,
        per_token_ms=args.per_token_ms,
    )
    manifest["output"] = str(args.output)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
