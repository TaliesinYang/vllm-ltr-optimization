#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ltr_training.tier2 import replay_labels, summarize_results  # noqa: E402


PINNED_MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay ToolACE requests against local vLLM.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", default="qwen3.5-9b-tier2")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--capture-text", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    rows = replay_labels(
        labels_path=args.labels,
        ledger_path=args.ledger,
        endpoint=args.endpoint,
        model=args.model,
        model_revision=PINNED_MODEL_REVISION,
        max_tokens=args.max_tokens,
        limit=args.limit,
        concurrency=args.concurrency,
        capture_text=args.capture_text,
    )
    wall_elapsed = time.monotonic() - started
    expected = args.limit if args.limit is not None else sum(1 for _ in args.labels.open())
    selected = rows[:expected]
    report = summarize_results(
        selected, expected_count=expected, wall_elapsed_seconds=wall_elapsed
    )
    report["d4_passed"] = report["failure_rate"] <= 0.01
    report["model_revision"] = PINNED_MODEL_REVISION
    report["temperature"] = 0
    report["max_tokens"] = args.max_tokens
    report["concurrency"] = args.concurrency
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
