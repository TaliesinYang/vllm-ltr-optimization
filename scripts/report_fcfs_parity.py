#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ltr_training.fcfs_parity import compare_benchmark_results  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report stock-FCFS vs custom-shim metric parity."
    )
    parser.add_argument("--stock", required=True, type=Path)
    parser.add_argument("--shim", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = compare_benchmark_results(
        json.loads(args.stock.read_text()),
        json.loads(args.shim.read_text()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f'within_tolerance={report["within_tolerance"]} output={args.output}',
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
