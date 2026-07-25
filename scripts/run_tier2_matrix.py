#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ltr_training.tier2_training import (  # noqa: E402
    run_nested_learning_curve,
    run_tier2_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Tier-2 9+1 matrix and nested curve.")
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--tier1-results-dir", type=Path, required=True)
    parser.add_argument("--tier1-config-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix = run_tier2_matrix(
        sample_path=args.sample,
        ledger_path=args.ledger,
        tier1_results_dir=args.tier1_results_dir,
        tier1_config_dir=args.tier1_config_dir,
        work_dir=args.work_dir / "matrix",
        results_dir=args.results_dir,
    )
    curve = run_nested_learning_curve(
        sample_path=args.sample,
        ledger_path=args.ledger,
        tier1_checkpoint=(
            args.tier1_results_dir / "bert-full_context-tier1-seed42" / "final"
        ),
        tier1_config_path=(
            args.tier1_config_dir / "bert-full_context-tier1-seed42.json"
        ),
        work_dir=args.work_dir / "learning-curve",
        results_dir=args.results_dir / "tier2-learning-curve-runs",
        summary_path=args.results_dir / "tier2-learning-curve.json",
    )
    print(json.dumps({
        "matrix_completed_runs": matrix["completed_runs"],
        "learning_curve_completed_points": curve["completed_points"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
