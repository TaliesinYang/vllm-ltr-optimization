#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ltr_training.training_matrix import run_matrix  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Tier-1 9+1 predictor matrix.")
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_matrix(
        base_config=json.loads(args.base_config.read_text()),
        labels_path=args.labels,
        config_dir=args.config_dir,
        work_dir=args.work_dir,
        results_dir=args.results_dir,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
