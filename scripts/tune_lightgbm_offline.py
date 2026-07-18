#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ltr_training.offline_baselines import run_lightgbm_grid  # noqa: E402
from ltr_training.offline_io import read_json_records, sha256_file, write_json  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune the 20-config LightGBM baseline on validation."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report_path = args.output_dir / "lightgbm-search-report.json"
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        write_json(
            report_path,
            {
                "schema_version": "lightgbm-search-v1",
                "status": "blocked",
                "reason": "lightgbm_not_installed",
                "detail": str(exc),
                "grid_size": 20,
                "input_path": str(args.input),
            },
        )
        return 2

    rows = read_json_records(args.input)
    splits: dict[str, list[dict[str, object]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    aliases = {
        "val": "validation",
        "validation": "validation",
        "train": "train",
        "test": "test",
    }
    for row in rows:
        split = aliases.get(str(row.get("split", row.get("tier2_split", ""))))
        if split is None:
            raise ValueError(
                "each row must have split/tier2_split in train, val, validation, test"
            )
        copied = dict(row)
        copied["true_length"] = int(row.get("true_length", row.get("output_length")))
        splits[split].append(copied)
    report, model = run_lightgbm_grid(
        splits, model_factory=LGBMRegressor, seed=args.seed
    )
    model_path = args.output_dir / "lightgbm-model.txt"
    model.booster_.save_model(str(model_path))
    report.update(
        {
            "input_path": str(args.input),
            "input_sha256": sha256_file(args.input),
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
        }
    )
    write_json(report_path, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
