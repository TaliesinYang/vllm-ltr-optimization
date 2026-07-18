#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ltr_training.offline_baselines import LEGACY_FAMILIES, legacy_loader_status  # noqa: E402
from ltr_training.offline_io import write_json  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe optional P2 legacy zero-shot predictor families."
    )
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    families = [
        legacy_loader_status(name, args.checkpoint_root / name)
        for name in LEGACY_FAMILIES
    ]
    write_json(
        args.output,
        {
            "schema_version": "legacy-zero-shot-v1",
            "priority": "P2_optional",
            "r1_blocking": False,
            "status": "blocked"
            if all(row["status"] == "blocked" for row in families)
            else "partial",
            "score_contract": "all outputs transformed to longer-is-higher",
            "families": families,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
