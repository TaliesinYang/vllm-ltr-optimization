#!/usr/bin/env python3
"""Build the replay sidecar and uncalibrated empirical rank lookup."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scheduler_benchmark.rank_quantiles import (  # noqa: E402
    APPROXIMATION_NOTICE,
    build_rank_quantile_artifacts,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sidecar-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--expected-count", type=int, default=6_000)
    parser.add_argument("--structural-exclusions", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    exclusions: tuple = ()
    if args.structural_exclusions is not None:
        loaded = json.loads(args.structural_exclusions.read_text(encoding="utf-8"))
        exclusions = tuple(loaded)
    manifest = build_rank_quantile_artifacts(
        labels_path=args.labels,
        checkpoint=args.checkpoint,
        sidecar_path=args.sidecar_output,
        manifest_path=args.manifest_output,
        model_version=args.model_version,
        expected_count=args.expected_count,
        structural_exclusions=exclusions,
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.manifest_output),
                "sidecar_path": str(args.sidecar_output),
                "sample_count": manifest["sample_count"],
                "mapping_version": manifest["mapping_version"],
                "approximation_notice": APPROXIMATION_NOTICE,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
