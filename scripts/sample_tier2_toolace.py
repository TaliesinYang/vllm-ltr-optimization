#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ltr_training.tier2 import read_jsonl  # noqa: E402
from ltr_training.tier2_sampling import build_stratified_splits  # noqa: E402


STOPPING_CRITERION = (
    "2K→4K 若 val tau 提升 ≤0.01 且下游 utility 在 bootstrap CI 内, 判定 4K 饱和"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the fixed 6,000-row Tier-2 ToolACE sample.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    sampled, manifest = build_stratified_splits(
        read_jsonl(args.source),
        seed=args.seed,
        split_counts={"train": 4000, "validation": 1000, "test": 1000},
        stopping_criterion=STOPPING_CRITERION,
    )
    args.sample.parent.mkdir(parents=True, exist_ok=True)
    with args.sample.open("w", encoding="utf-8") as handle:
        for row in sampled:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest.update(
        {
            "source_path": str(args.source),
            "source_sha256": sha256(args.source),
            "sample_path": str(args.sample),
            "sample_sha256": sha256(args.sample),
        }
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: manifest[key] for key in (
        "sample_count", "split_counts", "session_counts", "sampling_seed",
        "sample_sha256", "stopping_criterion"
    )}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
