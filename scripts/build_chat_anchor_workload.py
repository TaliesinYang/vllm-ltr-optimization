#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ltr_training.chat_anchor import (  # noqa: E402
    build_sharegpt_anchor_workload,
    convert_sharegpt_row,
)
from ltr_training.offline_io import (  # noqa: E402
    read_json_records,
    sha256_file,
    write_json,
    write_jsonl,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a ShareGPT chat anchor workload."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--lengths", required=True, type=Path)
    parser.add_argument("--per-token-ms", required=True, type=float)
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    items = [
        convert_sharegpt_row(row, revision=args.source_revision)
        for row in read_json_records(args.input)
    ]
    lengths = {
        str(row.get("sample_id", row.get("request_id"))): int(
            row.get("output_length", row.get("true_length"))
        )
        for row in read_json_records(args.lengths)
    }
    rows, manifest = build_sharegpt_anchor_workload(
        items,
        lengths=lengths,
        per_token_ms=args.per_token_ms,
        seed=args.seed,
        sample_size=args.sample_size,
    )
    write_jsonl(args.output, rows)
    manifest.update(
        {
            "input_path": str(args.input),
            "input_sha256": sha256_file(args.input),
            "lengths_sha256": sha256_file(args.lengths),
            "output_path": str(args.output),
            "output_sha256": sha256_file(args.output),
        }
    )
    write_json(args.manifest, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
