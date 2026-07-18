#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ltr_training.label_input import LabelInput  # noqa: E402
from ltr_training.offline_io import (  # noqa: E402
    read_json_records,
    sha256_file,
    write_json,
    write_jsonl,
)
from ltr_training.workload_builder import build_workload, manifest_split_ids  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build ID/OOD runner workload JSONL.")
    parser.add_argument("--id-input", type=Path)
    parser.add_argument("--id-manifest", type=Path)
    parser.add_argument("--id-split", default="test")
    parser.add_argument("--ood-input", type=Path)
    parser.add_argument("--lengths", required=True, type=Path)
    parser.add_argument("--profile", required=True, choices=("id", "ood", "mixed"))
    parser.add_argument("--per-token-ms", required=True, type=float)
    parser.add_argument("--ood-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    return parser


def _inputs(path: Path | None) -> list[LabelInput]:
    return (
        []
        if path is None
        else [LabelInput.from_dict(row) for row in read_json_records(path)]
    )


def main() -> int:
    args = _parser().parse_args()
    length_rows = read_json_records(args.lengths)
    lengths = {
        str(row.get("sample_id", row.get("request_id"))): int(
            row.get("output_length", row.get("true_length"))
        )
        for row in length_rows
    }
    id_inputs = _inputs(args.id_input)
    selected_id_count = len(id_inputs)
    if id_inputs:
        if args.id_manifest is None:
            raise ValueError("--id-manifest is required when --id-input is used")
        manifest_payload = json.loads(args.id_manifest.read_text(encoding="utf-8"))
        selected_ids = manifest_split_ids(manifest_payload, split=args.id_split)
        id_inputs = [item for item in id_inputs if item.sample_id in selected_ids]
        selected_id_count = len(id_inputs)
    rows, manifest = build_workload(
        id_inputs=id_inputs,
        ood_inputs=_inputs(args.ood_input),
        lengths=lengths,
        profile=args.profile,
        per_token_ms=args.per_token_ms,
        ood_ratio=args.ood_ratio,
        seed=args.seed,
    )
    write_jsonl(args.output, rows)
    manifest.update(
        {
            "output_path": str(args.output),
            "output_sha256": sha256_file(args.output),
            "lengths_path": str(args.lengths),
            "lengths_sha256": sha256_file(args.lengths),
            "input_sha256": {
                name: sha256_file(path)
                for name, path in (("id", args.id_input), ("ood", args.ood_input))
                if path is not None
            },
            "id_selection": {
                "split": args.id_split,
                "selected_count": selected_id_count,
                "manifest_path": str(args.id_manifest) if args.id_manifest else None,
                "manifest_sha256": sha256_file(args.id_manifest)
                if args.id_manifest
                else None,
            },
        }
    )
    write_json(args.manifest, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
