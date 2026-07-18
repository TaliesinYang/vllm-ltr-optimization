#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ltr_training.offline_io import (  # noqa: E402
    read_json_records,
    sha256_file,
    write_json,
    write_jsonl,
)
from ltr_training.ood_conversion import (  # noqa: E402
    convert_bfcl_row,
    convert_toolathlon_row,
    sample_label_inputs,
    summarize_conversion,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert pinned OOD sources to LabelInput JSONL."
    )
    parser.add_argument("--source", required=True, choices=("bfcl", "toolathlon"))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--function-docs", type=Path)
    parser.add_argument("--category")
    parser.add_argument("--sample-size", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--source-declarations",
        type=Path,
        default=ROOT / "configs" / "source-declarations.json",
    )
    return parser


def _bfcl_class_name(stem: str) -> str:
    return "".join(
        "API" if part == "api" else part.capitalize() for part in stem.split("_")
    )


def _function_index(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    index: dict[str, object] = {}
    files = sorted(path.rglob("*.json")) if path.is_dir() else [path]
    by_class: dict[str, list[dict[str, object]]] = {}
    for file_path in files:
        class_name = _bfcl_class_name(file_path.stem)
        by_class.setdefault(class_name, [])
        for row in read_json_records(file_path):
            candidates = row.get("function", row.get("tools", row))
            if isinstance(candidates, dict):
                candidates = [candidates]
            if not isinstance(candidates, list):
                continue
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                function = candidate.get("function", candidate)
                if isinstance(function, dict) and function.get("name"):
                    function = dict(function)
                    name = str(function["name"])
                    by_class[class_name].append(function)
                    index[name] = function
                    index[f"{class_name}.{name}"] = function
    index.update(by_class)
    return index


def main() -> int:
    args = _parser().parse_args()
    declarations = json.loads(args.source_declarations.read_text(encoding="utf-8"))
    declaration = declarations[args.source]
    revision = str(declaration["revision"])
    raw_rows = read_json_records(args.input)
    function_index = _function_index(args.function_docs)
    converted = []
    errors: list[dict[str, object]] = []
    for row_number, row in enumerate(raw_rows, start=1):
        try:
            category = args.category or (
                str(row.get("modelname_run", "default"))
                if args.source == "toolathlon"
                else args.input.stem.removeprefix("BFCL_v3_")
            )
            if args.source == "bfcl":
                item = convert_bfcl_row(
                    row,
                    revision=revision,
                    category=category,
                    function_index=function_index,
                )
            else:
                item = convert_toolathlon_row(row, revision=revision, category=category)
            converted.append(item)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append({"row_number": row_number, "error": str(exc)})
    sampled = sample_label_inputs(
        converted, sample_size=args.sample_size, seed=args.seed
    )
    write_jsonl(args.output, (item.to_dict() for item in sampled))
    manifest = summarize_conversion(sampled, sampling_seed=args.seed)
    manifest.update(
        {
            "source": args.source,
            "repository": declaration["repository"],
            "revision": revision,
            "input_path": str(args.input),
            "input_sha256": sha256_file(args.input),
            "input_row_count": len(raw_rows),
            "converted_before_sampling": len(converted),
            "conversion_error_count": len(errors),
            "conversion_errors": errors,
            "sample_size_requested": args.sample_size,
            "extraction_scope": "first-assistant-invocation-only",
            "output_path": str(args.output),
            "output_sha256": sha256_file(args.output),
        }
    )
    write_json(args.manifest, manifest)
    return 0 if sampled else 2


if __name__ == "__main__":
    raise SystemExit(main())
