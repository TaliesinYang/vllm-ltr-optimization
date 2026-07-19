#!/usr/bin/env python3
"""Deterministic file transforms and NO-GO checks for server workloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ltr_training.label_input import LabelInput  # noqa: E402
from ltr_training.offline_io import (  # noqa: E402
    read_json_records,
    sha256_file,
    write_jsonl,
)


def _required_file(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"required file missing: {path}")


def _label_rows(path: Path) -> list[dict[str, object]]:
    _required_file(path)
    rows = read_json_records(path)
    seen: set[str] = set()
    for row in rows:
        item = LabelInput.from_dict(row)
        if item.sample_id in seen:
            raise ValueError(f"duplicate sample_id in {path}: {item.sample_id}")
        seen.add(item.sample_id)
    return rows


def materialize_source(source: str, snapshot: Path, output: Path) -> dict[str, object]:
    if not snapshot.is_dir():
        raise ValueError(f"snapshot directory missing: {snapshot}")
    if source == "bfcl":
        candidates = sorted(snapshot.glob("BFCL_v3_*.json"))
    elif source == "toolathlon":
        candidates = sorted(snapshot.glob("*.jsonl"))
        if not candidates:
            candidates = sorted(snapshot.glob("*.json"))
    else:
        raise ValueError(f"unsupported source: {source}")
    if not candidates:
        raise ValueError(f"no {source} source data files found in {snapshot}")

    rows: list[dict[str, object]] = []
    for path in candidates:
        rows.extend(read_json_records(path))
    if not rows:
        raise ValueError(f"{source} snapshot produced zero source rows")
    write_jsonl(output, rows)
    return {
        "source": source,
        "source_files": [str(path.relative_to(snapshot)) for path in candidates],
        "row_count": len(rows),
        "output": str(output),
        "output_sha256": sha256_file(output),
    }


def _validate_conversion(
    *,
    source: str,
    inputs: Path,
    manifest_path: Path,
    expected_count: int,
) -> list[dict[str, object]]:
    _required_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError(f"conversion manifest must be an object: {manifest_path}")
    checks = {
        "source": source,
        "row_count": expected_count,
        "sample_size_requested": expected_count,
    }
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"conversion manifest {key} mismatch for {source}: "
                f"got {manifest.get(key)!r}, expected {expected!r}"
            )
    actual_sha = sha256_file(inputs)
    if manifest.get("output_sha256") != actual_sha:
        raise ValueError(f"conversion output SHA-256 mismatch for {source}")
    rows = _label_rows(inputs)
    if len(rows) != expected_count:
        raise ValueError(
            f"{source} label input count mismatch: got {len(rows)}, "
            f"expected {expected_count}"
        )
    if any(row.get("source") != source for row in rows):
        raise ValueError(f"{source} label input contains a foreign source row")
    return rows


def combine_label_inputs(
    *,
    bfcl_input: Path,
    bfcl_manifest: Path,
    toolathlon_input: Path,
    toolathlon_manifest: Path,
    expected_per_source: int,
    output: Path,
) -> dict[str, object]:
    if expected_per_source < 1:
        raise ValueError("expected source count must be positive")
    bfcl_rows = _validate_conversion(
        source="bfcl",
        inputs=bfcl_input,
        manifest_path=bfcl_manifest,
        expected_count=expected_per_source,
    )
    toolathlon_rows = _validate_conversion(
        source="toolathlon",
        inputs=toolathlon_input,
        manifest_path=toolathlon_manifest,
        expected_count=expected_per_source,
    )
    combined = bfcl_rows + toolathlon_rows
    ids = [str(row["sample_id"]) for row in combined]
    if len(ids) != len(set(ids)):
        raise ValueError("combined OOD label inputs contain duplicate sample_id values")
    write_jsonl(output, combined)
    return {
        "bfcl_rows": len(bfcl_rows),
        "toolathlon_rows": len(toolathlon_rows),
        "combined_rows": len(combined),
        "output": str(output),
        "output_sha256": sha256_file(output),
    }


def _latest_ledger(path: Path) -> dict[str, dict[str, object]]:
    _required_file(path)
    latest: dict[str, dict[str, object]] = {}
    for row in read_json_records(path):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"ledger row lacks sample_id: {path}")
        latest[sample_id] = row
    return latest


def merge_lengths(
    *,
    id_input: Path,
    id_ledger: Path,
    ood_input: Path,
    ood_ledger: Path,
    output: Path,
) -> dict[str, object]:
    sources: list[tuple[dict[str, object], dict[str, object]]] = []
    for input_path, ledger_path in (
        (id_input, id_ledger),
        (ood_input, ood_ledger),
    ):
        latest = _latest_ledger(ledger_path)
        for source_row in _label_rows(input_path):
            sample_id = str(source_row["sample_id"])
            ledger_row = latest.get(sample_id)
            if ledger_row is None or ledger_row.get("status") != "ok":
                raise ValueError(f"missing latest successful label for {sample_id}")
            length = ledger_row.get("output_length")
            if isinstance(length, bool) or not isinstance(length, int) or length < 1:
                raise ValueError(f"invalid output_length for {sample_id}: {length!r}")
            sources.append((source_row, ledger_row))

    ids = [str(source["sample_id"]) for source, _ in sources]
    if len(ids) != len(set(ids)):
        raise ValueError("ID and OOD inputs overlap on sample_id")
    rows = [
        {
            "sample_id": str(source["sample_id"]),
            "output_length": int(ledger["output_length"]),
            "source": str(source["source"]),
        }
        for source, ledger in sorted(sources, key=lambda item: str(item[0]["sample_id"]))
    ]
    write_jsonl(output, rows)
    return {
        "row_count": len(rows),
        "output": str(output),
        "output_sha256": sha256_file(output),
    }


def _verify_profile(
    workload: Path, manifest_path: Path, profile: str
) -> tuple[list[dict[str, object]], dict[str, object]]:
    _required_file(workload)
    _required_file(manifest_path)
    rows = read_json_records(workload)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"workload manifest must be an object: {manifest_path}")
    if manifest.get("schema_version") != "offline-workload-v2":
        raise ValueError(f"{profile} workload schema_version mismatch")
    if manifest.get("profile") != profile:
        raise ValueError(f"{profile} workload manifest profile mismatch")
    if manifest.get("request_count") != len(rows) or not rows:
        raise ValueError(f"{profile} workload row count mismatch or empty output")
    if manifest.get("output_sha256") != sha256_file(workload):
        raise ValueError(f"{profile} workload SHA-256 mismatch")

    seen: set[str] = set()
    for row in rows:
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or not request_id or request_id in seen:
            raise ValueError(f"{profile} workload has missing/duplicate request_id")
        seen.add(request_id)
        if not isinstance(row.get("prompt"), str):
            raise ValueError(f"{profile} workload prompt must be a string")
        if not isinstance(row.get("tool_schema"), str):
            raise ValueError(f"{profile} workload tool_schema must be a string")
        if not isinstance(row.get("history"), list):
            raise ValueError(f"{profile} workload history must be a list")
        if row.get("max_tokens") != 4096:
            raise ValueError(f"{profile} workload max_tokens must be 4096")
        service_ms = row.get("baseline_service_ms")
        if isinstance(service_ms, bool) or not isinstance(service_ms, (int, float)) or service_ms <= 0:
            raise ValueError(f"{profile} workload baseline_service_ms must be positive")
        category = row.get("category")
        if not isinstance(category, str) or ":" not in category:
            raise ValueError(f"{profile} workload category is invalid")
    return rows, manifest


def verify_workloads(
    *,
    mixed: Path,
    mixed_manifest: Path,
    ood: Path,
    ood_manifest: Path,
    ood_input: Path,
    lengths: Path,
) -> dict[str, object]:
    mixed_rows, _ = _verify_profile(mixed, mixed_manifest, "mixed")
    ood_rows, _ = _verify_profile(ood, ood_manifest, "ood")
    ood_ids = {str(row["sample_id"]) for row in _label_rows(ood_input)}
    length_ids = {
        str(row.get("sample_id"))
        for row in read_json_records(lengths)
        if row.get("sample_id") is not None
    }
    mixed_ids = {str(row["request_id"]) for row in mixed_rows}
    workload_ood_ids = {str(row["request_id"]) for row in ood_rows}
    if workload_ood_ids != ood_ids:
        raise ValueError("OOD workload IDs do not exactly match OOD label inputs")
    if not ood_ids.issubset(mixed_ids):
        raise ValueError("mixed workload does not contain every OOD label input")
    if not mixed_ids.issubset(length_ids) or not workload_ood_ids.issubset(length_ids):
        raise ValueError("workload contains request IDs missing from combined lengths")
    mixed_categories = {str(row["category"]).split(":", 1)[0] for row in mixed_rows}
    if mixed_categories != {"id", "ood"}:
        raise ValueError("mixed workload must contain both ID and OOD categories")
    if any(not str(row["category"]).startswith("ood:") for row in ood_rows):
        raise ValueError("OOD workload contains a non-OOD category")
    return {
        "status": "ok",
        "mixed_rows": len(mixed_rows),
        "ood_rows": len(ood_rows),
        "mixed_sha256": sha256_file(mixed),
        "mixed_manifest_sha256": sha256_file(mixed_manifest),
        "ood_sha256": sha256_file(ood),
        "ood_manifest_sha256": sha256_file(ood_manifest),
        "ood_input_sha256": sha256_file(ood_input),
        "lengths_sha256": sha256_file(lengths),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    materialize = subparsers.add_parser("materialize-source")
    materialize.add_argument("--source", required=True, choices=("bfcl", "toolathlon"))
    materialize.add_argument("--snapshot", required=True, type=Path)
    materialize.add_argument("--output", required=True, type=Path)

    combine = subparsers.add_parser("combine-label-inputs")
    combine.add_argument("--bfcl-input", required=True, type=Path)
    combine.add_argument("--bfcl-manifest", required=True, type=Path)
    combine.add_argument("--toolathlon-input", required=True, type=Path)
    combine.add_argument("--toolathlon-manifest", required=True, type=Path)
    combine.add_argument("--expected-per-source", required=True, type=int)
    combine.add_argument("--output", required=True, type=Path)

    merge = subparsers.add_parser("merge-lengths")
    merge.add_argument("--id-input", required=True, type=Path)
    merge.add_argument("--id-ledger", required=True, type=Path)
    merge.add_argument("--ood-input", required=True, type=Path)
    merge.add_argument("--ood-ledger", required=True, type=Path)
    merge.add_argument("--output", required=True, type=Path)

    verify = subparsers.add_parser("verify-workloads")
    verify.add_argument("--mixed", required=True, type=Path)
    verify.add_argument("--mixed-manifest", required=True, type=Path)
    verify.add_argument("--ood", required=True, type=Path)
    verify.add_argument("--ood-manifest", required=True, type=Path)
    verify.add_argument("--ood-input", required=True, type=Path)
    verify.add_argument("--lengths", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "materialize-source":
            report = materialize_source(args.source, args.snapshot, args.output)
        elif args.command == "combine-label-inputs":
            report = combine_label_inputs(
                bfcl_input=args.bfcl_input,
                bfcl_manifest=args.bfcl_manifest,
                toolathlon_input=args.toolathlon_input,
                toolathlon_manifest=args.toolathlon_manifest,
                expected_per_source=args.expected_per_source,
                output=args.output,
            )
        elif args.command == "merge-lengths":
            report = merge_lengths(
                id_input=args.id_input,
                id_ledger=args.id_ledger,
                ood_input=args.ood_input,
                ood_ledger=args.ood_ledger,
                output=args.output,
            )
        else:
            report = verify_workloads(
                mixed=args.mixed,
                mixed_manifest=args.mixed_manifest,
                ood=args.ood,
                ood_manifest=args.ood_manifest,
                ood_input=args.ood_input,
                lengths=args.lengths,
            )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"NO-GO: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
