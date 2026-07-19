#!/usr/bin/env python3
"""Deterministic file transforms and NO-GO checks for server workloads."""

from __future__ import annotations

import argparse
import json
import random
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
    write_json,
    write_jsonl,
)
from ltr_training.workload_builder import manifest_split_ids  # noqa: E402


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


def _deterministic_sample(
    rows: list[dict[str, object]], *, target: int, seed: int, pool_name: str
) -> list[dict[str, object]]:
    if target < 1:
        raise ValueError(f"{pool_name} target must be positive")
    if target > len(rows):
        raise ValueError(
            f"{pool_name} target exceeds pool: target={target}, pool={len(rows)}"
        )
    ordered = sorted(rows, key=lambda row: str(row["sample_id"]))
    random.Random(seed).shuffle(ordered)
    return ordered[:target]


def select_workload_inputs(
    *,
    id_input: Path,
    id_manifest: Path,
    id_split: str,
    ood_input: Path,
    expected_id_pool_size: int | None,
    expected_ood_pool_size: int | None,
    mixed_id_target: int,
    mixed_ood_target: int,
    ood_target: int,
    seed: int,
    mixed_id_output: Path,
    mixed_ood_output: Path,
    ood_output: Path,
    manifest_path: Path,
) -> dict[str, object]:
    _required_file(id_manifest)
    manifest_payload = json.loads(id_manifest.read_text(encoding="utf-8"))
    split_ids = manifest_split_ids(manifest_payload, split=id_split)
    if not split_ids:
        raise ValueError(f"ID manifest split {id_split!r} contains no sample IDs")
    id_rows_by_id = {str(row["sample_id"]): row for row in _label_rows(id_input)}
    missing_ids = sorted(split_ids - set(id_rows_by_id))
    if missing_ids:
        raise ValueError(
            f"ID input is missing {len(missing_ids)} rows declared by split {id_split!r}"
        )
    id_pool = [id_rows_by_id[sample_id] for sample_id in split_ids]
    ood_pool = _label_rows(ood_input)
    for name, actual, expected in (
        ("ID test", len(id_pool), expected_id_pool_size),
        ("OOD", len(ood_pool), expected_ood_pool_size),
    ):
        if expected is not None and actual != expected:
            raise ValueError(
                f"{name} pool size mismatch: got {actual}, expected {expected}"
            )
    mixed_ids = _deterministic_sample(
        id_pool, target=mixed_id_target, seed=seed, pool_name="mixed ID"
    )
    shuffled_ood = _deterministic_sample(
        ood_pool, target=len(ood_pool), seed=seed, pool_name="OOD"
    )
    if mixed_ood_target > len(shuffled_ood) or ood_target > len(shuffled_ood):
        raise ValueError(
            "OOD workload target exceeds pool: "
            f"mixed={mixed_ood_target}, ood={ood_target}, pool={len(shuffled_ood)}"
        )
    if mixed_ood_target < 1 or ood_target < 1:
        raise ValueError("OOD workload targets must be positive")
    mixed_oods = shuffled_ood[:mixed_ood_target]
    oods = shuffled_ood[:ood_target]
    for path, rows in (
        (mixed_id_output, mixed_ids),
        (mixed_ood_output, mixed_oods),
        (ood_output, oods),
    ):
        write_jsonl(path, rows)
    manifest = {
        "schema_version": "server-workload-selection-v1",
        "sampling_seed": seed,
        "id_split": id_split,
        "pool_sizes": {"id_test": len(id_pool), "ood": len(ood_pool)},
        "selected_counts": {
            "mixed_id": len(mixed_ids),
            "mixed_ood": len(mixed_oods),
            "ood": len(oods),
        },
        "inputs": {
            "id": {"path": str(id_input), "sha256": sha256_file(id_input)},
            "id_manifest": {
                "path": str(id_manifest),
                "sha256": sha256_file(id_manifest),
            },
            "ood": {"path": str(ood_input), "sha256": sha256_file(ood_input)},
        },
        "outputs": {
            "mixed_id": {
                "path": str(mixed_id_output),
                "sha256": sha256_file(mixed_id_output),
            },
            "mixed_ood": {
                "path": str(mixed_ood_output),
                "sha256": sha256_file(mixed_ood_output),
            },
            "ood": {"path": str(ood_output), "sha256": sha256_file(ood_output)},
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def _selection_payload(
    path: Path,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    _required_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("workload selection manifest must be an object")
    if payload.get("schema_version") != "server-workload-selection-v1":
        raise ValueError("workload selection manifest schema_version mismatch")
    seed = payload.get("sampling_seed")
    pools = payload.get("pool_sizes")
    counts = payload.get("selected_counts")
    inputs = payload.get("inputs")
    outputs = payload.get("outputs")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("workload selection seed must be an integer")
    if not isinstance(pools, dict) or not isinstance(counts, dict):
        raise ValueError("workload selection manifest lacks pool/count mappings")
    if not isinstance(outputs, dict):
        raise ValueError("workload selection manifest lacks outputs")
    if not isinstance(inputs, dict):
        raise ValueError("workload selection manifest lacks inputs")
    for name in ("id_test", "ood"):
        value = pools.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"invalid workload pool size: {name}")

    input_paths: dict[str, Path] = {}
    for name in ("id", "id_manifest", "ood"):
        source = inputs.get(name)
        if not isinstance(source, dict):
            raise ValueError(f"selection manifest input missing: {name}")
        source_path = Path(str(source.get("path", "")))
        _required_file(source_path)
        if source.get("sha256") != sha256_file(source_path):
            raise ValueError(f"selection input SHA-256 mismatch: {name}")
        input_paths[name] = source_path
    id_manifest_payload = json.loads(
        input_paths["id_manifest"].read_text(encoding="utf-8")
    )
    id_split = str(payload.get("id_split", ""))
    id_split_ids = manifest_split_ids(id_manifest_payload, split=id_split)
    id_input_ids = {
        str(row["sample_id"]) for row in _label_rows(input_paths["id"])
    }
    if not id_split_ids or not id_split_ids.issubset(id_input_ids):
        raise ValueError("selection ID pool no longer matches its declared split")
    if pools.get("id_test") != len(id_split_ids):
        raise ValueError("selection ID pool size does not match its inputs")
    if pools.get("ood") != len(_label_rows(input_paths["ood"])):
        raise ValueError("selection OOD pool size does not match its input")

    selected_rows: dict[str, list[dict[str, object]]] = {}
    for name in ("mixed_id", "mixed_ood", "ood"):
        output = outputs.get(name)
        if not isinstance(output, dict):
            raise ValueError(f"selection manifest output missing: {name}")
        output_path = Path(str(output.get("path", "")))
        rows = _label_rows(output_path)
        if output.get("sha256") != sha256_file(output_path):
            raise ValueError(f"selection output SHA-256 mismatch: {name}")
        if counts.get(name) != len(rows):
            raise ValueError(f"selection output row count mismatch: {name}")
        selected_rows[name] = rows
    return (
        payload,
        selected_rows["mixed_id"],
        selected_rows["mixed_ood"],
        selected_rows["ood"],
    )


def _subsampling_provenance(
    selection: dict[str, object], *, profile: str, selection_manifest: Path
) -> dict[str, object]:
    counts = selection["selected_counts"]
    if not isinstance(counts, dict):
        raise ValueError("selection counts must be an object")
    selected_counts = (
        {"id": counts["mixed_id"], "ood": counts["mixed_ood"]}
        if profile == "mixed"
        else {"ood": counts["ood"]}
    )
    return {
        "sampling_seed": selection["sampling_seed"],
        "pool_sizes": selection["pool_sizes"],
        "selected_counts": selected_counts,
        "selection_manifest_path": str(selection_manifest),
        "selection_manifest_sha256": sha256_file(selection_manifest),
    }


def annotate_workload_manifests(
    *, selection_manifest: Path, mixed_manifest: Path, ood_manifest: Path
) -> dict[str, object]:
    selection, _, _, _ = _selection_payload(selection_manifest)
    for profile, path in (("mixed", mixed_manifest), ("ood", ood_manifest)):
        _required_file(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("profile") != profile:
            raise ValueError(f"cannot annotate invalid {profile} workload manifest")
        manifest["workload_subsampling"] = _subsampling_provenance(
            selection, profile=profile, selection_manifest=selection_manifest
        )
        write_json(path, manifest)
    return {
        "mixed_manifest_sha256": sha256_file(mixed_manifest),
        "ood_manifest_sha256": sha256_file(ood_manifest),
        "selection_manifest_sha256": sha256_file(selection_manifest),
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
    structural_exclusions: frozenset[str] = frozenset(),
    min_ood_labelable: int = 0,
) -> dict[str, object]:
    sources: list[tuple[dict[str, object], dict[str, object]]] = []
    excluded: list[str] = []
    ood_dropped: list[str] = []
    # ID side: fixed 6000 contract minus the DECLARED structural exclusions.
    # OOD side: a sampled pool — rows that fail labeling (context-length under
    # the frozen protocol) are dropped and counted; downstream sampling draws
    # from whatever remains, provided enough survive.
    for input_path, ledger_path, is_ood in (
        (id_input, id_ledger, False),
        (ood_input, ood_ledger, True),
    ):
        latest = _latest_ledger(ledger_path)
        for source_row in _label_rows(input_path):
            sample_id = str(source_row["sample_id"])
            ledger_row = latest.get(sample_id)
            if ledger_row is None or ledger_row.get("status") != "ok":
                if is_ood:
                    ood_dropped.append(sample_id)
                    continue
                if sample_id in structural_exclusions:
                    excluded.append(sample_id)
                    continue
                raise ValueError(f"missing latest successful label for {sample_id}")
            length = ledger_row.get("output_length")
            if isinstance(length, bool) or not isinstance(length, int) or length < 1:
                raise ValueError(f"invalid output_length for {sample_id}: {length!r}")
            sources.append((source_row, ledger_row))

    if set(excluded) != set(structural_exclusions):
        missing = sorted(set(structural_exclusions) - set(excluded))
        raise ValueError(
            f"declared structural exclusions not all absent from labels: {missing[:10]}"
        )
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
    ood_total = sum(1 for _ in _label_rows(ood_input))
    ood_labelable = ood_total - len(ood_dropped)
    if ood_labelable < min_ood_labelable:
        raise ValueError(
            f"NO-GO: only {ood_labelable} OOD rows labelable, need >= "
            f"{min_ood_labelable}; a systematic labeling failure may be masked"
        )
    return {
        "row_count": len(rows),
        "ood_pool_total": ood_total,
        "ood_pool_labelable": ood_labelable,
        "ood_dropped_unlabelable": len(ood_dropped),
        "id_structural_exclusions": len(excluded),
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
    selection_manifest: Path,
    lengths: Path,
) -> dict[str, object]:
    mixed_rows, mixed_payload = _verify_profile(mixed, mixed_manifest, "mixed")
    ood_rows, ood_payload = _verify_profile(ood, ood_manifest, "ood")
    selection, mixed_id_rows, mixed_ood_rows, selected_ood_rows = (
        _selection_payload(selection_manifest)
    )
    expected_mixed_provenance = _subsampling_provenance(
        selection, profile="mixed", selection_manifest=selection_manifest
    )
    expected_ood_provenance = _subsampling_provenance(
        selection, profile="ood", selection_manifest=selection_manifest
    )
    if mixed_payload.get("workload_subsampling") != expected_mixed_provenance:
        raise ValueError("mixed workload subsampling provenance mismatch")
    if ood_payload.get("workload_subsampling") != expected_ood_provenance:
        raise ValueError("OOD workload subsampling provenance mismatch")
    expected_mixed_ids = {
        str(row["sample_id"]) for row in mixed_id_rows + mixed_ood_rows
    }
    expected_ood_ids = {str(row["sample_id"]) for row in selected_ood_rows}
    length_ids = {
        str(row.get("sample_id"))
        for row in read_json_records(lengths)
        if row.get("sample_id") is not None
    }
    mixed_ids = {str(row["request_id"]) for row in mixed_rows}
    workload_ood_ids = {str(row["request_id"]) for row in ood_rows}
    if workload_ood_ids != expected_ood_ids:
        raise ValueError("OOD workload IDs do not exactly match its selected inputs")
    if mixed_ids != expected_mixed_ids:
        raise ValueError("mixed workload IDs do not exactly match its selected inputs")
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
        "selection_manifest_sha256": sha256_file(selection_manifest),
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

    select = subparsers.add_parser("select-workload-inputs")
    select.add_argument("--id-input", required=True, type=Path)
    select.add_argument("--id-manifest", required=True, type=Path)
    select.add_argument("--id-split", default="test")
    select.add_argument("--ood-input", required=True, type=Path)
    select.add_argument("--expected-id-pool-size", type=int)
    select.add_argument("--expected-ood-pool-size", type=int)
    select.add_argument("--mixed-id-target", required=True, type=int)
    select.add_argument("--mixed-ood-target", required=True, type=int)
    select.add_argument("--ood-target", required=True, type=int)
    select.add_argument("--seed", required=True, type=int)
    select.add_argument("--mixed-id-output", required=True, type=Path)
    select.add_argument("--mixed-ood-output", required=True, type=Path)
    select.add_argument("--ood-output", required=True, type=Path)
    select.add_argument("--manifest", required=True, type=Path)

    annotate = subparsers.add_parser("annotate-workload-manifests")
    annotate.add_argument("--selection-manifest", required=True, type=Path)
    annotate.add_argument("--mixed-manifest", required=True, type=Path)
    annotate.add_argument("--ood-manifest", required=True, type=Path)

    merge = subparsers.add_parser("merge-lengths")
    merge.add_argument("--id-input", required=True, type=Path)
    merge.add_argument("--id-ledger", required=True, type=Path)
    merge.add_argument("--ood-input", required=True, type=Path)
    merge.add_argument("--ood-ledger", required=True, type=Path)
    merge.add_argument("--output", required=True, type=Path)
    merge.add_argument("--structural-exclusions", type=Path)
    merge.add_argument("--min-ood-labelable", type=int, default=0)

    verify = subparsers.add_parser("verify-workloads")
    verify.add_argument("--mixed", required=True, type=Path)
    verify.add_argument("--mixed-manifest", required=True, type=Path)
    verify.add_argument("--ood", required=True, type=Path)
    verify.add_argument("--ood-manifest", required=True, type=Path)
    verify.add_argument("--selection-manifest", required=True, type=Path)
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
        elif args.command == "select-workload-inputs":
            report = select_workload_inputs(
                id_input=args.id_input,
                id_manifest=args.id_manifest,
                id_split=args.id_split,
                ood_input=args.ood_input,
                expected_id_pool_size=args.expected_id_pool_size,
                expected_ood_pool_size=args.expected_ood_pool_size,
                mixed_id_target=args.mixed_id_target,
                mixed_ood_target=args.mixed_ood_target,
                ood_target=args.ood_target,
                seed=args.seed,
                mixed_id_output=args.mixed_id_output,
                mixed_ood_output=args.mixed_ood_output,
                ood_output=args.ood_output,
                manifest_path=args.manifest,
            )
        elif args.command == "annotate-workload-manifests":
            report = annotate_workload_manifests(
                selection_manifest=args.selection_manifest,
                mixed_manifest=args.mixed_manifest,
                ood_manifest=args.ood_manifest,
            )
        elif args.command == "merge-lengths":
            exclusions: frozenset[str] = frozenset()
            if args.structural_exclusions is not None:
                entries = json.loads(
                    args.structural_exclusions.read_text(encoding="utf-8")
                )
                exclusions = frozenset(str(e["sample_id"]) for e in entries)
            report = merge_lengths(
                id_input=args.id_input,
                id_ledger=args.id_ledger,
                ood_input=args.ood_input,
                ood_ledger=args.ood_ledger,
                output=args.output,
                structural_exclusions=exclusions,
                min_ood_labelable=args.min_ood_labelable,
            )
        else:
            report = verify_workloads(
                mixed=args.mixed,
                mixed_manifest=args.mixed_manifest,
                ood=args.ood,
                ood_manifest=args.ood_manifest,
                selection_manifest=args.selection_manifest,
                lengths=args.lengths,
            )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"NO-GO: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
