#!/usr/bin/env python3
"""Deterministically rebuild and verify the fixed 6,000-row Tier-2 sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


SOURCE_SHA256 = "6dc808aa8f76a5391d33c22ecb0ae2a2967d01c923c71ec85d84ec537e5f227b"
SAMPLE_SHA256 = "ee5a5889ca3d9bbee7790e7a408bd1664a285b6410b4fee54e45786d3eecb709"
SPLIT_COUNTS = {"train": 4_000, "validation": 1_000, "test": 1_000}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def choose_exact_sessions(
    sessions: list[tuple[str, list[dict[str, object]], int]], target: int
) -> tuple[
    list[tuple[str, list[dict[str, object]], int]],
    list[tuple[str, list[dict[str, object]], int]],
]:
    reachable = [False] * (target + 1)
    reachable[0] = True
    parent: list[tuple[int, int] | None] = [None] * (target + 1)
    for index, (_, rows, _) in enumerate(sessions):
        size = len(rows)
        if size > target:
            continue
        for total in range(target, size - 1, -1):
            if not reachable[total] and reachable[total - size]:
                reachable[total] = True
                parent[total] = (total - size, index)
        if reachable[target]:
            break
    if not reachable[target]:
        raise ValueError(f"cannot select exactly {target} rows at session boundaries")
    selected_indexes: set[int] = set()
    total = target
    while total:
        item = parent[total]
        if item is None:
            raise RuntimeError("invalid subset-sum parent chain")
        total, index = item
        selected_indexes.add(index)
    return (
        [row for index, row in enumerate(sessions) if index in selected_indexes],
        [row for index, row in enumerate(sessions) if index not in selected_indexes],
    )


def build_sample(rows: list[dict[str, object]], seed: int) -> tuple[list[dict[str, object]], dict[str, list[str]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["session_id"])].append(row)
    ranked = sorted(
        grouped.items(),
        key=lambda item: (
            sum(len(str(row.get("prompt", ""))) for row in item[1]) / len(item[1]),
            item[0],
        ),
    )
    buckets: list[list[tuple[str, list[dict[str, object]], int]]] = [
        [] for _ in range(4)
    ]
    for rank, (session_id, session_rows) in enumerate(ranked):
        bucket = min(3, rank * 4 // len(ranked))
        buckets[bucket].append((session_id, session_rows, bucket))
    rng = random.Random(seed)
    for bucket in buckets:
        rng.shuffle(bucket)
    candidates = []
    depth = 0
    while any(depth < len(bucket) for bucket in buckets):
        candidates.extend(bucket[depth] for bucket in buckets if depth < len(bucket))
        depth += 1

    assigned = {}
    remaining = candidates
    for split, count in SPLIT_COUNTS.items():
        assigned[split], remaining = choose_exact_sessions(remaining, count)

    sampled = []
    sample_ids: dict[str, list[str]] = {}
    for split, sessions in assigned.items():
        sample_ids[split] = []
        for _, session_rows, bucket in sessions:
            for row in sorted(
                session_rows, key=lambda value: int(value.get("invocation_index", 0))
            ):
                annotated = dict(row)
                annotated["tier2_split"] = split
                annotated["prompt_length_bucket"] = bucket
                sampled.append(annotated)
                sample_ids[split].append(str(row["sample_id"]))
    return sampled, sample_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--reference-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.seed != 42:
        raise ValueError("the pinned Tier-2 sample requires sampling_seed=42")
    actual_source_sha = sha256(args.source)
    if actual_source_sha != SOURCE_SHA256:
        raise ValueError(
            f"NO-GO: tier1 source sha256 {actual_source_sha} != {SOURCE_SHA256}"
        )
    reference = json.loads(args.reference_manifest.read_text(encoding="utf-8"))
    if reference.get("sampling_seed") != 42 or reference.get("sample_count") != 6_000:
        raise ValueError("NO-GO: reference manifest seed/count contract is invalid")
    sampled, sample_ids = build_sample(read_jsonl(args.source), args.seed)
    if sample_ids != reference.get("sample_ids"):
        raise ValueError("NO-GO: rebuilt sample IDs differ from the full reference manifest")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in sampled:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    actual_sample_sha = sha256(temporary)
    if actual_sample_sha != SAMPLE_SHA256:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"NO-GO: rebuilt sample sha256 {actual_sample_sha} != {SAMPLE_SHA256}"
        )
    temporary.replace(args.output)
    print(json.dumps({"sample_count": len(sampled), "sha256": actual_sample_sha}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
