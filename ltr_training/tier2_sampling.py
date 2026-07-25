from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable


def _prompt_length(row: dict[str, object]) -> int:
    return len(str(row.get("prompt", "")))


def _choose_exact_sessions(
    sessions: list[tuple[str, list[dict[str, object]], int]], target: int
) -> tuple[list[tuple[str, list[dict[str, object]], int]], list[tuple[str, list[dict[str, object]], int]]]:
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
        raise ValueError(f"cannot create a session-boundary split of exactly {target} rows")

    selected_indexes: set[int] = set()
    total = target
    while total:
        previous, index = parent[total] or (-1, -1)
        if index < 0:
            raise RuntimeError("invalid subset-sum parent chain")
        selected_indexes.add(index)
        total = previous
    selected = [session for index, session in enumerate(sessions) if index in selected_indexes]
    remaining = [session for index, session in enumerate(sessions) if index not in selected_indexes]
    return selected, remaining


def build_stratified_splits(
    rows: Iterable[dict[str, object]],
    *,
    seed: int,
    split_counts: dict[str, int],
    stopping_criterion: str,
    length_bucket_count: int = 4,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if length_bucket_count < 1:
        raise ValueError("length_bucket_count must be at least 1")
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["session_id"])].append(dict(row))
    if sum(split_counts.values()) > sum(len(items) for items in grouped.values()):
        raise ValueError("requested sample is larger than source data")

    ranked = sorted(
        grouped.items(),
        key=lambda item: (
            sum(_prompt_length(row) for row in item[1]) / len(item[1]),
            item[0],
        ),
    )
    bucketed: list[list[tuple[str, list[dict[str, object]], int]]] = [
        [] for _ in range(length_bucket_count)
    ]
    for rank, (session_id, session_rows) in enumerate(ranked):
        bucket = min(length_bucket_count - 1, rank * length_bucket_count // len(ranked))
        bucketed[bucket].append((session_id, session_rows, bucket))

    rng = random.Random(seed)
    for bucket in bucketed:
        rng.shuffle(bucket)
    candidates: list[tuple[str, list[dict[str, object]], int]] = []
    depth = 0
    while any(depth < len(bucket) for bucket in bucketed):
        for bucket in bucketed:
            if depth < len(bucket):
                candidates.append(bucket[depth])
        depth += 1

    assigned: dict[str, list[tuple[str, list[dict[str, object]], int]]] = {}
    remaining = candidates
    for split, count in split_counts.items():
        assigned[split], remaining = _choose_exact_sessions(remaining, count)

    sampled: list[dict[str, object]] = []
    bucket_counts = {str(index): 0 for index in range(length_bucket_count)}
    session_counts: dict[str, int] = {}
    sample_ids: dict[str, list[str]] = {}
    for split, sessions in assigned.items():
        session_counts[split] = len(sessions)
        sample_ids[split] = []
        for _, session_rows, bucket in sessions:
            for row in sorted(session_rows, key=lambda value: int(value.get("invocation_index", 0))):
                annotated = dict(row)
                annotated["tier2_split"] = split
                annotated["prompt_length_bucket"] = bucket
                sampled.append(annotated)
                bucket_counts[str(bucket)] += 1
                sample_ids[split].append(str(row["sample_id"]))

    actual_counts = {
        split: sum(len(rows) for _, rows, _ in sessions)
        for split, sessions in assigned.items()
    }
    manifest: dict[str, object] = {
        "sampling_seed": seed,
        "sample_count": len(sampled),
        "split_counts": actual_counts,
        "session_counts": session_counts,
        "length_bucket_count": length_bucket_count,
        "length_bucket_counts": bucket_counts,
        "session_boundary_preserved": True,
        "sample_ids": sample_ids,
        "stopping_criterion": stopping_criterion,
    }
    return sampled, manifest
