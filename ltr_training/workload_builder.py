from __future__ import annotations

import math
import random
from typing import Iterable, Mapping, Sequence

from .label_input import LabelInput


def manifest_split_ids(payload: object, *, split: str = "test") -> set[str]:
    candidates: object = payload
    if isinstance(payload, Mapping):
        splits = payload.get("splits")
        sample_ids = payload.get("sample_ids")
        if isinstance(splits, Mapping) and split in splits:
            candidates = splits[split]
        elif isinstance(sample_ids, Mapping) and split in sample_ids:
            candidates = sample_ids[split]
        elif split in payload:
            candidates = payload[split]
        else:
            candidates = payload.get("samples", payload.get("rows", []))
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError("tier2 sample manifest must contain a sequence")
    selected: set[str] = set()
    for item in candidates:
        if isinstance(item, Mapping):
            item_split = str(item.get("split", item.get("tier2_split", split)))
            if item_split != split:
                continue
            sample_id = item.get("sample_id", item.get("request_id"))
        else:
            sample_id = item
        if sample_id is not None:
            selected.add(str(sample_id))
    return selected


def _choose_mixed(
    id_inputs: list[LabelInput],
    ood_inputs: list[LabelInput],
    *,
    ood_ratio: float,
    seed: int,
) -> list[LabelInput]:
    if not 0.0 <= ood_ratio <= 1.0:
        raise ValueError("ood_ratio must be between zero and one")
    rng = random.Random(seed)
    ids = sorted(id_inputs, key=lambda item: item.sample_id)
    oods = sorted(ood_inputs, key=lambda item: item.sample_id)
    rng.shuffle(ids)
    rng.shuffle(oods)
    if not ids or not oods:
        return ids + oods
    def stable_floor(value: float) -> int:
        return math.floor(value + 1e-12 * max(1.0, abs(value)))

    total = min(
        len(ids) + len(oods),
        stable_floor(len(ids) / max(1.0 - ood_ratio, 1e-12)),
        stable_floor(len(oods) / max(ood_ratio, 1e-12))
        if ood_ratio
        else len(ids),
    )
    if total < 1:
        total = 1
    ood_count = min(len(oods), round(total * ood_ratio))
    id_count = min(len(ids), total - ood_count)
    selected = ids[:id_count] + oods[:ood_count]
    rng.shuffle(selected)
    return selected


def build_workload(
    *,
    id_inputs: Iterable[LabelInput],
    ood_inputs: Iterable[LabelInput],
    lengths: Mapping[str, int],
    profile: str,
    per_token_ms: float,
    ood_ratio: float = 0.5,
    seed: int = 42,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if per_token_ms <= 0.0:
        raise ValueError("per_token_ms must be positive")
    ids = list(id_inputs)
    oods = list(ood_inputs)
    if profile == "id":
        selected = ids
    elif profile == "ood":
        selected = oods
    elif profile == "mixed":
        selected = _choose_mixed(ids, oods, ood_ratio=ood_ratio, seed=seed)
    else:
        raise ValueError("profile must be id, ood, or mixed")

    rows: list[dict[str, object]] = []
    for item in selected:
        if item.sample_id not in lengths:
            raise ValueError(f"missing output length for {item.sample_id}")
        output_length = int(lengths[item.sample_id])
        if output_length <= 0:
            raise ValueError(f"non-positive output length for {item.sample_id}")
        domain = "id" if item.source == "toolace" else "ood"
        rows.append(
            {
                "request_id": item.sample_id,
                "sample_id": item.sample_id,
                "prompt": item.prompt,
                "tool_schema": item.tool_schema,
                "history": [list(history_item) for history_item in item.history],
                "baseline_service_ms": round(output_length * per_token_ms, 6),
                "max_tokens": 4096,
                "kind": "tool",
                "category": f"{domain}:{item.source}",
                "profile": profile,
                "source": item.source,
                "source_revision": item.source_revision,
                "session_id": item.session_id,
                "task_id": item.task_id,
                "true_length": output_length,
            }
        )
    manifest = {
        "schema_version": "offline-workload-v2",
        "profile": profile,
        "sampling_seed": seed,
        "request_count": len(rows),
        "max_tokens": 4096,
        "baseline_service_ms": {
            "kind": "output_length_per_token_proxy",
            "formula": "output_length * per_token_ms",
            "per_token_ms": per_token_ms,
            "claim": "proxy; not isolated service timing",
        },
        "runner_compatibility": "unverified_scheduler_benchmark_frozen",
    }
    return rows, manifest
