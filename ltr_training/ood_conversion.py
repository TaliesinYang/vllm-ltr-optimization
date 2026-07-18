from __future__ import annotations

import json
import random
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

from .label_input import LabelInput, canonical_json, canonical_schema


def _normalize_schema_types(value: object) -> object:
    if isinstance(value, list):
        return [_normalize_schema_types(item) for item in value]
    if not isinstance(value, dict):
        return value
    normalized = {
        str(key): _normalize_schema_types(item) for key, item in value.items()
    }
    if normalized.get("type") == "dict":
        normalized["type"] = "object"
    elif normalized.get("type") == "list":
        normalized["type"] = "array"
    return normalized


def decode_json_string(value: object) -> object:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _normalize_tools(raw_tools: object) -> list[dict[str, object]]:
    decoded = decode_json_string(raw_tools)
    if isinstance(decoded, Mapping):
        decoded = decode_json_string(decoded.get("tools", decoded.get("functions", [])))
    if not isinstance(decoded, Sequence) or isinstance(decoded, (str, bytes)):
        return []
    tools: list[dict[str, object]] = []
    for raw in decoded:
        if not isinstance(raw, Mapping):
            continue
        if raw.get("type") == "function" and isinstance(raw.get("function"), Mapping):
            function = dict(raw["function"])
        elif isinstance(raw.get("name"), str):
            function = dict(raw)
        else:
            continue
        function.setdefault("description", "")
        function.setdefault("parameters", {"type": "object", "properties": {}})
        tools.append({"type": "function", "function": _normalize_schema_types(function)})
    return json.loads(canonical_json(tools))


def _message_pairs(messages: object) -> list[tuple[str, str]]:
    decoded = decode_json_string(messages)
    if not isinstance(decoded, Sequence) or isinstance(decoded, (str, bytes)):
        raise ValueError("messages must be an array")
    pairs: list[tuple[str, str]] = []
    for raw in decoded:
        if not isinstance(raw, Mapping):
            raise ValueError("message must be an object")
        role = raw.get("role")
        content = raw.get("content", "")
        if not isinstance(role, str):
            raise ValueError("message role must be a string")
        if not isinstance(content, str):
            content = canonical_json(content)
        pairs.append((role, content))
    return pairs


def _first_invocation(pairs: Sequence[tuple[str, str]]) -> tuple[str, tuple[tuple[str, str], ...]]:
    assistant_index = next(
        (index for index, (role, _) in enumerate(pairs) if role == "assistant"),
        len(pairs),
    )
    prefix = list(pairs[:assistant_index])
    user_indexes = [index for index, (role, _) in enumerate(prefix) if role == "user"]
    if not user_indexes:
        raise ValueError("first assistant invocation has no user prompt")
    prompt_index = user_indexes[-1]
    prompt = prefix[prompt_index][1]
    history = tuple(prefix[:prompt_index] + prefix[prompt_index + 1 :])
    return prompt, history


def convert_bfcl_row(
    row: Mapping[str, object],
    *,
    revision: str,
    category: str,
    function_index: Mapping[
        str, Mapping[str, object] | Sequence[Mapping[str, object]]
    ]
    | None = None,
) -> LabelInput:
    task_id = str(row["id"])
    question = row.get("question")
    if not isinstance(question, Sequence) or isinstance(question, (str, bytes)) or not question:
        raise ValueError(f"BFCL {task_id} has no question turns")
    first_turn = question[0]
    if isinstance(first_turn, Mapping):
        first_turn = [first_turn]
    pairs = _message_pairs(first_turn)
    prompt, history = _first_invocation(pairs)

    raw_tools = row.get("function", row.get("tools"))
    if raw_tools is None:
        selected: list[Mapping[str, object]] = []
        index = function_index or {}
        involved_classes = row.get("involved_classes", [])
        if isinstance(involved_classes, Sequence) and not isinstance(
            involved_classes, (str, bytes)
        ):
            for class_name in involved_classes:
                candidates = index.get(str(class_name))
                if isinstance(candidates, Mapping):
                    selected.append(candidates)
                elif isinstance(candidates, Sequence):
                    selected.extend(
                        candidate
                        for candidate in candidates
                        if isinstance(candidate, Mapping)
                    )
        if not selected:
            raw_path = row.get("path", [])
            if isinstance(raw_path, Sequence) and not isinstance(raw_path, (str, bytes)):
                for name in raw_path:
                    candidate = index.get(str(name)) or index.get(
                        str(name).rsplit(".", 1)[-1]
                    )
                    if isinstance(candidate, Mapping):
                        selected.append(candidate)
        if not selected:
            raise ValueError(f"BFCL {task_id} has no resolvable tool schema")
        raw_tools = selected
    tools = _normalize_tools(raw_tools)
    if not tools:
        raise ValueError(f"BFCL {task_id} resolved an empty tool schema")
    sample_id = f"bfcl:{task_id}:0000"
    return LabelInput(
        sample_id=sample_id,
        request_id=sample_id,
        prompt=prompt,
        tool_schema=canonical_schema(tools),
        history=history,
        session_id=task_id,
        task_id=task_id,
        source="bfcl",
        source_revision=revision,
        category=category,
    )


def convert_toolathlon_row(
    row: Mapping[str, object], *, revision: str, category: str
) -> LabelInput:
    task_id = str(row["task_name"])
    pairs = _message_pairs(row.get("messages", []))
    prompt, history = _first_invocation(pairs)
    tools = _normalize_tools(row.get("tool_calls", []))
    if not tools:
        raise ValueError(f"Toolathlon {task_id} resolved an empty tool schema")
    model_run = str(row.get("modelname_run", category))
    native_request = str(row.get("request_id", task_id))
    sample_id = f"toolathlon:{model_run}:{native_request}:0000"
    return LabelInput(
        sample_id=sample_id,
        request_id=sample_id,
        prompt=prompt,
        tool_schema=canonical_schema(tools),
        history=history,
        session_id=sample_id,
        task_id=task_id,
        source="toolathlon",
        source_revision=revision,
        category=category,
    )


def sample_label_inputs(
    items: Iterable[LabelInput], *, sample_size: int, seed: int
) -> list[LabelInput]:
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    buckets: dict[str, list[LabelInput]] = defaultdict(list)
    for item in items:
        buckets[item.category].append(item)
    rng = random.Random(seed)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: item.sample_id)
        rng.shuffle(bucket)
    selected: list[LabelInput] = []
    categories = sorted(buckets)
    while categories and len(selected) < sample_size:
        remaining: list[str] = []
        for category in categories:
            if buckets[category] and len(selected) < sample_size:
                selected.append(buckets[category].pop())
            if buckets[category]:
                remaining.append(category)
        categories = remaining
    return selected


def summarize_conversion(
    items: Iterable[LabelInput], *, sampling_seed: int
) -> dict[str, object]:
    rows = list(items)
    return {
        "schema_version": "label-input-v1",
        "sampling_seed": sampling_seed,
        "row_count": len(rows),
        "unique_task_count": len({item.task_id for item in rows}),
        "unique_input_hash_count": len({item.input_hash for item in rows}),
        "unique_schema_hash_count": len({item.schema_hash for item in rows}),
        "source_revisions": {
            source: sorted({item.source_revision for item in rows if item.source == source})
            for source in sorted({item.source for item in rows})
        },
        "sample_ids": [item.sample_id for item in rows],
        "domain_separation": {
            "source_identity": "disjoint",
            "schema_hash": "measured",
            "claim": "source-identity disjoint; schema overlap requires measured hashes",
        },
    }
