from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


def _normalize_json(value: object) -> object:
    if isinstance(value, list):
        return [_normalize_json(item) for item in value]
    if not isinstance(value, dict):
        return value
    normalized = {
        str(key): _normalize_json(item)
        for key, item in sorted(value.items(), key=lambda item: str(item[0]))
    }
    if normalized.get("type") == "dict":
        normalized["type"] = "object"
    elif normalized.get("type") == "list":
        normalized["type"] = "array"
    return normalized


def canonical_json(value: object) -> str:
    return json.dumps(
        _normalize_json(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_schema(schema: str | object) -> str:
    value: object = schema
    if isinstance(schema, str):
        try:
            value = json.loads(schema)
        except json.JSONDecodeError:
            value = schema
    return canonical_json(value)


def canonical_schema_hash(schema: str | object) -> str:
    return hashlib.sha256(canonical_schema(schema).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LabelInput:
    """Pre-label request contract, deliberately distinct from Tier1Label."""

    sample_id: str
    request_id: str
    prompt: str
    tool_schema: str
    history: tuple[tuple[str, str], ...]
    session_id: str
    task_id: str
    source: str
    source_revision: str
    category: str

    def __post_init__(self) -> None:
        if self.request_id != self.sample_id:
            raise ValueError("request_id must equal sample_id")
        for name in (
            "sample_id",
            "prompt",
            "session_id",
            "task_id",
            "source",
            "source_revision",
            "category",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")

    @property
    def schema_hash(self) -> str:
        return canonical_schema_hash(self.tool_schema)

    @property
    def input_hash(self) -> str:
        payload = {
            "history": self.history,
            "prompt": self.prompt,
            "tool_schema": json.loads(canonical_schema(self.tool_schema)),
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["history"] = [list(item) for item in self.history]
        row["schema_hash"] = self.schema_hash
        row["input_hash"] = self.input_hash
        return row

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> "LabelInput":
        raw_history = row.get("history", [])
        if not isinstance(raw_history, Sequence) or isinstance(raw_history, (str, bytes)):
            raise ValueError("history must be a sequence")
        history: list[tuple[str, str]] = []
        for item in raw_history:
            if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
                raise ValueError("history entries must be role/content pairs")
            history.append((str(item[0]), str(item[1])))
        return cls(
            sample_id=str(row["sample_id"]),
            request_id=str(row.get("request_id", row["sample_id"])),
            prompt=str(row["prompt"]),
            tool_schema=canonical_schema(row.get("tool_schema", [])),
            history=tuple(history),
            session_id=str(row["session_id"]),
            task_id=str(row.get("task_id", row["session_id"])),
            source=str(row["source"]),
            source_revision=str(row["source_revision"]),
            category=str(row.get("category", row["source"])),
        )
