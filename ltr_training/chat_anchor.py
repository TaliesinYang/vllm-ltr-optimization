from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .label_input import canonical_json


@dataclass(frozen=True)
class ChatAnchor:
    sample_id: str
    prompt: str
    history: tuple[tuple[str, str], ...]
    source_revision: str


def convert_sharegpt_row(row: Mapping[str, object], *, revision: str) -> ChatAnchor:
    conversations = row.get("conversations", row.get("messages", []))
    if not isinstance(conversations, Sequence) or isinstance(conversations, (str, bytes)):
        raise ValueError("ShareGPT conversations must be an array")
    messages: list[tuple[str, str]] = []
    role_aliases = {"human": "user", "gpt": "assistant"}
    for message in conversations:
        if not isinstance(message, Mapping):
            raise ValueError("ShareGPT message must be an object")
        raw_role = str(message.get("from", message.get("role", ""))).lower()
        role = role_aliases.get(raw_role, raw_role)
        content = message.get("value", message.get("content", ""))
        messages.append((role, content if isinstance(content, str) else canonical_json(content)))
    assistant_index = next(
        (index for index, (role, _) in enumerate(messages) if role == "assistant"),
        len(messages),
    )
    prefix = messages[:assistant_index]
    user_indexes = [index for index, (role, _) in enumerate(prefix) if role == "user"]
    if not user_indexes:
        raise ValueError("ShareGPT row has no user input before first assistant")
    prompt_index = user_indexes[-1]
    native_id = row.get("id", row.get("conversation_id"))
    if native_id is None:
        native_id = hashlib.sha256(canonical_json(row).encode("utf-8")).hexdigest()[:24]
    return ChatAnchor(
        sample_id=f"sharegpt:{native_id}:0000",
        prompt=prefix[prompt_index][1],
        history=tuple(prefix[:prompt_index] + prefix[prompt_index + 1 :]),
        source_revision=revision,
    )


def build_sharegpt_anchor_workload(
    items: Iterable[ChatAnchor],
    *,
    lengths: Mapping[str, int],
    per_token_ms: float,
    seed: int,
    sample_size: int | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if per_token_ms <= 0:
        raise ValueError("per_token_ms must be positive")
    selected = sorted(items, key=lambda item: item.sample_id)
    random.Random(seed).shuffle(selected)
    if sample_size is not None:
        selected = selected[:sample_size]
    rows: list[dict[str, object]] = []
    for item in selected:
        output_length = int(lengths[item.sample_id])
        if output_length <= 0:
            raise ValueError(f"non-positive output length for {item.sample_id}")
        history_text = "\n".join(f"[{role.upper()}]\n{content}" for role, content in item.history)
        prompt = f"{history_text}\n[USER]\n{item.prompt}" if history_text else item.prompt
        rows.append(
            {
                "request_id": item.sample_id,
                "sample_id": item.sample_id,
                "prompt": prompt,
                "baseline_service_ms": round(output_length * per_token_ms, 6),
                "max_tokens": 4096,
                "kind": "chat",
                "category": "anchor/sharegpt",
                "profile": "anchor",
                "source": "sharegpt",
                "source_revision": item.source_revision,
                "session_id": item.sample_id,
                "true_length": output_length,
            }
        )
    return rows, {
        "schema_version": "chat-anchor-workload-v1",
        "source": "sharegpt",
        "source_revision": sorted({item.source_revision for item in selected}),
        "sampling_seed": seed,
        "sample_size_requested": sample_size,
        "row_count": len(rows),
        "max_tokens": 4096,
        "extraction_scope": "first-assistant-invocation-only",
        "baseline_service_ms": {
            "kind": "output_length_per_token_proxy",
            "formula": "output_length * per_token_ms",
            "per_token_ms": per_token_ms,
            "claim": "proxy; not isolated service timing",
        },
    }
