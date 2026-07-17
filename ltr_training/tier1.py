from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import Iterable
from typing import Iterator

from .data import Invocation, iter_lmcache_invocations, iter_toolace_invocations


@dataclass(frozen=True)
class Tier1Label:
    sample_id: str
    source: str
    source_revision: str
    session_id: str
    invocation_index: int
    prompt: str
    tool_schema: str
    history: tuple[tuple[str, str], ...]
    output_length: int
    generator_id: str
    length_kind: str
    tokenizer_id: str | None


def extract_toolace_label(
    invocation: Invocation,
    *,
    count_tokens: Callable[[str], int],
    tokenizer_id: str,
) -> Tier1Label:
    if invocation.source != "toolace" or invocation.completion is None:
        raise ValueError("expected a ToolACE invocation with recorded completion")
    output_length = count_tokens(invocation.completion)
    if output_length < 0:
        raise ValueError(f"negative token label for {invocation.sample_id}")
    return _label_from_invocation(
        invocation,
        output_length=output_length,
        length_kind="retokenized_recorded_completion",
        tokenizer_id=tokenizer_id,
    )


def extract_lmcache_label(invocation: Invocation) -> Tier1Label:
    if invocation.source != "lmcache" or invocation.recorded_output_length is None:
        raise ValueError("expected an LMCache invocation with recorded output_length")
    return _label_from_invocation(
        invocation,
        output_length=invocation.recorded_output_length,
        length_kind="recorded_output_tokens",
        tokenizer_id=None,
    )


def _label_from_invocation(
    invocation: Invocation,
    *,
    output_length: int,
    length_kind: str,
    tokenizer_id: str | None,
) -> Tier1Label:
    return Tier1Label(
        sample_id=invocation.sample_id,
        source=invocation.source,
        source_revision=invocation.source_revision,
        session_id=invocation.session_id,
        invocation_index=invocation.invocation_index,
        prompt=invocation.prompt,
        tool_schema=invocation.tool_schema,
        history=invocation.history,
        output_length=output_length,
        generator_id=invocation.generator_id,
        length_kind=length_kind,
        tokenizer_id=tokenizer_id,
    )


def write_labels_jsonl(labels: Iterable[Tier1Label], output: str | Path) -> int:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for label in labels:
            handle.write(json.dumps(asdict(label), ensure_ascii=False) + "\n")
            count += 1
    return count


def iter_toolace_labels(
    path: str | Path,
    *,
    revision: str,
    generator_id: str,
    count_tokens: Callable[[str], int],
    tokenizer_id: str,
    limit: int | None = None,
) -> Iterator[Tier1Label]:
    for index, invocation in enumerate(
        iter_toolace_invocations(
            path,
            revision=revision,
            generator_id=generator_id,
        )
    ):
        if limit is not None and index >= limit:
            return
        yield extract_toolace_label(
            invocation,
            count_tokens=count_tokens,
            tokenizer_id=tokenizer_id,
        )


def iter_lmcache_labels(
    rows: Iterable[dict[str, object]],
    *,
    revision: str,
    limit: int | None = None,
) -> Iterator[Tier1Label]:
    for index, invocation in enumerate(
        iter_lmcache_invocations(rows, revision=revision)
    ):
        if limit is not None and index >= limit:
            return
        yield extract_lmcache_label(invocation)
