from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence


@dataclass(frozen=True)
class Invocation:
    sample_id: str
    source: str
    source_revision: str
    session_id: str
    invocation_index: int
    prompt: str
    completion: str | None
    tool_schema: str
    history: tuple[tuple[str, str], ...]
    generator_id: str
    recorded_output_length: int | None


def _iter_json_array(path: Path, chunk_size: int = 1 << 20) -> Iterator[object]:
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    is_started = False
    is_finished = False

    with path.open(encoding="utf-8") as handle:
        while not is_finished:
            chunk = handle.read(chunk_size)
            if chunk:
                buffer = buffer[position:] + chunk
                position = 0
            elif position >= len(buffer):
                break

            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if not is_started:
                    if position >= len(buffer):
                        break
                    if buffer[position] != "[":
                        raise ValueError(f"expected JSON array in {path}")
                    is_started = True
                    position += 1
                    continue

                while position < len(buffer) and (
                    buffer[position].isspace() or buffer[position] == ","
                ):
                    position += 1
                if position >= len(buffer):
                    break
                if buffer[position] == "]":
                    is_finished = True
                    position += 1
                    break
                try:
                    value, position = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    if not chunk:
                        raise
                    break
                yield value

    if not is_finished:
        raise ValueError(f"unterminated JSON array in {path}")


def iter_toolace_invocations(
    path: str | Path,
    *,
    revision: str,
    generator_id: str,
) -> Iterator[Invocation]:
    snapshot_path = Path(path)
    for session_index, raw_row in enumerate(_iter_json_array(snapshot_path)):
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"ToolACE row {session_index} is not an object")
        raw_messages = raw_row.get("conversations")
        system = raw_row.get("system")
        if not isinstance(raw_messages, Sequence) or isinstance(raw_messages, (str, bytes)):
            raise ValueError(f"ToolACE row {session_index} has no conversations array")
        if not isinstance(system, str):
            raise ValueError(f"ToolACE row {session_index} has no system string")

        session_id = f"toolace-{session_index:06d}"
        prior_messages: list[tuple[str, str]] = []
        invocation_index = 0
        for message_index, raw_message in enumerate(raw_messages):
            if not isinstance(raw_message, Mapping):
                raise ValueError(
                    f"ToolACE row {session_index} message {message_index} is not an object"
                )
            role = raw_message.get("from")
            value = raw_message.get("value")
            if not isinstance(role, str) or not isinstance(value, str):
                raise ValueError(
                    f"ToolACE row {session_index} message {message_index} is malformed"
                )
            if role == "assistant":
                if not prior_messages:
                    raise ValueError(
                        f"ToolACE row {session_index} starts with an assistant response"
                    )
                yield Invocation(
                    sample_id=f"{session_id}:{invocation_index:04d}",
                    source="toolace",
                    source_revision=revision,
                    session_id=session_id,
                    invocation_index=invocation_index,
                    prompt=prior_messages[-1][1],
                    completion=value,
                    tool_schema=system,
                    history=tuple(prior_messages[:-1]),
                    generator_id=generator_id,
                    recorded_output_length=None,
                )
                invocation_index += 1
            prior_messages.append((role, value))


def _message_content(message: Mapping[str, object]) -> str:
    content = message.get("content")
    if isinstance(content, str) and content:
        return content
    tool_calls = message.get("tool_calls")
    if tool_calls:
        return json.dumps(tool_calls, ensure_ascii=False, separators=(",", ":"))
    return ""


def iter_lmcache_invocations(
    rows: Iterable[Mapping[str, object]], *, revision: str
) -> Iterator[Invocation]:
    invocation_indexes: defaultdict[str, int] = defaultdict(int)
    for row_index, row in enumerate(rows):
        session_id = row.get("session_id")
        generator_id = row.get("model")
        raw_messages = row.get("input")
        output_length = row.get("output_length")
        if not isinstance(session_id, str) or not isinstance(generator_id, str):
            raise ValueError(f"LMCache row {row_index} lacks session_id/model")
        if not isinstance(output_length, int) or output_length < 0:
            raise ValueError(f"LMCache row {row_index} has invalid output_length")
        if not isinstance(raw_messages, Sequence) or isinstance(raw_messages, (str, bytes)):
            raise ValueError(f"LMCache row {row_index} has invalid input")

        messages: list[tuple[str, str]] = []
        for message in raw_messages:
            if not isinstance(message, Mapping):
                raise ValueError(f"LMCache row {row_index} has malformed input message")
            role = message.get("role")
            if not isinstance(role, str):
                raise ValueError(f"LMCache row {row_index} has message without role")
            messages.append((role, _message_content(message)))
        if not messages:
            raise ValueError(f"LMCache row {row_index} has empty input")

        invocation_index = invocation_indexes[session_id]
        invocation_indexes[session_id] += 1
        system_messages = [content for role, content in messages if role == "system"]
        yield Invocation(
            sample_id=f"lmcache:{session_id}:{invocation_index:04d}",
            source="lmcache",
            source_revision=revision,
            session_id=session_id,
            invocation_index=invocation_index,
            prompt=messages[-1][1],
            completion=None,
            tool_schema="\n".join(system_messages),
            history=tuple(messages[:-1]),
            generator_id=generator_id,
            recorded_output_length=output_length,
        )

