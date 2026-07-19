from __future__ import annotations

import math
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable


TOOL_SCHEMA_MARKER = "Here is a list of functions in JSON format that you can invoke:"


def _normalize_json_schema(value: object) -> object:
    if isinstance(value, list):
        return [_normalize_json_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    normalized = {
        key: _normalize_json_schema(item) for key, item in value.items()
    }
    if normalized.get("type") == "dict":
        normalized["type"] = "object"
    elif normalized.get("type") == "list":
        normalized["type"] = "array"
    return normalized


def _toolace_tools(tool_schema: str) -> list[dict[str, object]]:
    try:
        wrapped_tools = json.loads(tool_schema)
    except json.JSONDecodeError:
        wrapped_tools = None
    if isinstance(wrapped_tools, list) and all(
        isinstance(tool, dict)
        and tool.get("type") == "function"
        and isinstance(tool.get("function"), dict)
        and isinstance(tool["function"].get("name"), str)
        and bool(tool["function"]["name"])
        for tool in wrapped_tools
    ):
        return wrapped_tools
    marker_index = tool_schema.find(TOOL_SCHEMA_MARKER)
    if marker_index < 0:
        return []
    array_index = tool_schema.find("[", marker_index + len(TOOL_SCHEMA_MARKER))
    if array_index < 0:
        return []
    try:
        definitions, _ = json.JSONDecoder().raw_decode(tool_schema[array_index:])
    except json.JSONDecodeError:
        return []
    if not isinstance(definitions, list):
        return []
    tools: list[dict[str, object]] = []
    for definition in definitions:
        if not isinstance(definition, dict) or not isinstance(definition.get("name"), str):
            continue
        name = re.sub(r"[^A-Za-z0-9_-]+", "_", definition["name"]).strip("_")
        function = {
            "name": name or "tool",
            "description": str(definition.get("description", "")),
            "parameters": _normalize_json_schema(
                definition.get("parameters", {"type": "object", "properties": {}})
            ),
        }
        tools.append({"type": "function", "function": function})
    return tools


def build_request(
    row: dict[str, object], *, model: str, max_tokens: int = 4096
) -> dict[str, object]:
    role_aliases = {"human": "user", "gpt": "assistant"}
    messages: list[dict[str, str]] = []
    tool_schema = row.get("tool_schema")
    tools = _toolace_tools(tool_schema) if isinstance(tool_schema, str) else []
    is_empty_wrapped_tools = False
    if isinstance(tool_schema, str):
        try:
            parsed_tool_schema = json.loads(tool_schema)
        except json.JSONDecodeError:
            parsed_tool_schema = None
        is_empty_wrapped_tools = parsed_tool_schema == []
    if (
        isinstance(tool_schema, str)
        and tool_schema
        and not tools
        and not is_empty_wrapped_tools
    ):
        messages.append({"role": "system", "content": tool_schema})
    for raw_role, raw_content in row.get("history", []):
        role = role_aliases.get(str(raw_role), str(raw_role))
        messages.append({"role": role, "content": str(raw_content)})
    messages.append({"role": "user", "content": str(row["prompt"])})
    request: dict[str, object] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if tools:
        request["tools"] = tools
    return request


def completed_sample_ids(rows: Iterable[dict[str, object]]) -> set[str]:
    return {
        str(row["sample_id"])
        for row in rows
        if row.get("status") == "ok" and "sample_id" in row
    }


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def latest_results(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    latest: dict[str, dict[str, object]] = {}
    for row in rows:
        latest[str(row["sample_id"])] = row
    return list(latest.values())


def replay_labels(
    *,
    labels_path: Path,
    ledger_path: Path,
    endpoint: str,
    model: str,
    model_revision: str,
    max_tokens: int = 4096,
    limit: int | None = None,
    concurrency: int = 1,
    capture_text: bool = False,
) -> list[dict[str, object]]:
    import requests

    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    prior_rows = read_jsonl(ledger_path)
    completed = completed_sample_ids(prior_rows)
    pending: list[dict[str, object]] = []
    with labels_path.open(encoding="utf-8") as labels:
        for index, line in enumerate(labels):
            if limit is not None and index >= limit:
                break
            source = json.loads(line)
            if str(source["sample_id"]) not in completed:
                pending.append(source)

    def replay_one(source: dict[str, object]) -> dict[str, object]:
        sample_id = str(source["sample_id"])
        started = time.monotonic()
        try:
            response = requests.post(
                endpoint,
                json=build_request(source, model=model, max_tokens=max_tokens),
                timeout=900,
            )
            response.raise_for_status()
            payload = response.json()
            usage = payload["usage"]
            completion_tokens = int(usage["completion_tokens"])
            choice = payload["choices"][0]
            finish_reason = choice.get("finish_reason")
            row: dict[str, object] = {
                "sample_id": sample_id,
                "source": source["source"],
                "source_revision": source["source_revision"],
                "session_id": source.get("session_id"),
                "tier2_split": source.get("tier2_split"),
                "status": "ok",
                "output_length": completion_tokens,
                "censored": finish_reason == "length" or completion_tokens >= max_tokens,
                "finish_reason": finish_reason,
                "elapsed_seconds": time.monotonic() - started,
                "generator_id": f"{model}@{model_revision}",
                "temperature": 0,
                "max_tokens": max_tokens,
                "usage": usage,
            }
            if capture_text:
                message = choice.get("message") or {}
                row["response_text"] = message.get("content") or ""
                row["reasoning_content"] = message.get("reasoning_content")
                row["tool_calls"] = message.get("tool_calls")
            return row
        except Exception as error:
            return {
                "sample_id": sample_id,
                "source": source.get("source"),
                "session_id": source.get("session_id"),
                "tier2_split": source.get("tier2_split"),
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
                "elapsed_seconds": time.monotonic() - started,
            }

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as ledger, ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:
        # Workers only perform HTTP calls. The main thread is the sole JSONL writer,
        # so each completed request is appended and fsynced atomically for resume.
        futures = [executor.submit(replay_one, source) for source in pending]
        for future in as_completed(futures):
            row = future.result()
            ledger.write(json.dumps(row, ensure_ascii=False) + "\n")
            ledger.flush()
            os.fsync(ledger.fileno())
            print(json.dumps(row, sort_keys=True), flush=True)
    return latest_results(read_jsonl(ledger_path))


def _nearest_rank(values: list[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def summarize_results(
    rows: Iterable[dict[str, object]],
    *,
    expected_count: int,
    wall_elapsed_seconds: float | None = None,
) -> dict[str, object]:
    materialized = list(rows)
    successful = [row for row in materialized if row.get("status") == "ok"]
    lengths = [int(row["output_length"]) for row in successful]
    request_elapsed = sum(float(row.get("elapsed_seconds", 0.0)) for row in materialized)
    elapsed = wall_elapsed_seconds if wall_elapsed_seconds is not None else request_elapsed
    failures = len(materialized) - len(successful)
    censored = sum(bool(row.get("censored")) for row in successful)
    return {
        "expected": expected_count,
        "attempted": len(materialized),
        "successful": len(successful),
        "failures": failures,
        "failure_rate": failures / len(materialized) if materialized else 0.0,
        "censored": censored,
        "censor_rate": censored / len(successful) if successful else 0.0,
        "elapsed_seconds": elapsed,
        "summed_request_seconds": request_elapsed,
        "requests_per_second": len(materialized) / elapsed if elapsed else 0.0,
        "output_tokens_per_second": sum(lengths) / elapsed if elapsed else 0.0,
        "output_length": {
            "min": min(lengths) if lengths else None,
            "p50": _nearest_rank(lengths, 0.50),
            "p95": _nearest_rank(lengths, 0.95),
            "p99": _nearest_rank(lengths, 0.99),
            "max": max(lengths) if lengths else None,
        },
    }
