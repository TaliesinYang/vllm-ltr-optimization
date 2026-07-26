"""Block-1 workload: trace-calibrated synthetic multi-tenant agent traffic.

Every distributional parameter is MEASURED from the captured agent trace
(``probes/agent-traces-2026-07-26/``) at build time. Nothing is hardcoded: if
the trace is recollected the workload changes with it, and the manifest records
which measurement produced which parameter.

What the trace establishes, and what this generator reproduces:

  zero-tool share      a third of real agent requests carry no tools at all
                       (title generation and other utility calls)
  completion lengths   resampled from the empirical distribution, so p50/p99
                       match the trace rather than a fitted curve
  turn depth           resampled from the empirical message-count distribution
  schema constancy     a real client's tool schema does not change between
                       turns, so each synthetic client is pinned to one tool set

Multi-tenancy is the part the trace CANNOT supply — it was a single client, so
its 3 distinct schemas say nothing about a shared queue. Tool sets are
therefore drawn from the ToolACE vocabulary and deliberately spread across all
four Cold-Start strata, so the queue exercises the gate rather than landing
entirely in one stratum. That construction is a design choice and is recorded
as such in the manifest, not presented as trace-derived.

The 75 real trace requests are emitted alongside the synthetic ones, marked
with ``synthetic=False``, so replay can report on them separately.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .train_ranker import TrainingExample

SOURCE_SYNTHETIC = "block1-synthetic"
SOURCE_TRACE = "block1-trace"
STRATA = ("S1", "S2", "S3", "S4")

# Row schema of the existing run_matrix workload jsonl. Emitting anything else
# would break the consumer, so it is asserted rather than assumed.
WORKLOAD_FIELDS = (
    "baseline_service_ms",
    "category",
    "history",
    "kind",
    "max_tokens",
    "profile",
    "prompt",
    "request_id",
    "sample_id",
    "session_id",
    "source",
    "source_revision",
    "task_id",
    "tool_schema",
    "true_length",
)

_SCHEMA_HEADER = (
    "You are an expert in composing functions. You are given a question and a "
    "set of possible functions. \nBased on the question, you will need to make "
    "one or more function/tool calls to achieve the purpose. \n"
    "Here is a list of functions in JSON format that you can invoke:\n"
)
_SCHEMA_FOOTER = ". \nShould you decide to return the function call(s).\n"


def render_tool_schema(tool_names: Sequence[str]) -> str:
    """A ToolACE JSON-template schema advertising exactly these tools."""
    tools = [
        {
            "name": name,
            "description": f"Synthetic tool {name} for multi-tenant workload replay.",
            "parameters": {
                "type": "dict",
                "properties": {
                    "query": {"description": "Free-form argument.", "type": "string"}
                },
                "required": ["query"],
            },
        }
        for name in tool_names
    ]
    return _SCHEMA_HEADER + json.dumps(tools, ensure_ascii=False) + _SCHEMA_FOOTER


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile: ceil(fraction * n), 1-based.

    Same convention as ``e4_latency.py`` and the trace MANIFEST, so the numbers
    in this workload's manifest are directly comparable to the recorded
    characterisation. Note what that means at n=75: p99 lands on the single
    largest observation, so the "p99 = 328" anchor is one data point, not a
    tail estimate.
    """
    if not values:
        raise ValueError("cannot take a percentile of an empty sequence")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[index])


@dataclass(frozen=True)
class TraceCalibration:
    """Distributional parameters measured from the captured agent trace."""

    trace_path: str
    trace_sha256: str
    request_count: int
    zero_tool_fraction: float
    completion_tokens: tuple[int, ...]
    completion_p50: float
    completion_p90: float
    completion_p99: float
    turn_depths: tuple[int, ...]
    turn_depth_p50: float
    turn_depth_max: int
    distinct_schema_hashes: int
    rows: tuple[Mapping[str, object], ...] = field(repr=False, default=())

    def as_manifest(self) -> dict[str, object]:
        return {
            "source_measurement": self.trace_path,
            "trace_sha256": self.trace_sha256,
            "request_count": self.request_count,
            "zero_tool_fraction": self.zero_tool_fraction,
            "completion_tokens_p50": self.completion_p50,
            "completion_tokens_p90": self.completion_p90,
            "completion_tokens_p99": self.completion_p99,
            "turn_depth_p50": self.turn_depth_p50,
            "turn_depth_max": self.turn_depth_max,
            "distinct_tool_schema_hashes": self.distinct_schema_hashes,
        }


def _read_trace_rows(path: Path) -> list[Mapping[str, object]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        return [json.loads(line) for line in handle if line.strip()]


def measure_trace(path: Path) -> TraceCalibration:
    """Derive every calibration parameter from the trace file itself."""
    rows = _read_trace_rows(path)
    if not rows:
        raise ValueError(f"trace {path} is empty")

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    completions: list[int] = []
    depths: list[int] = []
    zero_tool = 0
    schema_hashes: set[str] = set()
    for row in rows:
        body = row.get("body") or {}
        tools = body.get("tools") or []
        if not tools:
            zero_tool += 1
        else:
            rendered = json.dumps(tools, sort_keys=True, ensure_ascii=False)
            schema_hashes.add(hashlib.sha256(rendered.encode("utf-8")).hexdigest())
        messages = body.get("messages") or []
        depths.append(len(messages))
        usage = row.get("usage") or {}
        tokens = usage.get("completion_tokens")
        if isinstance(tokens, int) and tokens > 0:
            completions.append(tokens)

    if not completions:
        raise ValueError(f"trace {path} carries no completion_tokens")

    return TraceCalibration(
        trace_path=str(path),
        trace_sha256=digest,
        request_count=len(rows),
        zero_tool_fraction=zero_tool / len(rows),
        completion_tokens=tuple(completions),
        completion_p50=percentile(completions, 0.50),
        completion_p90=percentile(completions, 0.90),
        completion_p99=percentile(completions, 0.99),
        turn_depths=tuple(depths),
        turn_depth_p50=percentile(depths, 0.50),
        turn_depth_max=max(depths),
        distinct_schema_hashes=len(schema_hashes),
        rows=tuple(rows),
    )


@dataclass(frozen=True)
class Client:
    """One synthetic tenant. Its tool schema is fixed for the whole run."""

    client_id: str
    stratum: str
    tool_names: tuple[str, ...]
    tool_schema: str


def _novel_name(index: int) -> str:
    return f"synthetic_tool_{index:05d}"


def build_clients(
    *,
    train_examples: Iterable[TrainingExample],
    per_stratum: int,
    seed: int,
    tool_names_of,
) -> list[Client]:
    """One cohort of clients per Cold-Start stratum.

    S1 reuses a training tool set verbatim; S2 recombines seen tool names into
    a combination training never saw; S3 mixes seen and novel; S4 is entirely
    novel. Spreading clients this way is what puts all four strata in one queue.
    """
    if per_stratum < 1:
        raise ValueError("per_stratum must be positive")
    rng = random.Random(seed)

    train_sets: list[tuple[str, ...]] = []
    seen_names: set[str] = set()
    for example in train_examples:
        names = tool_names_of(example.tool_schema)
        if names:
            train_sets.append(names)
            seen_names.update(names)
    if not train_sets:
        raise ValueError("training pool contains no readable tool sets")

    train_fingerprints = {tuple(names) for names in train_sets}
    seen_pool = sorted(seen_names)
    clients: list[Client] = []
    novel_counter = 0

    for index in range(per_stratum):
        # S1 - a combination training has seen.
        names = tuple(rng.choice(train_sets))
        clients.append(_client("S1", index, names))

    for index in range(per_stratum):
        # S2 - all names seen, combination unseen.
        for _ in range(64):
            size = rng.randint(2, min(4, len(seen_pool)))
            names = tuple(sorted(rng.sample(seen_pool, size)))
            if names not in train_fingerprints:
                break
        else:
            raise ValueError("could not synthesise an unseen combination of seen tools")
        clients.append(_client("S2", index, names))

    for index in range(per_stratum):
        # S3 - some seen, some novel.
        novel_counter += 1
        names = tuple(
            sorted({rng.choice(seen_pool), _novel_name(novel_counter)})
        )
        clients.append(_client("S3", index, names))

    for index in range(per_stratum):
        # S4 - nothing training has seen.
        novel_counter += 1
        first = _novel_name(novel_counter)
        novel_counter += 1
        names = tuple(sorted({first, _novel_name(novel_counter)}))
        clients.append(_client("S4", index, names))

    return clients


def _client(stratum: str, index: int, names: tuple[str, ...]) -> Client:
    return Client(
        client_id=f"{stratum.lower()}-client-{index:03d}",
        stratum=stratum,
        tool_names=names,
        tool_schema=render_tool_schema(names),
    )


def _prompt_for(rng: random.Random, client: Client, turn: int, depth: int) -> str:
    if not client.tool_names:
        return (
            "Generate a short title for this conversation. "
            f"(utility request, turn {turn} of {depth})"
        )
    tool = rng.choice(client.tool_names)
    return (
        f"Using the available tools, handle step {turn} of {depth} for tenant "
        f"{client.client_id}. Prefer {tool} if it fits the request."
    )


def generate_requests(
    *,
    calibration: TraceCalibration,
    clients: Sequence[Client],
    request_count: int,
    seed: int,
    max_tokens: int = 4096,
    source_revision: str = "block1-v1",
) -> list[dict[str, object]]:
    """Synthetic multi-tenant traffic with trace-calibrated marginals.

    Completion lengths and turn depths are resampled from the trace's empirical
    distributions rather than fitted, so the marginals match by construction
    instead of by parameter tuning.
    """
    if request_count < 1:
        raise ValueError("request_count must be positive")
    if not clients:
        raise ValueError("at least one client is required")

    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    for index in range(request_count):
        client = clients[index % len(clients)]
        zero_tool = rng.random() < calibration.zero_tool_fraction
        depth = int(rng.choice(calibration.turn_depths))
        depth = max(1, depth)
        turn = rng.randint(1, depth)
        true_length = int(rng.choice(calibration.completion_tokens))
        session_id = f"{client.client_id}-s{index // max(1, len(clients)):04d}"
        sample_id = f"block1-{index:06d}"
        # A zero-tool request is a utility call on the same tenant; the tenant's
        # schema is unchanged, this particular request just does not carry it.
        tool_schema = "" if zero_tool else client.tool_schema
        rows.append(
            {
                "baseline_service_ms": 0.0,
                "category": f"block1:{client.stratum.lower()}"
                if not zero_tool
                else "block1:zero_tool",
                "history": [],
                "kind": "utility" if zero_tool else "tool",
                "max_tokens": max_tokens,
                "profile": "block1",
                "prompt": _prompt_for(rng, client, turn, depth),
                "request_id": sample_id,
                "sample_id": sample_id,
                "session_id": session_id,
                "source": SOURCE_SYNTHETIC,
                "source_revision": source_revision,
                "task_id": session_id,
                "tool_schema": tool_schema,
                "true_length": true_length,
                # Block-1 extras, ignored by run_matrix but used for reporting.
                "synthetic": True,
                "client_id": client.client_id,
                "cold_start_stratum": "unknown" if zero_tool else client.stratum,
                "turn_index": turn,
                "turn_depth": depth,
            }
        )
    return rows


def trace_rows(
    calibration: TraceCalibration, *, max_tokens: int = 4096
) -> list[dict[str, object]]:
    """The 75 real requests, in workload shape, marked as not synthetic."""
    rows: list[dict[str, object]] = []
    for index, row in enumerate(calibration.rows):
        body = row.get("body") or {}
        messages = body.get("messages") or []
        tools = body.get("tools") or []
        usage = row.get("usage") or {}
        completion = usage.get("completion_tokens")
        if not isinstance(completion, int) or completion < 0:
            continue
        prompt = ""
        for message in reversed(messages):
            content = message.get("content")
            if isinstance(content, str) and content:
                prompt = content
                break
        system = next(
            (
                message.get("content")
                for message in messages
                if message.get("role") == "system"
                and isinstance(message.get("content"), str)
            ),
            "",
        )
        tool_schema = (
            json.dumps(tools, ensure_ascii=False) if tools else ""
        )
        sample_id = f"trace-{index:04d}"
        rows.append(
            {
                "baseline_service_ms": float(row.get("e2e_ms") or 0.0),
                "category": "block1:trace",
                "history": [],
                "kind": "tool" if tools else "utility",
                "max_tokens": max_tokens,
                "profile": "block1",
                "prompt": prompt or system,
                "request_id": str(row.get("request_id") or sample_id),
                "sample_id": sample_id,
                "session_id": f"trace-session-{index // 3:04d}",
                "source": SOURCE_TRACE,
                "source_revision": calibration.trace_sha256[:12],
                "task_id": f"trace-task-{index:04d}",
                "tool_schema": tool_schema,
                "true_length": completion,
                "synthetic": False,
                "client_id": "real-agent",
                "cold_start_stratum": "trace",
                "turn_index": 1,
                "turn_depth": len(messages),
            }
        )
    return rows


def build_manifest(
    *,
    calibration: TraceCalibration,
    clients: Sequence[Client],
    synthetic: Sequence[Mapping[str, object]],
    traces: Sequence[Mapping[str, object]],
    seed: int,
) -> dict[str, object]:
    lengths = [int(row["true_length"]) for row in synthetic]
    depths = [int(row["turn_depth"]) for row in synthetic]
    zero_tool = sum(1 for row in synthetic if not row["tool_schema"])
    by_stratum: dict[str, int] = {}
    for row in synthetic:
        key = str(row["cold_start_stratum"])
        by_stratum[key] = by_stratum.get(key, 0) + 1
    return {
        "schema_version": "block1-workload-v1",
        "seed": seed,
        "deterministic": "same seed and same trace file reproduce this file byte for byte",
        "calibration": {
            "note": "every parameter below is measured from the trace at build "
            "time; none is hardcoded",
            **calibration.as_manifest(),
        },
        "calibration_targets_vs_realized": {
            "zero_tool_fraction": {
                "trace": calibration.zero_tool_fraction,
                "workload": zero_tool / len(synthetic) if synthetic else 0.0,
            },
            "completion_p50": {
                "trace": calibration.completion_p50,
                "workload": percentile(lengths, 0.50) if lengths else 0.0,
            },
            "completion_p99": {
                "trace": calibration.completion_p99,
                "workload": percentile(lengths, 0.99) if lengths else 0.0,
                "note": "the trace's p99 at n=75 IS its single largest "
                "observation, so it is one data point rather than a tail "
                "estimate. Lengths are resampled from the full empirical "
                "distribution, so the workload's own p99 varies with the draw; "
                "compare completion_max and tail_count below to confirm the "
                "tail is present rather than truncated.",
            },
            "completion_max": {
                "trace": max(calibration.completion_tokens),
                "workload": max(lengths) if lengths else 0,
            },
            "completion_tail_count_at_trace_max": {
                "workload": sum(
                    1
                    for value in lengths
                    if value >= max(calibration.completion_tokens)
                ),
                "of_rows": len(lengths),
            },
            "turn_depth_p50": {
                "trace": calibration.turn_depth_p50,
                "workload": percentile(depths, 0.50) if depths else 0.0,
            },
        },
        "multi_tenancy": {
            "note": "NOT trace-derived. The trace was a single client with 3 "
            "schemas, which cannot characterise a shared queue. Client count "
            "and stratum spread are a design choice so the queue exercises all "
            "four Cold-Start strata.",
            "client_count": len(clients),
            "clients_per_stratum": {
                stratum: sum(1 for client in clients if client.stratum == stratum)
                for stratum in STRATA
            },
            "schema_constant_per_client": True,
        },
        "rows": {
            "synthetic": len(synthetic),
            "real_trace": len(traces),
            "total": len(synthetic) + len(traces),
            "synthetic_by_stratum": by_stratum,
        },
    }

