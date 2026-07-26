"""Cold-Start stratum classification and evidence-based gate confidence.

At admission the scheduler knows the request's tool schema but not its output.
Comparing that schema's tool set against the Ranker's training vocabulary
places the request in a Cold-Start Transfer stratum, and each stratum carries a
confidence measured offline (T5, issue #9):

    S1  seen combination        tool-set fingerprint appears in training
    S2  new combination         fingerprint unseen, every tool name seen
    S3  partial-new tools       fingerprint unseen, some tool names unseen
    S4  all-new tools           fingerprint unseen, no tool name seen

Confidence values are loaded from a committed artifact rather than written
here, so the numbers in the serving path and the numbers in the offline
evaluation cannot drift apart.

Requests whose tool set cannot be determined - an empty tool list, or a schema
the parser cannot read - are deliberately NOT guessed at. They receive the
artifact's conservative ``unknown`` confidence, which routes them to the
Fallback path instead of having the gate vouch for them.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

DEFAULT_ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "gate_confidence.json"

STRATA = ("S1", "S2", "S3", "S4")
UNKNOWN_STRATUM = "unknown"

_INVOKE_MARKER = "you can invoke:"

# ToolACE ships several system-prompt templates; each names its tools
# differently. Kept in one place so the serving path and the offline
# experiments classify a schema identically.
_NAME_PATTERNS = (
    re.compile(r"^\s*(?:[-*]\s*)?\**tool_name\**\s*:\s*(?!\s*$)(.+?)\s*$", re.MULTILINE),
    re.compile(r'"tool_name"\s*:\s*"((?:[^"\\]|\\.)*)"'),
    re.compile(r"<tool_name>\s*(.+?)\s*</tool_name>", re.DOTALL),
)
_HTML_ROW_PATTERN = re.compile(r"<tr>\s*<td>\s*(.*?)\s*</td>", re.DOTALL | re.IGNORECASE)
_LATEX_ROW_PATTERN = re.compile(r"^\s*([^&\\\n][^&\n]*?)\s*&", re.MULTILINE)


def _extract_schema_json(tool_schema: str) -> list | None:
    marker = tool_schema.find(_INVOKE_MARKER)
    start = tool_schema.find("[", marker if marker >= 0 else 0)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(tool_schema)):
        char = tool_schema[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(tool_schema[start : index + 1])
                except json.JSONDecodeError:
                    return None
                return value if isinstance(value, list) else None
    return None


def tool_names(tool_schema: str) -> tuple[str, ...]:
    """Sorted top-level tool names; empty when no tool set can be read."""
    parsed = _extract_schema_json(tool_schema)
    if parsed:
        names = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            if "name" in item:
                names.append(str(item["name"]))
            else:
                # OpenAI tools-array format: {"type": "function",
                # "function": {"name": ...}} - what gateways forward.
                function = item.get("function")
                if isinstance(function, dict) and "name" in function:
                    names.append(str(function["name"]))
        if names:
            return tuple(sorted(names))
    for pattern in _NAME_PATTERNS:
        found = [name for name in pattern.findall(tool_schema) if name]
        if found:
            return tuple(sorted(found))
    if "<th>tool_name</th>" in tool_schema.lower():
        found = [name for name in _HTML_ROW_PATTERN.findall(tool_schema) if name]
        if found:
            return tuple(sorted(found))
    if "begin{tabular}" in tool_schema:
        found = [
            name
            for name in _LATEX_ROW_PATTERN.findall(tool_schema)
            if name and name != "tool_name"
        ]
        if found:
            return tuple(sorted(found))
    return ()


def toolset_fingerprint(tool_schema: str) -> str:
    """SHA-256 over the sorted tool-name list.

    The raw ``tool_schema`` string is never hashed: many ToolACE prompts embed a
    per-request timestamp, which would make every request its own identity and
    silently classify all traffic as unseen.
    """
    payload = json.dumps(list(tool_names(tool_schema)), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GateVocabulary:
    """Training vocabulary plus the measured confidence for each stratum."""

    fingerprints: frozenset[str]
    tools: frozenset[str]
    confidence_by_stratum: Mapping[str, float]
    unknown_confidence: float
    fingerprint_prefix_length: int
    provenance: Mapping[str, object]

    @classmethod
    def from_artifact(cls, path: Path | str = DEFAULT_ARTIFACT) -> "GateVocabulary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        confidence = {
            stratum: float(payload["confidence_by_stratum"][stratum])
            for stratum in STRATA
        }
        for stratum, value in confidence.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"confidence for {stratum} outside [0, 1]: {value}")
        unknown = float(payload["unknown_confidence"])
        if not 0.0 <= unknown <= 1.0:
            raise ValueError(f"unknown_confidence outside [0, 1]: {unknown}")
        return cls(
            fingerprints=frozenset(payload["train_fingerprints"]),
            tools=frozenset(payload["train_tool_names"]),
            confidence_by_stratum=confidence,
            unknown_confidence=unknown,
            fingerprint_prefix_length=int(payload["fingerprint_prefix_length"]),
            provenance=payload.get("provenance", {}),
        )

    def stratum(self, tool_schema: str) -> str:
        names = tool_names(tool_schema)
        if not names:
            # No readable tool set: do not guess a stratum.
            return UNKNOWN_STRATUM
        fingerprint = toolset_fingerprint(tool_schema)[: self.fingerprint_prefix_length]
        if fingerprint in self.fingerprints:
            return "S1"
        seen = sum(1 for name in names if name in self.tools)
        if seen == len(names):
            return "S2"
        if seen == 0:
            return "S4"
        return "S3"

    def confidence(self, tool_schema: str) -> float:
        stratum = self.stratum(tool_schema)
        if stratum == UNKNOWN_STRATUM:
            return self.unknown_confidence
        return self.confidence_by_stratum[stratum]
