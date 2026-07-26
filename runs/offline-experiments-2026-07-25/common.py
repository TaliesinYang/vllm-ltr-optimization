"""Shared data loading + tool-set parsing for the E1/E2/E3 offline experiments.

Splits, features and the label filter are taken verbatim from the tier-2 recipe
(ltr_training.tier2_training.load_tier2_split_examples +
ltr_training.training_matrix.structural_features) so every model in these
experiments sees exactly the rows the BERT tier-2 matrix saw.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ltr_training.tier2_training import load_tier2_split_examples  # noqa: E402
from ltr_training.training_matrix import structural_features  # noqa: E402

RESULTS = Path("/Volumes/T7 Shield/vllm-ltr-results")
SAMPLE_PATH = RESULTS / "rebuild" / "tier2-toolace-sample-6000.jsonl"
LEDGER_PATH = RESULTS / "extracted" / "tier2-toolace-6000-ledger.jsonl"
SAMPLE_SHA = "ee5a5889ca3d9bbee7790e7a408bd1664a285b6410b4fee54e45786d3eecb709"

SPLITS = ("train", "validation", "test")
SEEDS = (17, 42, 73)

_INVOKE_MARKER = "you can invoke:"

# ToolACE ships several system-prompt templates. Besides the embedded JSON list
# there is a YAML-ish one ("tool_name: X") and a markdown one
# ("- **tool_name**: X"); both name the tools in plain text.
_NAME_PATTERNS = (
    # YAML-ish and markdown: "tool_name: X" / "- **tool_name**: X"
    re.compile(r"^\s*(?:[-*]\s*)?\**tool_name\**\s*:\s*(?!\s*$)(.+?)\s*$", re.MULTILINE),
    # JSON object stream keyed by tool_name rather than name
    re.compile(r'"tool_name"\s*:\s*"((?:[^"\\]|\\.)*)"'),
    # XML-ish
    re.compile(r"<tool_name>\s*(.+?)\s*</tool_name>", re.DOTALL),
)
# HTML table: first cell of each body row, when the header declares tool_name
_HTML_ROW_PATTERN = re.compile(r"<tr>\s*<td>\s*(.*?)\s*</td>", re.DOTALL | re.IGNORECASE)
# LaTeX tabular: first cell of each body row
_LATEX_ROW_PATTERN = re.compile(r"^\s*([^&\\\n][^&\n]*?)\s*&", re.MULTILINE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_inputs() -> dict[str, str]:
    sample_sha = sha256_file(SAMPLE_PATH)
    if sample_sha != SAMPLE_SHA:
        raise SystemExit(
            f"sample-6000 sha mismatch: expected {SAMPLE_SHA}, got {sample_sha}"
        )
    return {
        "sample_path": str(SAMPLE_PATH),
        "sample_sha256": sample_sha,
        "ledger_path": str(LEDGER_PATH),
        "ledger_sha256": sha256_file(LEDGER_PATH),
    }


def load_splits():
    return load_tier2_split_examples(sample_path=SAMPLE_PATH, ledger_path=LEDGER_PATH)


def _extract_schema_json(tool_schema: str) -> list[dict] | None:
    """Pull the embedded tool-list JSON out of a ToolACE system prompt.

    The prompt wraps the list in prose and carries a per-row timestamp, so the
    raw string is not a usable identity key; the tool list is.
    """
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
    """Sorted top-level tool names across every ToolACE prompt template.

    Empty tuple only when the prompt genuinely advertises no tool.
    """
    parsed = _extract_schema_json(tool_schema)
    if parsed:
        names = [
            str(item["name"])
            for item in parsed
            if isinstance(item, dict) and "name" in item
        ]
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
    """SHA-256 over the sorted tool-name list — the tool-set identity key."""
    names = tool_names(tool_schema)
    payload = json.dumps(list(names), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def schema_body_hash(tool_schema: str) -> str:
    """SHA-256 over the canonicalised tool-list JSON (content identity).

    Uses the parsed tool list rather than the raw prompt so the per-row
    ``The current time is ...`` stamp does not make every hash unique.
    """
    parsed = _extract_schema_json(tool_schema)
    if parsed is None:
        return hashlib.sha256(tool_schema.encode("utf-8")).hexdigest()
    payload = json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
