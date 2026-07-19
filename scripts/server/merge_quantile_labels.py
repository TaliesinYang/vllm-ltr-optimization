#!/usr/bin/env python3
"""Join the fixed Tier-2 sample with the append-only replay ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_COUNT = 6_000


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def merge(samples_path: Path, ledger_path: Path) -> list[dict[str, object]]:
    samples = read_jsonl(samples_path)
    if len(samples) != EXPECTED_COUNT:
        raise ValueError(
            f"NO-GO: sample must contain exactly {EXPECTED_COUNT} rows, got {len(samples)}"
        )
    sample_by_id: dict[str, dict[str, object]] = {}
    for row in samples:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in sample_by_id:
            raise ValueError(f"NO-GO: missing or duplicate sample_id {sample_id!r}")
        sample_by_id[sample_id] = row

    latest: dict[str, dict[str, object]] = {}
    for row in read_jsonl(ledger_path):
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            raise ValueError("NO-GO: ledger row is missing sample_id")
        latest[sample_id] = row  # append-only ledger: last row wins

    unknown = sorted(set(latest) - set(sample_by_id))
    if unknown:
        raise ValueError(f"NO-GO: ledger contains unknown sample IDs: {unknown[:5]}")
    if len(latest) != EXPECTED_COUNT:
        raise ValueError(
            f"NO-GO: ledger has {len(latest)} latest sample IDs, expected {EXPECTED_COUNT}"
        )

    merged: list[dict[str, object]] = []
    failures: list[str] = []
    for sample_id, sample in sample_by_id.items():
        result = latest.get(sample_id)
        if result is None or result.get("status") != "ok":
            failures.append(sample_id)
            continue
        output_length = result.get("output_length")
        if isinstance(output_length, bool) or not isinstance(output_length, int):
            failures.append(sample_id)
            continue
        merged.append(
            {
                "sample_id": sample_id,
                "prompt": str(sample.get("prompt", "")),
                "tool_schema": str(sample.get("tool_schema", "")),
                "output_length": output_length,
            }
        )
    if failures:
        raise ValueError(
            f"NO-GO: {len(failures)} samples lack a latest successful output length: "
            f"{failures[:10]}"
        )
    if len(merged) != EXPECTED_COUNT:
        raise ValueError(f"NO-GO: merged count is {len(merged)}, expected {EXPECTED_COUNT}")
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = merge(args.samples, args.ledger)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "sample_count": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
