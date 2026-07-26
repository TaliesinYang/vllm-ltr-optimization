"""Build the committed gate-confidence artifact from t5-gate.json.

Derives, rather than restates, the numbers the Decision Service serves: the
per-stratum confidences come straight out of the T5 evaluation, and the
training vocabulary comes from the same fixed tier-2 split T5 fit on. Nothing
here is a literal typed by hand.

Fingerprints are stored truncated to 32 hex characters (128 bits). For ~3k
fingerprints the collision probability is on the order of 1e-33, and it keeps
the committed artifact a third of the size.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import common

from scheduler_benchmark.tool_vocabulary import (
    STRATA,
    tool_names as library_tool_names,
    toolset_fingerprint,
)

HERE = Path(__file__).resolve().parent
T5 = HERE / "t5-gate.json"
OUT = common.REPO / "scheduler_benchmark" / "artifacts" / "gate_confidence.json"
PREFIX = 32


def main() -> None:
    started = time.time()
    inputs = common.verify_inputs()
    splits, _ = common.load_splits()
    gate = json.loads(T5.read_text(encoding="utf-8"))

    # Parity guard: the parser now lives in the library so serving and the
    # experiments cannot diverge. Prove the move changed nothing before the
    # artifact is built from it.
    mismatches = [
        item.sample_id
        for split in common.SPLITS
        for item in splits[split]
        if library_tool_names(item.tool_schema) != common.tool_names(item.tool_schema)
    ]
    if mismatches:
        raise SystemExit(
            f"parser parity broken on {len(mismatches)} rows, "
            f"first: {mismatches[:3]}"
        )
    print(f"parser parity verified on {sum(len(splits[s]) for s in common.SPLITS)} rows")

    confidence = {
        stratum: float(gate["assigned_confidence"][stratum]) for stratum in STRATA
    }
    # The conservative value the gate uses when it cannot read a tool set at
    # all. It must never exceed the lowest confidence any stratum earns.
    unknown = min(confidence.values())

    fingerprints = sorted(
        {
            toolset_fingerprint(item.tool_schema)[:PREFIX]
            for item in splits["train"]
        }
    )
    tools = sorted(
        {name for item in splits["train"] for name in common.tool_names(item.tool_schema)}
    )

    payload = {
        "schema_version": "gate-confidence-v1",
        "confidence_by_stratum": confidence,
        "unknown_confidence": unknown,
        "fingerprint_prefix_length": PREFIX,
        "train_fingerprints": fingerprints,
        "train_tool_names": tools,
        "provenance": {
            "derived_from": "runs/offline-experiments-2026-07-25/t5-gate.json",
            "rule": gate["method"]["selected_small_stratum_rule"],
            "fit_split": gate["method"]["fit_split"],
            "evaluation_split": gate["method"]["evaluation_split"],
            "confidence_rule": gate["method"]["confidence_rule"],
            "sample_sha256": inputs["sample_sha256"],
            "ledger_sha256": inputs["ledger_sha256"],
            "train_rows": len(splits["train"]),
            "unique_fingerprints": len(fingerprints),
            "unique_tool_names": len(tools),
            "meaning": "confidence is a measured Kendall tau-b lower bound in "
            "[0, 1], not a calibrated probability",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB) "
        f"fingerprints={len(fingerprints)} tools={len(tools)} "
        f"in {time.time() - started:.1f}s"
    )
    print(json.dumps({**confidence, "unknown": unknown}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
