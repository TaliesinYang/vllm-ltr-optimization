#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ltr_training.tier2 import read_jsonl, replay_labels  # noqa: E402
from ltr_training.tier2_diagnostics import (  # noqa: E402
    aggregate_classifications,
    classify_long_response,
)
from replay_tier2_labels import PINNED_MODEL_REVISION  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay five censored rows and retain full responses.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pilot-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", default="qwen3.5-9b-tier2")
    parser.add_argument("--max-tokens", type=int, default=4096)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    censored_ids = [
        str(row["sample_id"])
        for row in read_jsonl(args.pilot_ledger)
        if row.get("status") == "ok" and row.get("censored")
    ][:5]
    if len(censored_ids) != 5:
        raise SystemExit(f"need five known censored rows, found {len(censored_ids)}")
    wanted = set(censored_ids)
    selected = [row for row in read_jsonl(args.source) if str(row["sample_id"]) in wanted]
    selected.sort(key=lambda row: censored_ids.index(str(row["sample_id"])))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = args.output_dir / "selected-source.jsonl"
    selected_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    ledger_path = args.output_dir / "replay-ledger.jsonl"
    rows = replay_labels(
        labels_path=selected_path,
        ledger_path=ledger_path,
        endpoint=args.endpoint,
        model=args.model,
        model_revision=PINNED_MODEL_REVISION,
        max_tokens=args.max_tokens,
        concurrency=5,
        capture_text=True,
    )
    successful = [row for row in rows if row.get("status") == "ok"]
    if len(successful) != 5:
        raise SystemExit(f"censored replay completed only {len(successful)}/5 rows")

    diagnoses: list[dict[str, object]] = []
    for row in successful:
        full_text = "\n".join(
            part for part in (
                str(row.get("reasoning_content") or ""),
                str(row.get("response_text") or ""),
                json.dumps(row.get("tool_calls"), ensure_ascii=False)
                if row.get("tool_calls") is not None else "",
            ) if part
        )
        diagnosis = {"sample_id": row["sample_id"], **classify_long_response(full_text)}
        diagnoses.append(diagnosis)
        artifact = {"replay": row, "full_text": full_text, "diagnosis": diagnosis}
        (args.output_dir / f"{row['sample_id'].replace(':', '_')}.json").write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    conclusion = aggregate_classifications(diagnoses)
    report = {
        "conclusion": conclusion,
        "sample_count": 5,
        "samples": diagnoses,
        "model_revision": PINNED_MODEL_REVISION,
        "temperature": 0,
        "max_tokens": args.max_tokens,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "classification.txt").write_text(
        f"{conclusion}: "
        + "; ".join(
            f"{item['sample_id']} ratio={item['repeated_ngram_ratio']:.3f} "
            f"evidence={item['evidence_snippets'][0] if item['evidence_snippets'] else 'none'}"
            for item in diagnoses
        )
        + "\n",
        encoding="utf-8",
    )
    print((args.output_dir / "classification.txt").read_text().strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
