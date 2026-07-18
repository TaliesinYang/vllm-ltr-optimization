#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ltr_training.label_input import LabelInput  # noqa: E402
from ltr_training.offline_io import (  # noqa: E402
    read_json_records,
    sha256_file,
    write_json,
    write_jsonl,
)
from ltr_training.offline_scoring import (  # noqa: E402
    TransformersCheckpointScorer,
    checkpoint_sha256,
    disagreement_diagnostic,
    score_ensemble_rows,
)

REQUIRED_SEEDS = (17, 42, 73)
DEFAULT_CHECKPOINTS = {
    17: ROOT / "checkpoints_best_predictor",
    42: ROOT / "checkpoints_best_predictor_seed42",
    73: ROOT / "checkpoints_best_predictor_seed73",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score LabelInput rows with three predictor seeds."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--lengths", required=True, type=Path)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="SEED=PATH",
        help="Override a default seed checkpoint path; repeat per seed as needed.",
    )
    parser.add_argument("--scores-output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--diagnostic", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    return parser


def _checkpoints(values: list[str]) -> dict[int, Path]:
    parsed = dict(DEFAULT_CHECKPOINTS)
    for value in values:
        seed_text, separator, path_text = value.partition("=")
        if not separator:
            raise ValueError(f"checkpoint must be SEED=PATH: {value}")
        parsed[int(seed_text)] = Path(path_text)
    if set(parsed) != set(REQUIRED_SEEDS):
        raise ValueError(f"checkpoint seeds must be exactly {list(REQUIRED_SEEDS)}")
    return parsed


def _checkpoint_inventory(checkpoints: dict[int, Path]) -> dict[str, dict[str, object]]:
    inventory: dict[str, dict[str, object]] = {}
    for seed, path in sorted(checkpoints.items()):
        record: dict[str, object] = {"path": str(path)}
        if not path.is_dir():
            record["status"] = "missing"
        else:
            has_config = (path / "config.json").is_file()
            has_tokenizer = any(
                (path / name).is_file()
                for name in ("tokenizer.json", "tokenizer_config.json")
            )
            has_weights = any(
                (path / name).is_file()
                for name in ("model.safetensors", "pytorch_model.bin")
            )
            missing_files = [
                name
                for name, present in (
                    ("config", has_config),
                    ("tokenizer", has_tokenizer),
                    ("weights", has_weights),
                )
                if not present
            ]
            if missing_files:
                record.update(
                    {"status": "incomplete", "missing_components": missing_files}
                )
            else:
                record.update({"status": "present", "sha256": checkpoint_sha256(path)})
        inventory[str(seed)] = record
    return inventory


def main() -> int:
    args = _parser().parse_args()
    checkpoints = _checkpoints(args.checkpoint)
    inventory = _checkpoint_inventory(checkpoints)
    missing_seeds = [
        seed for seed in REQUIRED_SEEDS if inventory[str(seed)]["status"] == "missing"
    ]
    unavailable_seeds = [
        seed for seed in REQUIRED_SEEDS if inventory[str(seed)]["status"] != "present"
    ]
    if unavailable_seeds:
        write_jsonl(args.scores_output, [])
        write_json(
            args.diagnostic,
            {
                "schema_version": "disagreement-diagnostic-v1",
                "status": "blocked",
                "reason": "missing_required_checkpoints",
                "domains": {},
            },
        )
        write_json(
            args.report,
            {
                "schema_version": "offline-ensemble-scores-v1",
                "status": "blocked",
                "reason": "missing_required_checkpoints",
                "required_seeds": list(REQUIRED_SEEDS),
                "missing_seeds": missing_seeds,
                "unavailable_seeds": unavailable_seeds,
                "checkpoints": inventory,
                "input_path": str(args.input),
            },
        )
        return 2

    items = [LabelInput.from_dict(row) for row in read_json_records(args.input)]
    length_rows = read_json_records(args.lengths)
    lengths = {
        str(row.get("sample_id", row.get("request_id"))): int(
            row.get("output_length", row.get("true_length"))
        )
        for row in length_rows
    }
    rows = []
    for item in items:
        if item.sample_id not in lengths:
            raise ValueError(f"missing output length for {item.sample_id}")
        row = item.to_dict()
        row.update(
            {
                "request_id": item.sample_id,
                "true_length": lengths[item.sample_id],
                "domain": "id" if item.source == "toolace" else "ood",
            }
        )
        rows.append(row)
    scorers = {
        seed: TransformersCheckpointScorer(
            path, batch_size=args.batch_size, max_length=args.max_length
        )
        for seed, path in sorted(checkpoints.items())
    }
    scored, report = score_ensemble_rows(
        rows, scorers=scorers, max_length=args.max_length
    )
    diagnostic = disagreement_diagnostic(scored)
    write_jsonl(args.scores_output, scored)
    write_json(args.diagnostic, diagnostic)
    report.update(
        {
            "status": "done",
            "row_count": len(scored),
            "input_sha256": sha256_file(args.input),
            "lengths_sha256": sha256_file(args.lengths),
            "scores_sha256": sha256_file(args.scores_output),
            "diagnostic_sha256": sha256_file(args.diagnostic),
            "request_id_mapping": "request_id == sample_id",
        }
    )
    write_json(args.report, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
