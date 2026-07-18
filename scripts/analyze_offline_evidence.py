#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ltr_training.offline_io import read_json_records, sha256_file, write_json
from ltr_training.offline_statistics import (
    canonical_schema_overlap_report,
    cluster_bootstrap_tau_b,
    session_overlap_report,
    tie_proportions,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute frozen offline CIs, ties, and leakage checks.")
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--ood", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _cluster_id(row: dict[str, object]) -> str:
    source = str(row.get("source", ""))
    if source == "bfcl":
        return str(row.get("task_id", row.get("session_id")))
    if source == "toolathlon":
        return str(row.get("task_id"))
    return str(row.get("session_id"))


def main() -> int:
    args = _parser().parse_args()
    scores = read_json_records(args.scores)
    by_domain: dict[str, list[dict[str, object]]] = {}
    for row in scores:
        copied = dict(row)
        copied["_cluster_id"] = _cluster_id(copied)
        by_domain.setdefault(str(copied["domain"]), []).append(copied)
    statistics: dict[str, object] = {}
    for domain, rows in sorted(by_domain.items()):
        prediction_keys = sorted(
            {"ensemble_rank"}
            | {key for row in rows for key in row if key.startswith("rank_seed")}
        )
        statistics[domain] = {
            key: {
                "tau": cluster_bootstrap_tau_b(
                    rows,
                    truth_key="true_length",
                    prediction_key=key,
                    cluster_key="_cluster_id",
                    iterations=args.iterations,
                    seed=args.seed,
                ),
                "ties": tie_proportions(
                    [row["true_length"] for row in rows],
                    [row[key] for row in rows],
                ),
            }
            for key in prediction_keys
        }

    split_rows = {
        name: read_json_records(path)
        for name, path in (
            ("train", args.train),
            ("validation", args.validation),
            ("test", args.test),
            ("ood", args.ood),
        )
    }
    report = {
        "schema_version": "offline-evidence-analysis-v1",
        "status": "done",
        "kendalltau_variant": "b",
        "bootstrap": {
            "kind": "session-level cluster bootstrap",
            "ci": "95% percentile",
            "iterations": args.iterations,
            "seed": args.seed,
            "cluster_units": {
                "bfcl": "id/task_id",
                "toolathlon": "canonical task_name/task_id",
                "toolace": "session_id",
            },
        },
        "statistics": statistics,
        "leakage": {
            "schema_hash": canonical_schema_overlap_report(
                (str(row["tool_schema"]) for row in split_rows["train"]),
                (str(row["tool_schema"]) for row in split_rows["validation"]),
                (str(row["tool_schema"]) for row in split_rows["test"]),
                (str(row["tool_schema"]) for row in split_rows["ood"]),
            ),
            "session_hash": session_overlap_report(
                (row["session_id"] for row in split_rows["train"]),
                (row["session_id"] for row in split_rows["validation"]),
                (row["session_id"] for row in split_rows["test"]),
                (row["session_id"] for row in split_rows["ood"]),
            ),
        },
        "inputs_sha256": {
            "scores": sha256_file(args.scores),
            "train": sha256_file(args.train),
            "validation": sha256_file(args.validation),
            "test": sha256_file(args.test),
            "ood": sha256_file(args.ood),
        },
    }
    write_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
