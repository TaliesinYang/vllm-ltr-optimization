#!/usr/bin/env python3
"""Run the CPU `/v1/decision` development service for VeloxMesh."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scheduler_benchmark.decision_service import (
    FEATURE_VARIANTS,
    DecisionApplication,
    create_decision_server,
)
from scheduler_benchmark.predictor import BertPredictor, ConstantPredictor, Predictor


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BERT_CHECKPOINT = REPO_ROOT / "checkpoints_best_predictor"
DEFAULT_STUB_REVISION = "stub-constant-v1"
DEFAULT_BERT_REVISION = "bert-prompt_schema-tier2-seed17"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--predictor", choices=("stub", "bert"), default="stub")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_BERT_CHECKPOINT)
    parser.add_argument("--score", type=float, default=0.5)
    parser.add_argument("--confidence", type=float, default=0.9)
    parser.add_argument("--ood", action="store_true")
    parser.add_argument(
        "--feature-variant",
        choices=tuple(FEATURE_VARIANTS),
        default=None,
    )
    parser.add_argument("--predictor-revision")
    parser.add_argument("--max-body-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--max-concurrency", type=int, default=8)
    return parser.parse_args(argv)


def build_predictor(args: argparse.Namespace) -> Predictor:
    if args.predictor == "stub":
        return ConstantPredictor(args.score, args.confidence, args.ood)
    return BertPredictor(args.checkpoint)


def effective_feature_variant(args: argparse.Namespace) -> str:
    if args.predictor == "bert":
        if args.feature_variant not in (None, "prompt_schema"):
            raise ValueError("BERT checkpoint requires feature variant prompt_schema")
        return "prompt_schema"
    return args.feature_variant or "prompt_schema_history_workflow"


def effective_predictor_revision(args: argparse.Namespace) -> str:
    if args.predictor_revision:
        return args.predictor_revision
    if args.predictor == "bert":
        return DEFAULT_BERT_REVISION
    return DEFAULT_STUB_REVISION


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    application = DecisionApplication(
        predictor=build_predictor(args),
        predictor_revision=effective_predictor_revision(args),
        feature_variant=effective_feature_variant(args),
        max_concurrency=args.max_concurrency,
    )
    server = create_decision_server(
        application,
        host=args.host,
        port=args.port,
        max_body_bytes=args.max_body_bytes,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
