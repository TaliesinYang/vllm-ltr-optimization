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
from scheduler_benchmark.predictor import ConstantPredictor


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--score", type=float, default=0.5)
    parser.add_argument("--confidence", type=float, default=0.9)
    parser.add_argument("--ood", action="store_true")
    parser.add_argument(
        "--feature-variant",
        choices=tuple(FEATURE_VARIANTS),
        default="prompt_schema_history_workflow",
    )
    parser.add_argument("--predictor-revision", default="stub-constant-v1")
    parser.add_argument("--max-body-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--max-concurrency", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    application = DecisionApplication(
        predictor=ConstantPredictor(args.score, args.confidence, args.ood),
        predictor_revision=args.predictor_revision,
        feature_variant=args.feature_variant,
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
