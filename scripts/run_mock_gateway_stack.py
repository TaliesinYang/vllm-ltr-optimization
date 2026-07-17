#!/usr/bin/env python3
"""Run a CPU-only mock VeloxMesh, decision service, and SSE engine stack."""

import argparse
import logging
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scheduler_benchmark.decision_service import DecisionApplication
from scheduler_benchmark.mock_stack import MockGatewayStack
from scheduler_benchmark.predictor import ConstantPredictor


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--gateway-port", type=int, default=8080)
    parser.add_argument("--decision-port", type=int, default=8081)
    parser.add_argument("--engine-port", type=int, default=8082)
    parser.add_argument("--score", type=float, default=0.5)
    parser.add_argument("--confidence", type=float, default=0.9)
    parser.add_argument("--ood", action="store_true")
    parser.add_argument("--decision-timeout-s", type=float, default=2.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    application = DecisionApplication(
        predictor=ConstantPredictor(args.score, args.confidence, args.ood),
        predictor_revision="stub-constant-v1",
        feature_variant="prompt",
    )
    stack = MockGatewayStack(
        application,
        host=args.host,
        gateway_port=args.gateway_port,
        decision_port=args.decision_port,
        engine_port=args.engine_port,
        decision_timeout_s=args.decision_timeout_s,
    )
    with stack:
        logging.info("gateway: %s", stack.gateway_endpoint)
        logging.info("decision: %s", stack.decision_endpoint)
        logging.info("engine: %s", stack.engine_endpoint)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
