#!/usr/bin/env python3
"""Run the replay-harness BERT predictor gRPC worker."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scheduler_benchmark.grpc_worker import load_worker, start_server  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--addr", default=":50052")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--model-version")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    worker = load_worker(
        checkpoint=args.checkpoint,
        sidecar_path=args.sidecar,
        manifest_path=args.manifest,
    )
    if args.model_version and args.model_version != worker.model_version:
        raise SystemExit(
            "model version mismatch: "
            f"manifest={worker.model_version} expected={args.model_version}"
        )
    server, port = start_server(worker, args.addr, max_workers=args.max_workers)
    print(
        json.dumps(
            {
                "ready": True,
                "addr": args.addr,
                "bound_port": port,
                "model_version": worker.model_version,
                "mapping_version": worker.mapping_version,
                "calibrated": False,
                "warning": worker.approximation_notice,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=1).wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
