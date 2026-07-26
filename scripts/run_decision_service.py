#!/usr/bin/env python3
"""Run the CPU `/v1/decision` development service for VeloxMesh."""

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scheduler_benchmark.decision_service import (
    FEATURE_VARIANTS,
    DecisionApplication,
    create_decision_server,
)
from scheduler_benchmark.predictor import BertPredictor, ConstantPredictor, Predictor
from scheduler_benchmark.rank_quantiles import RankQuantileMapper


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
    parser.add_argument("--quantile-manifest", type=Path, required=True)
    parser.add_argument("--max-body-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument(
        "--reliability-threshold",
        type=float,
        default=None,
        help="Override DEFAULT_RELIABILITY_THRESHOLD (0.8). Rule C confidences "
        "top out at 0.6233, so a functional gate needs <= 0.5787.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Predictor device. cuda falls back to cpu, with a startup log line, "
        "when torch.cuda.is_available() is False.",
    )
    parser.add_argument(
        "--batch-max",
        type=int,
        default=1,
        help="Coalesce up to this many concurrent requests into one forward; "
        "1 disables batching. The 201-box GPU prototype used 8.",
    )
    parser.add_argument(
        "--batch-window-ms",
        type=float,
        default=3.0,
        help="How long the batcher waits to fill a batch (prototype: 3ms). "
        "Ignored when --batch-max is 1.",
    )
    return parser.parse_args(argv)


def build_predictor(args: argparse.Namespace) -> Predictor:
    if args.predictor == "stub":
        return ConstantPredictor(args.score, args.confidence, args.ood)
    return BertPredictor(
        args.checkpoint,
        device=args.device,
        batch_max=args.batch_max,
        batch_window_ms=args.batch_window_ms,
    )


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


def load_quantile_manifest(path: Path) -> tuple[RankQuantileMapper, str]:
    try:
        raw_manifest = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"quantile manifest could not be read: {exc}") from exc
    manifest_sha256 = hashlib.sha256(raw_manifest).hexdigest()
    try:
        decoded = raw_manifest.decode("utf-8")
        manifest = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"quantile manifest is malformed JSON: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("quantile manifest must be a JSON object")
    try:
        mapper = RankQuantileMapper(manifest)
    except (KeyError, OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid quantile manifest: {exc}") from exc
    return mapper, manifest_sha256


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        quantile_mapper, quantile_manifest_sha256 = load_quantile_manifest(
            args.quantile_manifest
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    application_kwargs = dict(
        predictor=build_predictor(args),
        predictor_revision=effective_predictor_revision(args),
        feature_variant=effective_feature_variant(args),
        max_concurrency=args.max_concurrency,
        quantile_mapper=quantile_mapper,
        quantile_manifest_sha256=quantile_manifest_sha256,
    )
    if args.reliability_threshold is not None:
        application_kwargs["reliability_threshold"] = args.reliability_threshold
    application = DecisionApplication(**application_kwargs)
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
