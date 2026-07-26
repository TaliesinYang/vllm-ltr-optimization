"""T5 step 1 - score the validation split with the prompt_schema checkpoints.

The gate's confidence values must be FIT on data that is not the data they are
evaluated on. T1's per-stratum tau is measured on the test split, so deriving
confidence from it and then validating on test would be circular: the numbers
would agree by construction and prove nothing.

This scores the validation split so the gate can be fit there and evaluated on
test. Test scores already exist in e2-bert-test-scores.jsonl.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import common

from ltr_training.offline_scoring import TransformersCheckpointScorer
from ltr_training.train_ranker import render_example

HERE = Path(__file__).resolve().parent
MATRIX = Path("/Volumes/T7 Shield/vllm-ltr-results/extracted/tier2-matrix")
OUT = HERE / "t5-bert-validation-scores.jsonl"
META = HERE / "t5-validation-scoring.json"
VARIANT = "prompt_schema"


def main() -> None:
    started = time.time()
    inputs = common.verify_inputs()
    splits, _ = common.load_splits()
    rows = splits["validation"]
    texts = [render_example(item, variant=VARIANT) for item in rows]

    scores: dict[int, list[float]] = {}
    hashes: dict[str, str] = {}
    for seed in common.SEEDS:
        checkpoint = MATRIX / f"bert-{VARIANT}-tier2-seed{seed}" / "final"
        if not checkpoint.exists():
            raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
        elapsed = time.time()
        scorer = TransformersCheckpointScorer(checkpoint, batch_size=32, max_length=512)
        values, _ = scorer.score(texts)
        scores[seed] = values
        hashes[f"seed{seed}"] = scorer.checkpoint_sha256
        print(f"scored validation seed{seed} in {time.time() - elapsed:.1f}s", flush=True)

    with OUT.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(rows):
            handle.write(
                json.dumps(
                    {
                        "sample_id": item.sample_id,
                        "session_id": item.session_id,
                        "true_length": item.output_length,
                        **{
                            f"bert_{VARIANT}_seed{seed}": scores[seed][index]
                            for seed in common.SEEDS
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    META.write_text(
        json.dumps(
            {
                "schema_version": "t5-validation-scoring-v1",
                "variant": VARIANT,
                "rows": len(rows),
                "inputs": inputs,
                "checkpoint_root": str(MATRIX),
                "checkpoint_sha256": hashes,
                "scoring": {"max_length": 512, "batch_size": 32, "device": "cpu"},
                "scores_path": str(OUT),
                "wall_clock_seconds": time.time() - started,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"done in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
