"""E2 - cold-start evaluation on unseen tool sets.

No retraining. The tier-2 split is reconstructed exactly as the recipe builds it
(it is fixed by the sample file's tier2_split column, not by seed), then the
test split is cut into nested subsets by how much of its tool vocabulary the
train split ever saw:

  seen_combination   tool-set fingerprint appears in train
  unseen_combination tool-set fingerprint never appears in train
  unseen_tools       no individual tool name appears in any train row (strict;
                     rows advertising no tools at all are excluded)

unseen_tools is nested inside unseen_combination by construction.

Existing checkpoints are scored once over the full test split and then sliced,
so tau on a subset is computed from exactly that subset's rows. LightGBM scalar
(E3) and the schema-identity models (E1) are read back from their saved
per-row test predictions, so every model is compared on identical rows.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from scipy.stats import kendalltau

import common

from ltr_training.offline_scoring import TransformersCheckpointScorer
from ltr_training.offline_statistics import kendall_tau_b
from ltr_training.train_ranker import render_example

HERE = Path(__file__).resolve().parent
OUT = HERE / "e2-cold-start.json"
SCORES_OUT = HERE / "e2-bert-test-scores.jsonl"
MATRIX = Path("/Volumes/T7 Shield/vllm-ltr-results/extracted/tier2-matrix")

BERT_VARIANTS = ("prompt_schema", "prompt_only")
SUBSETS = ("all", "seen_combination", "unseen_combination", "unseen_tools")


def checkpoint_path(variant: str, seed: int) -> Path:
    return MATRIX / f"bert-{variant}-tier2-seed{seed}" / "final"


def build_subsets(splits) -> tuple[dict[str, list[int]], dict[str, object]]:
    train_fingerprints = {
        common.toolset_fingerprint(item.tool_schema) for item in splits["train"]
    }
    train_tools = {
        name for item in splits["train"] for name in common.tool_names(item.tool_schema)
    }
    test = splits["test"]
    subsets: dict[str, list[int]] = {name: [] for name in SUBSETS}
    toolless = 0
    for index, item in enumerate(test):
        subsets["all"].append(index)
        fingerprint = common.toolset_fingerprint(item.tool_schema)
        names = common.tool_names(item.tool_schema)
        if fingerprint in train_fingerprints:
            subsets["seen_combination"].append(index)
        else:
            subsets["unseen_combination"].append(index)
        if not names:
            toolless += 1
        elif not any(name in train_tools for name in names):
            subsets["unseen_tools"].append(index)

    strict = set(subsets["unseen_tools"])
    nested = strict.issubset(set(subsets["unseen_combination"]))
    definition = {
        "identity_key": "sha256 of sorted top-level tool-name list",
        "train_unique_fingerprints": len(train_fingerprints),
        "train_unique_tool_names": len(train_tools),
        "test_rows_advertising_no_tools": toolless,
        "unseen_tools_nested_in_unseen_combination": nested,
        "sizes": {name: len(indices) for name, indices in subsets.items()},
    }
    return subsets, definition


def session_bootstrap_ci(examples, predictions, indices, *, iterations=1000, seed=42):
    import random

    grouped: dict[str, list[int]] = {}
    for index in indices:
        grouped.setdefault(examples[index].session_id, []).append(index)
    clusters = sorted(grouped)
    if len(clusters) < 2:
        return None
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(iterations):
        sampled = [
            index
            for cluster in (rng.choice(clusters) for _ in clusters)
            for index in grouped[cluster]
        ]
        statistic = kendalltau(
            [predictions[index] for index in sampled],
            [float(examples[index].output_length) for index in sampled],
        ).statistic
        samples.append(0.0 if statistic != statistic else float(statistic))
    samples.sort()
    return {
        "iterations": iterations,
        "cluster_count": len(clusters),
        "ci95_percentile": [
            samples[int(0.025 * (iterations - 1))],
            samples[int(0.975 * (iterations - 1))],
        ],
    }


def tau_on(examples, predictions: list[float], indices: list[int]) -> float:
    return kendall_tau_b(
        [float(examples[index].output_length) for index in indices],
        [predictions[index] for index in indices],
    )


def load_saved_predictions(path: Path, key: str, test_examples) -> dict[int, list[float]]:
    rows = {json.loads(line)["sample_id"]: json.loads(line) for line in path.open()}
    if len(rows) != len(test_examples):
        raise ValueError(f"{path} has {len(rows)} rows, expected {len(test_examples)}")
    return {
        seed: [float(rows[item.sample_id][key][str(seed)]) for item in test_examples]
        for seed in common.SEEDS
    }


def main() -> None:
    started = time.time()
    inputs = common.verify_inputs()
    splits, counts = common.load_splits()
    test = splits["test"]
    subsets, definition = build_subsets(splits)
    print(json.dumps(definition, indent=2, sort_keys=True), flush=True)

    predictions: dict[str, dict[int, list[float]]] = {}
    checkpoint_hashes: dict[str, str] = {}

    for variant in BERT_VARIANTS:
        texts = [render_example(item, variant=variant) for item in test]
        for seed in common.SEEDS:
            path = checkpoint_path(variant, seed)
            if not path.exists():
                raise FileNotFoundError(f"missing checkpoint: {path}")
            elapsed = time.time()
            scorer = TransformersCheckpointScorer(path, batch_size=32, max_length=512)
            scores, _ = scorer.score(texts)
            predictions.setdefault(f"bert_{variant}", {})[seed] = scores
            checkpoint_hashes[f"bert_{variant}_seed{seed}"] = scorer.checkpoint_sha256
            print(
                f"scored bert-{variant}-seed{seed} in {time.time() - elapsed:.1f}s",
                flush=True,
            )

    predictions["lightgbm_scalar"] = load_saved_predictions(
        HERE / "e3-lightgbm-test-predictions.jsonl", "prediction", test
    )
    for key, label in (("e1a_categorical", "schema_hash_categorical"), ("e1b_lookup", "schema_hash_lookup")):
        predictions[label] = load_saved_predictions(
            HERE / "e1-schema-hash-test-predictions.jsonl", key, test
        )

    with SCORES_OUT.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(test):
            handle.write(
                json.dumps(
                    {
                        "sample_id": item.sample_id,
                        "session_id": item.session_id,
                        "true_length": item.output_length,
                        "subsets": [
                            name
                            for name in SUBSETS
                            if name != "all" and index in set(subsets[name])
                        ],
                        **{
                            f"{model}_seed{seed}": values[seed][index]
                            for model, values in predictions.items()
                            for seed in common.SEEDS
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    table: dict[str, dict[str, object]] = {}
    for model, by_seed in predictions.items():
        table[model] = {}
        for subset in SUBSETS:
            indices = subsets[subset]
            per_seed = {
                seed: tau_on(test, by_seed[seed], indices) for seed in common.SEEDS
            }
            values = list(per_seed.values())
            table[model][subset] = {
                "n": len(indices),
                "per_seed_tau_b": {str(seed): value for seed, value in per_seed.items()},
                "mean_tau_b": statistics.fmean(values),
                "stdev_tau_b": statistics.stdev(values),
                "bootstrap_seed17": session_bootstrap_ci(
                    test, by_seed[common.SEEDS[0]], indices
                ),
            }
            print(
                f"{model:28s} {subset:20s} n={len(indices):4d} "
                f"tau={statistics.fmean(values):.4f}+-{statistics.stdev(values):.4f}",
                flush=True,
            )

    report = {
        "schema_version": "e2-cold-start-v1",
        "status": "done",
        "experiment": "E2",
        "question": "does schema TEXT hold its tau on unseen tool sets while "
        "hash/lookup collapses?",
        "split_note": "the tier-2 split is fixed by the sample file's tier2_split "
        "column and does not depend on seed, so there is one split, not three",
        "subset_definition": definition,
        "inputs": inputs,
        "split_sizes": {split: len(splits[split]) for split in common.SPLITS},
        "censor_exclusion_counts": counts,
        "checkpoint_root": str(MATRIX),
        "checkpoint_sha256": checkpoint_hashes,
        "scoring": {"max_length": 512, "batch_size": 32, "device": "cpu"},
        "scores_path": str(SCORES_OUT),
        "results": table,
        "wall_clock_seconds": time.time() - started,
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"done in {report['wall_clock_seconds']:.1f}s")


if __name__ == "__main__":
    main()
