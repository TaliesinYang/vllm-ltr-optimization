"""E3 - LightGBM scalar baseline rerun under the tier-2 recipe for 3 seeds.

Recipe is copied verbatim from ltr_training.tier2_training.train_lightgbm_tier2
(LGBMRegressor n_estimators=300, learning_rate=0.05, num_leaves=31) over
ltr_training.training_matrix.structural_features, on the same fixed tier-2
splits the BERT matrix used. Only random_state changes across seeds, because
the tier-2 splits are pre-assigned in the sample file and are seed-independent.

Both the recipe's scipy kendalltau and offline_statistics.kendall_tau_b are
reported so the rerun can be checked against the recorded seed-42 number.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from lightgbm import LGBMRegressor
from scipy.stats import kendalltau

import common

from ltr_training.offline_statistics import kendall_tau_b

OUT = Path(__file__).resolve().parent / "e3-lightgbm-seeds.json"
PREDICTIONS_OUT = Path(__file__).resolve().parent / "e3-lightgbm-test-predictions.jsonl"


def session_bootstrap_ci(
    examples, predictions: list[float], *, iterations: int = 1000, seed: int = 42
) -> dict[str, object]:
    """Session-clustered bootstrap CI for test tau-b.

    Mirrors offline_statistics.cluster_bootstrap_tau_b (same clustering, same
    percentile CI) but calls scipy's kendalltau inside the loop; the two tau
    implementations agree to machine precision on this data (checked per seed
    in the report), and scipy's O(n log n) kernel makes 1000 iterations cheap.
    """
    import random

    grouped: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        grouped.setdefault(example.session_id, []).append(index)
    clusters = sorted(grouped)
    rng = random.Random(seed)
    truths = [float(item.output_length) for item in examples]
    samples: list[float] = []
    for _ in range(iterations):
        indices = [
            index
            for cluster in (rng.choice(clusters) for _ in clusters)
            for index in grouped[cluster]
        ]
        statistic = kendalltau(
            [predictions[index] for index in indices],
            [truths[index] for index in indices],
        ).statistic
        samples.append(float(statistic))
    samples.sort()
    return {
        "iterations": iterations,
        "bootstrap_seed": seed,
        "cluster_key": "session_id",
        "cluster_count": len(clusters),
        "ci95_percentile": [
            samples[int(0.025 * (iterations - 1))],
            samples[int(0.975 * (iterations - 1))],
        ],
    }


def main() -> None:
    started = time.time()
    inputs = common.verify_inputs()
    splits, counts = common.load_splits()

    features = {
        split: [common.structural_features(item) for item in splits[split]]
        for split in common.SPLITS
    }
    labels = {
        split: [float(item.output_length) for item in splits[split]]
        for split in common.SPLITS
    }

    runs: list[dict[str, object]] = []
    test_predictions_by_seed: dict[int, list[float]] = {}
    for seed in common.SEEDS:
        model = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=seed,
            verbosity=-1,
        )
        model.fit(features["train"], labels["train"])
        record: dict[str, object] = {"seed": seed}
        for split in ("validation", "test"):
            predictions = [float(value) for value in model.predict(features[split])]
            scipy_tau = kendalltau(predictions, labels[split])
            record[f"{split}_tau_scipy"] = float(scipy_tau.statistic)
            record[f"{split}_pvalue_scipy"] = float(scipy_tau.pvalue)
            record[f"{split}_tau_b_repo"] = kendall_tau_b(labels[split], predictions)
            record[f"{split}_examples"] = len(splits[split])
            if split == "test":
                test_predictions_by_seed[seed] = predictions
        record["tau_implementations_agree"] = (
            abs(float(record["test_tau_scipy"]) - float(record["test_tau_b_repo"])) < 1e-12
        )
        record["test_bootstrap"] = session_bootstrap_ci(
            splits["test"], test_predictions_by_seed[seed]
        )
        runs.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    with PREDICTIONS_OUT.open("w", encoding="utf-8") as handle:
        for index, example in enumerate(splits["test"]):
            handle.write(
                json.dumps(
                    {
                        "sample_id": example.sample_id,
                        "session_id": example.session_id,
                        "true_length": example.output_length,
                        "prediction": {
                            str(seed): test_predictions_by_seed[seed][index]
                            for seed in common.SEEDS
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    test_taus = [float(run["test_tau_b_repo"]) for run in runs]
    validation_taus = [float(run["validation_tau_b_repo"]) for run in runs]
    report = {
        "schema_version": "e3-lightgbm-seeds-v1",
        "status": "done",
        "experiment": "E3",
        "model": "lightgbm-structural (scalar tool features)",
        "recipe_source": "ltr_training.tier2_training.train_lightgbm_tier2",
        "features": [
            "prompt_char_count",
            "prompt_word_count",
            "tool_schema_char_count",
            "tool_name_token_count",
            "history_turn_count",
        ],
        "hyperparameters": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
        },
        "seed_controls": "LGBMRegressor.random_state only; tier-2 splits are "
        "fixed by the sample file's tier2_split column and do not depend on seed",
        "determinism_note": "subsample=1.0, subsample_freq=0, colsample_bytree=1.0 "
        "in this recipe, so no stochastic sampling consumes random_state and the "
        "fit is deterministic; zero seed variance is expected by construction",
        "predictions_path": str(PREDICTIONS_OUT),
        "inputs": inputs,
        "split_sizes": {split: len(splits[split]) for split in common.SPLITS},
        "censor_exclusion_counts": counts,
        "runs": runs,
        "test_tau_b_mean": statistics.fmean(test_taus),
        "test_tau_b_stdev": statistics.stdev(test_taus),
        "validation_tau_b_mean": statistics.fmean(validation_taus),
        "validation_tau_b_stdev": statistics.stdev(validation_taus),
        "wall_clock_seconds": time.time() - started,
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in (
        "test_tau_b_mean", "test_tau_b_stdev", "wall_clock_seconds"
    )}, indent=2))


if __name__ == "__main__":
    main()
