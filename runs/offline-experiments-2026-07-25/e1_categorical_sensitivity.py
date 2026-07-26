"""E1 robustness check: is the dead categorical feature just LightGBM's guardrails?

LightGBM refuses categorical splits on levels with fewer than
``min_data_per_group`` rows (default 100). The tool-set fingerprint averages
~1.3 train rows per level, so the default settings can never split on it. This
re-runs E1a with those guardrails relaxed to their loosest legal values, to
check that the null result is a property of the feature and not of the defaults.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from lightgbm import LGBMRegressor

import common
from e1_schema_hash import build_categorical_codes

from ltr_training.offline_statistics import kendall_tau_b

OUT = Path(__file__).resolve().parent / "e1-categorical-sensitivity.json"

SETTINGS = (
    {"name": "default_guardrails", "params": {}},
    {
        "name": "relaxed_guardrails",
        "params": {
            "min_data_per_group": 1,
            "cat_smooth": 1.0,
            "cat_l2": 1.0,
            "max_cat_threshold": 512,
            "max_cat_to_onehot": 1,
        },
    },
)


def main() -> None:
    started = time.time()
    common.verify_inputs()
    splits, _ = common.load_splits()
    scalars = {
        split: [common.structural_features(item) for item in splits[split]]
        for split in common.SPLITS
    }
    labels = {
        split: [float(item.output_length) for item in splits[split]]
        for split in common.SPLITS
    }
    train_codes, _ = build_categorical_codes(splits["train"], splits["train"])
    matrices = {
        "train": np.array(
            [row + [code] for row, code in zip(scalars["train"], train_codes)]
        )
    }
    for split in ("validation", "test"):
        _, codes = build_categorical_codes(splits["train"], splits[split])
        matrices[split] = np.array(
            [row + [code] for row, code in zip(scalars[split], codes)]
        )

    results: list[dict[str, object]] = []
    for setting in SETTINGS:
        for seed in common.SEEDS:
            model = LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                random_state=seed,
                verbosity=-1,
                **setting["params"],
            )
            model.fit(matrices["train"], labels["train"], categorical_feature=[5])
            predictions = [float(v) for v in model.predict(matrices["test"])]
            results.append(
                {
                    "setting": setting["name"],
                    "seed": seed,
                    "test_tau_b": kendall_tau_b(labels["test"], predictions),
                    "fingerprint_split_count": int(
                        model.booster_.feature_importance("split")[5]
                    ),
                    "fingerprint_gain": float(
                        model.booster_.feature_importance("gain")[5]
                    ),
                }
            )
            print(json.dumps(results[-1], sort_keys=True), flush=True)

    report = {
        "schema_version": "e1-categorical-sensitivity-v1",
        "status": "done",
        "purpose": "confirm the null categorical result is not an artifact of "
        "LightGBM's default min_data_per_group=100 guardrail",
        "settings": SETTINGS,
        "scalar_only_reference_test_tau_b": 0.4267985278708031,
        "results": results,
        "wall_clock_seconds": time.time() - started,
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
