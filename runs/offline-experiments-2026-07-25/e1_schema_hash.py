"""E1 - schema-IDENTITY baselines: does knowing *which* tool set it is suffice?

Tests whether the schema's identity (a hash), rather than its text, carries the
signal that the BERT schema-text model extracts. Two encodings of the identity
are run on top of the existing five scalars, on the same fixed tier-2 splits:

  E1a  scalar + LightGBM native categorical over the tool-set fingerprint
  E1b  scalar + per-fingerprint train-label lookup (target encoding)

E1b is the deployed per-client lookup table: for a fingerprint seen in
training, look up the mean/median output length of the rows that shared it.
Unseen fingerprints fall back to the global train mean/median, count 0, seen 0.

Target encoding is used rather than native categorical as the headline variant
because (a) the fingerprint has 3162 levels over 3997 train rows, so native
categorical splits degenerate into near-per-row memorisation, and (b) target
encoding *is* the lookup table whose deployability is the question. Train-row
encodings are computed out-of-fold (K=5, fold assignment seeded per run) so the
model never sees its own label; validation/test use full-train statistics.
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

OUT = Path(__file__).resolve().parent / "e1-schema-hash.json"
PREDICTIONS_OUT = Path(__file__).resolve().parent / "e1-schema-hash-test-predictions.jsonl"
FOLDS = 5


def _fingerprints(examples) -> list[str]:
    return [common.toolset_fingerprint(item.tool_schema) for item in examples]


def _fold_assignment(count: int, *, seed: int) -> list[int]:
    import random

    folds = [index % FOLDS for index in range(count)]
    random.Random(seed).shuffle(folds)
    return folds


def _lookup(labels: list[float], indices: list[int]) -> tuple[float, float, int]:
    values = [labels[index] for index in indices]
    return statistics.fmean(values), statistics.median(values), len(values)


def build_lookup_features(
    train_examples,
    train_labels: list[float],
    target_examples,
    *,
    seed: int,
    out_of_fold: bool,
) -> tuple[list[list[float]], int]:
    """Per-fingerprint (mean, median, count, seen) with global-mean fallback."""
    train_fingerprints = _fingerprints(train_examples)
    global_mean = statistics.fmean(train_labels)
    global_median = statistics.median(train_labels)

    by_fingerprint: dict[str, list[int]] = {}
    for index, fingerprint in enumerate(train_fingerprints):
        by_fingerprint.setdefault(fingerprint, []).append(index)

    if out_of_fold:
        folds = _fold_assignment(len(train_examples), seed=seed)
        rows: list[list[float]] = []
        hits = 0
        for index, fingerprint in enumerate(train_fingerprints):
            peers = [
                peer
                for peer in by_fingerprint[fingerprint]
                if folds[peer] != folds[index]
            ]
            if peers:
                mean, median, count = _lookup(train_labels, peers)
                hits += 1
            else:
                mean, median, count = global_mean, global_median, 0
            rows.append([mean, median, float(count), 1.0 if peers else 0.0])
        return rows, hits

    rows = []
    hits = 0
    for fingerprint in _fingerprints(target_examples):
        peers = by_fingerprint.get(fingerprint)
        if peers:
            mean, median, count = _lookup(train_labels, peers)
            hits += 1
        else:
            mean, median, count = global_mean, global_median, 0
        rows.append([mean, median, float(count), 1.0 if peers else 0.0])
    return rows, hits


def build_categorical_codes(train_examples, target_examples) -> tuple[list[float], list[float]]:
    """Ordinal codes for the fingerprint; unseen fingerprints share code -1."""
    train_fingerprints = _fingerprints(train_examples)
    codes = {fingerprint: index for index, fingerprint in enumerate(sorted(set(train_fingerprints)))}
    return (
        [float(codes[fingerprint]) for fingerprint in train_fingerprints],
        [float(codes.get(fingerprint, -1)) for fingerprint in _fingerprints(target_examples)],
    )


def evaluate(model, features, examples) -> tuple[float, list[float]]:
    predictions = [float(value) for value in model.predict(features)]
    truths = [float(item.output_length) for item in examples]
    return kendall_tau_b(truths, predictions), predictions


def main() -> None:
    started = time.time()
    inputs = common.verify_inputs()
    splits, counts = common.load_splits()

    scalars = {
        split: [common.structural_features(item) for item in splits[split]]
        for split in common.SPLITS
    }
    labels = {
        split: [float(item.output_length) for item in splits[split]]
        for split in common.SPLITS
    }
    train_examples = splits["train"]
    train_labels = labels["train"]

    # Identity coverage - the context that makes the result interpretable.
    train_fingerprints = set(_fingerprints(train_examples))
    coverage = {
        "identity_key": "sha256 of sorted top-level tool-name list",
        "train_unique_fingerprints": len(train_fingerprints),
        "train_rows": len(train_examples),
        **{
            f"{split}_rows_with_fingerprint_seen_in_train": sum(
                1
                for fingerprint in _fingerprints(splits[split])
                if fingerprint in train_fingerprints
            )
            for split in ("validation", "test")
        },
    }

    runs: list[dict[str, object]] = []
    test_predictions: dict[str, dict[int, list[float]]] = {"e1a": {}, "e1b": {}}
    for seed in common.SEEDS:
        # --- E1a: native categorical over the fingerprint ---------------------
        train_codes, _ = build_categorical_codes(train_examples, train_examples)
        categorical_features = {
            "train": [row + [code] for row, code in zip(scalars["train"], train_codes)]
        }
        for split in ("validation", "test"):
            _, target_codes = build_categorical_codes(train_examples, splits[split])
            categorical_features[split] = [
                row + [code] for row, code in zip(scalars[split], target_codes)
            ]
        model_a = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=seed,
            verbosity=-1,
        )
        model_a.fit(
            categorical_features["train"],
            train_labels,
            categorical_feature=[5],
        )

        # --- E1b: per-fingerprint train-label lookup --------------------------
        lookup_train, oof_hits = build_lookup_features(
            train_examples, train_labels, train_examples, seed=seed, out_of_fold=True
        )
        lookup_features = {
            "train": [row + extra for row, extra in zip(scalars["train"], lookup_train)]
        }
        lookup_hits: dict[str, int] = {}
        for split in ("validation", "test"):
            rows, hits = build_lookup_features(
                train_examples, train_labels, splits[split], seed=seed, out_of_fold=False
            )
            lookup_hits[split] = hits
            lookup_features[split] = [
                row + extra for row, extra in zip(scalars[split], rows)
            ]
        model_b = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=seed,
            verbosity=-1,
        )
        model_b.fit(lookup_features["train"], train_labels)

        record: dict[str, object] = {
            "seed": seed,
            "train_out_of_fold_lookup_hits": oof_hits,
            "lookup_hits": lookup_hits,
        }
        for name, model, feature_set, store in (
            ("e1a_categorical", model_a, categorical_features, "e1a"),
            ("e1b_lookup", model_b, lookup_features, "e1b"),
        ):
            for split in ("validation", "test"):
                tau, predictions = evaluate(model, feature_set[split], splits[split])
                record[f"{name}_{split}_tau_b"] = tau
                record[f"{name}_{split}_tau_scipy"] = float(
                    kendalltau(predictions, labels[split]).statistic
                )
                if split == "test":
                    test_predictions[store][seed] = predictions
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
                        "e1a_categorical": {
                            str(seed): test_predictions["e1a"][seed][index]
                            for seed in common.SEEDS
                        },
                        "e1b_lookup": {
                            str(seed): test_predictions["e1b"][seed][index]
                            for seed in common.SEEDS
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    summary = {}
    for name in ("e1a_categorical", "e1b_lookup"):
        for split in ("validation", "test"):
            values = [float(run[f"{name}_{split}_tau_b"]) for run in runs]
            summary[f"{name}_{split}_tau_b_mean"] = statistics.fmean(values)
            summary[f"{name}_{split}_tau_b_stdev"] = statistics.stdev(values)

    report = {
        "schema_version": "e1-schema-hash-v1",
        "status": "done",
        "experiment": "E1",
        "question": "does schema IDENTITY + history match schema TEXT on the seen split?",
        "variants": {
            "e1a_categorical": "5 scalars + LightGBM native categorical over tool-set "
            "fingerprint (unseen fingerprints share code -1)",
            "e1b_lookup": "5 scalars + per-fingerprint train-label lookup "
            "(mean, median, count, seen-flag); unseen falls back to global train "
            "mean/median, count 0, seen 0",
        },
        "encoding_justification": "fingerprint cardinality is "
        "high relative to train size, so native categorical splits degenerate "
        "into near-per-row memorisation; target encoding is also the literal "
        "deployed lookup table, which is what the experiment is testing",
        "leakage_control": f"train-row lookup features computed out-of-fold "
        f"(K={FOLDS}, fold assignment shuffled with the run seed); "
        "validation/test use full-train statistics",
        "seed_controls": "LGBMRegressor.random_state (inert - see E3) and the "
        "out-of-fold fold assignment, which does vary the E1b fit",
        "identity_coverage": coverage,
        "inputs": inputs,
        "split_sizes": {split: len(splits[split]) for split in common.SPLITS},
        "censor_exclusion_counts": counts,
        "predictions_path": str(PREDICTIONS_OUT),
        "runs": runs,
        **summary,
        "wall_clock_seconds": time.time() - started,
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**summary, "identity_coverage": coverage}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
