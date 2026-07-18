from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Iterable, Mapping, Sequence

from .label_input import canonical_schema_hash


def kendall_tau_b(truth: Sequence[float], prediction: Sequence[float]) -> float:
    if len(truth) != len(prediction):
        raise ValueError("truth and prediction must have equal length")
    concordant = 0
    discordant = 0
    truth_only_ties = 0
    prediction_only_ties = 0
    for left in range(len(truth)):
        for right in range(left + 1, len(truth)):
            truth_delta = float(truth[left]) - float(truth[right])
            prediction_delta = float(prediction[left]) - float(prediction[right])
            if truth_delta == 0.0 and prediction_delta == 0.0:
                continue
            if truth_delta == 0.0:
                truth_only_ties += 1
            elif prediction_delta == 0.0:
                prediction_only_ties += 1
            elif truth_delta * prediction_delta > 0.0:
                concordant += 1
            else:
                discordant += 1
    comparable = concordant + discordant
    denominator = math.sqrt(
        (comparable + truth_only_ties)
        * (comparable + prediction_only_ties)
    )
    return (concordant - discordant) / denominator if denominator else 0.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(
        ordered[lower] * (upper - position) + ordered[upper] * (position - lower)
    )


def cluster_bootstrap_tau_b(
    rows: Iterable[Mapping[str, object]],
    *,
    truth_key: str,
    prediction_key: str,
    cluster_key: str,
    iterations: int = 1000,
    seed: int = 42,
) -> dict[str, object]:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[cluster_key])].append(row)
    clusters = sorted(grouped)
    if not clusters:
        raise ValueError("no clusters")
    materialized = [row for cluster in clusters for row in grouped[cluster]]
    point = kendall_tau_b(
        [float(row[truth_key]) for row in materialized],
        [float(row[prediction_key]) for row in materialized],
    )
    rng = random.Random(seed)
    bootstrap: list[float] = []
    for _ in range(iterations):
        sampled_clusters = [rng.choice(clusters) for _ in clusters]
        sampled_rows = [row for cluster in sampled_clusters for row in grouped[cluster]]
        bootstrap.append(
            kendall_tau_b(
                [float(row[truth_key]) for row in sampled_rows],
                [float(row[prediction_key]) for row in sampled_rows],
            )
        )
    return {
        "variant": "b",
        "point_tau_b": point,
        "iterations": iterations,
        "seed": seed,
        "cluster_key": cluster_key,
        "cluster_count": len(clusters),
        "ci95_percentile": [
            _percentile(bootstrap, 0.025),
            _percentile(bootstrap, 0.975),
        ],
    }


def _tie_proportion(values: Sequence[object]) -> float:
    total_pairs = len(values) * (len(values) - 1) // 2
    if total_pairs == 0:
        return 0.0
    tied_pairs = sum(count * (count - 1) // 2 for count in Counter(values).values())
    return tied_pairs / total_pairs


def tie_proportions(
    true_lengths: Sequence[object], predictions: Sequence[object]
) -> dict[str, float]:
    return {
        "true_length_tie_proportion": _tie_proportion(true_lengths),
        "prediction_tie_proportion": _tie_proportion(predictions),
    }


def _overlap(left: Iterable[str], right: Iterable[str]) -> dict[str, object]:
    left_hashes = {canonical_schema_hash(value) for value in left}
    right_hashes = {canonical_schema_hash(value) for value in right}
    intersection = sorted(left_hashes & right_hashes)
    return {
        "left_unique": len(left_hashes),
        "right_unique": len(right_hashes),
        "intersection_count": len(intersection),
        "intersection_sha256": intersection,
    }


def canonical_schema_overlap_report(
    train: Iterable[str],
    validation: Iterable[str],
    test: Iterable[str],
    ood: Iterable[str],
) -> dict[str, object]:
    train_values = list(train)
    validation_values = list(validation)
    test_values = list(test)
    return {
        "canonicalization": "JSON sort_keys + type dict/list normalization + SHA-256",
        "train_validation": _overlap(train_values, validation_values),
        "train_test": _overlap(train_values, test_values),
        "toolace_ood": _overlap(
            train_values + validation_values + test_values,
            ood,
        ),
    }


def _identity_hash(value: object) -> str:
    import hashlib
    import json

    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _identity_overlap(left: Iterable[object], right: Iterable[object]) -> dict[str, object]:
    left_hashes = {_identity_hash(value) for value in left}
    right_hashes = {_identity_hash(value) for value in right}
    intersection = sorted(left_hashes & right_hashes)
    return {
        "left_unique": len(left_hashes),
        "right_unique": len(right_hashes),
        "intersection_count": len(intersection),
        "intersection_sha256": intersection,
    }


def session_overlap_report(
    train: Iterable[object],
    validation: Iterable[object],
    test: Iterable[object],
    ood: Iterable[object],
) -> dict[str, object]:
    train_values = list(train)
    validation_values = list(validation)
    test_values = list(test)
    return {
        "canonicalization": "canonical JSON sort_keys + SHA-256",
        "train_validation": _identity_overlap(train_values, validation_values),
        "train_test": _identity_overlap(train_values, test_values),
        "toolace_ood": _identity_overlap(
            train_values + validation_values + test_values,
            ood,
        ),
    }
