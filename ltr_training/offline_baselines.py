from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .offline_statistics import kendall_tau_b


def lightgbm_grid() -> list[dict[str, int | float]]:
    grid: list[dict[str, int | float]] = []
    for max_depth in (3, 5):
        for num_leaves in (7, 15):
            for learning_rate in (0.03, 0.1):
                for n_estimators in (100, 300):
                    grid.append(
                        {
                            "max_depth": max_depth,
                            "num_leaves": num_leaves,
                            "learning_rate": learning_rate,
                            "n_estimators": n_estimators,
                        }
                    )
    grid.extend(
        {
            "max_depth": depth,
            "num_leaves": leaves,
            "learning_rate": 0.05,
            "n_estimators": estimators,
        }
        for depth, leaves, estimators in (
            (7, 31, 200),
            (7, 63, 300),
            (-1, 31, 300),
            (-1, 63, 500),
        )
    )
    return grid


def _lightgbm_features(row: Mapping[str, object]) -> list[float]:
    prompt = str(row.get("prompt", ""))
    tool_schema = str(row.get("tool_schema", ""))
    history = row.get("history", [])
    history_count = len(history) if isinstance(history, Sequence) else 0
    return [
        float(len(prompt)),
        float(len(prompt.split())),
        float(len(tool_schema)),
        float(tool_schema.count('"name"')),
        float(history_count),
    ]


def run_lightgbm_grid(
    splits: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    model_factory: Callable[..., object],
    seed: int,
) -> tuple[dict[str, object], object]:
    required = ("train", "validation", "test")
    for split in required:
        if not splits.get(split):
            raise ValueError(f"{split} split is empty")

    features = {
        split: [_lightgbm_features(row) for row in splits[split]] for split in required
    }
    labels = {
        split: [float(row["true_length"]) for row in splits[split]] for split in required
    }
    search_results: list[dict[str, object]] = []
    best_model: object | None = None
    best_config: dict[str, int | float] | None = None
    best_tau = float("-inf")
    for config in lightgbm_grid():
        model = model_factory(**config, random_state=seed, verbosity=-1)
        model.fit(features["train"], labels["train"])
        predictions = model.predict(features["validation"])
        tau = kendall_tau_b(labels["validation"], predictions)
        search_results.append({"config": dict(config), "validation_tau_b": tau})
        if tau > best_tau:
            best_tau = tau
            best_config = dict(config)
            best_model = model

    assert best_model is not None and best_config is not None
    test_predictions = best_model.predict(features["test"])
    test_tau = kendall_tau_b(labels["test"], test_predictions)
    report: dict[str, object] = {
        "schema_version": "lightgbm-search-v1",
        "status": "done",
        "seed": seed,
        "grid_size": len(search_results),
        "search_range": {
            "max_depth": sorted({item["max_depth"] for item in lightgbm_grid()}),
            "num_leaves": sorted({item["num_leaves"] for item in lightgbm_grid()}),
            "learning_rate": sorted({item["learning_rate"] for item in lightgbm_grid()}),
            "n_estimators": sorted({item["n_estimators"] for item in lightgbm_grid()}),
        },
        "selection_split": "validation",
        "search_results": search_results,
        "best_config": best_config,
        "best_validation_tau_b": best_tau,
        "test_tau_b": test_tau,
        "test_evaluations": 1,
        "split_counts": {split: len(splits[split]) for split in required},
        "features": [
            "prompt_char_count",
            "prompt_word_count",
            "tool_schema_char_count",
            "tool_name_count",
            "history_turn_count",
        ],
    }
    return report, best_model


@dataclass(frozen=True)
class LegacyFamily:
    backbone: str
    head: str
    shorter_is_higher: bool = True


LEGACY_FAMILIES = {
    "listmle-opt": LegacyFamily("opt", "rank"),
    "classification-opt": LegacyFamily("opt", "classification"),
    "pars-bert": LegacyFamily("bert", "rank"),
    "a1-opt": LegacyFamily("opt", "rank"),
    "a2-bert": LegacyFamily("bert", "rank"),
}


def _expected_class(logits: Sequence[float]) -> float:
    if not logits:
        raise ValueError("classification logits are empty")
    maximum = max(float(value) for value in logits)
    weights = [math.exp(float(value) - maximum) for value in logits]
    denominator = sum(weights)
    return sum(index * weight for index, weight in enumerate(weights)) / denominator


def legacy_length_score(
    family: str,
    *,
    raw_score: float | None = None,
    logits: Sequence[float] | None = None,
) -> float:
    try:
        spec = LEGACY_FAMILIES[family]
    except KeyError as exc:
        raise ValueError(f"unknown legacy family: {family}") from exc
    if spec.head == "classification":
        if logits is None:
            raise ValueError("classification family requires logits")
        native = _expected_class(logits)
    else:
        if raw_score is None:
            raise ValueError("rank family requires raw_score")
        native = float(raw_score)
    return -native if spec.shorter_is_higher else native


def legacy_loader_status(family: str, checkpoint: Path) -> dict[str, object]:
    if family not in LEGACY_FAMILIES:
        raise ValueError(f"unknown legacy family: {family}")
    spec = LEGACY_FAMILIES[family]
    if not checkpoint.exists():
        return {
            "family": family,
            "status": "blocked",
            "reason": "checkpoint_missing",
            "checkpoint": str(checkpoint),
            "backbone": spec.backbone,
            "head": spec.head,
            "score_direction": "reversed_to_longer_is_higher",
        }
    return {
        "family": family,
        "status": "blocked",
        "reason": "course_vllm_runtime_required",
        "checkpoint": str(checkpoint),
        "backbone": spec.backbone,
        "head": spec.head,
        "score_direction": "reversed_to_longer_is_higher",
    }
