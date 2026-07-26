"""T1 (issue #5) - S1-S4 Cold-Start Transfer re-stratification + grid-searched baseline.

Strata follow the CONTEXT.md Cold-Start Transfer glossary entry:

  S1  seen-combination        tool-set fingerprint appears in train
  S2  new-combination         fingerprint unseen, but every tool name appears in train
  S3  partial-new tools       fingerprint unseen, some tool names seen and some not
  S4  all-new tools           fingerprint unseen, no tool name appears in train

S1 is defined by fingerprint alone; S2-S4 partition the remainder by tool-name
novelty and are defined only for rows advertising at least one tool.

Model scores are read back from e2-bert-test-scores.jsonl - zero new BERT
inference. The grid-searched LightGBM (offline_baselines.run_lightgbm_grid, 40
configs) is the one model trained here, so that the +0.20 headline is compared
against a tuned scalar baseline rather than a fixed-hyperparameter one.

Tracer-bullet order per the ticket: stratum sizes, then table, then grid.
"""

from __future__ import annotations

# lightgbm MUST be imported before torch (pulled in by common -> ltr_training).
# Importing it after torch loads a second OpenMP runtime and segfaults (SIGSEGV,
# exit 139) on the first fit; E3 only worked because its import came first.
from lightgbm import LGBMRegressor  # isort: skip

import json
import random
import statistics
import time
from pathlib import Path

from scipy.stats import kendalltau

import common

from ltr_training.offline_baselines import (
    _lightgbm_features,
    run_lightgbm_grid,
)
from ltr_training.offline_statistics import kendall_tau_b

HERE = Path(__file__).resolve().parent
SCORES = HERE / "e2-bert-test-scores.jsonl"
OUT = HERE / "t1-strata.json"
GRID_PREDICTIONS = HERE / "t1-lightgbm-grid-test-predictions.jsonl"

STRATA = ("S1", "S2", "S3", "S4", "all")
MIN_TAU_ROWS = 100  # ticket: strata with n<100 report size only, not tau

FIXED_BASELINE_TAU = 0.4267985278708031  # E3, e3-lightgbm-seeds.json

MODEL_LABELS = {
    "bert_prompt_schema": "BERT prompt_schema (schema TEXT)",
    "bert_prompt_only": "BERT prompt_only (control, no schema)",
    "lightgbm_scalar": "LightGBM scalar (fixed hyperparameters, E3)",
    "schema_hash_categorical": "schema-hash categorical (E1a)",
    "schema_hash_lookup": "schema-hash lookup table (E1b)",
    "lightgbm_grid": "LightGBM scalar (grid-searched, 40 configs)",
}


def assign_strata(splits) -> tuple[dict[str, list[int]], dict[str, object]]:
    train_fingerprints = {
        common.toolset_fingerprint(item.tool_schema) for item in splits["train"]
    }
    train_tools = {
        name for item in splits["train"] for name in common.tool_names(item.tool_schema)
    }
    test = splits["test"]
    strata: dict[str, list[int]] = {name: [] for name in STRATA}
    toolless_by_stratum: dict[str, int] = {name: 0 for name in STRATA}
    unstratified: list[int] = []

    for index, item in enumerate(test):
        strata["all"].append(index)
        names = common.tool_names(item.tool_schema)
        fingerprint = common.toolset_fingerprint(item.tool_schema)
        if fingerprint in train_fingerprints:
            label = "S1"
        elif not names:
            # New combination but no tools to judge novelty by; S2-S4 undefined.
            unstratified.append(index)
            continue
        else:
            seen = sum(1 for name in names if name in train_tools)
            if seen == len(names):
                label = "S2"
            elif seen == 0:
                label = "S4"
            else:
                label = "S3"
        strata[label].append(index)
        if not names:
            toolless_by_stratum[label] += 1

    definition = {
        "source": "CONTEXT.md glossary, Cold-Start Transfer",
        "identity_key": "sha256 over sorted top-level tool names (E1 multi-template parser); "
        "the raw tool_schema string is never hashed - 2383 rows embed a per-row timestamp",
        "train_unique_fingerprints": len(train_fingerprints),
        "train_unique_tool_names": len(train_tools),
        "sizes": {name: len(indices) for name, indices in strata.items()},
        "toolless_rows_by_stratum": toolless_by_stratum,
        "rows_unstratified_no_tools_new_combination": len(unstratified),
        "partition_check_S1_S2_S3_S4_plus_unstratified": (
            sum(len(strata[name]) for name in ("S1", "S2", "S3", "S4"))
            + len(unstratified)
        ),
        "tau_reporting_threshold": MIN_TAU_ROWS,
    }
    return strata, definition


def load_scores(test_examples) -> dict[str, dict[int, list[float]]]:
    rows = {}
    for line in SCORES.open(encoding="utf-8"):
        row = json.loads(line)
        rows[row["sample_id"]] = row
    if len(rows) != len(test_examples):
        raise ValueError(f"{SCORES} has {len(rows)} rows, expected {len(test_examples)}")
    models = sorted(
        {
            key.rsplit("_seed", 1)[0]
            for key in next(iter(rows.values()))
            if "_seed" in key
        }
    )
    return {
        model: {
            seed: [float(rows[item.sample_id][f"{model}_seed{seed}"]) for item in test_examples]
            for seed in common.SEEDS
        }
        for model in models
    }


def bootstrap_ci(examples, predictions, indices, *, iterations=1000, seed=42):
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
    return [
        samples[int(0.025 * (iterations - 1))],
        samples[int(0.975 * (iterations - 1))],
    ]


def tau_on(examples, predictions, indices) -> float:
    return kendall_tau_b(
        [float(examples[index].output_length) for index in indices],
        [predictions[index] for index in indices],
    )


def build_table(test, predictions, strata) -> dict[str, dict[str, object]]:
    table: dict[str, dict[str, object]] = {}
    for model, by_seed in predictions.items():
        table[model] = {}
        for stratum in STRATA:
            indices = strata[stratum]
            cell: dict[str, object] = {"n": len(indices)}
            if len(indices) < MIN_TAU_ROWS:
                cell["tau_withheld"] = (
                    f"n<{MIN_TAU_ROWS}; ticket requires size only, no tau"
                )
            else:
                per_seed = {
                    seed: tau_on(test, by_seed[seed], indices) for seed in common.SEEDS
                }
                values = list(per_seed.values())
                cell["per_seed_tau_b"] = {
                    str(seed): value for seed, value in per_seed.items()
                }
                cell["mean_tau_b"] = statistics.fmean(values)
                cell["stdev_tau_b"] = statistics.stdev(values)
                cell["ci95_seed17"] = bootstrap_ci(
                    test, by_seed[common.SEEDS[0]], indices
                )
            table[model][stratum] = cell
    return table


def to_rows(examples) -> list[dict[str, object]]:
    return [
        {
            "prompt": item.prompt,
            "tool_schema": item.tool_schema,
            "history": list(item.history),
            "true_length": item.output_length,
        }
        for item in examples
    ]


def run_grid(splits) -> tuple[dict[str, object], dict[int, list[float]]]:
    """offline_baselines.run_lightgbm_grid over the same fixed tier-2 split."""
    rows = {split: to_rows(splits[split]) for split in common.SPLITS}
    reports: list[dict[str, object]] = []
    predictions: dict[int, list[float]] = {}
    features = None
    for seed in common.SEEDS:
        report, model = run_lightgbm_grid(rows, model_factory=LGBMRegressor, seed=seed)
        reports.append(report)
        if features is None:
            features = report["features"]
        predictions[seed] = [
            float(value)
            for value in model.predict([_lightgbm_features(row) for row in rows["test"]])
        ]
        print(
            f"grid seed{seed}: best={report['best_config']} "
            f"val={report['best_validation_tau_b']:.4f} test={report['test_tau_b']:.4f}",
            flush=True,
        )
    test_taus = [float(item["test_tau_b"]) for item in reports]
    summary = {
        "recipe_source": "ltr_training.offline_baselines.run_lightgbm_grid",
        "grid_size": reports[0]["grid_size"],
        "selection_split": "validation",
        "features": features,
        "per_seed": [
            {
                "seed": item["seed"],
                "best_config": item["best_config"],
                "best_validation_tau_b": item["best_validation_tau_b"],
                "test_tau_b": item["test_tau_b"],
            }
            for item in reports
        ],
        "test_tau_b_mean": statistics.fmean(test_taus),
        "test_tau_b_stdev": statistics.stdev(test_taus),
    }
    return summary, predictions


def evaluate_criteria(table, baseline_model: str) -> dict[str, object]:
    """Re-evaluate the frozen pre-registered criteria, verbatim.

    primary   session-clustered bootstrap 95% CI separation on unseen strata
    secondary delta-tau >= 0.05 material-effect bar (self-imposed)
    tie bar   seen stratum delta-tau < 0.02
    """
    claim = "bert_prompt_schema"
    findings: list[dict[str, object]] = []
    for stratum in ("S1", "S2", "S3", "S4"):
        claim_cell = table[claim][stratum]
        base_cell = table[baseline_model][stratum]
        entry: dict[str, object] = {"stratum": stratum, "n": claim_cell["n"]}
        if "mean_tau_b" not in claim_cell or "mean_tau_b" not in base_cell:
            entry["evaluated"] = False
            entry["reason"] = f"n<{MIN_TAU_ROWS}; tau withheld per ticket"
            findings.append(entry)
            continue
        claim_ci = claim_cell["ci95_seed17"]
        base_ci = base_cell["ci95_seed17"]
        delta = float(claim_cell["mean_tau_b"]) - float(base_cell["mean_tau_b"])
        entry.update(
            {
                "evaluated": True,
                "claim_tau_b": claim_cell["mean_tau_b"],
                "baseline_tau_b": base_cell["mean_tau_b"],
                "delta_tau_b": delta,
                "claim_ci95": claim_ci,
                "baseline_ci95": base_ci,
                "primary_ci_separated": bool(
                    claim_ci and base_ci and claim_ci[0] > base_ci[1]
                ),
                "secondary_material_effect_delta_ge_0.05": delta >= 0.05,
            }
        )
        if stratum == "S1":
            entry["tie_bar_delta_lt_0.02"] = abs(delta) < 0.02
        findings.append(entry)
    return {
        "criteria_text": {
            "primary": "session-clustered bootstrap 95% CI separation on unseen strata "
            "(community-standard)",
            "secondary": "delta-tau >= 0.05 material-effect bar (self-imposed, stated as such)",
            "seen_stratum_tie_bar": "delta-tau < 0.02 (expected, supports lookup-table "
            "deployment story, does not kill)",
            "frozen_before_results": True,
            "source": "docs/DIRECTION-DECISION-2026-07-25.md",
        },
        "claim_model": claim,
        "baseline_model_of_record": baseline_model,
        "per_stratum": findings,
    }


def main() -> None:
    started = time.time()
    inputs = common.verify_inputs()
    splits, counts = common.load_splits()
    test = splits["test"]

    # --- tracer bullet 1: stratum sizes -----------------------------------
    strata, definition = assign_strata(splits)
    print("=== STRATUM SIZES ===", flush=True)
    print(json.dumps(definition, indent=2, sort_keys=True), flush=True)

    # --- tracer bullet 2: table from saved scores -------------------------
    predictions = load_scores(test)
    print("\n=== TABLE (saved scores, zero new inference) ===", flush=True)
    table = build_table(test, predictions, strata)
    for model in table:
        for stratum in STRATA:
            cell = table[model][stratum]
            if "mean_tau_b" in cell:
                print(
                    f"{model:26s} {stratum:4s} n={cell['n']:4d} "
                    f"tau={cell['mean_tau_b']:.4f}+-{cell['stdev_tau_b']:.4f}",
                    flush=True,
                )
            else:
                print(
                    f"{model:26s} {stratum:4s} n={cell['n']:4d} tau=withheld",
                    flush=True,
                )

    # --- tracer bullet 3: grid-searched baseline --------------------------
    print("\n=== GRID-SEARCHED LIGHTGBM ===", flush=True)
    grid_summary, grid_predictions = run_grid(splits)
    predictions["lightgbm_grid"] = grid_predictions
    with GRID_PREDICTIONS.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(test):
            handle.write(
                json.dumps(
                    {
                        "sample_id": item.sample_id,
                        "session_id": item.session_id,
                        "true_length": item.output_length,
                        "prediction": {
                            str(seed): grid_predictions[seed][index]
                            for seed in common.SEEDS
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    table = build_table(test, predictions, strata)
    grid_mean = float(grid_summary["test_tau_b_mean"])
    baseline_model = (
        "lightgbm_grid" if grid_mean > FIXED_BASELINE_TAU else "lightgbm_scalar"
    )
    baseline_note = (
        f"grid mean {grid_mean:.4f} vs fixed {FIXED_BASELINE_TAU:.4f}; "
        f"stronger = {baseline_model}"
    )
    print(f"\nbaseline of record: {baseline_note}", flush=True)

    criteria = evaluate_criteria(table, baseline_model)
    print("\n=== PRE-REGISTERED CRITERIA ===", flush=True)
    print(json.dumps(criteria, indent=2, sort_keys=True), flush=True)

    report = {
        "schema_version": "t1-strata-v1",
        "status": "done",
        "ticket": "issue #5 (T1); spec issue #4",
        "inputs": inputs,
        "scores_reused": str(SCORES),
        "new_bert_inference": False,
        "split_sizes": {split: len(splits[split]) for split in common.SPLITS},
        "censor_exclusion_counts": counts,
        "stratum_definition": definition,
        "model_labels": MODEL_LABELS,
        "results": table,
        "lightgbm_grid": grid_summary,
        "baseline_of_record": {
            "model": baseline_model,
            "fixed_test_tau_b": FIXED_BASELINE_TAU,
            "grid_test_tau_b_mean": grid_mean,
            "note": baseline_note,
        },
        "pre_registered_criteria": criteria,
        "wall_clock_seconds": time.time() - started,
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\ndone in {report['wall_clock_seconds']:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
