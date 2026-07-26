"""T5 - evidence-based Reliability Gate confidence from Cold-Start strata.

Replaces BertPredictor's hardcoded confidence = 0.9 with a value derived from
measured ranking quality, selected at request time from a signal the gateway
already has: whether the request's tool-set fingerprint (and its individual
tool names) appear in the Ranker's training vocabulary.

  request -> tool-set fingerprint -> stratum (S1/S2/S3/S4) -> confidence

The /v1/decision contract does not change: this only alters what value the
predictor puts in the existing ``confidence`` field.

METHOD - fit and evaluation are on different splits.
T1's per-stratum tau is measured on the TEST split. Deriving confidence from
those numbers and then "validating" on test would be circular - the table would
agree by construction. So confidence is fit on the VALIDATION split and
evaluated on TEST, which is genuinely held out for this purpose.

CONFIDENCE RULE (uniform, conservative).
For each stratum, confidence = max(0, lower bound of the 95% session-clustered
bootstrap CI of validation tau). The lower CI bound rather than the point
estimate, because a reliability gate that overstates its own reliability is the
failure mode being fixed - the placeholder 0.9 is exactly that failure.

SMALL STRATA (ticket's n<100 rule).
Strata below 100 validation rows cannot support their own estimate, so they are
pooled into one bucket and share the pooled CI lower bound. Which strata get
pooled is data-dependent and reported.

Cross-Workload Transfer is NOT handled here; it remains a Fallback trigger, as
specified.
"""

from __future__ import annotations

import json
import random
import statistics
import time
from pathlib import Path

from scipy.stats import kendalltau

import common
from t1_strata import MIN_TAU_ROWS

from ltr_training.offline_statistics import kendall_tau_b

HERE = Path(__file__).resolve().parent
TEST_SCORES = HERE / "e2-bert-test-scores.jsonl"
VALIDATION_SCORES = HERE / "t5-bert-validation-scores.jsonl"
OUT = HERE / "t5-gate.json"

MODEL = "bert_prompt_schema"
PLACEHOLDER_CONFIDENCE = 0.9  # what BertPredictor ships today
STRATA_ORDER = ("S1", "S2", "S3", "S4")


def train_vocabulary(splits) -> tuple[set[str], set[str]]:
    fingerprints = {
        common.toolset_fingerprint(item.tool_schema) for item in splits["train"]
    }
    tools = {
        name for item in splits["train"] for name in common.tool_names(item.tool_schema)
    }
    return fingerprints, tools


def stratum_of(tool_schema: str, fingerprints: set[str], tools: set[str]) -> str:
    """The request-time signal: what the gateway can know before generation."""
    if common.toolset_fingerprint(tool_schema) in fingerprints:
        return "S1"
    names = common.tool_names(tool_schema)
    if not names:
        return "S1"
    seen = sum(1 for name in names if name in tools)
    if seen == len(names):
        return "S2"
    if seen == 0:
        return "S4"
    return "S3"


def assign(rows, fingerprints, tools) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {name: [] for name in STRATA_ORDER}
    for index, item in enumerate(rows):
        groups[stratum_of(item.tool_schema, fingerprints, tools)].append(index)
    return groups


def load_scores(path: Path, rows) -> dict[int, list[float]]:
    by_id = {}
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        by_id[row["sample_id"]] = row
    if len(by_id) != len(rows):
        raise ValueError(f"{path} has {len(by_id)} rows, expected {len(rows)}")
    return {
        seed: [float(by_id[item.sample_id][f"{MODEL}_seed{seed}"]) for item in rows]
        for seed in common.SEEDS
    }


def bootstrap(rows, predictions, indices, *, iterations=1000, seed=42):
    grouped: dict[str, list[int]] = {}
    for index in indices:
        grouped.setdefault(rows[index].session_id, []).append(index)
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
            [float(rows[index].output_length) for index in sampled],
        ).statistic
        samples.append(0.0 if statistic != statistic else float(statistic))
    samples.sort()
    return [
        samples[int(0.025 * (iterations - 1))],
        samples[int(0.975 * (iterations - 1))],
    ]


def tau_of(rows, predictions, indices) -> float:
    return kendall_tau_b(
        [float(rows[index].output_length) for index in indices],
        [predictions[index] for index in indices],
    )


def summarize(rows, by_seed, indices) -> dict[str, object]:
    per_seed = {seed: tau_of(rows, by_seed[seed], indices) for seed in common.SEEDS}
    values = list(per_seed.values())
    return {
        "n": len(indices),
        "per_seed_tau_b": {str(k): v for k, v in per_seed.items()},
        "mean_tau_b": statistics.fmean(values),
        "stdev_tau_b": statistics.stdev(values) if len(values) > 1 else 0.0,
        "ci95_seed17": bootstrap(rows, by_seed[common.SEEDS[0]], indices),
    }


def main() -> None:
    started = time.time()
    inputs = common.verify_inputs()
    splits, _ = common.load_splits()
    fingerprints, tools = train_vocabulary(splits)

    validation, test = splits["validation"], splits["test"]
    validation_groups = assign(validation, fingerprints, tools)
    test_groups = assign(test, fingerprints, tools)
    validation_scores = load_scores(VALIDATION_SCORES, validation)
    test_scores = load_scores(TEST_SCORES, test)

    # --- fit on validation ------------------------------------------------
    fit: dict[str, dict[str, object]] = {}
    small = [s for s in STRATA_ORDER if len(validation_groups[s]) < MIN_TAU_ROWS]
    large = [s for s in STRATA_ORDER if len(validation_groups[s]) >= MIN_TAU_ROWS]
    for stratum in STRATA_ORDER:
        fit[stratum] = summarize(
            validation, validation_scores, validation_groups[stratum]
        )

    pooled_indices = [i for s in small for i in validation_groups[s]]
    pooled = (
        summarize(validation, validation_scores, pooled_indices)
        if len(pooled_indices) >= 2
        else None
    )

    def lower_bound(stratum: str) -> float:
        ci = fit[stratum]["ci95_seed17"]
        return max(0.0, float(ci[0])) if ci else 0.0

    measured = {stratum: lower_bound(stratum) for stratum in large}
    pooled_value = max(0.0, float(pooled["ci95_seed17"][0])) if pooled else 0.0
    floor_value = min(measured.values()) if measured else 0.0

    # Three candidate rules for the strata too small to estimate. They differ
    # only on `small`; measured strata are identical across all three.
    candidates = {
        "A_pooled": {
            "description": "small strata share the pooled small-stratum CI lower bound",
            "small_value": pooled_value,
        },
        "B_floor": {
            "description": "small strata inherit the lowest measured stratum's "
            "CI lower bound",
            "small_value": floor_value,
        },
        "C_abstain": {
            "description": "the gate declines to vouch for strata it cannot "
            "measure; confidence 0 routes them to the Fallback path",
            "small_value": 0.0,
        },
    }

    # Global (non-stratified) control: one measured value for every request.
    all_validation = list(range(len(validation)))
    global_fit = summarize(validation, validation_scores, all_validation)
    global_confidence = max(0.0, float(global_fit["ci95_seed17"][0]))

    selected_rule = "C_abstain"
    confidence: dict[str, float] = {}
    basis: dict[str, str] = {}
    for stratum in large:
        confidence[stratum] = measured[stratum]
        basis[stratum] = f"own validation CI lower bound (n={fit[stratum]['n']})"
    for stratum in small:
        confidence[stratum] = float(candidates[selected_rule]["small_value"])
        basis[stratum] = (
            f"rule {selected_rule}: {candidates[selected_rule]['description']} "
            f"(own n={fit[stratum]['n']} < {MIN_TAU_ROWS})"
        )

    # --- evaluate on test -------------------------------------------------
    realized: dict[str, dict[str, object]] = {}
    reliability: list[dict[str, object]] = []
    for stratum in STRATA_ORDER:
        indices = test_groups[stratum]
        cell = summarize(test, test_scores, indices) if len(indices) >= 2 else {"n": len(indices)}
        realized[stratum] = cell
        entry: dict[str, object] = {
            "stratum": stratum,
            "assigned_confidence": confidence[stratum],
            "confidence_basis": basis[stratum],
            "validation_n": fit[stratum]["n"],
            "test_n": len(indices),
        }
        if "mean_tau_b" in cell:
            realized_tau = float(cell["mean_tau_b"])
            entry.update(
                {
                    "realized_test_tau_b": realized_tau,
                    "tau_withheld": len(indices) < MIN_TAU_ROWS,
                    "gap_realized_minus_assigned": realized_tau - confidence[stratum],
                    "conservative": realized_tau >= confidence[stratum],
                    "placeholder_confidence": PLACEHOLDER_CONFIDENCE,
                    "placeholder_overstatement": PLACEHOLDER_CONFIDENCE - realized_tau,
                }
            )
        reliability.append(entry)

    # Score every candidate rule, plus the global control, on test.
    rule_comparison: dict[str, object] = {}
    for name, spec in candidates.items():
        assigned = {
            **measured,
            **{stratum: float(spec["small_value"]) for stratum in small},
        }
        rows_out = []
        for stratum in STRATA_ORDER:
            cell = realized[stratum]
            if "mean_tau_b" not in cell:
                continue
            rows_out.append(
                {
                    "stratum": stratum,
                    "assigned": assigned[stratum],
                    "realized": float(cell["mean_tau_b"]),
                    "overstates_by": assigned[stratum] - float(cell["mean_tau_b"]),
                }
            )
        rule_comparison[name] = {
            "description": spec["description"],
            "small_strata_value": float(spec["small_value"]),
            "per_stratum": rows_out,
            "never_overstates": all(r["overstates_by"] <= 0 for r in rows_out),
            "max_overstatement": max(r["overstates_by"] for r in rows_out),
        }
    global_rows = [
        {
            "stratum": stratum,
            "assigned": global_confidence,
            "realized": float(realized[stratum]["mean_tau_b"]),
            "overstates_by": global_confidence - float(realized[stratum]["mean_tau_b"]),
        }
        for stratum in STRATA_ORDER
        if "mean_tau_b" in realized[stratum]
    ]
    rule_comparison["global_control_no_stratification"] = {
        "description": "single measured confidence for every request; the control "
        "that says whether stratification earns its complexity",
        "small_strata_value": global_confidence,
        "per_stratum": global_rows,
        "never_overstates": all(r["overstates_by"] <= 0 for r in global_rows),
        "max_overstatement": max(r["overstates_by"] for r in global_rows),
    }
    rule_comparison["placeholder_0.9"] = {
        "description": "what BertPredictor ships today",
        "small_strata_value": PLACEHOLDER_CONFIDENCE,
        "per_stratum": [
            {
                "stratum": stratum,
                "assigned": PLACEHOLDER_CONFIDENCE,
                "realized": float(realized[stratum]["mean_tau_b"]),
                "overstates_by": PLACEHOLDER_CONFIDENCE
                - float(realized[stratum]["mean_tau_b"]),
            }
            for stratum in STRATA_ORDER
            if "mean_tau_b" in realized[stratum]
        ],
        "never_overstates": False,
        "max_overstatement": max(
            PLACEHOLDER_CONFIDENCE - float(realized[s]["mean_tau_b"])
            for s in STRATA_ORDER
            if "mean_tau_b" in realized[s]
        ),
    }

    evaluated = [e for e in reliability if "realized_test_tau_b" in e]
    reportable = [e for e in evaluated if not e["tau_withheld"]]
    spread_assigned = (
        max(e["assigned_confidence"] for e in evaluated)
        - min(e["assigned_confidence"] for e in evaluated)
    )
    spread_realized = (
        max(e["realized_test_tau_b"] for e in evaluated)
        - min(e["realized_test_tau_b"] for e in evaluated)
    )
    all_conservative = all(e["conservative"] for e in evaluated)
    placeholder_overstatement = [
        e["placeholder_overstatement"] for e in evaluated
    ]

    report = {
        "schema_version": "t5-gate-v1",
        "status": "done",
        "ticket": "issue #9 (T5); spec issue #4",
        "contract_change": "none - only the value written to the existing "
        "Prediction.confidence field changes",
        "request_time_signal": "tool-set fingerprint + tool names vs the Ranker's "
        "training vocabulary; both are available at admission, neither needs generation",
        "method": {
            "fit_split": "validation",
            "evaluation_split": "test",
            "why": "T1's per-stratum tau is measured on test; fitting confidence there "
            "and evaluating there would be circular",
            "confidence_rule": "max(0, lower bound of 95% session-clustered bootstrap "
            "CI of validation tau) - the lower bound, not the point estimate, because "
            "an overconfident reliability gate is the defect being fixed",
            "small_stratum_rule": f"strata with fewer than {MIN_TAU_ROWS} validation "
            "rows are pooled and share the pooled CI lower bound",
            "pooled_strata": small,
            "own_estimate_strata": large,
            "selected_small_stratum_rule": selected_rule,
            "cross_workload_transfer": "out of scope here; remains a Fallback trigger",
        },
        "rule_comparison": rule_comparison,
        "global_control": {
            "confidence": global_confidence,
            "validation_fit": global_fit,
        },
        "inputs": inputs,
        "train_vocabulary": {
            "unique_fingerprints": len(fingerprints),
            "unique_tool_names": len(tools),
        },
        "validation_fit": fit,
        "validation_pooled_small_strata": pooled,
        "assigned_confidence": confidence,
        "confidence_basis": basis,
        "test_realized": realized,
        "reliability_table": reliability,
        "findings": {
            "all_strata_conservative_on_test": all_conservative,
            "assigned_confidence_spread": spread_assigned,
            "realized_tau_spread": spread_realized,
            "placeholder_overstatement_min": min(placeholder_overstatement),
            "placeholder_overstatement_max": max(placeholder_overstatement),
            "reportable_strata": [e["stratum"] for e in reportable],
        },
        "wall_clock_seconds": time.time() - started,
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"{'stratum':8s} {'val n':>6s} {'test n':>7s} {'assigned':>9s} "
          f"{'realized':>9s} {'gap':>8s} {'0.9 over':>9s}")
    for entry in reliability:
        if "realized_test_tau_b" in entry:
            print(
                f"{entry['stratum']:8s} {entry['validation_n']:6d} {entry['test_n']:7d} "
                f"{entry['assigned_confidence']:9.4f} {entry['realized_test_tau_b']:9.4f} "
                f"{entry['gap_realized_minus_assigned']:+8.4f} "
                f"{entry['placeholder_overstatement']:+9.4f}"
                + ("   [tau withheld, n<100]" if entry["tau_withheld"] else "")
            )
        else:
            print(
                f"{entry['stratum']:8s} {entry['validation_n']:6d} {entry['test_n']:7d} "
                f"{entry['assigned_confidence']:9.4f}  (too few test rows)"
            )
    print(f"\nassigned spread {spread_assigned:.4f} vs realized spread {spread_realized:.4f}")
    print(f"all strata conservative on test: {all_conservative}")
    print(f"done in {report['wall_clock_seconds']:.1f}s")


if __name__ == "__main__":
    main()
