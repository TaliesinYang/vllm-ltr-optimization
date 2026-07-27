"""Paired confidence interval for the schema's own contribution to ranking.

The paper's headline gap, +0.19 tau, compares BERT-with-schema against a
LightGBM baseline and therefore changes two things at once: the model class
and the input representation. The prompt-only control isolates the second, at
+0.0437 -- and that number has been quoted without an interval, which is the
one place the paper's central claim could fail silently.

This computes the difference the way the paper computes everything else:
paired (both models scored on the same resampled requests, never on two
independent draws) and session-clustered (sessions resampled whole, because
requests within a session are not independent). The seeds are averaged inside
each replicate rather than treated as extra samples.

Run: python3 schema_effect_ci.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from _common import REPO

SCORES = REPO / "runs" / "offline-experiments-2026-07-25" / "e2-bert-test-scores.jsonl"
SEEDS = ("seed17", "seed42", "seed73")
DRAWS = 10_000
SEED = 20260727


def load():
    rows = [json.loads(line) for line in SCORES.open(encoding="utf-8") if line.strip()]
    by_session: dict[str, list[dict]] = {}
    for row in rows:
        by_session.setdefault(row["session_id"], []).append(row)
    return rows, by_session


def tau(rows: list[dict], model: str, seed: str) -> float:
    """Kendall tau-b between predicted and realized length order."""
    predicted = [r[f"{model}_{seed}"] for r in rows]
    actual = [r["true_length"] for r in rows]
    return stats.kendalltau(predicted, actual, variant="b").statistic


def mean_tau(rows: list[dict], model: str) -> float:
    values = [tau(rows, model, seed) for seed in SEEDS]
    return float(np.mean(values))


def main() -> None:
    rows, by_session = load()
    sessions = sorted(by_session)
    print(f"{len(rows)} held-out requests across {len(sessions)} sessions")

    point_schema = mean_tau(rows, "bert_prompt_schema")
    point_only = mean_tau(rows, "bert_prompt_only")
    point_delta = point_schema - point_only
    print(f"prompt+schema  tau_b = {point_schema:.4f}")
    print(f"prompt only    tau_b = {point_only:.4f}")
    print(f"paired delta         = {point_delta:+.4f}")

    rng = np.random.default_rng(SEED)
    index = np.arange(len(sessions))
    deltas = np.empty(DRAWS)
    for draw in range(DRAWS):
        picked = rng.choice(index, size=len(sessions), replace=True)
        resampled = [row for i in picked for row in by_session[sessions[i]]]
        deltas[draw] = mean_tau(resampled, "bert_prompt_schema") - \
            mean_tau(resampled, "bert_prompt_only")

    low, high = np.percentile(deltas, [2.5, 97.5])
    crosses_zero = low <= 0 <= high
    print(f"\n95% session-clustered paired CI = [{low:+.4f}, {high:+.4f}]")
    print(f"share of replicates below zero  = {(deltas < 0).mean():.4f}")
    print("VERDICT:", "CONTAINS ZERO -- the schema effect is not established"
          if crosses_zero else "excludes zero -- the schema effect is established")


if __name__ == "__main__":
    main()
