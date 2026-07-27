"""Figure 4 - ranking quality on the pooled tier-2 test split.

Bars are the 3-seed mean Kendall tau-b, dots are the individual seeds, error
bars are the session-clustered bootstrap 95% CI carried in the artifact. The
grid-searched LightGBM is the baseline of record and is emphasised, because
comparing against the untuned one would flatter the headline.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from _common import (
    COLOR,
    IEEE_SINGLE_WIDTH,
    LABEL,
    OFFLINE,
    OKABE_ITO,
    load_json,
    record_provenance,
    save,
)

T1 = OFFLINE / "t1-strata.json"
ORDER = (
    "lightgbm_scalar",
    "schema_hash_lookup",
    "lightgbm_grid",
    "bert_prompt_only",
    "bert_prompt_schema",
)
BASELINE = "lightgbm_grid"
CLAIM = "bert_prompt_schema"


def main() -> None:
    payload = load_json(T1)
    cells = {name: payload["results"][name]["all"] for name in ORDER}

    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_WIDTH, 2.9), constrained_layout=True)
    positions = np.arange(len(ORDER))

    for index, name in enumerate(ORDER):
        cell = cells[name]
        mean = float(cell["mean_tau_b"])
        low, high = (float(value) for value in cell["ci95_seed17"])
        emphasised = name in (BASELINE, CLAIM)
        ax.bar(
            index,
            mean,
            width=0.62,
            color=COLOR[name.replace("bert_", "")],
            edgecolor=OKABE_ITO["black"] if emphasised else "none",
            linewidth=1.0 if emphasised else 0.0,
            zorder=2,
        )
        ax.errorbar(
            index,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="none",
            ecolor=OKABE_ITO["black"],
            elinewidth=0.9,
            capsize=2.5,
            zorder=4,
        )
        seeds = [float(v) for v in cell["per_seed_tau_b"].values()]
        ax.scatter(
            np.full(len(seeds), index),
            seeds,
            s=9,
            facecolor="white",
            edgecolor=OKABE_ITO["black"],
            linewidth=0.7,
            zorder=5,
        )

    claim_mean = float(cells[CLAIM]["mean_tau_b"])
    base_mean = float(cells[BASELINE]["mean_tau_b"])
    delta = claim_mean - base_mean

    # Delta bracket between the baseline of record and the claim.
    left, right = ORDER.index(BASELINE), ORDER.index(CLAIM)
    top = max(float(cells[n]["ci95_seed17"][1]) for n in ORDER) + 0.035
    ax.plot([left, left, right, right], [base_mean, top, top, claim_mean],
            color=OKABE_ITO["black"], linewidth=0.8, zorder=6)
    ax.text(
        (left + right) / 2,
        top + 0.012,
        f"$\\Delta\\tau_b$ = +{delta:.4f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [LABEL[name] for name in ORDER], rotation=32, ha="right"
    )
    ax.set_ylabel("Kendall $\\tau_b$ (test, n=%d)" % int(cells[CLAIM]["n"]))
    ax.set_ylim(0, top + 0.09)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

    save(fig, "fig4.pdf")
    record_provenance("fig4.pdf", [T1])
    print(
        json.dumps(
            {
                name: {
                    "mean_tau_b": float(cells[name]["mean_tau_b"]),
                    "ci95": [float(v) for v in cells[name]["ci95_seed17"]],
                    "per_seed": {k: float(v) for k, v in cells[name]["per_seed_tau_b"].items()},
                }
                for name in ORDER
            }
            | {"delta_claim_minus_baseline_of_record": delta},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
