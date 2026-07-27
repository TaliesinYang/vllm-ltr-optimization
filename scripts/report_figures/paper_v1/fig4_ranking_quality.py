"""Figure 4 - ranking quality by predictor input family.

A horizontal dot-and-interval (forest) plot rather than bars. Kendall's tau is
an estimate with an interval, not a quantity accumulated from zero: bars spend
most of their ink on the uninteresting distance from zero, and truncating them
to fix that exaggerates the differences. The horizontal layout also removes the
rotated category labels, which were themselves a symptom of the wrong
orientation.

Forest-plot convention: the proposed input reads first (top row), its
ablation directly under it, and the baseline group sits below a visual gap.
Colour encodes ownership, never identity: grey baselines, one accent for the
proposed input. The tuned-baseline mean is a labelled reference line, and the
headline delta is printed in the right margin against a bracket spanning the
two rows it compares.
"""

from __future__ import annotations

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
# Top-to-bottom: ours, its ablation, then the baseline group after a gap.
ROWS = (
    "bert_prompt_schema",
    "bert_prompt_only",
    "lightgbm_grid",
    "lightgbm_scalar",
    "schema_hash_lookup",
)
PITCH = 0.62          # row pitch within a group (tight forest rhythm)
GROUP_GAP = 0.55      # extra separation between the BERT and baseline groups
KEY = {
    "bert_prompt_schema": "prompt_schema",
    "bert_prompt_only": "prompt_only",
    "lightgbm_grid": "lightgbm_grid",
    "lightgbm_scalar": "lightgbm_scalar",
    "schema_hash_lookup": "schema_hash_lookup",
}


def main() -> None:
    payload = load_json(T1)
    results = payload["results"]
    record = payload["baseline_of_record"]["model"]

    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_WIDTH, 2.05),
                           layout="constrained")

    # Manual y positions: tight pitch inside a group, a gap between groups.
    ys, cursor = [], 0.0
    for index, name in enumerate(ROWS):
        if index == 2:
            cursor += GROUP_GAP
        ys.append(cursor)
        cursor += PITCH
    ys = np.asarray(ys)

    highs, means = [], {}
    for name, row_y in zip(ROWS, ys):
        cell = results[name]["all"]
        mean = float(cell["mean_tau_b"])
        low, high = (float(v) for v in cell["ci95_seed17"])
        highs.append(high)
        means[name] = mean
        colour = COLOR[KEY[name]]
        is_ours = name == "bert_prompt_schema"
        err = ax.errorbar(
            mean, row_y,
            xerr=[[mean - low], [high - mean]],
            fmt="o", markersize=6.0 if is_ours else 5.0,
            linewidth=1.4 if is_ours else 1.1, capsize=0,
            color=colour, markerfacecolor=colour, markeredgecolor=colour,
            zorder=3,
        )
        for bar in err[2]:
            bar.set_capstyle("butt")
        # Per-seed ticks sit just below the error-bar line so they stay
        # visible instead of vanishing under the same-colour interval.
        for seed_value in cell["per_seed_tau_b"].values():
            ax.plot(float(seed_value), row_y + 0.17, marker="|",
                    markersize=3.2, markeredgewidth=0.9, color=colour,
                    alpha=0.7, zorder=2)
        # Point estimate printed at the whisker end (mean only, never both
        # CI endpoints: the whisker already draws the interval).
        ax.text(high + 0.012, row_y, f"{mean:.3f}", ha="left", va="center",
                fontsize=8, color=OKABE_ITO["black"], zorder=4)

    baseline = float(results[record]["all"]["mean_tau_b"])
    proposed = float(results["bert_prompt_schema"]["all"]["mean_tau_b"])

    # Reference line at the tuned-baseline mean, labelled once at the top edge
    # (inside the axes so the legend band above stays clear of it).
    ax.axvline(baseline, color=OKABE_ITO["gray"], linewidth=0.7,
               linestyle=(0, (4, 3)), zorder=1)
    ax.text(baseline + 0.008, -0.44, "tuned LightGBM grid", ha="left",
            va="center", fontsize=9, color=OKABE_ITO["dark_gray"])

    # Two effect callouts in the right margin, framed EXION-style. The inner
    # bracket spans the two BERT rows (schema-text ablation); the outer
    # bracket spans ours -> tuned baseline of record. Each bracket carries a
    # midpoint leader into its framed box so ownership stays unambiguous.
    FRAME = dict(boxstyle="square,pad=0.25", facecolor="#f2f2f2",
                 edgecolor="#888888", linewidth=0.6)
    x1 = max(highs) + 0.092          # clears the 8pt point-estimate labels
    x2 = x1 + 0.034
    x_text = x2 + 0.016
    tick = 0.012

    def bracket(x_line, y_a, y_b, y_leader, text):
        ax.plot([x_line, x_line], [y_a, y_b], color=OKABE_ITO["black"],
                linewidth=0.8, solid_capstyle="butt", clip_on=False, zorder=3)
        for row_y in (y_a, y_b):
            ax.plot([x_line - tick, x_line], [row_y, row_y],
                    color=OKABE_ITO["black"], linewidth=0.8,
                    solid_capstyle="butt", clip_on=False, zorder=3)
        ax.plot([x_line, x_text - 0.006], [y_leader, y_leader],
                color=OKABE_ITO["black"], linewidth=0.6, clip_on=False,
                zorder=3)
        ax.text(x_text, y_leader, text, ha="left", va="center", fontsize=8,
                color=OKABE_ITO["black"], linespacing=1.3, bbox=FRAME,
                zorder=5)

    schema_delta = proposed - means["bert_prompt_only"]
    bracket(x1, ys[0], ys[1], (ys[0] + ys[1]) / 2,
            f"schema text\n$+{schema_delta:.3f}$")
    bracket(x2, ys[0], ys[ROWS.index(record)], 1.42,
            f"$+{proposed - baseline:.3f}$ vs\ntuned scalar")

    # Row labels left-aligned on a common edge; no y tick marks.
    ax.set_yticks(ys)
    ax.set_yticklabels([LABEL[name] for name in ROWS])
    for label in ax.get_yticklabels():
        label.set_ha("left")
    ax.set_ylim(ys[-1] + 0.45, -0.62)
    # Left-align row labels on a common edge: pad derives from the widest
    # label's rendered extent instead of a hand-tuned constant.
    fig.canvas.draw()
    widest_px = max(lbl.get_window_extent().width
                    for lbl in ax.get_yticklabels())
    ax.tick_params(axis="y", length=0, pad=widest_px * 72.0 / fig.dpi + 3)

    n_test = int(payload["split_sizes"]["test"])
    ax.set_xlabel(f"Kendall $\\tau_b$ (test split, $n{{=}}{n_test}$)")
    ax.set_xlim(0.35, 1.0)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7])
    # The measured axis ends at the last tick; everything to its right is
    # annotation margin, so the spine stops there instead of underlining it.
    ax.spines["bottom"].set_bounds(0.35, 0.72)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    # The per-seed tick is the one mark a reader cannot decode from the row
    # labels; the key names it in a boxed band above the axes.
    from matplotlib.lines import Line2D

    legend = fig.legend(handles=[
        Line2D([], [], marker="o", color=OKABE_ITO["black"], linewidth=1.1,
               markersize=5.0, label="mean $\\pm$ 95% CI"),
        Line2D([], [], marker="|", linestyle="none", color=OKABE_ITO["black"],
               markersize=4.0, markeredgewidth=0.9, label="per-seed $\\tau_b$"),
    ], loc="outside upper center", ncols=2, fontsize=8, frameon=True,
        handletextpad=0.4, columnspacing=1.4, handlelength=1.6,
        borderaxespad=0.15)
    legend.get_frame().set_facecolor("#f2f2f2")
    legend.get_frame().set_edgecolor("#888888")
    legend.get_frame().set_linewidth(0.6)

    save(fig, "ranking.pdf")
    record_provenance("ranking.pdf", [T1])
    print(f"baseline of record {record}={baseline:.4f} proposed={proposed:.4f}")


if __name__ == "__main__":
    main()
