#!/usr/bin/env python3
"""Build publication-v3 Figure 4 from committed Tier-2 evidence."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.report_figures.publication_v2.figures_04_06 import (  # noqa: E402
    load_predictor_evidence,
)


OUTPUT_DIR = REPO_ROOT / "latex_source" / "figures" / "publication-v3"
CREATOR = "publication_v3/figure_04.py"

WIDTH_MM = 181.9
HEIGHT_MM = 86.0
MM_PER_INCH = 25.4

TEXT = "#1A1A1A"
LEARNED = "#0072B2"
NON_LEARNED = "#E69F00"
NEUTRAL = "#4A4A4A"
WARNING = "#D55E00"
GRID = "#D9D9D9"

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 8.0,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "axes.edgecolor": NEUTRAL,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.0,
    "axes.titleweight": "bold",
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.0,
    "axes.linewidth": 0.75,
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": None,
    "savefig.pad_inches": 0.0,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


DISPLAY_LABELS = {
    "prompt_schema": "Prompt + schema",
    "full_context": "Full context",
    "prompt_only": "Prompt only",
    "lightgbm": "LightGBM structural",
}
DISPLAY_ORDER = ("prompt_schema", "full_context", "prompt_only", "lightgbm")


def load_evidence() -> dict[str, object]:
    """Reuse the audited loader while preserving test/validation separation."""

    return load_predictor_evidence()


def _plot_group(ax, group, y: float, *, color: str, single_seed: bool) -> None:
    if single_seed:
        ax.scatter(
            [group.mean],
            [y],
            marker="D",
            s=37,
            facecolor=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
    else:
        ax.hlines(y, group.low, group.high, color=color, linewidth=2.0, zorder=2)
        ax.vlines(
            [group.low, group.high],
            y - 0.10,
            y + 0.10,
            color=color,
            linewidth=1.0,
            zorder=2,
        )
        jitter = np.linspace(-0.12, 0.12, len(group.values))
        ax.scatter(
            group.values,
            y + jitter,
            marker="o",
            s=22,
            facecolor="white",
            edgecolor=color,
            linewidth=0.9,
            zorder=3,
        )
        ax.scatter(
            [group.mean],
            [y],
            marker="o",
            s=46,
            facecolor=color,
            edgecolor="white",
            linewidth=0.9,
            zorder=4,
        )


def _add_break_marks(ax_left, ax_right) -> None:
    size = 0.014
    kwargs = {"color": NEUTRAL, "clip_on": False, "linewidth": 0.9}
    ax_left.plot((1 - size, 1 + size), (-size, +size), transform=ax_left.transAxes, **kwargs)
    ax_left.plot((1 - size, 1 + size), (1 - size, 1 + size), transform=ax_left.transAxes, **kwargs)
    ax_right.plot((-size, +size), (-size, +size), transform=ax_right.transAxes, **kwargs)
    ax_right.plot((-size, +size), (1 - size, 1 + size), transform=ax_right.transAxes, **kwargs)


def build_figure() -> tuple[Figure, dict[str, object]]:
    """Return the full-width forest plot plus separate validation ablation."""

    evidence = load_evidence()
    groups = {group.key: group for group in evidence["groups"]}
    with plt.rc_context(STYLE):
        fig = plt.figure(
            figsize=(WIDTH_MM / MM_PER_INCH, HEIGHT_MM / MM_PER_INCH)
        )
        outer = fig.add_gridspec(
            1,
            2,
            width_ratios=(1.28, 1.0),
            left=0.165,
            right=0.985,
            bottom=0.215,
            top=0.635,
            wspace=0.30,
        )
        forest = outer[0].subgridspec(
            1,
            2,
            width_ratios=(1.0, 3.2),
            wspace=0.08,
        )
        ax_structural = fig.add_subplot(forest[0])
        ax_bert = fig.add_subplot(forest[1], sharey=ax_structural)
        ax_curve = fig.add_subplot(outer[1])
        ax_structural.set_gid("panel-a-structural-range")
        ax_bert.set_gid("panel-a-bert-ranges")
        ax_curve.set_gid("panel-b-validation-curve")

        y_by_key = {key: 3 - index for index, key in enumerate(DISPLAY_ORDER)}
        _plot_group(
            ax_structural,
            groups["lightgbm"],
            y_by_key["lightgbm"],
            color=NON_LEARNED,
            single_seed=True,
        )
        for key in ("prompt_schema", "full_context", "prompt_only"):
            _plot_group(
                ax_bert,
                groups[key],
                y_by_key[key],
                color=LEARNED,
                single_seed=False,
            )

        ax_structural.set_xlim(0.410, 0.452)
        ax_structural.set_xticks((0.42, 0.44))
        ax_bert.set_xlim(0.565, 0.668)
        ax_bert.set_xticks((0.58, 0.60, 0.62, 0.64, 0.66))
        ax_structural.set_yticks(
            [y_by_key[key] for key in DISPLAY_ORDER],
            [DISPLAY_LABELS[key] for key in DISPLAY_ORDER],
        )
        ax_structural.set_ylim(-0.55, 3.55)
        ax_bert.tick_params(axis="y", left=False, labelleft=False)
        for ax in (ax_structural, ax_bert):
            ax.grid(axis="x")
            ax.grid(axis="y", visible=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        ax_structural.spines["right"].set_visible(False)
        ax_bert.spines["left"].set_visible(False)
        _add_break_marks(ax_structural, ax_bert)

        for key in DISPLAY_ORDER:
            group = groups[key]
            target_ax = ax_structural if key == "lightgbm" else ax_bert
            target_ax.text(
                group.mean + (0.0020 if key == "lightgbm" else 0.0030),
                y_by_key[key] + (0.20 if key != "lightgbm" else 0.0),
                f"{group.mean:.3f}",
                fontsize=7,
                va="center",
                ha="left",
            )

        pools = evidence["pool_sizes"]
        validation = evidence["validation_tau"]
        ax_curve.plot(
            pools,
            validation,
            color=LEARNED,
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.0,
            linewidth=1.6,
            markersize=4.5,
            zorder=3,
        )
        for pool, tau in zip(pools, validation):
            ax_curve.text(
                pool,
                tau + 0.0017,
                f"{tau:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        ax_curve.set_xscale("log", base=2)
        ax_curve.set_xlim(430, 4_650)
        ax_curve.set_xticks(pools, [f"{pool:,}" for pool in pools])
        ax_curve.minorticks_off()
        ax_curve.set_ylim(0.598, 0.644)
        ax_curve.set_yticks((0.60, 0.61, 0.62, 0.63, 0.64))
        ax_curve.set_xlabel("Nominal training-pool size")
        ax_curve.set_ylabel("Validation Kendall τ-b")
        ax_curve.grid(axis="y")
        ax_curve.grid(axis="x", visible=False)
        ax_curve.spines["top"].set_visible(False)
        ax_curve.spines["right"].set_visible(False)
        ax_curve.text(
            0.5,
            0.055,
            "effective n = 499 / 999 / 1,997 / 3,997",
            transform=ax_curve.transAxes,
            ha="center",
            va="bottom",
            fontsize=7,
        )

        fig.suptitle(
            "Schema-aware BERT ranks highest; prompt + schema and full-context seed ranges overlap",
            fontsize=10,
            fontweight="bold",
            y=0.965,
        )
        fig.text(
            0.075,
            0.775,
            "(a)",
            fontsize=10,
            fontweight="bold",
            va="bottom",
        )
        fig.text(
            0.355,
            0.775,
            "Held-out test ranking",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="bottom",
        )
        fig.text(
            0.675,
            0.775,
            "(b)",
            fontsize=10,
            fontweight="bold",
            va="bottom",
        )
        fig.text(
            0.830,
            0.775,
            "Full-context data scale",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="bottom",
        )
        fig.text(
            0.830,
            0.655,
            "validation only · single seed\nnot deployment evidence",
            fontsize=8,
            fontweight="bold",
            color=WARNING,
            ha="center",
            va="bottom",
            linespacing=0.9,
            bbox={
                "facecolor": "white",
                "edgecolor": WARNING,
                "linewidth": 0.8,
                "boxstyle": "round,pad=0.22",
            },
        )
        legend_handles = [
            Line2D(
                [0],
                [0],
                color=LEARNED,
                marker="o",
                markerfacecolor=LEARNED,
                linewidth=2.0,
                label="mean + observed seed min–max (BERT n=3)",
            ),
            Line2D(
                [0],
                [0],
                color=NON_LEARNED,
                marker="D",
                linewidth=0,
                label="single seed (LightGBM n=1)",
            ),
        ]
        legend = fig.legend(
            handles=legend_handles,
            ncol=2,
            loc="lower left",
            bbox_to_anchor=(0.165, 0.020),
            frameon=False,
            handlelength=2.0,
            columnspacing=1.5,
        )
        legend.set_gid("forest-legend")
        fig.text(
            0.350,
            0.145,
            "Held-out test Kendall τ-b",
            fontsize=8,
            ha="center",
        )
    return fig, evidence


def _fix_svg_canvas(svg_path: Path) -> None:
    raw = svg_path.read_text(encoding="utf-8")
    raw = re.sub(r'width="[^"]+"', f'width="{WIDTH_MM:.1f}mm"', raw, count=1)
    raw = re.sub(r'height="[^"]+"', f'height="{HEIGHT_MM:.1f}mm"', raw, count=1)
    svg_path.write_text(raw, encoding="utf-8")


def render(output_dir: Path = OUTPUT_DIR) -> tuple[Path, Path, Path]:
    """Write editable SVG/PDF and a fixed-canvas 300 dpi PNG."""

    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / "fig4.svg"
    pdf_path = output_dir / "fig4.pdf"
    png_path = output_dir / "fig4.png"
    fig, _ = build_figure()
    with plt.rc_context(STYLE):
        fig.savefig(svg_path, metadata={"Creator": CREATOR, "Date": None})
        _fix_svg_canvas(svg_path)
        fig.savefig(
            pdf_path,
            metadata={"Creator": CREATOR, "CreationDate": None, "ModDate": None},
        )
        fig.savefig(png_path, dpi=300, metadata={"Software": CREATOR})
    plt.close(fig)
    return svg_path, pdf_path, png_path


def main() -> int:
    for path in render():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
