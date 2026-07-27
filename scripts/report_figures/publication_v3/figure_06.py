#!/usr/bin/env python3
"""Build publication-v3 Figure 6 from common-complete BFCL evidence."""

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
    load_ood_evidence,
    paired_hierarchical_summaries,
)


OUTPUT_DIR = REPO_ROOT / "latex_source" / "figures" / "publication-v3"
CREATOR = "publication_v3/figure_06.py"

WIDTH_MM = 181.9
HEIGHT_MM = 78.0
MM_PER_INCH = 25.4

TEXT = "#1A1A1A"
LEARNED = "#0072B2"
NEUTRAL = "#4A4A4A"
WARNING = "#D55E00"
GRID = "#D9D9D9"

BASELINE = "StockFCFSShim"
DISPLAY_ORDER = ("TailSafe", "GatedHybrid", "PureLTR")
DISPLAY_LABELS = {
    "PureLTR": "Pure LTR",
    "GatedHybrid": "Gated hybrid",
    "TailSafe": "Tail safe",
}
REPEAT_MARKERS = ("o", "s", "D")

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


def _repeat_reductions(
    arrays: dict[str, np.ndarray],
) -> dict[str, dict[str, tuple[float, ...]]]:
    """Compute reductions against the matched FCFS-shim repeat."""

    output: dict[str, dict[str, tuple[float, ...]]] = {}
    for policy in DISPLAY_ORDER:
        output[policy] = {}
        for metric, statistic in (
            ("mean", np.mean),
            ("p99", lambda values: np.percentile(values, 99)),
        ):
            reductions = []
            for repeat_index in range(3):
                baseline_value = float(statistic(arrays[BASELINE][repeat_index]))
                policy_value = float(statistic(arrays[policy][repeat_index]))
                reductions.append(
                    100.0 * (baseline_value - policy_value) / baseline_value
                )
            output[policy][metric] = tuple(reductions)
    return output


def load_evidence() -> dict[str, object]:
    """Load the invalid-provenance, common-complete BFCL subset."""

    evidence = load_ood_evidence()
    summaries, _ = paired_hierarchical_summaries(evidence["arrays"], BASELINE)
    return {
        **evidence,
        "summaries": summaries,
        "repeat_reductions": _repeat_reductions(evidence["arrays"]),
    }


def _plot_reduction_panel(ax, evidence: dict[str, object], metric: str) -> None:
    summaries = evidence["summaries"]
    repeat_reductions = evidence["repeat_reductions"]
    y_by_policy = {
        policy: len(DISPLAY_ORDER) - 1 - index
        for index, policy in enumerate(DISPLAY_ORDER)
    }
    ax.axvline(0.0, color=NEUTRAL, linestyle=(0, (3, 2)), linewidth=0.9, zorder=1)
    for policy in DISPLAY_ORDER:
        y = y_by_policy[policy]
        summary = summaries[policy][metric]
        low, high = summary.improvement_interval
        ax.hlines(y, low, high, color=LEARNED, linewidth=1.9, zorder=2)
        ax.vlines((low, high), y - 0.10, y + 0.10, color=LEARNED, linewidth=0.9)
        for jitter, marker, value in zip(
            (-0.14, 0.0, 0.14),
            REPEAT_MARKERS,
            repeat_reductions[policy][metric],
        ):
            ax.scatter(
                value,
                y + jitter,
                marker=marker,
                s=24,
                facecolor="white",
                edgecolor=LEARNED,
                linewidth=0.9,
                zorder=3,
            )
        ax.scatter(
            summary.improvement,
            y,
            marker="o",
            s=48,
            facecolor=LEARNED,
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
    ax.set_ylim(-0.50, len(DISPLAY_ORDER) - 0.50)
    ax.set_yticks(
        [y_by_policy[policy] for policy in DISPLAY_ORDER],
        [DISPLAY_LABELS[policy] for policy in DISPLAY_ORDER],
    )
    ax.set_xlabel("Reduction vs FCFS shim (%) ↓")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_figure() -> tuple[Figure, dict[str, object]]:
    """Return two paired-reduction panels with one compact scope line."""

    evidence = load_evidence()
    summaries = evidence["summaries"]
    with plt.rc_context(STYLE):
        fig = plt.figure(
            figsize=(WIDTH_MM / MM_PER_INCH, HEIGHT_MM / MM_PER_INCH)
        )
        ax_mean = fig.add_axes((0.130, 0.240, 0.340, 0.410))
        ax_p99 = fig.add_axes((0.620, 0.240, 0.340, 0.410), sharey=ax_mean)
        ax_mean.set_gid("panel-a-mean-reduction")
        ax_p99.set_gid("panel-b-p99-reduction")

        _plot_reduction_panel(ax_mean, evidence, "mean")
        _plot_reduction_panel(ax_p99, evidence, "p99")
        ax_mean.set_xlim(-2, 25)
        ax_mean.set_xticks((0, 5, 10, 15, 20, 25))
        ax_p99.set_xlim(-3, 52)
        ax_p99.set_xticks((0, 10, 20, 30, 40, 50))
        ax_p99.tick_params(axis="y", left=False, labelleft=False)

        mean_values = [summaries[policy]["mean"].improvement for policy in DISPLAY_ORDER]
        p99_values = [summaries[policy]["p99"].improvement for policy in DISPLAY_ORDER]
        fig.suptitle(
            "One invalid-provenance BFCL subset: learned policies show lower TTLT than FCFS shim",
            fontsize=10,
            fontweight="bold",
            y=0.965,
        )
        scope = fig.text(
            0.545,
            0.825,
            "119 IDs × 3 common-complete · 7 error rows excluded · "
            "source summaries valid=false · one workload · no prompt control",
            fontsize=7,
            fontweight="bold",
            color=WARNING,
            ha="center",
            va="center",
        )
        scope.set_gid("scope-note")
        for label_x, title_x, label, title in (
            (
                0.070,
                0.300,
                "(a)",
                f"Mean TTLT: {min(mean_values):.1f}–{max(mean_values):.1f}% lower",
            ),
            (
                0.560,
                0.790,
                "(b)",
                f"p99 TTLT: {min(p99_values):.1f}–{max(p99_values):.1f}% lower",
            ),
        ):
            fig.text(
                label_x,
                0.710,
                label,
                fontsize=10,
                fontweight="bold",
                va="bottom",
            )
            fig.text(
                title_x,
                0.710,
                title,
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="bottom",
            )

        paired_handles = [
            Line2D(
                [0],
                [0],
                color=LEARNED,
                marker="o",
                markerfacecolor=LEARNED,
                linewidth=1.9,
                label="Paired hierarchical 95% CI",
            )
        ]
        paired_handles.extend(
            Line2D(
                [0],
                [0],
                color="none",
                marker=marker,
                markerfacecolor="white",
                markeredgecolor=LEARNED,
                markeredgewidth=0.9,
                label=f"repeat {index}",
            )
            for index, marker in enumerate(REPEAT_MARKERS, start=1)
        )
        legend = fig.legend(
            handles=paired_handles,
            ncol=4,
            loc="lower center",
            bbox_to_anchor=(0.545, 0.045),
            frameon=False,
            handlelength=2.0,
            columnspacing=1.5,
        )
        legend.set_gid("paired-repeat-legend")
    return fig, evidence


def _fix_svg_canvas(svg_path: Path) -> None:
    raw = svg_path.read_text(encoding="utf-8")
    raw = re.sub(r'width="[^"]+"', f'width="{WIDTH_MM:.1f}mm"', raw, count=1)
    raw = re.sub(r'height="[^"]+"', f'height="{HEIGHT_MM:.1f}mm"', raw, count=1)
    svg_path.write_text(raw, encoding="utf-8")


def render(output_dir: Path = OUTPUT_DIR) -> tuple[Path, Path, Path]:
    """Write editable SVG/PDF and a fixed-canvas 300 dpi PNG."""

    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / "fig6.svg"
    pdf_path = output_dir / "fig6.pdf"
    png_path = output_dir / "fig6.png"
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
