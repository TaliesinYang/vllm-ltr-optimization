#!/usr/bin/env python3
"""Build publication-v3 Figure 5 from paired live-serving evidence."""

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
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.report_figures.publication_v2.figures_04_06 import (  # noqa: E402
    load_mixed_evidence,
    paired_hierarchical_summaries,
)


OUTPUT_DIR = REPO_ROOT / "latex_source" / "figures" / "publication-v3"
CREATOR = "publication_v3/figure_05.py"

WIDTH_MM = 181.9
HEIGHT_MM = 92.0
MM_PER_INCH = 25.4

TEXT = "#1A1A1A"
LEARNED = "#0072B2"
NON_LEARNED = "#E69F00"
NEUTRAL = "#4A4A4A"
GRID = "#D9D9D9"

BASELINE = "stock_fcfs"
REPRESENTATIVE_LEARNED = "GatedHybrid"
CCDF_POLICIES = (BASELINE, REPRESENTATIVE_LEARNED, "PromptLengthSJF")
DISPLAY_ORDER = (
    "PromptLengthSJF",
    "LTRAging",
    "TailSafe",
    "GatedHybrid",
    "PureLTR",
    "StockFCFSShim",
)
DISPLAY_LABELS = {
    "stock_fcfs": "Stock FCFS",
    "StockFCFSShim": "FCFS shim",
    "PureLTR": "Pure LTR",
    "GatedHybrid": "Gated hybrid",
    "TailSafe": "Tail safe",
    "LTRAging": "LTR aging",
    "PromptLengthSJF": "Prompt SJF (non-learned)",
}
POLICY_COLORS = {
    "stock_fcfs": NEUTRAL,
    "StockFCFSShim": NEUTRAL,
    "PureLTR": LEARNED,
    "GatedHybrid": LEARNED,
    "TailSafe": LEARNED,
    "LTRAging": LEARNED,
    "PromptLengthSJF": NON_LEARNED,
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
    """Compute repeat-matched reductions against Stock FCFS."""

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
    """Load balanced repeated requests and compute paired summaries."""

    evidence = load_mixed_evidence()
    arrays = evidence["arrays"]
    summaries, _ = paired_hierarchical_summaries(arrays, BASELINE)
    return {
        **evidence,
        "summaries": summaries,
        "repeat_reductions": _repeat_reductions(arrays),
        "representative_learned": REPRESENTATIVE_LEARNED,
    }


def _set_plain_log_ticks(ax) -> None:
    ticks = (0.01, 0.1, 1.0)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter(("0.01", "0.1", "1")))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _plot_reduction_panel(
    ax,
    evidence: dict[str, object],
    metric: str,
) -> None:
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
        color = POLICY_COLORS[policy]
        ax.hlines(y, low, high, color=color, linewidth=1.8, zorder=2)
        ax.vlines((low, high), y - 0.10, y + 0.10, color=color, linewidth=0.9)
        for jitter, marker, value in zip(
            (-0.13, 0.0, 0.13),
            REPEAT_MARKERS,
            repeat_reductions[policy][metric],
        ):
            ax.scatter(
                value,
                y + jitter,
                marker=marker,
                s=21,
                facecolor="white",
                edgecolor=color,
                linewidth=0.8,
                zorder=3,
            )
        ax.scatter(
            summary.improvement,
            y,
            marker="o",
            s=43,
            facecolor=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )

    prompt_summary = summaries["PromptLengthSJF"][metric]
    ax.text(
        prompt_summary.improvement,
        y_by_policy["PromptLengthSJF"] + 0.31,
        f"{prompt_summary.improvement:.1f}% ↓",
        color=NON_LEARNED,
        fontsize=8,
        fontweight="bold",
        ha="center",
        va="bottom",
    )
    ax.set_ylim(-0.55, len(DISPLAY_ORDER) - 0.20)
    ax.set_yticks(
        [y_by_policy[policy] for policy in DISPLAY_ORDER],
        [DISPLAY_LABELS[policy] for policy in DISPLAY_ORDER],
    )
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_figure() -> tuple[Figure, dict[str, object]]:
    """Return representative CCDFs and paired mean/p99 reductions."""

    evidence = load_evidence()
    arrays = evidence["arrays"]
    with plt.rc_context(STYLE):
        fig = plt.figure(
            figsize=(WIDTH_MM / MM_PER_INCH, HEIGHT_MM / MM_PER_INCH)
        )
        ax_ccdf = fig.add_axes((0.075, 0.315, 0.300, 0.365))
        ax_mean = fig.add_axes((0.570, 0.315, 0.180, 0.365))
        ax_p99 = fig.add_axes((0.805, 0.315, 0.180, 0.365), sharey=ax_mean)
        ax_ccdf.set_gid("panel-a-ccdf")
        ax_mean.set_gid("panel-b-mean-reduction")
        ax_p99.set_gid("panel-c-p99-reduction")

        ccdf_styles = {
            BASELINE: {"color": NEUTRAL, "linestyle": (0, (5, 2))},
            REPRESENTATIVE_LEARNED: {"color": LEARNED, "linestyle": "-"},
            "PromptLengthSJF": {"color": NON_LEARNED, "linestyle": "-"},
        }
        for policy in CCDF_POLICIES:
            values = np.sort(arrays[policy].ravel()) / 1000.0
            survival = (values.size - np.arange(values.size)) / values.size
            ax_ccdf.step(
                values,
                survival,
                where="post",
                linewidth=1.6,
                zorder=3,
                **ccdf_styles[policy],
            )
        ax_ccdf.set_yscale("log")
        ax_ccdf.set_xlim(0, 28)
        ax_ccdf.set_ylim(0.002, 1.05)
        _set_plain_log_ticks(ax_ccdf)
        ax_ccdf.set_xlabel("TTLT (s)")
        ax_ccdf.set_ylabel("CCDF, Pr(TTLT > t)")
        ax_ccdf.grid(axis="y")
        ax_ccdf.grid(axis="x", visible=False)
        ax_ccdf.spines["top"].set_visible(False)
        ax_ccdf.spines["right"].set_visible(False)

        _plot_reduction_panel(ax_mean, evidence, "mean")
        _plot_reduction_panel(ax_p99, evidence, "p99")
        ax_mean.set_xlim(-6, 21)
        ax_mean.set_xticks((-5, 0, 5, 10, 15, 20))
        ax_mean.set_xlabel("Mean TTLT reduction\nvs Stock FCFS (%) ↓", linespacing=0.95)
        ax_p99.set_xlim(-12, 42)
        ax_p99.set_xticks((-10, 0, 10, 20, 30, 40))
        ax_p99.set_xlabel("p99 TTLT reduction\nvs Stock FCFS (%) ↓", linespacing=0.95)
        ax_p99.tick_params(axis="y", left=False, labelleft=False)

        fig.suptitle(
            "Short-job ordering lowers live TTLT; learned policies do not beat Prompt SJF",
            fontsize=10,
            fontweight="bold",
            y=0.965,
        )
        for x, label, title in (
            (0.055, "(a)", "Representative TTLT distributions"),
            (0.515, "(b)", "Paired mean reduction ↓"),
            (0.765, "(c)", "Paired p99 reduction ↓"),
        ):
            fig.text(x, 0.735, label, fontsize=10, fontweight="bold", va="bottom")
            title_x = {"(a)": 0.225, "(b)": 0.660, "(c)": 0.895}[label]
            fig.text(
                title_x,
                0.735,
                title,
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="bottom",
            )

        ccdf_handles = [
            Line2D(
                [0],
                [0],
                color=NEUTRAL,
                linestyle=(0, (5, 2)),
                linewidth=1.6,
                label="Stock FCFS",
            ),
            Line2D(
                [0],
                [0],
                color=LEARNED,
                linewidth=1.6,
                label="Gated hybrid (representative learned)",
            ),
            Line2D(
                [0],
                [0],
                color=NON_LEARNED,
                linewidth=1.6,
                label="Prompt SJF (non-learned)",
            ),
        ]
        ccdf_legend = fig.legend(
            handles=ccdf_handles,
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.53, 0.880),
            frameon=False,
            handlelength=2.0,
            columnspacing=1.5,
        )
        ccdf_legend.set_gid("shared-ccdf-legend")

        paired_handles = [
            Line2D(
                [0],
                [0],
                color=NEUTRAL,
                marker="o",
                markerfacecolor=NEUTRAL,
                linewidth=1.8,
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
                markeredgecolor=NEUTRAL,
                markeredgewidth=0.8,
                label=f"repeat {index}",
            )
            for index, marker in enumerate(REPEAT_MARKERS, start=1)
        )
        paired_legend = fig.legend(
            handles=paired_handles,
            ncol=4,
            loc="lower center",
            bbox_to_anchor=(0.64, 0.078),
            frameon=False,
            handlelength=2.0,
            columnspacing=1.2,
        )
        paired_legend.set_gid("paired-repeat-legend")
        fig.text(
            0.55,
            0.018,
            "3 paired repeats · 150 recurring request IDs per repeat · "
            "repeat + request-ID clusters resampled jointly",
            fontsize=7,
            ha="center",
            va="bottom",
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
    svg_path = output_dir / "fig5.svg"
    pdf_path = output_dir / "fig5.pdf"
    png_path = output_dir / "fig5.png"
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
