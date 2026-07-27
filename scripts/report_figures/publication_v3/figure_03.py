#!/usr/bin/env python3
"""Build publication-v3 Figure 3 from the committed legacy sweep."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plot_final_report_figures import (
    DEFAULT_BASELINE_SUMMARY,
    parse_baseline_summary,
)


OUTPUT_DIR = REPO_ROOT / "latex_source" / "figures" / "publication-v3"
CREATOR = "publication_v3/figure_03.py"

WIDTH_MM = 181.9
HEIGHT_MM = 84.0
MM_PER_INCH = 25.4

TEXT = "#1A1A1A"
LEARNED = "#0072B2"
NEUTRAL = "#4A4A4A"
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


def load_evidence(path: Path = DEFAULT_BASELINE_SUMMARY) -> dict[str, object]:
    """Load measured rows and compute only the two locked comparisons."""

    data = parse_baseline_summary(path)
    rates = data["FCFS"]["rate"]
    index_16 = rates.index(16.0)
    index_64 = rates.index(64.0)
    ttft_reduction = 100.0 * (
        1.0
        - data["LTR"]["ttft_ms"][index_16]
        / data["FCFS"]["ttft_ms"][index_16]
    )
    p99_ratio = (
        data["LTR"]["p99_tpot_ms"][index_64]
        / data["FCFS"]["p99_tpot_ms"][index_64]
    )
    return {
        "data": data,
        "ttft_rate_qps": 16.0,
        "tail_rate_qps": 64.0,
        "ttft_reduction_pct": ttft_reduction,
        "p99_tpot_ratio": p99_ratio,
    }


def _plain_log_ticks(ax, values: tuple[float, ...]) -> None:
    ax.yaxis.set_major_locator(FixedLocator(values))
    ax.yaxis.set_major_formatter(
        FixedFormatter([f"{value:,.0f}" if value >= 1_000 else f"{value:g}" for value in values])
    )
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_minor_formatter(NullFormatter())


def build_figure() -> tuple[Figure, dict[str, object]]:
    """Return the full-width two-panel baseline trade-off figure."""

    evidence = load_evidence()
    data = evidence["data"]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(WIDTH_MM / MM_PER_INCH, HEIGHT_MM / MM_PER_INCH),
        )
        fig.subplots_adjust(
            left=0.102,
            right=0.985,
            bottom=0.205,
            top=0.695,
            wspace=0.30,
        )
        ax_ttft, ax_tail = axes
        ax_ttft.set_gid("panel-a-mean-ttft")
        ax_tail.set_gid("panel-b-tail-tpot")

        styles = {
            "FCFS": {
                "color": NEUTRAL,
                "marker": "o",
                "linestyle": "--",
                "label": "Stock FCFS",
            },
            "LTR": {
                "color": LEARNED,
                "marker": "s",
                "linestyle": "-",
                "label": "Legacy shortest-first",
            },
        }
        legend_handles = []
        for method in ("FCFS", "LTR"):
            style = styles[method]
            line_ttft = ax_ttft.plot(
                data[method]["rate"],
                data[method]["ttft_ms"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.5,
                markersize=4.4,
                markerfacecolor="white",
                markeredgewidth=1.0,
                label=style["label"],
                zorder=3,
            )[0]
            line_ttft.set_gid(f"mean-ttft-{method.lower()}")
            line_tail = ax_tail.plot(
                data[method]["rate"],
                data[method]["p99_tpot_ms"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.5,
                markersize=4.4,
                markerfacecolor="white",
                markeredgewidth=1.0,
                zorder=3,
            )[0]
            line_tail.set_gid(f"p99-tpot-{method.lower()}")
            legend_handles.append(line_ttft)

        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlim(1.75, 70)
            ax.set_xticks(data["FCFS"]["rate"])
            ax.set_xticklabels([f"{value:g}" for value in data["FCFS"]["rate"]])
            ax.set_xlabel("Request rate (queries/s)")
            ax.grid(axis="y", which="major")
            ax.grid(axis="x", visible=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        ax_ttft.set_ylim(40, 400_000)
        _plain_log_ticks(ax_ttft, (50, 100, 1_000, 10_000, 100_000))
        ax_ttft.set_ylabel("Mean TTFT (ms)")
        ax_ttft.set_title("Mean TTFT — lower is better", pad=8)

        ax_tail.set_ylim(22, 2_000)
        _plain_log_ticks(ax_tail, (25, 50, 100, 250, 500, 1_000))
        ax_tail.set_ylabel("p99 TPOT (ms)")
        ax_tail.set_title("Tail per-token cost — lower is better", pad=8)

        index_16 = data["LTR"]["rate"].index(evidence["ttft_rate_qps"])
        ax_ttft.annotate(
            "16 qps · onset of saturation\n"
            f"{evidence['ttft_reduction_pct']:.1f}% lower",
            xy=(16, data["LTR"]["ttft_ms"][index_16]),
            xycoords="data",
            xytext=(0.53, 0.23),
            textcoords="axes fraction",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=LEARNED,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2},
            arrowprops={"arrowstyle": "-", "color": LEARNED, "linewidth": 0.9},
        )
        ax_tail.text(
            0.02,
            0.97,
            "64 qps · highest tested load\n"
            f"{evidence['p99_tpot_ratio']:.2f}× higher",
            transform=ax_tail.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            fontweight="bold",
            color=LEARNED,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2},
        )

        ax_ttft.text(
            -0.11,
            1.08,
            "(a)",
            transform=ax_ttft.transAxes,
            fontsize=10,
            fontweight="bold",
            va="bottom",
        )
        ax_tail.text(
            -0.11,
            1.08,
            "(b)",
            transform=ax_tail.transAxes,
            fontsize=10,
            fontweight="bold",
            va="bottom",
        )
        fig.suptitle(
            "Shortest-first cuts mean TTFT near saturation but raises tail TPOT at highest load",
            fontsize=10,
            fontweight="bold",
            y=0.965,
        )
        legend = fig.legend(
            legend_handles,
            [handle.get_label() for handle in legend_handles],
            ncol=2,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.865),
            frameon=False,
            handlelength=2.2,
            columnspacing=1.8,
        )
        legend.set_gid("shared-policy-legend")
        fig.text(
            0.985,
            0.035,
            "single sweep · no repeated runs",
            ha="right",
            va="bottom",
            fontsize=7,
            color=TEXT,
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
    svg_path = output_dir / "fig3.svg"
    pdf_path = output_dir / "fig3.pdf"
    png_path = output_dir / "fig3.png"
    fig, _ = build_figure()
    with plt.rc_context(STYLE):
        fig.savefig(
            svg_path,
            metadata={"Creator": CREATOR, "Date": None},
        )
        _fix_svg_canvas(svg_path)
        fig.savefig(
            pdf_path,
            metadata={"Creator": CREATOR, "CreationDate": None, "ModDate": None},
        )
        fig.savefig(
            png_path,
            dpi=300,
            metadata={"Software": CREATOR},
        )
    plt.close(fig)
    return svg_path, pdf_path, png_path


def main() -> int:
    for path in render():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
