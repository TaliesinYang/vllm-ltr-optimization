#!/usr/bin/env python3
"""Build publication-v3 Figures 7 and 8 without changing v2 statistics."""

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
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_FIGURES_DIR = REPO_ROOT / "scripts" / "report_figures"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPORT_FIGURES_DIR) not in sys.path:
    sys.path.insert(0, str(REPORT_FIGURES_DIR))

from scripts.report_figures import fig7_gate, fig8_overhead  # noqa: E402
from scripts.report_figures.style import bootstrap_ci  # noqa: E402


OUTPUT_DIR = REPO_ROOT / "latex_source" / "figures" / "publication-v3"
CREATOR = "publication_v3/figure_07_08.py"

WIDTH_MM = 181.9
FIG7_HEIGHT_MM = 86.0
FIG8_HEIGHT_MM = 80.0
MM_PER_INCH = 25.4
BOOTSTRAP_RESAMPLES = 2_000
RANDOM_SEED = 1_234

TEXT = "#1A1A1A"
LEARNED = "#0072B2"
NEUTRAL = "#4A4A4A"
WARNING = "#D55E00"
GRID = "#D9D9D9"
STRIP = "#F2F2F2"

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


def load_fig7_evidence() -> dict[str, object]:
    """Load the v2 live-probe and simulated-mixture evidence."""

    records = fig7_gate.load_live_records()
    return {
        "records": records,
        "tau": fig7_gate.live_tau_with_ci(records),
        "sweep": fig7_gate.load_sweep_summary(),
    }


def _delta_summary(deltas_ms: np.ndarray) -> dict[str, object]:
    values_s = np.asarray(deltas_ms, dtype=float) / 1_000.0
    return {
        "mean_s": float(np.mean(values_s)),
        "request_bootstrap_ci_s": bootstrap_ci(
            values_s,
            np.mean,
            n=BOOTSTRAP_RESAMPLES,
            seed=RANDOM_SEED,
        ),
    }


def load_fig8_evidence() -> dict[str, object]:
    """Load paired path deltas and preserve the v2 request bootstrap."""

    pairs = fig8_overhead.load_overhead_pairs()
    return {
        "pairs": pairs,
        "summaries": {
            "ttft": _delta_summary(pairs["all"]["ttft_delta"]),
            "ttlt": _delta_summary(pairs["matched"]["ttlt_delta"]),
        },
    }


def _add_scope_strip(
    fig: Figure,
    text: str,
    *,
    gid: str,
    fontsize: float,
) -> None:
    strip = Rectangle(
        (0.015, 0.018),
        0.970,
        0.080,
        transform=fig.transFigure,
        facecolor=STRIP,
        edgecolor="none",
        zorder=0,
    )
    strip.set_gid(gid)
    fig.add_artist(strip)
    fig.text(
        0.5,
        0.058,
        text,
        fontsize=fontsize,
        color=WARNING,
        ha="center",
        va="center",
        zorder=1,
    )


def _draw_live_probe(ax, tau: dict[str, object]) -> None:
    rows = (
        ("Chat ·\nPure LTR", "ungated", "chat_pred"),
        ("Chat ·\nGated policy bundle", "gated", "chat_pred"),
        ("Tool ·\nPure LTR", "ungated", "tool_pred"),
        ("Tool ·\nGated policy bundle", "gated", "tool_pred"),
    )
    y_positions = np.arange(len(rows))[::-1]
    for y, (_, mode, metric) in zip(y_positions, rows):
        estimate = tau[mode][metric]["value"]
        low, high = tau[mode][metric]["ci"]
        marker = "o" if mode == "ungated" else "s"
        facecolor = "white" if mode == "ungated" else LEARNED
        ax.errorbar(
            estimate,
            y,
            xerr=[[estimate - low], [high - estimate]],
            fmt=marker,
            color=LEARNED,
            markerfacecolor=facecolor,
            markeredgecolor=LEARNED,
            markeredgewidth=0.9,
            markersize=5.2,
            elinewidth=1.2,
            capsize=3.0,
            zorder=3,
        )
    ax.axvline(0.0, color=NEUTRAL, linewidth=0.8, linestyle=(0, (2, 2)), zorder=0)
    ax.set_yticks(y_positions, [label for label, _, _ in rows])
    ax.set_xlim(-1.08, 1.08)
    ax.set_xticks((-1.0, -0.5, 0.0, 0.5, 1.0))
    ax.set_ylim(-0.55, 3.55)
    ax.set_xlabel("Kendall τ-b vs first-token order")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    gated = tau["gated"]["tool_pred"]
    ax.text(
        -1.02,
        0.47,
        f"Tool gated: τ={gated['value']:.2f}\n95% CI [{gated['ci'][0]:.2f}, {gated['ci'][1]:.2f}]",
        fontsize=8,
        color=LEARNED,
        ha="left",
        va="center",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
    )


def _draw_simulation(ax, sweep: dict[str, object]) -> None:
    style_by_policy = {
        "pure_ltr": {
            "label": "Pure LTR",
            "linestyle": (0, (5, 2)),
            "marker": "o",
            "markerfacecolor": "white",
            "alpha": 0.07,
        },
        "gated_hybrid": {
            "label": "Gated policy bundle",
            "linestyle": "-",
            "marker": "s",
            "markerfacecolor": LEARNED,
            "alpha": 0.13,
        },
    }
    for policy in fig7_gate.POLICIES:
        points = sweep[policy]
        shares = np.array([point["tool_ratio"] for point in points]) * 100.0
        estimates = np.array([point["p99"] for point in points])
        intervals = np.array([point["p99_ci"] for point in points])
        style = style_by_policy[policy]
        ax.plot(
            shares,
            estimates,
            color=LEARNED,
            linestyle=style["linestyle"],
            marker=style["marker"],
            markerfacecolor=style["markerfacecolor"],
            markeredgecolor=LEARNED,
            markeredgewidth=0.9,
            linewidth=1.6,
            markersize=4.5,
            label=style["label"],
            zorder=3,
        )
        ax.fill_between(
            shares,
            intervals[:, 0],
            intervals[:, 1],
            color=LEARNED,
            alpha=style["alpha"],
            linewidth=0,
            zorder=1,
        )
    ax.axhline(1.0, color=NEUTRAL, linewidth=0.8, linestyle=(0, (2, 2)), zorder=0)
    ax.set_xlim(-3, 103)
    ax.set_ylim(0.45, 5.15)
    ax.set_xticks((0, 25, 50, 75, 100))
    ax.set_xlabel("Tool traffic share (%)")
    ax.set_ylabel("Median per-run p99 wait ratio / FCFS")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend = ax.legend(loc="upper right", frameon=False, handlelength=2.0)
    legend.set_gid("policy-legend")


def build_fig7() -> tuple[Figure, dict[str, object]]:
    """Return the live ordering probe beside the simulated policy comparison."""

    evidence = load_fig7_evidence()
    with plt.rc_context(STYLE):
        fig = plt.figure(
            figsize=(WIDTH_MM / MM_PER_INCH, FIG7_HEIGHT_MM / MM_PER_INCH)
        )
        live_ax = fig.add_axes((0.170, 0.225, 0.285, 0.480))
        simulation_ax = fig.add_axes((0.575, 0.225, 0.400, 0.480))
        live_ax.set_gid("panel-a-live-ordering")
        simulation_ax.set_gid("panel-b-simulation")
        _draw_live_probe(live_ax, evidence["tau"])
        _draw_simulation(simulation_ax, evidence["sweep"])

        fig.suptitle(
            "Small live probe is inconclusive; simulation favors the gated policy bundle",
            fontsize=10,
            fontweight="bold",
            y=0.965,
        )
        fig.text(0.065, 0.760, "(a)", fontsize=10, fontweight="bold", va="bottom")
        fig.text(
            0.305,
            0.760,
            "Live ordering probe · opt-125m · n=6/class",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="bottom",
        )
        fig.text(0.515, 0.760, "(b)", fontsize=10, fontweight="bold", va="bottom")
        fig.text(
            0.775,
            0.760,
            "Simulation · 10 seeds × 4 QPS per share",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="bottom",
        )
        _add_scope_strip(
            fig,
            "panel (a): live ordering only, no latency · panel (b): simulation · "
            "gated bundle changes aging/tail-safe policy; not an isolated gate ablation",
            gid="scope-strip-fig7",
            fontsize=8,
        )
    return fig, evidence


def _draw_delta_distribution(
    ax,
    deltas_ms: np.ndarray,
    summary: dict[str, object],
    *,
    metric: str,
) -> None:
    deltas_s = np.asarray(deltas_ms, dtype=float) / 1_000.0
    violin = ax.violinplot(
        [deltas_s],
        positions=[0.0],
        widths=0.72,
        showmeans=False,
        showmedians=True,
        showextrema=False,
        orientation="horizontal",
    )
    body = violin["bodies"][0]
    body.set_facecolor(LEARNED)
    body.set_edgecolor(LEARNED)
    body.set_alpha(0.18)
    violin["cmedians"].set_color(NEUTRAL)
    violin["cmedians"].set_linewidth(1.0)

    rng = np.random.default_rng(RANDOM_SEED)
    jitter = rng.uniform(-0.115, 0.115, size=deltas_s.size)
    ax.scatter(
        deltas_s,
        jitter,
        s=7,
        color=LEARNED,
        alpha=0.34,
        linewidth=0,
        zorder=2,
    )
    mean_s = summary["mean_s"]
    low, high = summary["request_bootstrap_ci_s"]
    ax.errorbar(
        mean_s,
        0.0,
        xerr=[[mean_s - low], [high - mean_s]],
        fmt="o",
        color="#000000",
        markerfacecolor="white",
        markeredgecolor="#000000",
        markeredgewidth=0.9,
        markersize=5.2,
        elinewidth=1.4,
        capsize=3.0,
        zorder=4,
    )
    ax.axvline(0.0, color=NEUTRAL, linewidth=0.8, linestyle=(0, (2, 2)), zorder=0)
    ax.set_yticks([])
    ax.set_ylim(-0.42, 0.42)
    ax.set_xlabel(f"{metric} paired delta: gateway − direct (s)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.98,
        0.93,
        f"Observed mean difference\n{mean_s:+.3f} s",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        fontweight="bold",
    )
    ax.text(
        0.98,
        0.10,
        f"Request-bootstrap 95% CI\n[{low:+.3f}, {high:+.3f}] s",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )


def build_fig8() -> tuple[Figure, dict[str, object]]:
    """Return paired delta distributions for one ordered replay."""

    evidence = load_fig8_evidence()
    pairs = evidence["pairs"]
    summaries = evidence["summaries"]
    with plt.rc_context(STYLE):
        fig = plt.figure(
            figsize=(WIDTH_MM / MM_PER_INCH, FIG8_HEIGHT_MM / MM_PER_INCH)
        )
        ttft_ax = fig.add_axes((0.070, 0.235, 0.420, 0.445))
        ttlt_ax = fig.add_axes((0.560, 0.235, 0.420, 0.445))
        ttft_ax.set_gid("panel-a-ttft-delta")
        ttlt_ax.set_gid("panel-b-ttlt-delta")
        _draw_delta_distribution(
            ttft_ax,
            pairs["all"]["ttft_delta"],
            summaries["ttft"],
            metric="TTFT",
        )
        _draw_delta_distribution(
            ttlt_ax,
            pairs["matched"]["ttlt_delta"],
            summaries["ttlt"],
            metric="TTLT",
        )
        ttft_ax.set_xlim(0.15, 1.30)
        ttft_ax.set_xticks((0.2, 0.4, 0.6, 0.8, 1.0, 1.2))
        ttlt_ax.set_xlim(-1.5, 11.0)
        ttlt_ax.set_xticks((-1, 0, 2, 4, 6, 8, 10))

        fig.suptitle(
            "One ordered replay shows observed gateway − direct latency differences",
            fontsize=10,
            fontweight="bold",
            y=0.965,
        )
        fig.text(0.030, 0.730, "(a)", fontsize=10, fontweight="bold", va="bottom")
        fig.text(
            0.280,
            0.730,
            "TTFT paired requests · n=150",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="bottom",
        )
        fig.text(0.520, 0.730, "(b)", fontsize=10, fontweight="bold", va="bottom")
        fig.text(
            0.770,
            0.730,
            f"TTLT matched outputs · n=118 · {pairs['dropped']} excluded",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="bottom",
        )
        key = fig.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color="#000000",
                    marker="o",
                    markerfacecolor="white",
                    markeredgecolor="#000000",
                    linewidth=1.4,
                    label="white point=mean; black bar=95% CI",
                )
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.855),
            frameon=False,
            handlelength=2.2,
        )
        key.set_gid("mean-ci-key")
        _add_scope_strip(
            fig,
            "one ordered replay · direct first, gateway second · request bootstrap conditional on replay · "
            "run-order and run-level uncertainty excluded",
            gid="scope-strip-fig8",
            fontsize=7,
        )
    return fig, evidence


def _fix_svg_canvas(svg_path: Path, height_mm: float) -> None:
    raw = svg_path.read_text(encoding="utf-8")
    raw = re.sub(r'width="[^"]+"', f'width="{WIDTH_MM:.1f}mm"', raw, count=1)
    raw = re.sub(r'height="[^"]+"', f'height="{height_mm:.1f}mm"', raw, count=1)
    svg_path.write_text(raw, encoding="utf-8")


def _render(
    builder,
    stem: str,
    height_mm: float,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{stem}.svg"
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig, _ = builder()
    with plt.rc_context(STYLE):
        fig.savefig(svg_path, metadata={"Creator": CREATOR, "Date": None})
        _fix_svg_canvas(svg_path, height_mm)
        fig.savefig(
            pdf_path,
            metadata={"Creator": CREATOR, "CreationDate": None, "ModDate": None},
        )
        fig.savefig(png_path, dpi=300, metadata={"Software": CREATOR})
    plt.close(fig)
    return svg_path, pdf_path, png_path


def render_fig7(output_dir: Path = OUTPUT_DIR) -> tuple[Path, Path, Path]:
    return _render(build_fig7, "fig7", FIG7_HEIGHT_MM, output_dir)


def render_fig8(output_dir: Path = OUTPUT_DIR) -> tuple[Path, Path, Path]:
    return _render(build_fig8, "fig8", FIG8_HEIGHT_MM, output_dir)


def main() -> int:
    for path in (*render_fig7(), *render_fig8()):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
