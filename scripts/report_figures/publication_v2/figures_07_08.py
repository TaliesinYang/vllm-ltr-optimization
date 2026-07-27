"""Build publication-v2 gate and gateway-path figures (Fig. 7-8)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


REPORT_FIGURES_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "latex_source" / "figures" / "publication-v2"
if str(REPORT_FIGURES_DIR) not in sys.path:
    sys.path.append(str(REPORT_FIGURES_DIR))

import fig7_gate  # noqa: E402
import fig8_overhead  # noqa: E402
from style import IEEE_DOUBLE_WIDTH, OKABE_ITO, POLICY_COLOR, bootstrap_ci  # noqa: E402


BOOTSTRAP_RESAMPLES = 2_000
RANDOM_SEED = 1_234


def _fig7_inputs() -> tuple[dict, dict]:
    records = fig7_gate.load_live_records()
    tau = fig7_gate.live_tau_with_ci(records)
    sweep = fig7_gate.load_sweep_summary()
    return tau, sweep


def _draw_live_probe(ax, tau: dict) -> None:
    rows = (
        ("Chat · ungated", "ungated", "chat_pred"),
        ("Chat · gated", "gated", "chat_pred"),
        ("Tool · ungated", "ungated", "tool_pred"),
        ("Tool · gated", "gated", "tool_pred"),
    )
    y = np.arange(len(rows))[::-1]
    estimates = np.array([tau[mode][metric]["value"] for _, mode, metric in rows])
    intervals = np.array([tau[mode][metric]["ci"] for _, mode, metric in rows])
    colors = [
        POLICY_COLOR["pure_ltr"] if mode == "ungated" else POLICY_COLOR["gated_hybrid"]
        for _, mode, _ in rows
    ]

    for position, estimate, interval, color in zip(y, estimates, intervals, colors):
        ax.errorbar(
            estimate,
            position,
            xerr=[[estimate - interval[0]], [interval[1] - estimate]],
            fmt="o",
            color=color,
            markeredgecolor=OKABE_ITO["black"],
            markeredgewidth=0.55,
            markersize=5.5,
            elinewidth=1.15,
            capsize=3.0,
            zorder=3,
        )

    ax.axvline(0.0, color=OKABE_ITO["gray"], linewidth=0.8, linestyle=":", zorder=0)
    ax.set_yticks(y, [label for label, _, _ in rows])
    ax.set_xlim(-1.08, 1.08)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xlabel("Kendall tau vs first-token order")
    ax.set_title("Live ordering probe\nopt-125m; n=6/class; bootstrap 95% CI", loc="left")
    ax.xaxis.grid(True, zorder=0)
    ax.annotate(
        "Tool gated: tau=0.07\nCI [-0.78, 1.00]",
        xy=(estimates[-1], y[-1]),
        xytext=(-1.02, 0.63),
        textcoords="data",
        ha="left",
        va="bottom",
        fontsize=10,
        arrowprops={"arrowstyle": "-", "color": POLICY_COLOR["gated_hybrid"], "linewidth": 0.8},
    )
    ax.text(-0.18, 1.08, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=10)


def _draw_simulation(ax, sweep: dict) -> None:
    for policy in fig7_gate.POLICIES:
        points = sweep[policy]
        shares = np.array([point["tool_ratio"] for point in points]) * 100.0
        estimates = np.array([point["p99"] for point in points])
        intervals = np.array([point["p99_ci"] for point in points])
        label = "Pure LTR" if policy == "pure_ltr" else "Gated policy bundle"
        color = POLICY_COLOR[policy]
        ax.plot(shares, estimates, color=color, marker="o", label=label, zorder=3)
        ax.fill_between(
            shares,
            intervals[:, 0],
            intervals[:, 1],
            color=color,
            alpha=0.10,
            linewidth=0,
            zorder=1,
        )

    ax.axhline(1.0, color=OKABE_ITO["gray"], linewidth=0.8, linestyle=":", zorder=0)
    ax.set_xlim(-3, 103)
    ax.set_ylim(0.45, 5.15)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Tool traffic share (%)")
    ax.set_ylabel("Median per-run p99 wait ratio / FCFS")
    ax.set_title("Simulated workload mixture (10 seeds x 4 QPS)", loc="left")
    ax.yaxis.grid(True, zorder=0)
    ax.legend(loc="upper right", handlelength=1.5)
    ax.text(-0.14, 1.08, "(b)", transform=ax.transAxes, fontweight="bold", fontsize=10)


def build_fig7() -> Figure:
    """Build Fig. 7 from live ordering and simulated mixture evidence."""

    tau, sweep = _fig7_inputs()
    figure, (live_ax, simulation_ax) = plt.subplots(
        1,
        2,
        figsize=(IEEE_DOUBLE_WIDTH, 3.85),
        gridspec_kw={"width_ratios": [0.88, 1.35]},
    )
    _draw_live_probe(live_ax, tau)
    _draw_simulation(simulation_ax, sweep)
    figure.subplots_adjust(left=0.145, right=0.985, top=0.86, bottom=0.29, wspace=0.34)
    figure.text(
        0.5,
        0.04,
        "Panel (b) is simulation. Gated bundle also changes aging/tail-safe policy;\n"
        "it is not an isolated gate ablation.",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    return figure


def _draw_delta_distribution(
    ax,
    deltas_ms: np.ndarray,
    *,
    metric: str,
    panel: str,
    sample_note: str,
) -> dict[str, float | tuple[float, float]]:
    deltas_s = np.asarray(deltas_ms, dtype=float) / 1_000.0
    mean_s = float(np.mean(deltas_s))
    interval_s = bootstrap_ci(
        deltas_s,
        np.mean,
        n=BOOTSTRAP_RESAMPLES,
        seed=RANDOM_SEED,
    )
    violin = ax.violinplot(
        [deltas_s],
        positions=[0.0],
        widths=0.72,
        showmeans=False,
        showmedians=True,
        showextrema=False,
        vert=False,
    )
    body = violin["bodies"][0]
    body.set_facecolor(OKABE_ITO["sky_blue"])
    body.set_edgecolor(OKABE_ITO["blue"])
    body.set_alpha(0.26)
    violin["cmedians"].set_color(OKABE_ITO["black"])
    violin["cmedians"].set_linewidth(1.0)

    rng = np.random.default_rng(RANDOM_SEED)
    jitter = rng.uniform(-0.115, 0.115, size=deltas_s.size)
    ax.scatter(
        deltas_s,
        jitter,
        s=7,
        color=OKABE_ITO["blue"],
        alpha=0.34,
        linewidth=0,
        zorder=2,
    )
    ax.errorbar(
        mean_s,
        0.0,
        xerr=[[mean_s - interval_s[0]], [interval_s[1] - mean_s]],
        fmt="o",
        color=OKABE_ITO["black"],
        markerfacecolor="white",
        markersize=5.0,
        elinewidth=1.4,
        capsize=3.0,
        zorder=4,
    )
    ax.axvline(0.0, color=OKABE_ITO["gray"], linewidth=0.8, linestyle=":", zorder=0)
    ax.set_yticks([])
    ax.set_ylim(-0.45, 0.45)
    ax.set_xlabel(f"{metric} paired delta: gateway - direct (s)")
    ax.set_title(sample_note, loc="left")
    ax.xaxis.grid(True, zorder=0)
    ax.text(
        0.98,
        0.94,
        f"Observed mean difference\n{mean_s:+.3f} s",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )
    ax.text(
        0.98,
        0.08,
        f"Request-bootstrap 95% CI\n[{interval_s[0]:+.3f}, {interval_s[1]:+.3f}] s",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
    )
    ax.text(-0.11, 1.10, panel, transform=ax.transAxes, fontweight="bold", fontsize=10)
    return {"mean_s": mean_s, "request_bootstrap_ci_s": interval_s}


def build_fig8() -> Figure:
    """Build Fig. 8 as paired request-level delta distributions."""

    pairs = fig8_overhead.load_overhead_pairs()
    figure, (ttft_ax, ttlt_ax) = plt.subplots(
        1,
        2,
        figsize=(IEEE_DOUBLE_WIDTH, 3.35),
    )
    _draw_delta_distribution(
        ttft_ax,
        pairs["all"]["ttft_delta"],
        metric="TTFT",
        panel="(a)",
        sample_note="TTFT paired requests (n=150)",
    )
    _draw_delta_distribution(
        ttlt_ax,
        pairs["matched"]["ttlt_delta"],
        metric="TTLT",
        panel="(b)",
        sample_note=f"TTLT matched outputs (n=118; {pairs['dropped']} excluded)",
    )
    figure.subplots_adjust(left=0.06, right=0.985, top=0.82, bottom=0.34, wspace=0.24)
    figure.text(
        0.5,
        0.035,
        "One ordered replay: direct first, gateway second.\n"
        "Request bootstrap is conditional on this replay;\n"
        "run-order and run-level uncertainty are excluded.",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    return figure


def _save(figure: Figure, stem: str) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    png_path = OUTPUT_DIR / f"{stem}.png"
    figure.savefig(pdf_path)
    figure.savefig(png_path, dpi=300)
    return pdf_path, png_path


def main() -> None:
    for stem, builder in (("fig7", build_fig7), ("fig8", build_fig8)):
        figure = builder()
        paths = _save(figure, stem)
        plt.close(figure)
        print("FILES " + " ".join(str(path.relative_to(REPO_ROOT)) for path in paths))


if __name__ == "__main__":
    main()
