import csv
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.patches import Patch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from style import (
    IEEE_DOUBLE_WIDTH,
    OKABE_ITO,
    POLICY_COLOR,
    POLICY_LABEL,
    bootstrap_ci,
    save_figure,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "rental-20260719T231309Z" / "matrix"
OUTPUT_DIR = ROOT / "out"
POLICY_DIR = {
    "stock_fcfs": "stock_fcfs.runs",
    "StockFCFSShim": "StockFCFSShim.runs",
    "PureLTR": "PureLTRScheduler.runs",
    "GatedHybrid": "GatedHybridScheduler.runs",
    "TailSafe": "TailSafeScheduler.runs",
    "LTRAging": "LTRAgingScheduler.runs",
    "PromptLengthSJF": "PromptLengthSJFScheduler.runs",
}
LINESTYLE = {
    "stock_fcfs": (0, (5, 2)),
    "StockFCFSShim": (0, (2, 1)),
    "PureLTR": "-",
    "GatedHybrid": "-",
    "TailSafe": "-",
    "LTRAging": "-",
    "PromptLengthSJF": "-",
}
def load_mixed_vectors(data_dir: Path = DATA_DIR) -> dict[str, np.ndarray]:
    vectors = {}
    for policy, directory in POLICY_DIR.items():
        files = sorted((data_dir / directory).glob("*.samples.csv"))
        if len(files) != 3:
            raise ValueError(f"expected 3 repeats for {policy}, found {len(files)}")
        values = []
        for path in files:
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    if (row.get("error") or "").strip():
                        continue
                    values.append(float(row["ttlt_ms"]))
        vectors[policy] = np.asarray(values, dtype=float)
    return vectors


def summarize(values: np.ndarray) -> dict[str, float | tuple[float, float]]:
    return {
        "mean": float(np.mean(values)),
        "mean_ci": bootstrap_ci(values, np.mean),
        "p99": float(np.percentile(values, 99)),
        "p99_ci": bootstrap_ci(values, lambda sample: np.percentile(sample, 99)),
    }


def survival_band(
    values_ms: np.ndarray,
    grid_s: np.ndarray,
    n: int = 2000,
    seed: int = 1234,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values_ms.size, size=(n, values_ms.size))
    curves = np.empty((n, grid_s.size), dtype=float)
    chunk_size = 128
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        samples_s = values_ms[indices[start:stop]] / 1000.0
        curves[start:stop] = np.mean(
            samples_s[:, :, None] > grid_s[None, None, :],
            axis=1,
        )
    lower, upper = np.percentile(curves, [2.5, 97.5], axis=0)
    return lower, upper


def build_figure(vectors: dict[str, np.ndarray]):
    policies = list(POLICY_DIR)
    summaries = {policy: summarize(vectors[policy]) for policy in policies}
    fig, (ax_ccdf, ax_bar) = plt.subplots(
        1,
        2,
        figsize=(IEEE_DOUBLE_WIDTH, 3.55),
        gridspec_kw={"width_ratios": [1.12, 1.0]},
        constrained_layout=True,
    )

    maximum = max(float(np.percentile(values, 99.7)) for values in vectors.values()) / 1000.0
    minimum = min(float(np.min(values)) for values in vectors.values()) / 1000.0
    grid = np.geomspace(max(minimum, 0.05), maximum, 64)
    for index, policy in enumerate(policies):
        values = np.sort(vectors[policy]) / 1000.0
        survival = (values.size - np.arange(values.size)) / values.size
        color = POLICY_COLOR[policy]
        linewidth = 1.65 if policy == "GatedHybrid" else 1.25
        ax_ccdf.step(
            values,
            survival,
            where="post",
            color=color,
            linestyle=LINESTYLE[policy],
            linewidth=linewidth,
            label=POLICY_LABEL[policy],
            zorder=3,
        )
        lower, upper = survival_band(vectors[policy], grid, seed=1234 + index)
        ax_ccdf.fill_between(
            grid,
            np.maximum(lower, 1e-4),
            np.maximum(upper, 1e-4),
            color=color,
            alpha=0.06,
            linewidth=0,
            zorder=1,
        )

    ax_ccdf.set_yscale("log")
    ax_ccdf.set_ylim(0.002, 1.05)
    ax_ccdf.set_xlim(left=0)
    ax_ccdf.set_xlabel("TTLT (s)")
    ax_ccdf.set_ylabel("CCDF, Pr(TTLT > t)")
    ax_ccdf.yaxis.grid(True, which="major")
    ax_ccdf.legend(ncol=2, loc="upper right", columnspacing=0.8, handlelength=2.0)
    ax_ccdf.text(
        0.02,
        0.03,
        "Request-bootstrap 95% CI · all curves",
        transform=ax_ccdf.transAxes,
        fontsize=10,
    )
    ax_ccdf.text(-0.11, 1.03, "(a)", transform=ax_ccdf.transAxes, fontweight="bold", fontsize=10)

    x = np.arange(len(policies))
    width = 0.34
    for metric, offset, alpha, hatch in (
        ("mean", -width / 2, 0.58, ""),
        ("p99", width / 2, 1.0, "///"),
    ):
        estimates = np.array([summaries[policy][metric] for policy in policies]) / 1000.0
        intervals = np.array([summaries[policy][f"{metric}_ci"] for policy in policies]) / 1000.0
        errors = np.vstack((estimates - intervals[:, 0], intervals[:, 1] - estimates))
        ax_bar.bar(
            x + offset,
            estimates,
            width=width,
            color=[POLICY_COLOR[policy] for policy in policies],
            alpha=alpha,
            hatch=hatch,
            edgecolor="white" if not hatch else OKABE_ITO["black"],
            zorder=2,
        )
        ax_bar.errorbar(
            x + offset,
            estimates,
            yerr=errors,
            fmt="none",
            ecolor="#222222",
            elinewidth=0.65,
            capsize=1.5,
            capthick=0.65,
            zorder=4,
        )

    ax_bar.axvline(1.5, color=OKABE_ITO["light_gray"], linewidth=0.8, zorder=0)
    ax_bar.set_xticks(x, [POLICY_LABEL[policy] for policy in policies], rotation=28, ha="right")
    ax_bar.set_ylabel("TTLT (s)")
    highest_ci_s = max(summary["p99_ci"][1] for summary in summaries.values()) / 1000.0
    ax_bar.set_ylim(0, highest_ci_s * 1.12)
    ax_bar.yaxis.grid(True, zorder=0)
    ax_bar.legend(
        handles=[
            Patch(facecolor=OKABE_ITO["gray"], alpha=0.58, label="Mean"),
            Patch(facecolor=OKABE_ITO["gray"], hatch="///", edgecolor=OKABE_ITO["black"], label="Pooled p99"),
        ],
        loc="upper right",
    )
    ax_bar.text(-0.11, 1.03, "(b)", transform=ax_bar.transAxes, fontweight="bold", fontsize=10)
    return fig, summaries


def main() -> None:
    vectors = load_mixed_vectors()
    fig, summaries = build_figure(vectors)
    paths = save_figure(fig, OUTPUT_DIR, "fig6_mixed_double")
    plt.close(fig)
    print("FILES " + " ".join(str(path.relative_to(ROOT)) for path in paths))
    for policy in ("stock_fcfs", "PureLTR", "GatedHybrid", "TailSafe", "PromptLengthSJF"):
        summary = summaries[policy]
        print(
            f"{policy} n={vectors[policy].size} "
            f"mean_ms={summary['mean']:.1f} CI95={summary['mean_ci'][0]:.1f},{summary['mean_ci'][1]:.1f} "
            f"p99_ms={summary['p99']:.1f} CI95={summary['p99_ci'][0]:.1f},{summary['p99_ci'][1]:.1f}"
        )


if __name__ == "__main__":
    main()
