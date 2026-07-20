import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from style import (
    IEEE_SINGLE_WIDTH,
    POLICY_COLOR,
    POLICY_LABEL,
    bootstrap_ci,
    save_figure,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "rental-20260719T231309Z" / "matrix-ood"
OUTPUT_DIR = ROOT / "out"
POLICY_DIR = {
    "StockFCFSShim": "StockFCFSShim.runs",
    "PureLTR": "PureLTRScheduler.runs",
    "GatedHybrid": "GatedHybridScheduler.runs",
    "TailSafe": "TailSafeScheduler.runs",
}


def load_ood_vectors(data_dir: Path = DATA_DIR) -> dict[str, np.ndarray]:
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


def build_figure(vectors: dict[str, np.ndarray]):
    policies = list(POLICY_DIR)
    summaries = {policy: summarize(vectors[policy]) for policy in policies}
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_WIDTH, 3.15), constrained_layout=True)

    centers = np.arange(2)
    width = 0.19
    offsets = (np.arange(len(policies)) - 1.5) * width
    p99_label_positions = []
    for offset, policy in zip(offsets, policies):
        summary = summaries[policy]
        estimates = np.array([summary["mean"], summary["p99"]]) / 1000.0
        intervals = np.array([summary["mean_ci"], summary["p99_ci"]]) / 1000.0
        errors = np.vstack((estimates - intervals[:, 0], intervals[:, 1] - estimates))
        ax.bar(
            centers + offset,
            estimates,
            width=width,
            color=POLICY_COLOR[policy],
            edgecolor="white",
            label=POLICY_LABEL[policy],
            zorder=2,
        )
        ax.errorbar(
            centers + offset,
            estimates,
            yerr=errors,
            fmt="none",
            ecolor="#222222",
            elinewidth=0.7,
            capsize=1.8,
            capthick=0.7,
            zorder=3,
        )
        p99_label_y = intervals[1, 1] + 0.16
        p99_label_positions.append(p99_label_y)
        ax.text(
            centers[1] + offset,
            p99_label_y,
            f"{estimates[1]:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#222222",
            zorder=5,
        )

    ax.set_xticks(centers, ["Mean", "Pooled p99"])
    ax.set_ylabel("TTLT (s)")
    ax.set_ylim(0, max(p99_label_positions) + 0.45)
    ax.yaxis.grid(True, zorder=0)
    ax.legend(
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        columnspacing=0.8,
        handlelength=1.2,
    )
    return fig, summaries


def main() -> None:
    vectors = load_ood_vectors()
    fig, summaries = build_figure(vectors)
    paths = save_figure(fig, OUTPUT_DIR, "fig4_ood_single")
    plt.close(fig)
    print("FILES " + " ".join(str(path.relative_to(ROOT)) for path in paths))
    for policy in POLICY_DIR:
        summary = summaries[policy]
        print(
            f"{policy} n={vectors[policy].size} "
            f"mean_ms={summary['mean']:.1f} CI95={summary['mean_ci'][0]:.1f},{summary['mean_ci'][1]:.1f} "
            f"p99_ms={summary['p99']:.1f} CI95={summary['p99_ci'][0]:.1f},{summary['p99_ci'][1]:.1f}"
        )


if __name__ == "__main__":
    main()
