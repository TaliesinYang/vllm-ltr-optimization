from pathlib import Path
from typing import Callable

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure


IEEE_SINGLE_WIDTH = 3.5
IEEE_DOUBLE_WIDTH = 7.16

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "sky_blue": "#56B4E9",
    "purple": "#CC79A7",
    "gray": "#999999",
    "dark_gray": "#4D4D4D",
    "light_gray": "#D9D9D9",
    "black": "#111111",
}

POLICY_COLOR = {
    "stock_fcfs": OKABE_ITO["gray"],
    "StockFCFSShim": OKABE_ITO["dark_gray"],
    "PureLTR": OKABE_ITO["blue"],
    "PureLTRScheduler": OKABE_ITO["blue"],
    "pure_ltr": OKABE_ITO["blue"],
    "GatedHybrid": OKABE_ITO["vermillion"],
    "GatedHybridScheduler": OKABE_ITO["vermillion"],
    "gated_hybrid": OKABE_ITO["vermillion"],
    "TailSafe": OKABE_ITO["green"],
    "TailSafeScheduler": OKABE_ITO["green"],
    "tail_safe": OKABE_ITO["green"],
    "LTRAging": OKABE_ITO["purple"],
    "LTRAgingScheduler": OKABE_ITO["purple"],
    "ltr_aging": OKABE_ITO["purple"],
    "PromptLengthSJF": OKABE_ITO["orange"],
    "PromptLengthSJFScheduler": OKABE_ITO["orange"],
}

POLICY_LABEL = {
    "stock_fcfs": "Stock FCFS",
    "StockFCFSShim": "FCFS shim",
    "PureLTR": "Pure LTR",
    "PureLTRScheduler": "Pure LTR",
    "pure_ltr": "Pure LTR",
    "GatedHybrid": "Gated hybrid",
    "GatedHybridScheduler": "Gated hybrid",
    "gated_hybrid": "Gated hybrid",
    "TailSafe": "Tail safe",
    "TailSafeScheduler": "Tail safe",
    "tail_safe": "Tail safe",
    "LTRAging": "LTR aging",
    "LTRAgingScheduler": "LTR aging",
    "ltr_aging": "LTR aging",
    "PromptLengthSJF": "Prompt SJF",
    "PromptLengthSJFScheduler": "Prompt SJF",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10.0,
        "axes.labelsize": 10.0,
        "axes.titlesize": 10.0,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.fontsize": 10.0,
        "legend.frameon": False,
        "lines.linewidth": 1.35,
        "lines.markersize": 4.0,
        "patch.linewidth": 0.65,
        "grid.color": OKABE_ITO["light_gray"],
        "grid.linewidth": 0.5,
        "grid.alpha": 0.65,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.025,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# Percentiles use pooled request rows, never per-run-averaged percentile summaries.
def bootstrap_ci(
    vec: np.ndarray,
    stat: Callable[[np.ndarray], float],
    n: int = 2000,
    seed: int = 1234,
) -> tuple[float, float]:
    values = np.asarray(vec, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bootstrap_ci requires a non-empty 1D vector")
    if not np.all(np.isfinite(values)):
        raise ValueError("bootstrap_ci requires finite values")
    if n < 2000:
        raise ValueError("bootstrap_ci requires at least 2000 resamples")

    rng = np.random.default_rng(seed)
    estimates = np.empty(n, dtype=float)
    for index in range(n):
        sample = rng.choice(values, size=values.size, replace=True)
        estimates[index] = float(stat(sample))
    if not np.all(np.isfinite(estimates)):
        raise ValueError("bootstrap statistic returned a non-finite value")
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return float(lower), float(upper)


def save_figure(fig: Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    return [png_path, pdf_path]
