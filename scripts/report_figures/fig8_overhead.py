import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from style import IEEE_DOUBLE_WIDTH, OKABE_ITO, bootstrap_ci, save_figure, set_log_axis_plain


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "rental-20260719T231309Z" / "gateway-overhead.json"
OUTPUT_DIR = ROOT / "out"
ARM_COLOR = {"direct": OKABE_ITO["dark_gray"], "gateway": OKABE_ITO["sky_blue"]}


def _has_error(row: dict) -> bool:
    error = row.get("error")
    return bool(str(error).strip()) if error is not None else False


def _build_subset(direct: dict[str, dict], gateway: dict[str, dict], ids: list[str]) -> dict[str, np.ndarray]:
    result = {}
    for metric in ("ttft", "ttlt"):
        direct_values = np.array([float(direct[request_id][f"{metric}_ms"]) for request_id in ids])
        gateway_values = np.array([float(gateway[request_id][f"{metric}_ms"]) for request_id in ids])
        result[f"{metric}_direct"] = direct_values
        result[f"{metric}_gateway"] = gateway_values
        result[f"{metric}_delta"] = gateway_values - direct_values
    return result


def load_overhead_pairs(data_path: Path = DATA_PATH) -> dict:
    payload = json.loads(data_path.read_text())
    direct = {row["request_id"]: row for row in payload["direct"]["samples"] if not _has_error(row)}
    gateway = {row["request_id"]: row for row in payload["gateway"]["samples"] if not _has_error(row)}
    if len(direct) != len(payload["direct"]["samples"]) or len(gateway) != len(payload["gateway"]["samples"]):
        raise ValueError("overhead input contains errored rows")
    ids = sorted(direct.keys() & gateway.keys())
    if len(ids) != 150:
        raise ValueError(f"expected 150 paired requests, found {len(ids)}")
    matched_ids = [
        request_id
        for request_id in ids
        if direct[request_id]["output_tokens"] == gateway[request_id]["output_tokens"]
    ]
    return {
        "all": _build_subset(direct, gateway, ids),
        "matched": _build_subset(direct, gateway, matched_ids),
        "dropped": len(ids) - len(matched_ids),
    }


def _paired_panel(
    ax,
    subset: dict[str, np.ndarray],
    metric: str,
    panel: str,
    note: str,
    use_log_scale: bool = False,
):
    direct = subset[f"{metric}_direct"]
    gateway = subset[f"{metric}_gateway"]
    delta = subset[f"{metric}_delta"]
    positions = np.array([0.0, 1.0])

    violin = ax.violinplot([direct, gateway], positions=positions, widths=0.62, showextrema=False)
    for body, arm in zip(violin["bodies"], ("direct", "gateway")):
        body.set_facecolor(ARM_COLOR[arm])
        body.set_edgecolor(ARM_COLOR[arm])
        body.set_alpha(0.18)

    for left, right in zip(direct, gateway):
        ax.plot(positions, [left, right], color=OKABE_ITO["gray"], alpha=0.055, linewidth=0.45, zorder=1)
    rng = np.random.default_rng(1234)
    jitter = rng.uniform(-0.085, 0.085, size=direct.size)
    ax.scatter(jitter, direct, s=5, color=ARM_COLOR["direct"], alpha=0.32, linewidth=0, zorder=2)
    ax.scatter(1 + jitter, gateway, s=5, color=ARM_COLOR["gateway"], alpha=0.32, linewidth=0, zorder=2)

    means = np.array([direct.mean(), gateway.mean()])
    intervals = np.array([bootstrap_ci(direct, np.mean), bootstrap_ci(gateway, np.mean)])
    errors = np.vstack((means - intervals[:, 0], intervals[:, 1] - means))
    ax.errorbar(
        positions,
        means,
        yerr=errors,
        fmt="o",
        color=OKABE_ITO["black"],
        markerfacecolor="white",
        markersize=4.5,
        elinewidth=1.0,
        capsize=2.5,
        zorder=5,
    )

    delta_mean = float(delta.mean())
    delta_ci = bootstrap_ci(delta, np.mean)
    delta_x = 0.98 if use_log_scale else 0.5
    delta_y = 1.08 if use_log_scale else 0.86
    delta_ha = "right" if use_log_scale else "center"
    ax.text(
        delta_x,
        delta_y,
        f"Δmean {delta_mean:.0f} ms [{delta_ci[0]:.0f}, {delta_ci[1]:.0f}]",
        transform=ax.transAxes,
        ha=delta_ha,
        va="bottom",
        fontsize=10,
    )
    ax.set_xticks(positions, ["Direct", "Gateway"])
    scale_note = ", log scale" if use_log_scale else ""
    ax.set_ylabel(f"{metric.upper()} (ms{scale_note})")
    ax.set_xlim(-0.5, 1.5)
    if use_log_scale:
        ax.set_yscale("log")
        minimum = min(float(np.min(direct)), float(np.min(gateway)))
        maximum = max(float(np.max(direct)), float(np.max(gateway)))
        ax.set_ylim(max(minimum * 0.72, 1.0), maximum * 1.22)
        set_log_axis_plain(
            ax,
            "y",
            [1000, 2000, 3000, 5000, 10000, 20000, 30000],
            fmt=lambda value: f"{value:g}",
        )
    else:
        ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, zorder=0)
    note_y = 1.015 if use_log_scale else 0.97
    note_va = "bottom" if use_log_scale else "top"
    ax.text(0.02, note_y, note, transform=ax.transAxes, ha="left", va=note_va, fontsize=10)
    ax.text(-0.09, 1.03, panel, transform=ax.transAxes, fontweight="bold", fontsize=10)
    return {
        "direct_mean": float(direct.mean()),
        "direct_mean_ci": tuple(intervals[0]),
        "gateway_mean": float(gateway.mean()),
        "gateway_mean_ci": tuple(intervals[1]),
        "delta_mean": delta_mean,
        "delta_mean_ci": delta_ci,
    }


def build_figure(pairs: dict):
    fig, (ax_ttft, ax_ttlt) = plt.subplots(
        1,
        2,
        figsize=(IEEE_DOUBLE_WIDTH, 3.55),
        constrained_layout=True,
    )
    ttft = _paired_panel(ax_ttft, pairs["all"], "ttft", "(a)", "paired n=150")
    ttlt = _paired_panel(
        ax_ttlt,
        pairs["matched"],
        "ttlt",
        "(b)",
        f"matched n=118 · {pairs['dropped']} dropped",
        use_log_scale=True,
    )
    return fig, {"ttft": ttft, "ttlt": ttlt}


def main() -> None:
    pairs = load_overhead_pairs()
    fig, summaries = build_figure(pairs)
    paths = save_figure(fig, OUTPUT_DIR, "fig8_overhead_double")
    plt.close(fig)
    print("FILES " + " ".join(str(path.relative_to(ROOT)) for path in paths))
    ttft = summaries["ttft"]
    ttlt = summaries["ttlt"]
    print(
        f"TTFT n=150 direct_mean_ms={ttft['direct_mean']:.1f} CI95={ttft['direct_mean_ci'][0]:.1f},{ttft['direct_mean_ci'][1]:.1f} "
        f"gateway_mean_ms={ttft['gateway_mean']:.1f} CI95={ttft['gateway_mean_ci'][0]:.1f},{ttft['gateway_mean_ci'][1]:.1f} "
        f"delta_mean_ms={ttft['delta_mean']:.1f} CI95={ttft['delta_mean_ci'][0]:.1f},{ttft['delta_mean_ci'][1]:.1f}"
    )
    print(
        f"TTLT matched_n=118 dropped={pairs['dropped']} "
        f"direct_mean_ms={ttlt['direct_mean']:.1f} CI95={ttlt['direct_mean_ci'][0]:.1f},{ttlt['direct_mean_ci'][1]:.1f} "
        f"gateway_mean_ms={ttlt['gateway_mean']:.1f} CI95={ttlt['gateway_mean_ci'][0]:.1f},{ttlt['gateway_mean_ci'][1]:.1f} "
        f"delta_mean_ms={ttlt['delta_mean']:.1f} CI95={ttlt['delta_mean_ci'][0]:.1f},{ttlt['delta_mean_ci'][1]:.1f}"
    )


if __name__ == "__main__":
    main()
