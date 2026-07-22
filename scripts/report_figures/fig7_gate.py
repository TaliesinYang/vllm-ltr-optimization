import csv
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.lines import Line2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from style import (
    IEEE_DOUBLE_WIDTH,
    OKABE_ITO,
    POLICY_COLOR,
    POLICY_LABEL,
    save_figure,
)


ROOT = Path(__file__).resolve().parent
LIVE_DIR = ROOT / "data" / "gateway_policy_probe" / "results_clite"
SWEEP_PATH = ROOT / "data" / "gateway_policy_probe" / "results_hybrid10" / "hybrid_matrix.csv"
OUTPUT_DIR = ROOT / "out"
MODES = ("ungated", "gated")
POLICIES = ("pure_ltr", "gated_hybrid")


def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    left = np.asarray(x, dtype=float)
    right = np.asarray(y, dtype=float)
    if left.shape != right.shape or left.ndim != 1 or left.size < 2:
        raise ValueError("kendall_tau_b requires equal 1D vectors with at least 2 values")
    concordant = discordant = tied_left = tied_right = 0
    for first in range(left.size - 1):
        for second in range(first + 1, left.size):
            delta_left = np.sign(left[second] - left[first])
            delta_right = np.sign(right[second] - right[first])
            if delta_left == 0 and delta_right == 0:
                continue
            if delta_left == 0:
                tied_left += 1
            elif delta_right == 0:
                tied_right += 1
            elif delta_left == delta_right:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + tied_left)
        * (concordant + discordant + tied_right)
    )
    return (concordant - discordant) / denominator if denominator else float("nan")


def bootstrap_tau_ci(
    x: np.ndarray,
    y: np.ndarray,
    n: int = 2000,
    seed: int = 1234,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n):
        indices = rng.integers(0, len(x), size=len(x))
        estimate = kendall_tau_b(x[indices], y[indices])
        if np.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        raise ValueError("all Kendall bootstrap resamples were degenerate")
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return float(lower), float(upper)


def load_live_records(live_dir: Path = LIVE_DIR) -> dict[str, list[dict]]:
    records = {}
    for mode in MODES:
        path = live_dir / f"gated_run_{mode}.jsonl"
        records[mode] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return records


def load_live_tau(live_dir: Path = LIVE_DIR) -> dict[str, dict[str, float]]:
    result = {}
    for mode, rows in load_live_records(live_dir).items():
        result[mode] = {}
        for cls in ("chat", "tool"):
            subset = [row for row in rows if row["cls"] == cls]
            order = np.array([row["first_token_index"] for row in subset], dtype=float)
            predictions = np.array([row["pred_est"] for row in subset], dtype=float)
            truth = np.array([row["true_tokens"] for row in subset], dtype=float)
            result[mode][f"{cls}_pred"] = kendall_tau_b(predictions, order)
            result[mode][f"{cls}_true"] = kendall_tau_b(truth, order)
    return result


def seed_clustered_ci(
    rows: list[dict],
    field: str,
    n: int = 2000,
    seed: int = 1234,
) -> tuple[float, float]:
    by_seed = {}
    for row in rows:
        by_seed.setdefault(row["seed"], []).append(float(row[field]))
    seeds = sorted(by_seed)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n, dtype=float)
    for index in range(n):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        sample = [value for sampled_seed in sampled_seeds for value in by_seed[sampled_seed]]
        estimates[index] = np.median(sample)
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return float(lower), float(upper)


def load_sweep_summary(sweep_path: Path = SWEEP_PATH) -> dict[str, list[dict]]:
    with sweep_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {policy: [] for policy in POLICIES}
    for policy in POLICIES:
        policy_rows = [row for row in rows if row["policy"] == policy]
        for ratio in sorted({float(row["tool_ratio"]) for row in policy_rows}):
            cell = [row for row in policy_rows if float(row["tool_ratio"]) == ratio]
            p99_values = np.array([float(row["p99_ratio_vs_fcfs"]) for row in cell])
            mean_values = np.array([float(row["mean_speedup_vs_fcfs"]) for row in cell])
            result[policy].append(
                {
                    "tool_ratio": ratio,
                    "p99": float(np.median(p99_values)),
                    "p99_ci": seed_clustered_ci(cell, "p99_ratio_vs_fcfs"),
                    "mean_speedup": float(np.median(mean_values)),
                    "mean_speedup_ci": seed_clustered_ci(cell, "mean_speedup_vs_fcfs"),
                }
            )
    return result


def live_tau_with_ci(records: dict[str, list[dict]]):
    result = {}
    for mode, rows in records.items():
        result[mode] = {}
        for cls in ("chat", "tool"):
            subset = [row for row in rows if row["cls"] == cls]
            order = np.array([row["first_token_index"] for row in subset], dtype=float)
            for value_name, field in (("pred", "pred_est"), ("true", "true_tokens")):
                values = np.array([row[field] for row in subset], dtype=float)
                result[mode][f"{cls}_{value_name}"] = {
                    "value": kendall_tau_b(values, order),
                    "ci": bootstrap_tau_ci(values, order),
                }
    return result


def build_figure(records: dict[str, list[dict]], sweep: dict[str, list[dict]]):
    tau = live_tau_with_ci(records)
    fig, (ax_live, ax_sweep) = plt.subplots(
        1,
        2,
        figsize=(IEEE_DOUBLE_WIDTH, 4.15),
        gridspec_kw={"width_ratios": [0.82, 1.45]},
        constrained_layout=True,
    )

    classes = ("chat", "tool")
    x = np.arange(len(classes))
    width = 0.32
    mode_color = {"ungated": POLICY_COLOR["pure_ltr"], "gated": POLICY_COLOR["gated_hybrid"]}
    for mode_index, mode in enumerate(MODES):
        offset = (-0.5 if mode_index == 0 else 0.5) * width
        estimates = np.array([tau[mode][f"{cls}_pred"]["value"] for cls in classes])
        intervals = np.array([tau[mode][f"{cls}_pred"]["ci"] for cls in classes])
        errors = np.vstack((estimates - intervals[:, 0], intervals[:, 1] - estimates))
        ax_live.bar(
            x + offset,
            estimates,
            width=width,
            color=mode_color[mode],
            alpha=0.88,
            label=mode.capitalize(),
            zorder=2,
        )
        ax_live.errorbar(
            x + offset,
            estimates,
            yerr=errors,
            fmt="none",
            ecolor=OKABE_ITO["black"],
            elinewidth=0.7,
            capsize=2.0,
            capthick=0.7,
            zorder=4,
        )
        tool_true = tau[mode]["tool_true"]
        true_offset = -0.055 if mode == "ungated" else 0.055
        tool_x = x[1] + offset + true_offset
        tool_true_error = np.array(
            [
                [tool_true["value"] - tool_true["ci"][0]],
                [tool_true["ci"][1] - tool_true["value"]],
            ]
        )
        ax_live.errorbar(
            [tool_x],
            [tool_true["value"]],
            yerr=tool_true_error,
            fmt="none",
            ecolor=OKABE_ITO["black"],
            elinewidth=0.7,
            capsize=2.0,
            capthick=0.7,
            zorder=4,
        )
        ax_live.scatter(
            [tool_x],
            [tool_true["value"]],
            marker="D",
            facecolor="white",
            edgecolor=OKABE_ITO["black"],
            linewidth=0.75,
            s=18,
            zorder=5,
        )

    ungated_tool_x = x[1] - width / 2
    gated_tool_x = x[1] + width / 2
    ax_live.annotate(
        "obeys\nwrong\nhint",
        xy=(ungated_tool_x, tau["ungated"]["tool_pred"]["value"]),
        xytext=(0.45, -0.30),
        ha="center",
        va="center",
        fontsize=10,
        color=mode_color["ungated"],
        arrowprops={"arrowstyle": "-", "color": mode_color["ungated"], "linewidth": 0.65},
    )
    ax_live.annotate(
        "τ≈0.07, n=6\n(wide CI)\npredefined tool→fallback",
        xy=(gated_tool_x, tau["gated"]["tool_pred"]["value"]),
        xytext=(1.20, 0.48),
        ha="center",
        va="center",
        fontsize=10,
        color=mode_color["gated"],
        arrowprops={"arrowstyle": "-", "color": mode_color["gated"], "linewidth": 0.65},
    )
    ax_live.text(0, 1.055, "chat: τ=1 both", ha="center", va="bottom", fontsize=10)
    ax_live.text(
        0.5,
        1.03,
        "Live opt-125m ordering probe · n=6/class · no latency",
        transform=ax_live.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
    )

    ax_live.axhline(0, color=OKABE_ITO["light_gray"], linewidth=0.8, zorder=0)
    ax_live.set_xticks(x, ["Chat", "Tool"])
    ax_live.set_ylabel("Kendall τ vs service order")
    ax_live.set_ylim(-1.18, 1.18)
    ax_live.yaxis.grid(True, zorder=0)
    handles, labels = ax_live.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="D", linestyle="none", markerfacecolor="white", markeredgecolor="black", markersize=4, label="True tokens (tool)"))
    labels.append("True tokens (tool)")
    ax_live.legend(handles, labels, loc="lower left", handlelength=1.4)
    ax_live.text(-0.16, 1.10, "(a)", transform=ax_live.transAxes, fontweight="bold", fontsize=10)

    ax_speed = ax_sweep.twinx()
    ax_speed.spines["right"].set_visible(True)
    for policy in POLICIES:
        points = sweep[policy]
        ratios = np.array([point["tool_ratio"] for point in points]) * 100
        color = POLICY_COLOR[policy]
        p99 = np.array([point["p99"] for point in points])
        p99_ci = np.array([point["p99_ci"] for point in points])
        speed = np.array([point["mean_speedup"] for point in points])
        speed_ci = np.array([point["mean_speedup_ci"] for point in points])
        ax_sweep.plot(ratios, p99, color=color, marker="o", label=POLICY_LABEL[policy], zorder=4)
        ax_sweep.fill_between(ratios, p99_ci[:, 0], p99_ci[:, 1], color=color, alpha=0.11, linewidth=0)
        ax_speed.plot(ratios, speed, color=color, marker="s", linestyle="--", zorder=3)
        ax_speed.fill_between(ratios, speed_ci[:, 0], speed_ci[:, 1], color=color, alpha=0.07, linewidth=0)

    ax_sweep.axhline(1.0, color=OKABE_ITO["gray"], linewidth=0.75, linestyle=":", zorder=0)
    ax_speed.axhline(1.0, color=OKABE_ITO["gray"], linewidth=0.75, linestyle="--", zorder=0)
    ax_sweep.set_xlabel("Wrong-hint requests (%)")
    ax_sweep.set_ylabel("Median per-run p99 ratio / FCFS")
    ax_speed.set_ylabel("Mean speedup / FCFS (×)")
    ax_sweep.set_xticks(
        [0, 25, 50, 75, 100],
        ["0\n(all accurate)", "25", "50", "75", "100"],
    )
    ax_sweep.set_xlim(-3, 103)
    ax_sweep.set_ylim(0.45, 5.15)
    ax_speed.set_ylim(0.75, 2.25)
    ax_sweep.yaxis.grid(True, zorder=0)
    policy_handles = [Line2D([], [], color=POLICY_COLOR[policy], label=POLICY_LABEL[policy]) for policy in POLICIES]
    metric_handles = [
        Line2D([], [], color=OKABE_ITO["black"], marker="o", label="Median per-run p99 ratio / FCFS"),
        Line2D([], [], color=OKABE_ITO["black"], marker="s", linestyle="--", label="Mean speedup"),
    ]
    first_legend = ax_sweep.legend(handles=policy_handles, loc="center right", handlelength=1.4)
    ax_sweep.add_artist(first_legend)
    ax_sweep.legend(handles=metric_handles, loc="upper center", handlelength=1.4)
    ax_sweep.text(
        0.5,
        1.03,
        "Simulation · 10 seeds × 4 QPS",
        transform=ax_sweep.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
    )
    ax_sweep.text(
        0.03,
        0.035,
        "Gated < PureLTR at every tested level\nGated > FCFS through 75%",
        transform=ax_sweep.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
    )
    ax_sweep.text(-0.11, 1.03, "(b)", transform=ax_sweep.transAxes, fontweight="bold", fontsize=10)
    return fig, tau


def main() -> None:
    records = load_live_records()
    sweep = load_sweep_summary()
    fig, tau = build_figure(records, sweep)
    paths = save_figure(fig, OUTPUT_DIR, "fig7_gate_double")
    plt.close(fig)
    print("FILES " + " ".join(str(path.relative_to(ROOT)) for path in paths))
    for mode in MODES:
        print(
            f"{mode} chat_pred_tau={tau[mode]['chat_pred']['value']:.4f} CI95={tau[mode]['chat_pred']['ci'][0]:.4f},{tau[mode]['chat_pred']['ci'][1]:.4f} "
            f"tool_pred_tau={tau[mode]['tool_pred']['value']:.4f} CI95={tau[mode]['tool_pred']['ci'][0]:.4f},{tau[mode]['tool_pred']['ci'][1]:.4f} "
            f"tool_true_tau={tau[mode]['tool_true']['value']:.4f} CI95={tau[mode]['tool_true']['ci'][0]:.4f},{tau[mode]['tool_true']['ci'][1]:.4f}"
        )
    for policy in POLICIES:
        point = next(item for item in sweep[policy] if item["tool_ratio"] == 0.5)
        print(
            f"{policy} wrong_hint_pct=50 p99_ratio={point['p99']:.3f} CI95={point['p99_ci'][0]:.3f},{point['p99_ci'][1]:.3f} "
            f"mean_speedup={point['mean_speedup']:.3f} CI95={point['mean_speedup_ci'][0]:.3f},{point['mean_speedup_ci'][1]:.3f}"
        )


if __name__ == "__main__":
    main()
