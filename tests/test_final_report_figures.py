from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import plot_final_report_figures as report_figures


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_parse_baseline_summary_reads_measured_rows(tmp_path: Path) -> None:
    source = tmp_path / "RESULTS-summary.txt"
    source.write_text(
        "method rate TTFT_ms TPOT_ms p99_TPOT\n"
        "FCFS 2 50.5 24.40 28.13\n"
        "LTR 2 103.2 30.44 44.18\n"
        "FCFS 16 17274.2 115.06 145.19\n"
        "LTR 16 6043.7 159.26 332.94\n",
        encoding="utf-8",
    )

    parsed = report_figures.parse_baseline_summary(source)

    assert parsed["FCFS"]["rate"] == [2.0, 16.0]
    assert parsed["LTR"]["ttft_ms"] == [103.2, 6043.7]
    assert parsed["FCFS"]["p99_tpot_ms"] == [28.13, 145.19]


def test_tier2_aggregation_uses_test_tau_and_three_seed_range(
    tmp_path: Path,
) -> None:
    summary = _write_json(
        tmp_path / "tier2.json",
        {
            "runs": [
                {"run_name": "bert-prompt_only-tier2-seed17", "test_tau": 0.5},
                {"run_name": "bert-prompt_only-tier2-seed42", "test_tau": 0.6},
                {"run_name": "bert-prompt_only-tier2-seed73", "test_tau": 0.7},
                {"run_name": "bert-prompt_schema-tier2-seed17", "test_tau": 0.7},
                {"run_name": "bert-prompt_schema-tier2-seed42", "test_tau": 0.8},
                {"run_name": "bert-prompt_schema-tier2-seed73", "test_tau": 0.9},
                {"run_name": "bert-full_context-tier2-seed17", "test_tau": 0.4},
                {"run_name": "bert-full_context-tier2-seed42", "test_tau": 0.5},
                {"run_name": "bert-full_context-tier2-seed73", "test_tau": 0.6},
                {
                    "run_name": "lightgbm-structural-tier2-seed42",
                    "test_tau": 0.3,
                },
            ]
        },
    )

    groups = report_figures.aggregate_predictor_groups(summary, metric="test_tau")

    prompt = next(group for group in groups if group.key == "prompt_only")
    assert prompt.values == pytest.approx((0.5, 0.6, 0.7))
    assert prompt.mean == pytest.approx(0.6)
    assert prompt.low == pytest.approx(0.5)
    assert prompt.high == pytest.approx(0.7)
    lightgbm = next(group for group in groups if group.key == "lightgbm")
    assert lightgbm.values == pytest.approx((0.3,))


def test_pending_figure_prints_and_does_not_create_pdf(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "fig4.pdf"

    generated = report_figures.draw_fig4(
        output, sources=(tmp_path / "missing-scores.jsonl",)
    )

    assert generated is False
    assert not output.exists()
    assert "PENDING fig4" in capsys.readouterr().out


def test_cli_accepts_fig_and_all_modes() -> None:
    selected = report_figures.parse_args(["--fig", "5"])
    all_figures = report_figures.parse_args(["--all"])

    assert selected.fig == 5
    assert selected.all is False
    assert all_figures.fig is None
    assert all_figures.all is True


def test_publication_style_never_drops_below_ten_points() -> None:
    point_keys = {
        "font.size",
        "axes.labelsize",
        "axes.titlesize",
        "xtick.labelsize",
        "ytick.labelsize",
        "legend.fontsize",
    }

    assert point_keys <= report_figures.PUBLICATION_STYLE.keys()
    assert all(report_figures.PUBLICATION_STYLE[key] >= 10 for key in point_keys)


def test_layout_contract_reserves_space_for_labels() -> None:
    assert report_figures.FIGURE_SIZES[1][0] >= 7.1
    assert report_figures.FIGURE_SIZES[2][0] >= 7.1
    assert report_figures.PREDICTOR_BAR_ORIENTATION == "horizontal"


def test_architecture_figures_use_two_column_ieee_float() -> None:
    root = Path(__file__).resolve().parents[1]
    main = (root / "latex_source/main.tex").read_text(encoding="utf-8")
    background = (root / "latex_source/sections/background.tex").read_text(
        encoding="utf-8"
    )

    assert r"\newcommand{\ReportWideFigure}" in main
    assert background.count(r"\ReportWideFigure{") == 2


def test_incomplete_seed_count_is_part_of_axis_category() -> None:
    group = report_figures.PredictorAggregate(
        key="prompt_only",
        label="BERT\nprompt only",
        values=(0.60, 0.61),
    )

    labeled = report_figures.with_seed_count(group, expected=3)

    assert labeled.label == "BERT\nprompt only\n(2 seeds)"


def test_fig1_control_annotations_have_separate_lanes() -> None:
    lanes = report_figures.FIG1_CONTROL_LANES

    assert max(lanes["decision_request"], lanes["decision_response"]) < lanes[
        "sse_response"
    ]


def test_seed_count_label_uses_singular_grammar() -> None:
    group = report_figures.PredictorAggregate(
        key="lightgbm",
        label="LightGBM\nstructural",
        values=(0.468,),
    )

    labeled = report_figures.with_seed_count(group, expected=3)

    assert labeled.label.endswith("(1 seed)")
