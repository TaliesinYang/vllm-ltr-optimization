import importlib.util
import unittest
from pathlib import Path

import numpy as np
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from matplotlib.text import Text


ROOT = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    if not path.exists():
        raise AssertionError(f"missing required module: {path.name}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class StyleContractTests(unittest.TestCase):
    def test_bootstrap_ci_is_deterministic_and_contains_mean(self):
        style = load_module("style", ROOT / "style.py")
        values = np.arange(1.0, 11.0)
        first = style.bootstrap_ci(values, np.mean)
        second = style.bootstrap_ci(values, np.mean)
        self.assertEqual(first, second)
        self.assertLess(first[0], values.mean())
        self.assertGreater(first[1], values.mean())

    def test_ieee_dimensions_and_policy_colors(self):
        style = load_module("style", ROOT / "style.py")
        self.assertEqual(style.IEEE_SINGLE_WIDTH, 3.5)
        self.assertEqual(style.IEEE_DOUBLE_WIDTH, 7.16)
        self.assertEqual(style.POLICY_COLOR["stock_fcfs"], style.OKABE_ITO["gray"])
        self.assertEqual(style.POLICY_COLOR["StockFCFSShim"], style.OKABE_ITO["dark_gray"])
        self.assertEqual(
            style.POLICY_COLOR["GatedHybrid"],
            style.POLICY_COLOR["GatedHybridScheduler"],
        )
        aliases = {
            "PureLTR": ("PureLTRScheduler", "pure_ltr"),
            "GatedHybrid": ("GatedHybridScheduler", "gated_hybrid"),
            "TailSafe": ("TailSafeScheduler", "tail_safe"),
            "LTRAging": ("LTRAgingScheduler", "ltr_aging"),
            "PromptLengthSJF": ("PromptLengthSJFScheduler",),
        }
        for canonical, policy_aliases in aliases.items():
            for policy_alias in policy_aliases:
                self.assertEqual(style.POLICY_COLOR[canonical], style.POLICY_COLOR[policy_alias])

    def test_all_required_rcparams_are_at_least_ten_points(self):
        style = load_module("style_fonts", ROOT / "style.py")
        required = (
            "font.size",
            "axes.labelsize",
            "axes.titlesize",
            "xtick.labelsize",
            "ytick.labelsize",
            "legend.fontsize",
        )
        for key in required:
            self.assertGreaterEqual(float(style.mpl.rcParams[key]), 10.0, key)


class DataContractTests(unittest.TestCase):
    def test_fig6_survival_band_resamples_pooled_requests_as_whole_curves(self):
        module = load_module("fig6_survival", ROOT / "fig6_mixed.py")
        values_ms = np.array([100.0, 200.0, 300.0, 400.0])
        grid_s = np.array([0.15, 0.20, 0.25, 0.30, 0.35])
        lower, upper = module.survival_band(values_ms, grid_s, n=17, seed=0)

        rng = np.random.default_rng(0)
        curves = np.empty((17, grid_s.size))
        for index in range(17):
            sample_s = values_ms[rng.integers(0, values_ms.size, size=values_ms.size)] / 1000.0
            curves[index] = np.mean(sample_s[:, None] > grid_s[None, :], axis=0)
        expected_lower, expected_upper = np.percentile(curves, [2.5, 97.5], axis=0)

        np.testing.assert_allclose(lower, expected_lower)
        np.testing.assert_allclose(upper, expected_upper)

    def test_fig4_ood_pooled_statistics(self):
        module = load_module("fig4_ood", ROOT / "fig4_ood.py")
        vectors = module.load_ood_vectors()
        expected = {
            "StockFCFSShim": (358, 3050.090380, 13993.824380),
            "PureLTR": (359, 2518.247203, 12845.755548),
            "GatedHybrid": (358, 2483.552831, 11513.000154),
            "TailSafe": (358, 2501.709972, 11523.023547),
        }
        for policy, (count, mean, p99) in expected.items():
            values = vectors[policy]
            self.assertEqual(values.size, count)
            self.assertAlmostEqual(float(values.mean()), mean, places=5)
            self.assertAlmostEqual(float(np.percentile(values, 99)), p99, places=5)

    def test_fig6_mixed_pooled_statistics(self):
        module = load_module("fig6_mixed", ROOT / "fig6_mixed.py")
        vectors = module.load_mixed_vectors()
        expected = {
            "stock_fcfs": (3349.488523, 17157.528497),
            "StockFCFSShim": (3377.536297, 17863.018120),
            "PureLTR": (2853.792466, 15692.816194),
            "GatedHybrid": (2851.741695, 14743.639633),
            "TailSafe": (2842.773929, 14689.892705),
            "LTRAging": (2842.421476, 14031.225506),
            "PromptLengthSJF": (2837.398644, 13764.291480),
        }
        for policy, (mean, p99) in expected.items():
            values = vectors[policy]
            self.assertEqual(values.size, 450)
            self.assertAlmostEqual(float(values.mean()), mean, places=5)
            self.assertAlmostEqual(float(np.percentile(values, 99)), p99, places=5)

    def test_fig7_live_tau_and_seed_clustered_sweep(self):
        module = load_module("fig7_gate", ROOT / "fig7_gate.py")
        tau = module.load_live_tau()
        self.assertAlmostEqual(tau["ungated"]["chat_pred"], 1.0)
        self.assertAlmostEqual(tau["ungated"]["tool_pred"], 1.0)
        self.assertAlmostEqual(tau["ungated"]["tool_true"], -1.0)
        self.assertAlmostEqual(tau["gated"]["chat_pred"], 1.0)
        self.assertAlmostEqual(tau["gated"]["tool_pred"], 1.0 / 15.0)
        self.assertAlmostEqual(tau["gated"]["tool_true"], -1.0 / 15.0)
        sweep = module.load_sweep_summary()
        self.assertEqual(sorted(sweep), ["gated_hybrid", "pure_ltr"])
        self.assertEqual([point["tool_ratio"] for point in sweep["pure_ltr"]], [0, 0.25, 0.5, 0.75, 1])
        self.assertAlmostEqual(sweep["pure_ltr"][2]["p99"], 2.582, places=3)
        self.assertAlmostEqual(sweep["gated_hybrid"][2]["mean_speedup"], 1.313, places=3)
        self.assertEqual(len(sweep["pure_ltr"][0]["p99_ci"]), 2)

    def test_fig8_pairing_and_matched_subset(self):
        module = load_module("fig8_overhead", ROOT / "fig8_overhead.py")
        pairs = module.load_overhead_pairs()
        self.assertEqual(pairs["all"]["ttft_delta"].size, 150)
        self.assertEqual(pairs["matched"]["ttlt_delta"].size, 118)
        self.assertEqual(pairs["dropped"], 32)
        self.assertAlmostEqual(float(pairs["all"]["ttft_delta"].mean()), 737.106670, places=5)
        self.assertAlmostEqual(float(pairs["matched"]["ttft_delta"].mean()), 711.444631, places=5)
        self.assertAlmostEqual(float(pairs["matched"]["ttlt_delta"].mean()), 839.504672, places=5)


class RenderContractTests(unittest.TestCase):
    def test_all_rendered_text_is_at_least_ten_points(self):
        cases = (
            ("fig4_fonts", "fig4_ood.py", lambda module: module.build_figure(module.load_ood_vectors())[0]),
            ("fig6_fonts", "fig6_mixed.py", lambda module: module.build_figure(module.load_mixed_vectors())[0]),
            (
                "fig7_fonts",
                "fig7_gate.py",
                lambda module: module.build_figure(module.load_live_records(), module.load_sweep_summary())[0],
            ),
            ("fig8_fonts", "fig8_overhead.py", lambda module: module.build_figure(module.load_overhead_pairs())[0]),
        )
        for name, filename, build in cases:
            module = load_module(name, ROOT / filename)
            figure = build(module)
            for text in figure.findobj(Text):
                self.assertGreaterEqual(text.get_fontsize(), 10.0, f"{filename}: {text.get_text()!r}")
            module.plt.close(figure)

    def test_png_and_pdf_outputs_exist_and_are_nonempty(self):
        stems = (
            "fig4_ood_single",
            "fig6_mixed_double",
            "fig7_gate_double",
            "fig8_overhead_double",
        )
        for stem in stems:
            for suffix in (".png", ".pdf"):
                path = ROOT / "out" / f"{stem}{suffix}"
                self.assertTrue(path.exists(), f"missing render: {path.name}")
                self.assertGreater(path.stat().st_size, 5_000, f"render too small: {path.name}")

    def test_fig6_bar_axis_contains_all_ci_whiskers(self):
        module = load_module("fig6_layout", ROOT / "fig6_mixed.py")
        figure, summaries = module.build_figure(module.load_mixed_vectors())
        upper_limit_ms = figure.axes[1].get_ylim()[1] * 1000.0
        highest_ci_ms = max(summary["p99_ci"][1] for summary in summaries.values())
        self.assertGreaterEqual(upper_limit_ms, highest_ci_ms * 1.10)
        module.plt.close(figure)

    def test_fig4_only_horizontal_p99_labels_above_ci_caps(self):
        module = load_module("fig4_layout", ROOT / "fig4_ood.py")
        figure, summaries = module.build_figure(module.load_ood_vectors())
        axis = figure.axes[0]
        value_labels = [text for text in axis.texts if text.get_text().replace(".", "").isdigit()]
        self.assertEqual(len(value_labels), len(module.POLICY_DIR))
        for text, policy in zip(value_labels, module.POLICY_DIR):
            self.assertEqual(text.get_rotation(), 0.0)
            self.assertGreater(text.get_position()[1] * 1000.0, summaries[policy]["p99_ci"][1])
        self.assertGreater(axis.get_ylim()[1], max(text.get_position()[1] for text in value_labels))
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        legend_box = axis.get_legend().get_window_extent(renderer)
        self.assertTrue(all(not legend_box.overlaps(text.get_window_extent(renderer)) for text in value_labels))
        module.plt.close(figure)

    def test_fig6_ccdf_bootstraps_all_curves_and_omits_unbounded_p99_dots(self):
        module = load_module("fig6_ccdf", ROOT / "fig6_mixed.py")
        figure, _ = module.build_figure(module.load_mixed_vectors())
        axis = figure.axes[0]
        bands = [collection for collection in axis.collections if isinstance(collection, PolyCollection)]
        self.assertEqual(len(bands), len(module.POLICY_DIR))
        self.assertTrue(all(collection.get_alpha() <= 0.10 for collection in bands))
        self.assertFalse(any(isinstance(collection, PathCollection) for collection in axis.collections))
        self.assertIn("Request-bootstrap 95% CI · all curves", {text.get_text() for text in axis.texts})
        module.plt.close(figure)

    def test_fig7_live_panel_draws_tau_ci_and_states_probe_scope(self):
        module = load_module("fig7_live", ROOT / "fig7_gate.py")
        figure, _ = module.build_figure(module.load_live_records(), module.load_sweep_summary())
        axis = figure.axes[0]
        self.assertTrue(any(isinstance(collection, LineCollection) for collection in axis.collections))
        labels = {text.get_text() for text in axis.texts}
        self.assertIn("Live opt-125m ordering probe · n=6/class · no latency", labels)
        self.assertIn("obeys\nwrong\nhint", labels)
        wide_ci_label = next(label for label in labels if label.startswith("τ≈0.07, n=6"))
        self.assertIn("(wide CI)", wide_ci_label)
        self.assertIn("predefined tool→fallback", wide_ci_label)
        self.assertNotIn("decoupled", labels)
        self.assertIn("chat: τ=1 both", labels)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        obeys = next(text for text in axis.texts if text.get_text() == "obeys\nwrong\nhint")
        self.assertTrue(
            all(not Text.get_window_extent(obeys, renderer).overlaps(bar.get_window_extent(renderer)) for bar in axis.patches)
        )
        title = next(text for text in axis.texts if text.get_text().startswith("Live opt-125m"))
        panel = next(text for text in axis.texts if text.get_text() == "(a)")
        self.assertFalse(title.get_window_extent(renderer).overlaps(panel.get_window_extent(renderer)))
        module.plt.close(figure)

    def test_fig7_sweep_uses_median_per_run_label_and_honest_scope(self):
        module = load_module("fig7_sweep_labels", ROOT / "fig7_gate.py")
        figure, _ = module.build_figure(module.load_live_records(), module.load_sweep_summary())
        axis = figure.axes[1]
        self.assertEqual(axis.get_ylabel(), "Median per-run p99 ratio / FCFS")
        legend_labels = {
            text.get_text()
            for legend in [*axis.artists, axis.get_legend()]
            if legend is not None and legend.__class__.__name__ == "Legend"
            for text in legend.get_texts()
        }
        self.assertIn("Median per-run p99 ratio / FCFS", legend_labels)
        labels = {text.get_text() for text in axis.texts}
        self.assertIn("Simulation · 10 seeds × 4 QPS", labels)
        self.assertIn("Gated < PureLTR at every tested level\nGated > FCFS through 75%", labels)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        legends = [
            legend
            for legend in [*axis.artists, axis.get_legend()]
            if legend is not None and legend.__class__.__name__ == "Legend"
        ]
        self.assertTrue(
            all(
                not left.get_window_extent(renderer).overlaps(right.get_window_extent(renderer))
                for index, left in enumerate(legends)
                for right in legends[index + 1 :]
            )
        )
        tick_labels = [
            text
            for sweep_axis in (axis, figure.axes[2])
            for text in sweep_axis.get_yticklabels()
            if text.get_visible()
        ]
        self.assertTrue(
            all(
                not legend.get_window_extent(renderer).overlaps(text.get_window_extent(renderer))
                for legend in legends
                for text in tick_labels
            )
        )
        module.plt.close(figure)

    def test_evidence_chain_uses_pooled_claims_and_honest_gate_scope(self):
        evidence_path = ROOT / "EVIDENCE_CHAIN.md"
        evidence = evidence_path.read_text()
        required = (
            "On this OOD workload",
            "17.4–18.6%",
            "8.2% for PureLTR",
            "17.7% for gated/tail",
            "8.5–18.2%",
            "Prompt SJF reaches 19.8%",
            "predefined workload-class fallback",
            "exceeds FCFS through 75%",
        )
        for phrase in required:
            self.assertIn(phrase, evidence)
        forbidden = ("18–23%", "detects unreliable predictions", "gated stays safe")
        for phrase in forbidden:
            self.assertNotIn(phrase, evidence)

    def test_fig7_sweep_marks_zero_as_all_accurate_and_legends_clear_peak(self):
        module = load_module("fig7_sweep", ROOT / "fig7_gate.py")
        sweep = module.load_sweep_summary()
        figure, _ = module.build_figure(module.load_live_records(), sweep)
        axis = figure.axes[1]
        figure.canvas.draw()
        self.assertIn("all accurate", axis.get_xticklabels()[0].get_text())
        peak = max(sweep["pure_ltr"], key=lambda point: point["p99"])
        peak_display = axis.transData.transform((peak["tool_ratio"] * 100, peak["p99"]))
        legends = [artist for artist in axis.artists if artist.__class__.__name__ == "Legend"]
        legends.append(axis.get_legend())
        renderer = figure.canvas.get_renderer()
        self.assertTrue(all(not legend.get_window_extent(renderer).contains(*peak_display) for legend in legends))
        module.plt.close(figure)

    def test_fig8_matched_ttlt_panel_uses_log_scale(self):
        module = load_module("fig8_layout", ROOT / "fig8_overhead.py")
        figure, _ = module.build_figure(module.load_overhead_pairs())
        self.assertEqual(figure.axes[1].get_yscale(), "log")
        module.plt.close(figure)

    def test_fig8_matched_annotations_do_not_overlap(self):
        module = load_module("fig8_annotations", ROOT / "fig8_overhead.py")
        figure, _ = module.build_figure(module.load_overhead_pairs())
        axis = figure.axes[1]
        figure.canvas.draw()
        note = next(text for text in axis.texts if text.get_text().startswith("matched"))
        delta = next(text for text in axis.texts if text.get_text().startswith("Δmean"))
        renderer = figure.canvas.get_renderer()
        self.assertFalse(note.get_window_extent(renderer).overlaps(delta.get_window_extent(renderer)))
        self.assertEqual(note.get_ha(), "left")
        self.assertEqual(delta.get_ha(), "right")
        self.assertLessEqual(note.get_position()[0], 0.05)
        self.assertGreaterEqual(delta.get_position()[0], 0.95)
        axes_top = axis.get_window_extent(renderer).y1
        self.assertGreaterEqual(note.get_window_extent(renderer).y0, axes_top)
        self.assertGreaterEqual(delta.get_window_extent(renderer).y0, axes_top)
        module.plt.close(figure)


if __name__ == "__main__":
    unittest.main()
