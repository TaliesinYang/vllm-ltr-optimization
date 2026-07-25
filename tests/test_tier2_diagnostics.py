from __future__ import annotations

import unittest

from ltr_training.tier2_diagnostics import classify_long_response


class Tier2DiagnosticsTests(unittest.TestCase):
    def test_detects_repetition_loop(self) -> None:
        repeated = " ".join(["call tool with the same arguments again"] * 100)

        diagnosis = classify_long_response(repeated)

        self.assertEqual(diagnosis["classification"], "repetition_loop")
        self.assertGreater(diagnosis["repeated_ngram_ratio"], 0.35)
        self.assertTrue(diagnosis["evidence_snippets"])

    def test_detects_genuine_varied_long_text(self) -> None:
        varied = " ".join(f"token-{index} explains item-{index * 7}" for index in range(200))

        diagnosis = classify_long_response(varied)

        self.assertEqual(diagnosis["classification"], "genuine_long")

    def test_aggregate_mixed_is_left_to_caller(self) -> None:
        partly_repeated = " ".join(
            [f"unique-{index}" for index in range(80)]
            + ["repeat this exact tool invocation"] * 20
        )

        diagnosis = classify_long_response(partly_repeated)

        self.assertIn(diagnosis["classification"], {"mixed", "repetition_loop"})


if __name__ == "__main__":
    unittest.main()
