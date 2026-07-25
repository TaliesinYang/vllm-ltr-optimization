from __future__ import annotations

import unittest

from ltr_training.tier2_sampling import build_stratified_splits


class Tier2SamplingTests(unittest.TestCase):
    def test_exact_deterministic_session_boundary_splits(self) -> None:
        rows = []
        for session_index in range(80):
            size = 1 if session_index < 60 else 2
            for invocation_index in range(size):
                rows.append(
                    {
                        "sample_id": f"s{session_index}:{invocation_index}",
                        "session_id": f"s{session_index}",
                        "invocation_index": invocation_index,
                        "prompt": "x" * (10 + session_index * 5),
                    }
                )

        sampled, manifest = build_stratified_splits(
            rows,
            seed=42,
            split_counts={"train": 50, "validation": 20, "test": 10},
            stopping_criterion=(
                "2K→4K 若 val tau 提升 ≤0.01 且下游 utility 在 bootstrap CI 内, 判定 4K 饱和"
            ),
        )
        sampled_again, manifest_again = build_stratified_splits(
            rows,
            seed=42,
            split_counts={"train": 50, "validation": 20, "test": 10},
            stopping_criterion=manifest["stopping_criterion"],
        )

        self.assertEqual(sampled, sampled_again)
        self.assertEqual(manifest, manifest_again)
        self.assertEqual(manifest["split_counts"], {"train": 50, "validation": 20, "test": 10})
        self.assertEqual(manifest["sample_count"], 80)
        self.assertEqual(manifest["sampling_seed"], 42)
        self.assertEqual(
            manifest["stopping_criterion"],
            "2K→4K 若 val tau 提升 ≤0.01 且下游 utility 在 bootstrap CI 内, 判定 4K 饱和",
        )

        session_splits: dict[str, set[str]] = {}
        for row in sampled:
            session_splits.setdefault(row["session_id"], set()).add(row["tier2_split"])
        self.assertTrue(all(len(splits) == 1 for splits in session_splits.values()))
        self.assertEqual(sum(manifest["length_bucket_counts"].values()), 80)


if __name__ == "__main__":
    unittest.main()
