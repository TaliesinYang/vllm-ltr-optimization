import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SourceManifestTest(unittest.TestCase):
    def test_training_sources_are_pinned_to_canonical_repositories(self) -> None:
        manifest = json.loads((ROOT / "configs" / "training_sources.json").read_text())

        self.assertEqual(manifest["toolace"]["repository"], "Team-ACE/ToolACE")
        self.assertEqual(
            manifest["toolace"]["revision"],
            "6bda777c88d21e5a204703c1ee45597a8fa4f734",
        )
        self.assertEqual(
            manifest["lmcache"]["repository"],
            "sammshen/lmcache-agentic-traces",
        )
        self.assertEqual(
            manifest["lmcache"]["revision"],
            "6e043b9e89865df3aec19fd5679286b683bfd70e",
        )
        self.assertNotIn("DiscoPosse", json.dumps(manifest))

    def test_prompt_only_seed_42_run_is_explicit(self) -> None:
        run = json.loads((ROOT / "configs" / "bert_prompt_only_seed42.json").read_text())

        self.assertEqual(run["variant"], "prompt_only")
        self.assertEqual(run["seed"], 42)
        self.assertEqual(run["backbone"], "google-bert/bert-base-uncased")
        self.assertEqual(run["label_tier"], 1)

    def test_ood_sources_are_pinned_in_mainline(self) -> None:
        declarations = json.loads(
            (ROOT / "configs" / "source-declarations.json").read_text()
        )

        self.assertEqual(
            declarations["bfcl"],
            {
                "repository": "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
                "revision": "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda",
            },
        )
        self.assertEqual(
            declarations["toolathlon"],
            {
                "repository": "hkust-nlp/Toolathlon-Trajectories",
                "revision": "6194034105bc27fa438447172be0e7b4e35396e4",
            },
        )


if __name__ == "__main__":
    unittest.main()
