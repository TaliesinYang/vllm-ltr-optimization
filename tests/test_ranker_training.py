import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from ltr_training.train_ranker import (
    TrainingExample,
    build_pair_indices,
    load_tier1_examples,
    pairwise_margin_loss,
    should_save_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]


class RankerDataTest(unittest.TestCase):
    def test_first_step_is_checkpointed_then_configured_interval_is_used(self) -> None:
        self.assertTrue(should_save_checkpoint(1, save_steps=10))
        self.assertFalse(should_save_checkpoint(2, save_steps=10))
        self.assertTrue(should_save_checkpoint(10, save_steps=10))

    def test_loader_uses_prompt_only_and_source_filter(self) -> None:
        rows = [
            {
                "sample_id": "a",
                "source": "toolace",
                "session_id": "s1",
                "prompt": "short prompt",
                "tool_schema": "must not enter prompt-only input",
                "history": [["user", "must not enter prompt-only input"]],
                "output_length": 10,
                "generator_id": "g1",
            },
            {
                "sample_id": "b",
                "source": "lmcache",
                "session_id": "s2",
                "prompt": "excluded source",
                "tool_schema": "",
                "history": [],
                "output_length": 100,
                "generator_id": "g2",
            },
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            examples = load_tier1_examples(path, sources={"toolace"})

        self.assertEqual(
            examples,
            [TrainingExample("a", "s1", "short prompt", 10, "g1")],
        )

    def test_pair_builder_is_seeded_filtered_and_generator_safe(self) -> None:
        examples = [
            TrainingExample("a", "s1", "one", 10, "g1"),
            TrainingExample("b", "s2", "two", 11, "g1"),
            TrainingExample("c", "s3", "three", 30, "g1"),
            TrainingExample("d", "s4", "four", 100, "g2"),
        ]

        pairs = build_pair_indices(examples, seed=42, relative_delta=0.2)

        self.assertTrue(pairs)
        self.assertEqual(pairs, build_pair_indices(examples, seed=42, relative_delta=0.2))
        for left, right in pairs:
            self.assertEqual(examples[left].generator_id, examples[right].generator_id)
            difference = abs(examples[left].output_length - examples[right].output_length)
            self.assertGreaterEqual(
                difference / max(examples[left].output_length, examples[right].output_length, 1),
                0.2,
            )

    def test_pairwise_margin_loss_rewards_correct_length_order(self) -> None:
        correct = pairwise_margin_loss(
            torch.tensor([3.0]),
            torch.tensor([1.0]),
            torch.tensor([30]),
            torch.tensor([10]),
            margin=1.0,
        )
        incorrect = pairwise_margin_loss(
            torch.tensor([1.0]),
            torch.tensor([3.0]),
            torch.tensor([30]),
            torch.tensor([10]),
            margin=1.0,
        )

        self.assertEqual(correct.item(), 0.0)
        self.assertGreater(incorrect.item(), correct.item())


class TrainRankerCliTest(unittest.TestCase):
    def test_help_requires_config_labels_and_output(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "train_bert_ranker.py"), "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--config", result.stdout)
        self.assertIn("--labels", result.stdout)
        self.assertIn("--output-dir", result.stdout)


if __name__ == "__main__":
    unittest.main()
