import json
import os
import unittest
import urllib.parse
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory

from ltr_training.data import iter_lmcache_invocations, iter_toolace_invocations
from ltr_training.tier1 import (
    extract_lmcache_label,
    extract_toolace_label,
    iter_lmcache_labels,
    iter_toolace_labels,
    write_labels_jsonl,
)


TOOLACE_REVISION = "6bda777c88d21e5a204703c1ee45597a8fa4f734"
LMCACHE_REVISION = "6e043b9e89865df3aec19fd5679286b683bfd70e"
TOOLACE_GENERATOR = "toolace-synthetic-generator-unspecified"
DEFAULT_TOOLACE_SNAPSHOT = Path(
    "/Users/alex/.cache/vllm-ltr-optimization/datasets/toolace"
) / TOOLACE_REVISION / "data.json"


class ToolAceProductionSnapshotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.snapshot = Path(os.environ.get("TOOLACE_SNAPSHOT", DEFAULT_TOOLACE_SNAPSHOT))
        if not cls.snapshot.exists():
            raise RuntimeError(f"real ToolACE snapshot missing: {cls.snapshot}")

    def test_loader_yields_invocation_level_records_from_real_snapshot(self) -> None:
        first = next(
            iter_toolace_invocations(
                self.snapshot,
                revision=TOOLACE_REVISION,
                generator_id=TOOLACE_GENERATOR,
            )
        )

        self.assertEqual(first.source, "toolace")
        self.assertEqual(first.source_revision, TOOLACE_REVISION)
        self.assertEqual(first.session_id, "toolace-000000")
        self.assertEqual(first.invocation_index, 0)
        self.assertEqual(first.generator_id, TOOLACE_GENERATOR)
        self.assertIn("top market trends", first.prompt)
        self.assertEqual(
            first.completion,
            '[Market Trends API(trend_type="MARKET_INDEXES", country="us")]',
        )
        self.assertIn("newAddress", first.tool_schema)
        self.assertEqual(first.history, ())

    def test_loader_streams_every_recorded_assistant_invocation(self) -> None:
        count = sum(
            1
            for _ in iter_toolace_invocations(
                self.snapshot,
                revision=TOOLACE_REVISION,
                generator_id=TOOLACE_GENERATOR,
            )
        )

        self.assertEqual(count, 13_819)

    def test_tier1_label_retokenizes_recorded_completion_and_keeps_generator(self) -> None:
        invocation = next(
            iter_toolace_invocations(
                self.snapshot,
                revision=TOOLACE_REVISION,
                generator_id=TOOLACE_GENERATOR,
            )
        )

        label = extract_toolace_label(
            invocation,
            count_tokens=lambda text: len(text.encode("utf-8")),
            tokenizer_id="byte-counter-for-test",
        )

        self.assertEqual(label.output_length, len(invocation.completion.encode("utf-8")))
        self.assertEqual(label.generator_id, TOOLACE_GENERATOR)
        self.assertEqual(label.length_kind, "retokenized_recorded_completion")
        self.assertEqual(label.tokenizer_id, "byte-counter-for-test")

    def test_label_writer_persists_training_fields_as_jsonl(self) -> None:
        invocation = next(
            iter_toolace_invocations(
                self.snapshot,
                revision=TOOLACE_REVISION,
                generator_id=TOOLACE_GENERATOR,
            )
        )
        label = extract_toolace_label(
            invocation,
            count_tokens=lambda text: len(text.encode("utf-8")),
            tokenizer_id="byte-counter-for-test",
        )

        with TemporaryDirectory() as directory:
            output = Path(directory) / "labels.jsonl"
            written = write_labels_jsonl([label], output)
            row = json.loads(output.read_text())

        self.assertEqual(written, 1)
        self.assertEqual(row["sample_id"], "toolace-000000:0000")
        self.assertEqual(row["output_length"], label.output_length)
        self.assertEqual(row["generator_id"], TOOLACE_GENERATOR)
        self.assertIsInstance(row["history"], list)

    def test_toolace_label_pipeline_honors_limit(self) -> None:
        labels = list(
            iter_toolace_labels(
                self.snapshot,
                revision=TOOLACE_REVISION,
                generator_id=TOOLACE_GENERATOR,
                count_tokens=lambda text: len(text.encode("utf-8")),
                tokenizer_id="byte-counter-for-test",
                limit=2,
            )
        )

        self.assertEqual(len(labels), 2)
        self.assertEqual(labels[1].sample_id, "toolace-000000:0001")

    def test_empty_recorded_completion_is_preserved_as_zero_tokens(self) -> None:
        invocation = next(
            item
            for item in iter_toolace_invocations(
                self.snapshot,
                revision=TOOLACE_REVISION,
                generator_id=TOOLACE_GENERATOR,
            )
            if item.sample_id == "toolace-000027:0000"
        )

        label = extract_toolace_label(
            invocation,
            count_tokens=lambda text: len(text.encode("utf-8")),
            tokenizer_id="byte-counter-for-test",
        )

        self.assertEqual(invocation.completion, "")
        self.assertEqual(label.output_length, 0)


class LmCacheCanonicalSnapshotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        query = urllib.parse.urlencode(
            {
                "dataset": "sammshen/lmcache-agentic-traces",
                "config": "default",
                "split": "train",
            }
        )
        with urllib.request.urlopen(
            f"https://datasets-server.huggingface.co/first-rows?{query}", timeout=30
        ) as response:
            cls.rows = [item["row"] for item in json.load(response)["rows"]]

    def test_tier1_label_preserves_recorded_length_and_generator(self) -> None:
        invocation = next(
            iter_lmcache_invocations(self.rows, revision=LMCACHE_REVISION)
        )
        label = extract_lmcache_label(invocation)

        self.assertEqual(label.source, "lmcache")
        self.assertEqual(label.source_revision, LMCACHE_REVISION)
        self.assertEqual(label.generator_id, "minimax-m2.5")
        self.assertGreater(label.output_length, 0)
        self.assertEqual(label.length_kind, "recorded_output_tokens")
        self.assertIsNone(label.tokenizer_id)
        self.assertTrue(label.prompt)

    def test_lmcache_label_pipeline_honors_limit(self) -> None:
        labels = list(
            iter_lmcache_labels(self.rows, revision=LMCACHE_REVISION, limit=2)
        )

        self.assertEqual(len(labels), 2)
        self.assertEqual(labels[0].generator_id, "minimax-m2.5")


if __name__ == "__main__":
    unittest.main()
