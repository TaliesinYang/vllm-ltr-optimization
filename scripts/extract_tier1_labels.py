#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ltr_training.tier1 import (  # noqa: E402
    iter_lmcache_labels,
    iter_toolace_labels,
    write_labels_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Tier-1 observational labels without mixing generators."
    )
    parser.add_argument("--source", choices=("toolace", "lmcache"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--toolace-snapshot", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cache-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads((ROOT / "configs" / "training_sources.json").read_text())

    if args.source == "toolace":
        if args.toolace_snapshot is None:
            raise SystemExit("--toolace-snapshot is required for source=toolace")
        from transformers import AutoTokenizer

        tokenizer_source = manifest["toolace_label_tokenizer"]
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source["repository"],
            revision=tokenizer_source["revision"],
            cache_dir=args.cache_dir,
        )
        labels = iter_toolace_labels(
            args.toolace_snapshot,
            revision=manifest["toolace"]["revision"],
            generator_id=manifest["toolace"]["generator_id"],
            count_tokens=lambda text: len(
                tokenizer.encode(text, add_special_tokens=False)
            ),
            tokenizer_id=(
                f'{tokenizer_source["repository"]}'
                f'@{tokenizer_source["revision"]}'
            ),
            limit=args.limit,
        )
    else:
        from datasets import load_dataset

        source = manifest["lmcache"]
        rows = load_dataset(
            source["repository"],
            revision=source["revision"],
            split="train",
            streaming=True,
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
        )
        labels = iter_lmcache_labels(
            rows,
            revision=source["revision"],
            limit=args.limit,
        )

    written = write_labels_jsonl(labels, args.output)
    print(f"source={args.source} labels={written} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
