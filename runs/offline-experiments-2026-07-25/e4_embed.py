"""E4 step 1 - precompute frozen two-tower embeddings.

Two towers over the same frozen encoder (the fine-tuned prompt_schema seed-17
tier-2 checkpoint, so the encoder stays in-family with the single-tower
baseline):

  prompt tower  [USER]\\n{prompt}                    per-request, 512 cap
  schema tower  [TOOLS]\\n{tool_schema}              precomputed per schema body

The schema tower is computed twice, which is the whole point of the experiment:

  trunc512  first 512 tokens only - reproduces the single-tower truncation
  full      512-token windows over the entire schema, mean-pooled - no truncation

Both variants share one encoder and one pooling rule, so the difference between
them isolates truncation with nothing else moving.

Schema embeddings are cached by schema-body hash. That cache IS the deployment
artifact whose latency E4 measures: in serving, the schema tower runs once per
fingerprint, not once per request.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

import common

HERE = Path(__file__).resolve().parent
ENCODER = Path(
    "/Volumes/T7 Shield/vllm-ltr-results/extracted/tier2-matrix"
    "/bert-prompt_schema-tier2-seed17/final"
)
OUT = HERE / "e4-embeddings.pt"
META_OUT = HERE / "e4-embeddings-meta.json"
MAX_LENGTH = 512
BATCH = 32
MAX_CHUNKS = 8  # cap so a pathological schema cannot dominate the run


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * expanded).sum(dim=1)
    counts = expanded.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class Encoder:
    def __init__(self, checkpoint: Path) -> None:
        torch.set_num_threads(max(1, torch.get_num_threads()))
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        self.model = AutoModel.from_pretrained(checkpoint, local_files_only=True)
        self.model.eval()

    @torch.inference_mode()
    def encode_truncated(self, texts: list[str]) -> torch.Tensor:
        out = []
        for start in range(0, len(texts), BATCH):
            batch = texts[start : start + BATCH]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            hidden = self.model(**inputs).last_hidden_state
            out.append(mean_pool(hidden, inputs["attention_mask"]))
        return torch.cat(out) if out else torch.empty(0)

    @torch.inference_mode()
    def encode_windowed(self, texts: list[str]) -> tuple[torch.Tensor, list[int]]:
        """Encode each text as up to MAX_CHUNKS windows of 512 tokens, mean-pooled."""
        # transformers 5.x dropped build_inputs_with_special_tokens; wrap the
        # window in CLS/SEP by hand, which is what BERT's builder did.
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        body = MAX_LENGTH - 2
        vectors: list[torch.Tensor] = []
        chunk_counts: list[int] = []
        for text in texts:
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            windows = [ids[i : i + body] for i in range(0, len(ids), body)] or [[]]
            windows = windows[:MAX_CHUNKS]
            chunk_counts.append(len(windows))
            built = [[cls_id, *window, sep_id] for window in windows]
            width = max(len(item) for item in built)
            pad = self.tokenizer.pad_token_id or 0
            input_ids = torch.tensor(
                [item + [pad] * (width - len(item)) for item in built]
            )
            attention = torch.tensor(
                [[1] * len(item) + [0] * (width - len(item)) for item in built]
            )
            hidden = self.model(
                input_ids=input_ids, attention_mask=attention
            ).last_hidden_state
            pooled = mean_pool(hidden, attention)
            vectors.append(pooled.mean(dim=0))
        return torch.stack(vectors), chunk_counts


def main() -> None:
    started = time.time()
    inputs = common.verify_inputs()
    splits, counts = common.load_splits()
    encoder = Encoder(ENCODER)

    # Unique schema bodies - this is the per-deployment precompute set.
    schema_by_hash: dict[str, str] = {}
    row_schema_hash: dict[str, list[str]] = {}
    for split in common.SPLITS:
        keys = []
        for item in splits[split]:
            key = common.schema_body_hash(item.tool_schema)
            schema_by_hash.setdefault(key, item.tool_schema)
            keys.append(key)
        row_schema_hash[split] = keys

    hashes = sorted(schema_by_hash)
    schema_texts = [f"[TOOLS]\n{schema_by_hash[key]}" for key in hashes]
    total_rows = sum(len(splits[split]) for split in common.SPLITS)
    print(
        f"unique schema bodies: {len(hashes)} (vs {total_rows} rows) - "
        f"{total_rows / len(hashes):.2f} rows per schema encode",
        flush=True,
    )

    token_lengths = [
        len(encoder.tokenizer(text, add_special_tokens=True)["input_ids"])
        for text in schema_texts
    ]
    print(
        f"schema token length: median {statistics.median(token_lengths):.0f} "
        f"p95 {sorted(token_lengths)[int(0.95 * len(token_lengths))]} "
        f"max {max(token_lengths)}; "
        f"{sum(1 for n in token_lengths if n > MAX_LENGTH)} of {len(token_lengths)} "
        f"exceed the {MAX_LENGTH} cap",
        flush=True,
    )

    elapsed = time.time()
    schema_trunc = encoder.encode_truncated(schema_texts)
    print(f"schema trunc512 encoded in {time.time() - elapsed:.1f}s", flush=True)

    elapsed = time.time()
    schema_full, chunk_counts = encoder.encode_windowed(schema_texts)
    print(
        f"schema full encoded in {time.time() - elapsed:.1f}s "
        f"(mean {statistics.fmean(chunk_counts):.2f} windows/schema)",
        flush=True,
    )

    prompt_vectors: dict[str, torch.Tensor] = {}
    for split in common.SPLITS:
        elapsed = time.time()
        prompt_vectors[split] = encoder.encode_truncated(
            [f"[USER]\n{item.prompt}" for item in splits[split]]
        )
        print(
            f"prompts {split} ({len(splits[split])}) in {time.time() - elapsed:.1f}s",
            flush=True,
        )

    index = {key: position for position, key in enumerate(hashes)}
    payload = {
        "schema_hashes": hashes,
        "schema_trunc512": schema_trunc,
        "schema_full": schema_full,
        "prompt": prompt_vectors,
        "row_schema_index": {
            split: torch.tensor([index[key] for key in row_schema_hash[split]])
            for split in common.SPLITS
        },
        "labels": {
            split: torch.tensor(
                [float(item.output_length) for item in splits[split]]
            )
            for split in common.SPLITS
        },
    }
    torch.save(payload, OUT)

    meta = {
        "schema_version": "e4-embeddings-v1",
        "encoder_checkpoint": str(ENCODER),
        "encoder_sha256_note": "same checkpoint as bert-prompt_schema-tier2-seed17 "
        "used by the single-tower baseline; towers are FROZEN (feature extraction), "
        "only the fusion head is trained",
        "pooling": "mean over non-pad tokens; full variant additionally means over windows",
        "max_length": MAX_LENGTH,
        "max_chunks": MAX_CHUNKS,
        "inputs": inputs,
        "split_sizes": {split: len(splits[split]) for split in common.SPLITS},
        "censor_exclusion_counts": counts,
        "unique_schema_bodies": len(hashes),
        "rows_per_schema_encode": total_rows / len(hashes),
        "schema_token_length": {
            "median": statistics.median(token_lengths),
            "p95": sorted(token_lengths)[int(0.95 * len(token_lengths))],
            "max": max(token_lengths),
            "over_max_length": sum(1 for n in token_lengths if n > MAX_LENGTH),
            "count": len(token_lengths),
        },
        "windows_per_schema_mean": statistics.fmean(chunk_counts),
        "embedding_dim": int(schema_trunc.shape[1]),
        "wall_clock_seconds": time.time() - started,
    }
    META_OUT.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
