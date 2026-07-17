from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor


@dataclass(frozen=True)
class TrainingExample:
    sample_id: str
    session_id: str
    prompt: str
    output_length: int
    generator_id: str


def load_tier1_examples(
    path: str | Path, *, sources: set[str]
) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if row.get("source") not in sources:
                continue
            try:
                example = TrainingExample(
                    sample_id=row["sample_id"],
                    session_id=row["session_id"],
                    prompt=row["prompt"],
                    output_length=row["output_length"],
                    generator_id=row["generator_id"],
                )
            except KeyError as error:
                raise ValueError(f"label line {line_number} lacks {error.args[0]}") from error
            if not isinstance(example.prompt, str) or not example.prompt:
                continue
            if not isinstance(example.output_length, int) or example.output_length < 0:
                raise ValueError(f"label line {line_number} has invalid output_length")
            examples.append(example)
    return examples


def split_by_session(
    examples: Iterable[TrainingExample], *, validation_fraction: float = 0.1
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    threshold = int(validation_fraction * 10_000)
    train: list[TrainingExample] = []
    validation: list[TrainingExample] = []
    for example in examples:
        digest = hashlib.sha256(example.session_id.encode()).digest()
        bucket = int.from_bytes(digest[:8], "big") % 10_000
        (validation if bucket < threshold else train).append(example)
    if not train or not validation:
        raise ValueError("session split produced an empty partition")
    return train, validation


def build_pair_indices(
    examples: list[TrainingExample],
    *,
    seed: int,
    relative_delta: float,
) -> list[tuple[int, int]]:
    if relative_delta < 0:
        raise ValueError("relative_delta must be non-negative")
    grouped: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        grouped.setdefault(example.generator_id, []).append(index)

    rng = random.Random(seed)
    pairs: list[tuple[int, int]] = []
    for generator_id in sorted(grouped):
        indexes = sorted(
            grouped[generator_id], key=lambda index: examples[index].output_length
        )
        if len(indexes) < 2:
            continue
        candidates = indexes.copy()
        rng.shuffle(candidates)
        for left in candidates:
            right_candidates = indexes.copy()
            rng.shuffle(right_candidates)
            for right in right_candidates:
                if left == right:
                    continue
                left_length = examples[left].output_length
                right_length = examples[right].output_length
                denominator = max(left_length, right_length, 1)
                if abs(left_length - right_length) / denominator >= relative_delta:
                    pairs.append((left, right))
                    break
    rng.shuffle(pairs)
    if not pairs:
        raise ValueError("no same-generator pairs satisfy relative_pair_delta")
    return pairs


def pairwise_margin_loss(
    left_scores: Tensor,
    right_scores: Tensor,
    left_lengths: Tensor,
    right_lengths: Tensor,
    *,
    margin: float,
) -> Tensor:
    targets = torch.where(
        left_lengths > right_lengths,
        torch.ones_like(left_scores),
        -torch.ones_like(left_scores),
    )
    return torch.nn.functional.margin_ranking_loss(
        left_scores,
        right_scores,
        targets,
        margin=margin,
    )


def _batched(items: list[tuple[int, int]], size: int) -> Iterable[list[tuple[int, int]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def should_save_checkpoint(step: int, *, save_steps: int) -> bool:
    if step < 1 or save_steps < 1:
        raise ValueError("step and save_steps must be positive")
    return step == 1 or step % save_steps == 0


def _save_checkpoint(
    *,
    model: object,
    tokenizer: object,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    step: int,
    state: dict[str, object],
) -> Path:
    checkpoint = output_dir / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n"
    )
    return checkpoint


def train(
    *,
    config: dict[str, object],
    labels_path: Path,
    output_dir: Path,
    max_steps: int | None = None,
) -> None:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    examples = load_tier1_examples(
        labels_path, sources=set(config["train_sources"])
    )
    if len(examples) < 2:
        raise ValueError("at least two training examples are required")
    train_examples, validation_examples = split_by_session(examples)
    pairs = build_pair_indices(
        train_examples,
        seed=seed,
        relative_delta=float(config["relative_pair_delta"]),
    )

    backbone = str(config["backbone"])
    revision = str(config["backbone_revision"])
    tokenizer = AutoTokenizer.from_pretrained(backbone, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        backbone,
        revision=revision,
        num_labels=1,
    )
    device = _select_device()
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"])
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_name": config["run_name"],
        "variant": config["variant"],
        "seed": seed,
        "label_tier": config["label_tier"],
        "labels_path": str(labels_path.resolve()),
        "labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        "backbone": backbone,
        "backbone_revision": revision,
        "device": str(device),
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "train_pairs": len(pairs),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, sort_keys=True), flush=True)

    batch_size = int(config["batch_size"])
    epochs = int(config["epochs"])
    save_steps = int(config["save_steps"])
    max_length = int(config["max_length"])
    margin = float(config["margin"])
    total_steps = math.ceil(len(pairs) / batch_size) * epochs
    global_step = 0

    model.train()
    for epoch in range(epochs):
        epoch_pairs = pairs.copy()
        random.Random(seed + epoch).shuffle(epoch_pairs)
        for batch_pairs in _batched(epoch_pairs, batch_size):
            left = [train_examples[index] for index, _ in batch_pairs]
            right = [train_examples[index] for _, index in batch_pairs]
            left_inputs = tokenizer(
                [item.prompt for item in left],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            right_inputs = tokenizer(
                [item.prompt for item in right],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            left_scores = model(**left_inputs).logits.squeeze(-1)
            right_scores = model(**right_inputs).logits.squeeze(-1)
            left_lengths = torch.tensor(
                [item.output_length for item in left], device=device
            )
            right_lengths = torch.tensor(
                [item.output_length for item in right], device=device
            )
            loss = pairwise_margin_loss(
                left_scores,
                right_scores,
                left_lengths,
                right_lengths,
                margin=margin,
            )
            loss.backward()
            optimizer.step()
            global_step += 1
            print(
                f"step={global_step}/{total_steps} epoch={epoch + 1} loss={loss.item():.6f}",
                flush=True,
            )
            if should_save_checkpoint(global_step, save_steps=save_steps):
                checkpoint = _save_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    output_dir=output_dir,
                    step=global_step,
                    state={
                        "global_step": global_step,
                        "epoch": epoch + 1,
                        "loss": loss.item(),
                        "total_steps": total_steps,
                    },
                )
                print(f"checkpoint={checkpoint}", flush=True)
            if max_steps is not None and global_step >= max_steps:
                return
