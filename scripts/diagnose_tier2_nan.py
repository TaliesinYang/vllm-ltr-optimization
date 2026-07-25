#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ltr_training.tier2_training import load_tier2_split_examples  # noqa: E402
from ltr_training.train_ranker import (  # noqa: E402
    _batched,
    build_pair_indices,
    pairwise_margin_loss,
    render_example,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce and locate Tier-2 non-finite values.")
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initial-model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=40)
    return parser.parse_args()


def main() -> int:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    args = parse_args()
    config = json.loads(args.config.read_text())
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    splits, _ = load_tier2_split_examples(
        sample_path=args.sample,
        ledger_path=args.ledger,
    )
    train_examples = splits["train"]
    pairs = build_pair_indices(
        train_examples,
        seed=seed,
        relative_delta=float(config["relative_pair_delta"]),
    )
    epoch_pairs = pairs.copy()
    random.Random(seed).shuffle(epoch_pairs)
    tokenizer = AutoTokenizer.from_pretrained(args.initial_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.initial_model)
    device = torch.device("cuda")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    model.train()

    records: list[dict[str, object]] = []
    variant = str(config["variant"])
    batch_size = int(config["batch_size"])
    max_length = int(config["max_length"])
    for step, batch_pairs in enumerate(_batched(epoch_pairs, batch_size), start=1):
        left = [train_examples[index] for index, _ in batch_pairs]
        right = [train_examples[index] for _, index in batch_pairs]
        left_inputs = tokenizer(
            [render_example(item, variant=variant) for item in left],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        right_inputs = tokenizer(
            [render_example(item, variant=variant) for item in right],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        left_scores = model(**left_inputs).logits.squeeze(-1)
        right_scores = model(**right_inputs).logits.squeeze(-1)
        left_lengths = torch.tensor([item.output_length for item in left], device=device)
        right_lengths = torch.tensor([item.output_length for item in right], device=device)
        loss = pairwise_margin_loss(
            left_scores,
            right_scores,
            left_lengths,
            right_lengths,
            margin=float(config["margin"]),
        )
        loss.backward()
        finite_grads = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        gradients = [
            parameter.grad for parameter in model.parameters() if parameter.grad is not None
        ]
        grad_norm = float(torch.nn.utils.get_total_norm(gradients, 2.0))
        score_max = float(torch.cat((left_scores, right_scores)).detach().abs().max())
        optimizer.step()
        finite_parameters = all(torch.isfinite(parameter).all() for parameter in model.parameters())
        max_parameter = max(float(parameter.detach().abs().max()) for parameter in model.parameters())
        record = {
            "step": step,
            "loss": float(loss.detach()),
            "scores_finite": bool(torch.isfinite(left_scores).all() and torch.isfinite(right_scores).all()),
            "score_max_abs": score_max,
            "gradients_finite": finite_grads,
            "gradient_norm": grad_norm,
            "parameters_finite_after_step": finite_parameters,
            "parameter_max_abs_after_step": max_parameter,
            "left_sample_ids": [item.sample_id for item in left],
            "right_sample_ids": [item.sample_id for item in right],
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if not all((math.isfinite(float(loss)), finite_grads, finite_parameters)):
            break
        if step >= args.max_steps:
            break
    report = {
        "seed": seed,
        "initial_model": str(args.initial_model),
        "first_nonfinite": next(
            (row for row in records if not (
                math.isfinite(float(row["loss"]))
                and row["gradients_finite"]
                and row["parameters_finite_after_step"]
            )),
            None,
        ),
        "steps": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
