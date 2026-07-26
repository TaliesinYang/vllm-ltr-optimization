"""E4 step 2 - train the fusion head on cached embeddings and score S1-S4.

Frozen towers, trained fusion. The fusion head is an MLP over
[prompt_embedding, schema_embedding] optimised with the SAME pairwise margin
loss and the SAME same-generator pair construction the single-tower ranker used
(ltr_training.train_ranker.build_pair_indices / pairwise_margin_loss), on the
SAME fixed tier-2 split.

Two schema variants are trained identically and differ only in whether the
schema tower saw the whole schema:

  trunc512  schema truncated at 512 tokens (reproduces the single-tower cap)
  full      schema windowed and pooled, no truncation

The trunc512-vs-full contrast is the controlled test of the truncation
hypothesis: one encoder, one fusion recipe, one split, one thing moving.

Comparison against the single-tower prompt_schema baseline is reported too, but
it is NOT controlled - that baseline was fine-tuned end to end while these
towers are frozen. That confound is stated, not hidden.
"""

from __future__ import annotations

import json
import random
import statistics
import time
from pathlib import Path

import torch
from scipy.stats import kendalltau
from torch import nn

import common
from t1_strata import MIN_TAU_ROWS, STRATA, assign_strata, bootstrap_ci

from ltr_training.offline_statistics import kendall_tau_b
from ltr_training.train_ranker import build_pair_indices, pairwise_margin_loss

HERE = Path(__file__).resolve().parent
EMBEDDINGS = HERE / "e4-embeddings.pt"
OUT = HERE / "e4-fusion.json"
PREDICTIONS_OUT = HERE / "e4-two-tower-test-predictions.jsonl"

VARIANTS = ("trunc512", "full")
EPOCHS = 3
BATCH = 64
LEARNING_RATE = 1e-3
MARGIN = 1.0
RELATIVE_PAIR_DELTA = 0.1
HIDDEN = 256


class Fusion(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, prompt: torch.Tensor, schema: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([prompt, schema], dim=-1)).squeeze(-1)


def features(payload, split: str, variant: str) -> tuple[torch.Tensor, torch.Tensor]:
    schema_bank = payload[f"schema_{variant}"]
    index = payload["row_schema_index"][split]
    return payload["prompt"][split], schema_bank[index]


def train_one(payload, splits, *, variant: str, seed: int) -> dict[str, object]:
    torch.manual_seed(seed)
    random.seed(seed)
    prompt_train, schema_train = features(payload, "train", variant)
    dim = prompt_train.shape[1]
    model = Fusion(dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    pairs = build_pair_indices(
        splits["train"], seed=seed, relative_delta=RELATIVE_PAIR_DELTA
    )
    lengths = payload["labels"]["train"]
    started = time.time()
    model.train()
    for epoch in range(EPOCHS):
        shuffled = pairs.copy()
        random.Random(seed + epoch).shuffle(shuffled)
        for start in range(0, len(shuffled), BATCH):
            batch = shuffled[start : start + BATCH]
            left = torch.tensor([index for index, _ in batch])
            right = torch.tensor([index for _, index in batch])
            optimizer.zero_grad(set_to_none=True)
            loss = pairwise_margin_loss(
                model(prompt_train[left], schema_train[left]),
                model(prompt_train[right], schema_train[right]),
                lengths[left],
                lengths[right],
                margin=MARGIN,
            )
            loss.backward()
            optimizer.step()
    train_seconds = time.time() - started

    model.eval()
    scores: dict[str, list[float]] = {}
    with torch.inference_mode():
        for split in ("validation", "test"):
            prompt, schema = features(payload, split, variant)
            scores[split] = [float(value) for value in model(prompt, schema)]
    return {
        "variant": variant,
        "seed": seed,
        "train_pairs": len(pairs),
        "train_seconds": train_seconds,
        "scores": scores,
        "state_dict": model.state_dict(),
    }


def main() -> None:
    started = time.time()
    common.verify_inputs()
    splits, _ = common.load_splits()
    payload = torch.load(EMBEDDINGS, weights_only=False)
    strata, definition = assign_strata(splits)
    test = splits["test"]

    runs: list[dict[str, object]] = []
    predictions: dict[str, dict[int, list[float]]] = {}
    for variant in VARIANTS:
        predictions[f"two_tower_{variant}"] = {}
        for seed in common.SEEDS:
            run = train_one(payload, splits, variant=variant, seed=seed)
            predictions[f"two_tower_{variant}"][seed] = run["scores"]["test"]
            if seed == common.SEEDS[0]:
                # Weights the latency harness serves; hidden size travels with them.
                torch.save(
                    {"state_dict": run["state_dict"], "hidden": HIDDEN},
                    HERE / f"e4-fusion-{variant}-seed{seed}.pt",
                )
            validation_tau = kendall_tau_b(
                [float(v) for v in payload["labels"]["validation"]],
                run["scores"]["validation"],
            )
            test_tau = kendall_tau_b(
                [float(v) for v in payload["labels"]["test"]], run["scores"]["test"]
            )
            runs.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "train_pairs": run["train_pairs"],
                    "train_seconds": run["train_seconds"],
                    "validation_tau_b": validation_tau,
                    "test_tau_b": test_tau,
                }
            )
            print(
                f"{variant:9s} seed{seed}: val={validation_tau:.4f} test={test_tau:.4f} "
                f"({run['train_seconds']:.1f}s)",
                flush=True,
            )

    table: dict[str, dict[str, object]] = {}
    for model_name, by_seed in predictions.items():
        table[model_name] = {}
        for stratum in STRATA:
            indices = strata[stratum]
            cell: dict[str, object] = {"n": len(indices)}
            if len(indices) < MIN_TAU_ROWS:
                cell["tau_withheld"] = f"n<{MIN_TAU_ROWS}"
            else:
                per_seed = {
                    seed: kendall_tau_b(
                        [float(test[i].output_length) for i in indices],
                        [by_seed[seed][i] for i in indices],
                    )
                    for seed in common.SEEDS
                }
                values = list(per_seed.values())
                cell["per_seed_tau_b"] = {str(k): v for k, v in per_seed.items()}
                cell["mean_tau_b"] = statistics.fmean(values)
                cell["stdev_tau_b"] = statistics.stdev(values)
                cell["ci95_seed17"] = bootstrap_ci(
                    test, by_seed[common.SEEDS[0]], indices
                )
            table[model_name][stratum] = cell

    with PREDICTIONS_OUT.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(test):
            handle.write(
                json.dumps(
                    {
                        "sample_id": item.sample_id,
                        "session_id": item.session_id,
                        "true_length": item.output_length,
                        **{
                            f"{name}_seed{seed}": by_seed[seed][index]
                            for name, by_seed in predictions.items()
                            for seed in common.SEEDS
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    truncation_effect = {}
    for stratum in STRATA:
        full = table["two_tower_full"][stratum]
        trunc = table["two_tower_trunc512"][stratum]
        if "mean_tau_b" in full and "mean_tau_b" in trunc:
            truncation_effect[stratum] = {
                "n": full["n"],
                "full_tau_b": full["mean_tau_b"],
                "trunc512_tau_b": trunc["mean_tau_b"],
                "delta_tau_b": full["mean_tau_b"] - trunc["mean_tau_b"],
                "ci_overlap": not (
                    full["ci95_seed17"][0] > trunc["ci95_seed17"][1]
                    or trunc["ci95_seed17"][0] > full["ci95_seed17"][1]
                ),
            }
        else:
            truncation_effect[stratum] = {"n": full["n"], "evaluated": False}

    report = {
        "schema_version": "e4-fusion-v1",
        "status": "done",
        "ticket": "issue #8 (T4/E4)",
        "architecture": {
            "towers": "frozen bert-prompt_schema-tier2-seed17 encoder, mean-pooled",
            "fusion": f"MLP {HIDDEN} hidden over [prompt_emb, schema_emb]",
            "loss": "pairwise margin, same-generator pairs "
            "(train_ranker.build_pair_indices, relative_delta="
            f"{RELATIVE_PAIR_DELTA}, margin={MARGIN})",
            "epochs": EPOCHS,
            "batch_size": BATCH,
            "learning_rate": LEARNING_RATE,
        },
        "controlled_contrast": "trunc512 vs full - one encoder, one fusion recipe, "
        "one split; truncation is the only thing that moves",
        "uncontrolled_comparison_caveat": "the single-tower prompt_schema baseline was "
        "fine-tuned end to end; these towers are frozen. A two-tower vs single-tower "
        "gap therefore mixes frozen-vs-fine-tuned with two-tower-vs-single-tower and "
        "must not be read as a truncation result",
        "embedding_meta": json.loads((HERE / "e4-embeddings-meta.json").read_text()),
        "stratum_definition": definition,
        "runs": runs,
        "results": table,
        "truncation_effect_full_minus_trunc512": truncation_effect,
        "wall_clock_seconds": time.time() - started,
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(truncation_effect, indent=2, sort_keys=True))
    print(f"done in {report['wall_clock_seconds']:.1f}s")


if __name__ == "__main__":
    main()
