from __future__ import annotations

import hashlib
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Protocol, Sequence


def render_prompt_schema(prompt: str, tool_schema: str) -> str:
    return f"[USER]\n{prompt}\n[TOOLS]\n{tool_schema}"


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def percentile_ranks(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    if len(values) == 1:
        return [0.5]
    ordered = sorted((float(value), index) for index, value in enumerate(values))
    result = [0.0] * len(values)
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and ordered[end][0] == ordered[position][0]:
            end += 1
        average_rank = (position + end - 1) / 2.0
        percentile = average_rank / (len(values) - 1)
        for _, original_index in ordered[position:end]:
            result[original_index] = percentile
        position = end
    return result


class BatchScorer(Protocol):
    def score(self, texts: Sequence[str]) -> tuple[list[float], list[int]]: ...


def score_ensemble_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    scorers: Mapping[int, BatchScorer],
    max_length: int = 512,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    materialized = [dict(row) for row in rows]
    texts = [render_prompt_schema(str(row["prompt"]), str(row.get("tool_schema", ""))) for row in materialized]
    token_lengths_by_seed: dict[int, list[int]] = {}
    for seed, scorer in sorted(scorers.items()):
        scores, token_lengths = scorer.score(texts)
        if len(scores) != len(materialized) or len(token_lengths) != len(materialized):
            raise ValueError(f"seed {seed} scorer returned the wrong row count")
        token_lengths_by_seed[seed] = token_lengths
        for row, score in zip(materialized, scores):
            row[f"score_seed{seed}"] = float(score)

    by_domain: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(materialized):
        by_domain[str(row["domain"])].append(index)
    for domain_indexes in by_domain.values():
        for seed in sorted(scorers):
            ranks = percentile_ranks(
                [float(materialized[index][f"score_seed{seed}"]) for index in domain_indexes]
            )
            for index, rank in zip(domain_indexes, ranks):
                materialized[index][f"rank_seed{seed}"] = rank
        for index in domain_indexes:
            ranks = [float(materialized[index][f"rank_seed{seed}"]) for seed in sorted(scorers)]
            materialized[index]["ensemble_rank"] = statistics.mean(ranks)
            materialized[index]["rank_dispersion"] = statistics.pstdev(ranks)
            materialized[index]["request_id"] = str(
                materialized[index].get("request_id", materialized[index]["sample_id"])
            )

    domain_report: dict[str, dict[str, object]] = {}
    for domain, indexes in sorted(by_domain.items()):
        truncated = sum(
            any(token_lengths_by_seed[seed][index] > max_length for seed in scorers)
            for index in indexes
        )
        domain_report[domain] = {
            "rows": len(indexes),
            "truncated_rows": truncated,
            "truncation_ratio": truncated / len(indexes) if indexes else 0.0,
            "max_length": max_length,
        }
    hashes = {
        str(seed): getattr(scorer, "checkpoint_sha256")
        for seed, scorer in scorers.items()
        if hasattr(scorer, "checkpoint_sha256")
    }
    return materialized, {
        "schema_version": "offline-ensemble-scores-v1",
        "seeds": sorted(scorers),
        "checkpoint_sha256": hashes,
        "domains": domain_report,
        "disagreement_basis": "within-domain percentile-rank population standard deviation",
    }


def _tau_b(truth: Sequence[float], prediction: Sequence[float]) -> float:
    from .offline_statistics import kendall_tau_b

    return kendall_tau_b(truth, prediction)


def disagreement_diagnostic(
    rows: Iterable[Mapping[str, object]],
    *,
    coverages: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
) -> dict[str, object]:
    by_domain: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_domain[str(row["domain"])].append(row)
    domains: dict[str, list[dict[str, object]]] = {}
    for domain, domain_rows in sorted(by_domain.items()):
        ordered = sorted(domain_rows, key=lambda row: float(row["rank_dispersion"]))
        points: list[dict[str, object]] = []
        for coverage in coverages:
            if not 0.0 < coverage <= 1.0:
                raise ValueError("coverage must be in (0, 1]")
            retained_count = max(2, math.ceil(len(ordered) * coverage))
            retained = ordered[: min(len(ordered), retained_count)]
            tau = _tau_b(
                [float(row["true_length"]) for row in retained],
                [float(row["ensemble_rank"]) for row in retained],
            )
            points.append(
                {
                    "coverage": len(retained) / len(ordered) if ordered else 0.0,
                    "retained": len(retained),
                    "tau_b": tau,
                    "risk": 1.0 - tau,
                    "max_rank_dispersion": max(
                        (float(row["rank_dispersion"]) for row in retained),
                        default=0.0,
                    ),
                }
            )
        domains[domain] = points
    return {
        "schema_version": "disagreement-diagnostic-v1",
        "diagnostic_name": "disagreement-empirical-error diagnostic",
        "risk_definition": "1 - Kendall tau-b",
        "domains": domains,
    }


class TransformersCheckpointScorer:
    def __init__(self, checkpoint: Path, *, batch_size: int = 32, max_length: int = 512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._checkpoint = checkpoint
        self.checkpoint_sha256 = checkpoint_sha256(checkpoint)
        self._batch_size = batch_size
        self._max_length = max_length
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, local_files_only=True
        ).to(self._device)
        self._model.eval()

    def score(self, texts: Sequence[str]) -> tuple[list[float], list[int]]:
        scores: list[float] = []
        token_lengths = [
            len(self._tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
            for text in texts
        ]
        with self._torch.inference_mode():
            for start in range(0, len(texts), self._batch_size):
                batch = list(texts[start : start + self._batch_size])
                inputs = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self._max_length,
                    return_tensors="pt",
                ).to(self._device)
                logits = self._model(**inputs).logits.reshape(-1)
                if logits.numel() != len(batch):
                    raise ValueError("checkpoint must produce one scalar score per row")
                scores.extend(float(value) for value in logits.detach().cpu().tolist())
        return scores, token_lengths
