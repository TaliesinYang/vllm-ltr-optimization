"""Replay-only empirical mapping from BERT rank scores to token quantiles."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


APPROXIMATION_NOTICE = (
    "Quantiles are approximations obtained by mapping a rank score through "
    "the empirical training-length distribution, not calibrated intervals. "
    "Scheduling is driven by the rank score in signals."
)
MAPPING_VERSION = "uncalibrated-rank-lookup-v1"
MIN_PERCENTILE = 10
MAX_PERCENTILE = 99


@dataclass(frozen=True)
class ReplayRecord:
    request_id: str
    prompt_text: str
    tool_schema_text: str
    output_length: int


@dataclass(frozen=True)
class RankMappedPrediction:
    quantiles: dict[int, float]
    signals: dict[str, float]


class RankQuantileMapper:
    """Map rank position to approximate lengths, never calibrated intervals."""

    def __init__(self, manifest: Mapping[str, object]) -> None:
        _validate_manifest(manifest)
        self.mapping_version = str(manifest["mapping_version"])
        self.model_version = str(manifest["model_version"])
        self.approximation_notice = str(manifest["approximation_notice"])
        percentile_rows = manifest["percentiles"]
        global_rows = manifest["global_quantiles"]
        assert isinstance(percentile_rows, Mapping)
        assert isinstance(global_rows, Mapping)
        self._percentiles = {
            percentile: float(percentile_rows[str(percentile)])
            for percentile in range(MIN_PERCENTILE, MAX_PERCENTILE + 1)
        }
        self._global = {
            percentile: float(global_rows[str(percentile)])
            for percentile in (50, 70, 90)
        }

    @classmethod
    def from_path(cls, path: Path) -> "RankQuantileMapper":
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping):
            raise ValueError("rank quantile manifest must be a JSON object")
        return cls(manifest)

    def map_score(self, score: float) -> RankMappedPrediction:
        if not math.isfinite(score):
            raise ValueError("rank score must be finite")
        position = min(MAX_PERCENTILE, max(MIN_PERCENTILE, score * 100.0))
        lower = math.floor(position)
        upper = math.ceil(position)
        lower_length = self._percentiles[lower]
        upper_length = self._percentiles[upper]
        point_length = lower_length + (position - lower) * (
            upper_length - lower_length
        )
        global_p50 = self._global[50]
        # These are rank-score empirical-distribution approximations, not
        # calibrated intervals. Scheduling uses the raw rank_score signal.
        quantiles = {
            50: point_length,
            70: point_length * self._global[70] / global_p50,
            90: point_length * self._global[90] / global_p50,
        }
        signals = {
            "quantile_spread": self._global[90] - global_p50,
            "ood_distance": 0.0,
            "feature_coverage": 1.0,
            "rank_score": score,
        }
        return RankMappedPrediction(quantiles=quantiles, signals=signals)


class ReplayStore:
    def __init__(self, records: tuple[ReplayRecord, ...]) -> None:
        if not records:
            raise ValueError("replay sidecar must contain at least one record")
        self._records = records
        self._by_id = {record.request_id: record for record in records}

    @classmethod
    def from_path(cls, path: Path) -> "ReplayStore":
        records: list[ReplayRecord] = []
        seen: set[str] = set()
        for line_number, row in _iter_jsonl(path):
            request_id = _required_text(row, "request_id", line_number)
            if request_id in seen:
                raise ValueError(f"duplicate request_id at line {line_number}: {request_id}")
            if row.get("split") != "train":
                raise ValueError(f"sidecar line {line_number} must use split=train")
            record = ReplayRecord(
                request_id=request_id,
                prompt_text=_required_text(row, "prompt_text", line_number),
                tool_schema_text=_required_text(
                    row, "tool_schema_text", line_number
                ),
                output_length=_required_length(row, line_number),
            )
            seen.add(request_id)
            records.append(record)
        return cls(tuple(records))

    def get(self, request_id: str) -> ReplayRecord:
        return self._by_id[request_id]

    def first_request_id(self) -> str:
        return self._records[0].request_id


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"JSONL line {line_number} must be an object")
            yield line_number, row


def _required_text(row: Mapping[str, object], key: str, line_number: int) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"JSONL line {line_number} requires non-empty {key}")
    return value


def _required_length(row: Mapping[str, object], line_number: int) -> int:
    value = row.get("output_length")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"JSONL line {line_number} has invalid output_length")
    return value


def load_training_records(path: Path) -> tuple[ReplayRecord, ...]:
    records: list[ReplayRecord] = []
    seen: set[str] = set()
    for line_number, row in _iter_jsonl(path):
        request_id = _required_text(row, "sample_id", line_number)
        if request_id in seen:
            raise ValueError(f"duplicate sample_id at line {line_number}: {request_id}")
        record = ReplayRecord(
            request_id=request_id,
            prompt_text=_required_text(row, "prompt", line_number),
            tool_schema_text=_required_text(row, "tool_schema", line_number),
            output_length=_required_length(row, line_number),
        )
        seen.add(request_id)
        records.append(record)
    if not records:
        raise ValueError("training labels must contain at least one record")
    return tuple(records)


def nearest_rank(sorted_lengths: list[int], percentile: int) -> int:
    index = math.ceil(percentile / 100 * len(sorted_lengths)) - 1
    return sorted_lengths[max(0, min(index, len(sorted_lengths) - 1))]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_manifest(manifest: Mapping[str, object]) -> None:
    if manifest.get("mapping_version") != MAPPING_VERSION:
        raise ValueError(f"mapping_version must be {MAPPING_VERSION}")
    if manifest.get("approximation_notice") != APPROXIMATION_NOTICE:
        raise ValueError("manifest approximation_notice does not match contract")
    if not isinstance(manifest.get("model_version"), str) or not manifest[
        "model_version"
    ]:
        raise ValueError("manifest model_version must be non-empty")
    exclusions = manifest.get("structural_exclusions", [])
    if not isinstance(exclusions, list) or len(exclusions) > 5:
        raise ValueError("structural_exclusions must be a list of at most 5 entries")
    for entry in exclusions:
        if (
            not isinstance(entry, Mapping)
            or not isinstance(entry.get("sample_id"), str)
            or not entry["sample_id"]
            or not isinstance(entry.get("reason"), str)
            or not entry["reason"]
        ):
            raise ValueError("each structural exclusion needs sample_id and reason")
    if manifest.get("sample_count") != 6_000 - len(exclusions):
        raise ValueError(
            "manifest sample_count plus structural exclusions must equal 6000"
        )

    percentile_rows = manifest.get("percentiles")
    expected_keys = {
        str(percentile) for percentile in range(MIN_PERCENTILE, MAX_PERCENTILE + 1)
    }
    if not isinstance(percentile_rows, Mapping) or set(percentile_rows) != expected_keys:
        raise ValueError("manifest percentiles must contain exactly p10 through p99")
    percentile_values = [
        _finite_nonnegative(percentile_rows[str(percentile)], "percentile length")
        for percentile in range(MIN_PERCENTILE, MAX_PERCENTILE + 1)
    ]
    if percentile_values != sorted(percentile_values):
        raise ValueError("manifest percentile lengths must be ordered")

    global_rows = manifest.get("global_quantiles")
    if not isinstance(global_rows, Mapping) or set(global_rows) != {"50", "70", "90"}:
        raise ValueError("manifest global_quantiles must contain p50, p70, and p90")
    global_values = [
        _finite_nonnegative(global_rows[str(percentile)], "global quantile")
        for percentile in (50, 70, 90)
    ]
    if global_values[0] <= 0:
        raise ValueError("manifest global p50 must be positive")
    if global_values != sorted(global_values):
        raise ValueError("manifest global quantiles must be ordered")


def _finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return numeric


def build_rank_quantile_artifacts(
    *,
    labels_path: Path,
    checkpoint: Path,
    sidecar_path: Path,
    manifest_path: Path,
    model_version: str,
    expected_count: int,
    structural_exclusions: tuple[Mapping[str, object], ...] = (),
) -> dict[str, object]:
    if not model_version:
        raise ValueError("model_version must be non-empty")
    if expected_count < 1:
        raise ValueError("expected_count must be positive")
    records = load_training_records(labels_path)
    if len(records) != expected_count:
        raise ValueError(
            f"training label count mismatch: got {len(records)} want {expected_count}"
        )

    lengths = sorted(record.output_length for record in records)
    percentiles = {
        str(percentile): nearest_rank(lengths, percentile)
        for percentile in range(MIN_PERCENTILE, MAX_PERCENTILE + 1)
    }
    global_quantiles = {
        str(percentile): nearest_rank(lengths, percentile)
        for percentile in (50, 70, 90)
    }
    if global_quantiles["50"] <= 0:
        raise ValueError("global p50 must be positive")
    manifest: dict[str, object] = {
        "mapping_version": MAPPING_VERSION,
        "model_version": model_version,
        "approximation_notice": APPROXIMATION_NOTICE,
        "sample_count": len(records),
        "structural_exclusions": list(structural_exclusions),
        "source_sha256": sha256_file(labels_path),
        "checkpoint_sha256": sha256_file(checkpoint / "model.safetensors"),
        "percentiles": percentiles,
        "global_quantiles": global_quantiles,
    }
    _write_artifacts(
        records=records,
        manifest=manifest,
        sidecar_path=sidecar_path,
        manifest_path=manifest_path,
    )
    return manifest


def _write_artifacts(
    *,
    records: tuple[ReplayRecord, ...],
    manifest: Mapping[str, object],
    sidecar_path: Path,
    manifest_path: Path,
) -> None:
    sidecar_tmp = sidecar_path.with_suffix(sidecar_path.suffix + ".tmp")
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    for path in (sidecar_path, manifest_path, sidecar_tmp, manifest_tmp):
        if path.exists():
            raise FileExistsError(path)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    created_sidecar = False
    try:
        with sidecar_tmp.open("w", encoding="utf-8") as handle:
            for record in records:
                row = {
                    "request_id": record.request_id,
                    "prompt_text": record.prompt_text,
                    "tool_schema_text": record.tool_schema_text,
                    "output_length": record.output_length,
                    "split": "train",
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest_tmp.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        sidecar_tmp.replace(sidecar_path)
        created_sidecar = True
        manifest_tmp.replace(manifest_path)
    except Exception:
        sidecar_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)
        if created_sidecar:
            sidecar_path.unlink(missing_ok=True)
        raise
