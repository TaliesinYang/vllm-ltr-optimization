import json

import pytest

from scheduler_benchmark.predictor import (
    ConstantPredictor,
    OracleFromFilePredictor,
    Prediction,
    PredictorInput,
    RandomPredictor,
)


def make_input(request_id: str = "req-1") -> PredictorInput:
    return PredictorInput(
        request_id=request_id,
        prompt_token_ids=(10, 20, 30),
        metadata={"kind": "chat"},
    )


def test_constant_predictor_returns_complete_contract() -> None:
    predictor = ConstantPredictor(score=0.42, confidence=0.8, ood=False)

    result = predictor.predict(make_input())

    assert result.score == 0.42
    assert result.confidence == 0.8
    assert result.ood is False
    assert result.latency_ms >= 0.0


def test_prediction_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        Prediction(score=1.0, confidence=1.01, ood=False, latency_ms=0.0)


def test_prediction_rejects_score_outside_normalized_rank_space() -> None:
    with pytest.raises(ValueError, match="score"):
        Prediction(score=1.01, confidence=1.0, ood=False, latency_ms=0.0)


def test_random_predictor_is_reproducible_for_seed() -> None:
    left = RandomPredictor(seed=17)
    right = RandomPredictor(seed=17)

    left_values = [left.predict(make_input(str(index))).score for index in range(3)]
    right_values = [right.predict(make_input(str(index))).score for index in range(3)]

    assert left_values == right_values
    assert all(0.0 <= value <= 1.0 for value in left_values)


def test_oracle_predictor_reads_request_score_from_json(tmp_path) -> None:
    oracle_path = tmp_path / "oracle.json"
    oracle_path.write_text(
        json.dumps(
            {
                "req-1": {
                    "score": 0.123,
                    "confidence": 1.0,
                    "ood": False,
                }
            }
        )
    )

    result = OracleFromFilePredictor(oracle_path).predict(make_input())

    assert result.score == 0.123
    assert result.confidence == 1.0
    assert result.ood is False
    assert result.latency_ms >= 0.0


def test_oracle_predictor_rejects_unknown_request(tmp_path) -> None:
    oracle_path = tmp_path / "oracle.json"
    oracle_path.write_text("{}")
    predictor = OracleFromFilePredictor(oracle_path)

    with pytest.raises(KeyError, match="req-missing"):
        predictor.predict(make_input("req-missing"))
