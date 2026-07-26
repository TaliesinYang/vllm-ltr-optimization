import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scheduler_benchmark.predictor import (
    BertPredictor,
    ConstantPredictor,
    OracleFromFilePredictor,
    Prediction,
    PredictorInput,
    RandomPredictor,
)
from scheduler_benchmark.tool_vocabulary import (
    DEFAULT_ARTIFACT,
    GateVocabulary,
    toolset_fingerprint,
)


def schema_with(*names: str) -> str:
    """A ToolACE-shaped system prompt advertising the given tools."""
    tools = ", ".join(f'{{"name": "{name}", "description": "d"}}' for name in names)
    return (
        "You are an expert in composing functions.\n"
        "Here is a list of functions in JSON format that you can invoke:\n"
        f"[{tools}]. \n"
    )


@pytest.fixture
def gate_artifact(tmp_path) -> Path:
    """Small stand-in vocabulary: tools alpha+beta seen, combination alpha+beta seen."""
    path = tmp_path / "gate_confidence.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "gate-confidence-v1",
                "confidence_by_stratum": {
                    "S1": 0.0,
                    "S2": 0.0,
                    "S3": 0.25,
                    "S4": 0.5,
                },
                "unknown_confidence": 0.0,
                "fingerprint_prefix_length": 32,
                "train_fingerprints": [
                    toolset_fingerprint(schema_with("alpha", "beta"))[:32]
                ],
                "train_tool_names": ["alpha", "beta"],
            }
        )
    )
    return path


class FakeBatchEncoding(dict):
    def __init__(self) -> None:
        super().__init__(input_ids=torch.tensor([[101, 102]]))
        self.device = None

    def to(self, device):
        self.device = device
        return self


class RecordingTokenizer:
    def __init__(self) -> None:
        self.calls = []
        self.encoding = FakeBatchEncoding()

    def __call__(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return self.encoding


class ScalarLogitModel:
    def __init__(self, logit: float) -> None:
        self.logit = logit
        self.device = None
        self.is_eval = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.is_eval = True
        return self

    def __call__(self, **inputs):
        assert "input_ids" in inputs
        return SimpleNamespace(logits=torch.tensor([[self.logit]]))


def make_bert_predictor(monkeypatch, tmp_path, *, logit: float = 2.0, vocabulary=None):
    tokenizer = RecordingTokenizer()
    model = ScalarLogitModel(logit)
    loaded = {}

    def load_tokenizer(path, **kwargs):
        loaded["tokenizer"] = (path, kwargs)
        return tokenizer

    def load_model(path, **kwargs):
        loaded["model"] = (path, kwargs)
        return model

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", load_tokenizer)
    monkeypatch.setattr(
        "transformers.AutoModelForSequenceClassification.from_pretrained",
        load_model,
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    predictor = BertPredictor(checkpoint, vocabulary=vocabulary)
    return predictor, tokenizer, model, loaded


def predict_with_schema(predictor, tool_schema: str):
    return predictor.predict(
        PredictorInput(
            request_id="toolace-000000:0000",
            prompt_token_ids=(1, 2, 3),
            metadata={"prompt_text": "prompt", "tool_schema_text": tool_schema},
        )
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


def test_bert_predictor_replays_exact_prompt_schema_tokenization(
    monkeypatch, tmp_path
) -> None:
    predictor, tokenizer, model, loaded = make_bert_predictor(monkeypatch, tmp_path)
    predictor_input = PredictorInput(
        request_id="toolace-000000:0000",
        prompt_token_ids=(1, 2, 3),
        metadata={
            "prompt_text": "current prompt",
            "tool_schema_text": "raw schema",
        },
    )

    prediction = predictor.predict(predictor_input)

    assert tokenizer.calls == [
        (
            ["[USER]\ncurrent prompt\n[TOOLS]\nraw schema"],
            {
                "padding": True,
                "truncation": True,
                "max_length": 512,
                "return_tensors": "pt",
            },
        )
    ]
    assert loaded["tokenizer"][1] == {"local_files_only": True}
    assert loaded["model"][1] == {"local_files_only": True}
    assert str(model.device) == "cpu"
    assert model.is_eval is True
    assert str(tokenizer.encoding.device) == "cpu"
    assert prediction.score == pytest.approx(1.0 / (1.0 + math.exp(-2.0)))
    # "raw schema" advertises no readable tool set, so the gate abstains.
    assert prediction.confidence == 0.0
    assert prediction.ood is False
    assert prediction.latency_ms >= 0.0


def test_bert_predictor_maps_higher_longer_logit_to_higher_scheduler_cost(
    monkeypatch, tmp_path
) -> None:
    predictor, _, model, _ = make_bert_predictor(monkeypatch, tmp_path, logit=-2.0)
    predictor_input = PredictorInput(
        request_id="toolace-000000:0000",
        prompt_token_ids=(),
        metadata={"prompt_text": "prompt", "tool_schema_text": "schema"},
    )

    shorter_score = predictor.predict(predictor_input).score
    model.logit = 2.0
    longer_score = predictor.predict(predictor_input).score

    assert shorter_score == pytest.approx(1.0 / (1.0 + math.exp(2.0)))
    assert longer_score == pytest.approx(1.0 / (1.0 + math.exp(-2.0)))
    assert shorter_score < longer_score


def test_bert_predictor_does_not_report_the_hardcoded_placeholder_confidence(
    monkeypatch, tmp_path, gate_artifact
) -> None:
    """A request with entirely novel tools must not be vouched for at 0.9."""
    predictor, _, _, _ = make_bert_predictor(
        monkeypatch, tmp_path, vocabulary=GateVocabulary.from_artifact(gate_artifact)
    )

    prediction = predict_with_schema(predictor, schema_with("gamma", "delta"))

    assert prediction.confidence != 0.9
    assert prediction.confidence == 0.5
    assert not hasattr(BertPredictor, "PLACEHOLDER_CONFIDENCE")


@pytest.mark.parametrize(
    ("names", "stratum", "expected"),
    (
        (("alpha", "beta"), "S1", 0.0),
        (("alpha",), "S2", 0.0),
        (("alpha", "gamma"), "S3", 0.25),
        (("gamma", "delta"), "S4", 0.5),
    ),
)
def test_bert_predictor_confidence_follows_cold_start_stratum(
    monkeypatch, tmp_path, gate_artifact, names, stratum, expected
) -> None:
    vocabulary = GateVocabulary.from_artifact(gate_artifact)
    predictor, _, _, _ = make_bert_predictor(
        monkeypatch, tmp_path, vocabulary=vocabulary
    )
    schema = schema_with(*names)

    assert vocabulary.stratum(schema) == stratum
    assert predict_with_schema(predictor, schema).confidence == expected


@pytest.mark.parametrize(
    "tool_schema",
    (
        pytest.param(schema_with(), id="empty_tool_list"),
        pytest.param("raw schema", id="unparseable_schema"),
        pytest.param("", id="blank_schema_text_is_rejected_upstream"),
    ),
)
def test_bert_predictor_is_conservative_when_the_tool_set_is_unreadable(
    monkeypatch, tmp_path, gate_artifact, tool_schema
) -> None:
    """No readable tool set means no vouching - the gate abstains."""
    predictor, _, _, _ = make_bert_predictor(
        monkeypatch, tmp_path, vocabulary=GateVocabulary.from_artifact(gate_artifact)
    )
    if not tool_schema:
        with pytest.raises(ValueError, match="tool_schema_text"):
            predict_with_schema(predictor, tool_schema)
        return

    assert predict_with_schema(predictor, tool_schema).confidence == 0.0


def test_constant_confidence_env_override_remains_an_escape_hatch(
    monkeypatch, tmp_path, gate_artifact
) -> None:
    monkeypatch.setenv("LTR_CONSTANT_CONFIDENCE", "0.75")
    predictor, _, _, _ = make_bert_predictor(
        monkeypatch, tmp_path, vocabulary=GateVocabulary.from_artifact(gate_artifact)
    )

    prediction = predict_with_schema(predictor, schema_with("gamma", "delta"))

    assert prediction.confidence == 0.75


def test_committed_gate_artifact_matches_the_offline_evaluation() -> None:
    """The served numbers are the T5 numbers, not a hand-copied approximation."""
    gate = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "runs"
            / "offline-experiments-2026-07-25"
            / "t5-gate.json"
        ).read_text()
    )
    vocabulary = GateVocabulary.from_artifact(DEFAULT_ARTIFACT)

    assert vocabulary.confidence_by_stratum == {
        stratum: pytest.approx(float(value))
        for stratum, value in gate["assigned_confidence"].items()
    }
    # The abstain value must never exceed what any stratum earned.
    assert vocabulary.unknown_confidence <= min(vocabulary.confidence_by_stratum.values())
    assert vocabulary.provenance["rule"] == "C_abstain"


def test_bert_predictor_fails_closed_without_raw_training_features(
    monkeypatch, tmp_path
) -> None:
    predictor, _, _, _ = make_bert_predictor(monkeypatch, tmp_path)
    incomplete_metadata = (
        {"tool_schema_text": "schema"},
        {"prompt_text": "prompt"},
    )

    for metadata in incomplete_metadata:
        with pytest.raises(ValueError, match="prompt_text|tool_schema_text"):
            predictor.predict(
                PredictorInput(
                    request_id="toolace-000000:0000",
                    prompt_token_ids=(),
                    metadata=metadata,
                )
            )


def test_tool_names_parses_openai_tools_array_format() -> None:
    # Gateways forward the OpenAI array: name nested under "function".
    import json as _json

    from scheduler_benchmark.tool_vocabulary import tool_names

    schema = _json.dumps(
        [
            {"type": "function", "function": {"name": "glob", "parameters": {}}},
            {"type": "function", "function": {"name": "read", "parameters": {}}},
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    assert tool_names(schema) == ("glob", "read")
