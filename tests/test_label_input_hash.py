"""Regression: input_hash must accept raw non-JSON schema text (91179de)."""

from ltr_training.label_input import LabelInput, canonical_schema


def _label(schema: object) -> LabelInput:
    return LabelInput(
        sample_id="s", request_id="s", prompt="p", tool_schema=canonical_schema(schema),
        history=(), session_id="sess", task_id="t", source="toolace",
        source_revision="rev", category="id:toolace",
    )


def test_input_hash_accepts_raw_non_json_schema_text() -> None:
    raw = "You are an expert. Tools: [{\"name\": broken json"
    digest = _label(raw).input_hash
    assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)


def test_input_hash_still_canonicalizes_valid_json_schema() -> None:
    a = _label([{"name": "t", "parameters": {}}]).input_hash
    b = _label('[{"parameters": {}, "name": "t"}]').input_hash
    assert a == b  # key order must not matter for JSON schemas


def test_raw_and_json_schemas_hash_differently() -> None:
    assert _label("raw text").input_hash != _label(["raw text"]).input_hash
