from ltr_training.offline_baselines import (
    LEGACY_FAMILIES,
    lightgbm_grid,
    legacy_length_score,
    legacy_loader_status,
    run_lightgbm_grid,
)


def test_lightgbm_grid_has_twenty_declared_combinations() -> None:
    grid = lightgbm_grid()

    assert len(grid) == 20
    assert all(set(item) == {"max_depth", "num_leaves", "learning_rate", "n_estimators"} for item in grid)


def test_legacy_families_have_explicit_backbone_head_and_direction() -> None:
    assert set(LEGACY_FAMILIES) == {
        "listmle-opt",
        "classification-opt",
        "pars-bert",
        "a1-opt",
        "a2-bert",
    }
    assert LEGACY_FAMILIES["classification-opt"].head == "classification"
    assert LEGACY_FAMILIES["pars-bert"].backbone == "bert"
    assert all(spec.shorter_is_higher for spec in LEGACY_FAMILIES.values())


def test_old_shorter_is_higher_scores_are_reversed_to_length_cost() -> None:
    assert legacy_length_score("listmle-opt", raw_score=3.0) == -3.0
    assert legacy_length_score("classification-opt", logits=[0.0, 0.0, 10.0]) < -1.9


def test_missing_legacy_weights_are_typed_blocked(tmp_path) -> None:
    status = legacy_loader_status("a2-bert", tmp_path / "missing")

    assert status["status"] == "blocked"
    assert status["reason"] == "checkpoint_missing"


def test_lightgbm_selects_on_validation_and_reports_test_once() -> None:
    class FakeModel:
        def __init__(self, **config):
            self.config = config

        def fit(self, features, labels):
            return self

        def predict(self, features):
            direction = 1.0 if self.config["max_depth"] == 5 else -1.0
            return [direction * row[0] for row in features]

    rows = {
        "train": [
            {"prompt": "a", "tool_schema": "[]", "history": [], "true_length": 1},
            {"prompt": "aaaa", "tool_schema": "[]", "history": [], "true_length": 4},
        ],
        "validation": [
            {"prompt": "a", "tool_schema": "[]", "history": [], "true_length": 1},
            {"prompt": "aaa", "tool_schema": "[]", "history": [], "true_length": 3},
        ],
        "test": [
            {"prompt": "aa", "tool_schema": "[]", "history": [], "true_length": 2},
            {"prompt": "aaaaa", "tool_schema": "[]", "history": [], "true_length": 5},
        ],
    }

    report, _ = run_lightgbm_grid(rows, model_factory=FakeModel, seed=42)

    assert len(report["search_results"]) == 20
    assert report["selection_split"] == "validation"
    assert report["best_config"]["max_depth"] == 5
    assert report["test_evaluations"] == 1
    assert report["test_tau_b"] == 1.0


def test_read_json_records_tolerates_unicode_line_separators(tmp_path):
    from ltr_training.offline_io import read_json_records

    # U+2028 inside a JSON string value must NOT split the record
    path = tmp_path / "u2028.jsonl"
    path.write_text('{"text": "a b", "n": 1}\n{"text": "c", "n": 2}\n', encoding="utf-8")
    rows = read_json_records(path)
    assert len(rows) == 2
    assert rows[0]["text"] == "a b"
    assert rows[1]["n"] == 2
