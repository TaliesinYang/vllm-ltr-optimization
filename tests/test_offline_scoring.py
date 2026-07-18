import hashlib

from ltr_training.offline_scoring import (
    checkpoint_sha256,
    disagreement_diagnostic,
    percentile_ranks,
    render_prompt_schema,
    score_ensemble_rows,
)


class FakeScorer:
    def __init__(self, scores, token_lengths):
        self.scores = scores
        self.token_lengths = token_lengths

    def score(self, texts):
        return list(self.scores), list(self.token_lengths)


def test_checkpoint_hash_covers_relative_names_and_contents(tmp_path) -> None:
    (tmp_path / "config.json").write_text("config")
    (tmp_path / "model.safetensors").write_bytes(b"weights")

    expected = hashlib.sha256(
        b"config.json\0config\0model.safetensors\0weights\0"
    ).hexdigest()

    assert checkpoint_sha256(tmp_path) == expected


def test_percentiles_use_average_ranks_for_ties_within_domain() -> None:
    assert percentile_ranks([10.0, 20.0, 20.0, 40.0]) == [0.0, 0.5, 0.5, 1.0]


def test_ensemble_outputs_seed_scores_dispersion_and_truncation() -> None:
    rows = [
        {
            "sample_id": "a",
            "request_id": "a",
            "prompt": "p1",
            "tool_schema": "[]",
            "session_id": "s1",
            "domain": "id",
            "true_length": 10,
        },
        {
            "sample_id": "b",
            "request_id": "b",
            "prompt": "p2",
            "tool_schema": "[]",
            "session_id": "s2",
            "domain": "id",
            "true_length": 20,
        },
    ]
    scorers = {
        17: FakeScorer([1.0, 2.0], [100, 600]),
        42: FakeScorer([2.0, 1.0], [100, 600]),
        73: FakeScorer([1.0, 2.0], [100, 600]),
    }

    scored, report = score_ensemble_rows(rows, scorers=scorers, max_length=512)

    assert scored[0]["request_id"] == "a"
    assert set(scored[0]) >= {
        "score_seed17",
        "score_seed42",
        "score_seed73",
        "rank_dispersion",
    }
    assert report["domains"]["id"]["truncation_ratio"] == 0.5
    assert report["checkpoint_sha256"] == {}
    assert render_prompt_schema("p", "s") == "[USER]\np\n[TOOLS]\ns"


def test_diagnostic_is_not_named_calibration_and_risk_is_one_minus_tau_b() -> None:
    rows = [
        {"domain": "id", "true_length": 10, "ensemble_rank": 0.0, "rank_dispersion": 0.0},
        {"domain": "id", "true_length": 20, "ensemble_rank": 0.5, "rank_dispersion": 0.1},
        {"domain": "id", "true_length": 30, "ensemble_rank": 1.0, "rank_dispersion": 0.2},
    ]

    report = disagreement_diagnostic(rows, coverages=(1.0,))

    assert report["diagnostic_name"] == "disagreement-empirical-error diagnostic"
    assert report["domains"]["id"][0]["risk"] == 0.0
    assert "calibration" not in report
