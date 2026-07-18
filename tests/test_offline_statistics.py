from ltr_training.offline_statistics import (
    canonical_schema_overlap_report,
    cluster_bootstrap_tau_b,
    kendall_tau_b,
    tie_proportions,
)


def test_tau_b_and_seeded_cluster_bootstrap_are_reported() -> None:
    rows = [
        {"cluster": "a", "truth": 1, "prediction": 1},
        {"cluster": "a", "truth": 2, "prediction": 2},
        {"cluster": "b", "truth": 3, "prediction": 3},
        {"cluster": "b", "truth": 4, "prediction": 4},
    ]

    report = cluster_bootstrap_tau_b(
        rows,
        truth_key="truth",
        prediction_key="prediction",
        cluster_key="cluster",
        iterations=1000,
        seed=73,
    )

    assert kendall_tau_b([1, 2, 3], [1, 2, 3]) == 1.0
    assert report["variant"] == "b"
    assert report["iterations"] == 1000
    assert report["seed"] == 73
    assert report["ci95_percentile"] == [1.0, 1.0]


def test_true_and_prediction_ties_are_separate() -> None:
    report = tie_proportions([1, 1, 2], [1.0, 2.0, 2.0])

    assert report == {
        "true_length_tie_proportion": 1 / 3,
        "prediction_tie_proportion": 1 / 3,
    }


def test_schema_overlap_reports_three_requested_comparisons() -> None:
    train = ["{\"b\":2,\"a\":1}", "{\"name\":\"train\"}"]
    validation = ["{\"a\":1,\"b\":2}"]
    test = ["{\"name\":\"test\"}"]
    ood = ["{\"name\":\"train\"}"]

    report = canonical_schema_overlap_report(train, validation, test, ood)

    assert report["train_validation"]["intersection_count"] == 1
    assert report["train_test"]["intersection_count"] == 0
    assert report["toolace_ood"]["intersection_count"] == 1
