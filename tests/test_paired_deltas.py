import json

from scripts.analyze_paired_deltas import compute_paired_differences, main


def result(policy: str, values: list[float]) -> dict[str, object]:
    return {
        "schema_version": 2,
        "valid": True,
        "policy": policy,
        "model": "model",
        "workload_sha256": "abc",
        "capacity_rps": 10.0,
        "seed_derivation": "sha256(profile,load_pct,repeat)",
        "warmup_requested": {"count": 1, "ratio": None},
        "scenarios": [
            {
                "scenario": {"name": "saturation-90", "saturation": 0.9},
                "load_pct": 90,
                "profile": "mixed",
                "runs": [
                    {
                        "repeat": index,
                        "seed": index * 10,
                        "metrics": {"p99_ttlt_ms": value},
                    }
                    for index, value in enumerate(values, start=1)
                ],
            }
        ],
    }


def test_paired_differences_match_repeat_and_seed_and_report_scatter() -> None:
    output = compute_paired_differences(
        result("fcfs", [100.0, 120.0, 140.0]),
        result("ltr_aging", [90.0, 100.0, 150.0]),
        metrics=["p99_ttlt_ms"],
    )

    metric = output["groups"][0]["metrics"]["p99_ttlt_ms"]
    assert metric == {
        "values": [-10.0, -20.0, 10.0],
        "mean": -6.666667,
        "min": -20.0,
        "max": 10.0,
    }
    assert output["direction"] == "candidate_minus_baseline"


def test_paired_delta_cli_writes_json(tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "deltas.json"
    baseline.write_text(json.dumps(result("fcfs", [100.0])))
    candidate.write_text(json.dumps(result("ltr_aging", [90.0])))

    exit_code = main(
        [
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--metric",
            "p99_ttlt_ms",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert json.loads(output.read_text())["groups"][0]["pair_count"] == 1
