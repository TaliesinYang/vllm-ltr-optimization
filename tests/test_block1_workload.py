import gzip
import json
from pathlib import Path

import pytest

from ltr_training.block1_workload import (
    DEFAULT_PER_TOKEN_MS,
    SOURCE_SYNTHETIC,
    SOURCE_TRACE,
    STRATA,
    WORKLOAD_FIELDS,
    baseline_service_ms,
    build_clients,
    build_manifest,
    generate_requests,
    measure_trace,
    percentile,
    render_tool_schema,
    trace_rows,
)
from ltr_training.train_ranker import TrainingExample
from scheduler_benchmark.tool_vocabulary import tool_names

REPO = Path(__file__).resolve().parents[1]
REAL_TRACE = REPO / "probes" / "agent-traces-2026-07-26" / "agent_trace_vanilla.jsonl.gz"


def synthetic_trace(path: Path, *, rows: int = 60, zero_tool_every: int = 3) -> Path:
    """A trace with known marginals, so calibration can be checked exactly."""
    payload = []
    for index in range(rows):
        zero_tool = index % zero_tool_every == 0
        payload.append(
            {
                "request_id": f"r{index}",
                "status": 200,
                "e2e_ms": 100.0 + index,
                "usage": {"completion_tokens": 10 + index, "prompt_tokens": 500},
                "body": {
                    "messages": [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": f"hello {index}"},
                    ],
                    "tools": []
                    if zero_tool
                    else [{"type": "function", "function": {"name": "alpha"}}],
                },
            }
        )
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in payload:
            handle.write(json.dumps(row) + "\n")
    return path


def train_pool() -> list[TrainingExample]:
    def example(index: int, names: list[str]) -> TrainingExample:
        return TrainingExample(
            sample_id=f"t{index}",
            session_id=f"s{index}",
            prompt="p",
            output_length=10,
            generator_id="g",
            tool_schema=render_tool_schema(names),
        )

    return [
        example(0, ["alpha", "beta"]),
        example(1, ["gamma"]),
        example(2, ["delta", "epsilon"]),
        example(3, ["zeta"]),
    ]


@pytest.fixture
def calibration(tmp_path):
    return measure_trace(synthetic_trace(tmp_path / "trace.jsonl.gz"))


@pytest.fixture
def clients():
    return build_clients(
        train_examples=train_pool(), per_stratum=2, seed=17, tool_names_of=tool_names
    )


def test_calibration_is_measured_from_the_trace_not_assumed(tmp_path) -> None:
    result = measure_trace(synthetic_trace(tmp_path / "t.jsonl.gz", rows=60))

    assert result.request_count == 60
    # Every third row is zero-tool by construction.
    assert result.zero_tool_fraction == pytest.approx(20 / 60)
    assert result.completion_p50 == pytest.approx(percentile(range(10, 70), 0.50))
    assert result.turn_depth_p50 == 2
    assert result.trace_sha256


def test_real_trace_reproduces_the_documented_characterisation() -> None:
    """The committed manifest claims 75 requests, 33% zero-tool, p50 42, p99 328."""
    if not REAL_TRACE.exists():
        pytest.skip("real agent trace not present")
    result = measure_trace(REAL_TRACE)

    assert result.request_count == 75
    assert result.zero_tool_fraction == pytest.approx(25 / 75, abs=0.01)
    assert result.completion_p50 == pytest.approx(42, abs=1)
    assert result.completion_p99 == pytest.approx(328, abs=1)
    assert result.turn_depth_p50 == pytest.approx(3, abs=1)


def test_every_cold_start_stratum_is_represented(clients) -> None:
    present = {client.stratum for client in clients}

    assert present == set(STRATA)


def test_each_client_keeps_one_constant_tool_schema(calibration, clients) -> None:
    """Real clients do not change schema between turns; nor may synthetic ones."""
    rows = generate_requests(
        calibration=calibration, clients=clients, request_count=200, seed=5
    )
    by_client: dict[str, set[str]] = {}
    for row in rows:
        if not row["tool_schema"]:
            continue  # zero-tool utility call, carries no schema by design
        by_client.setdefault(str(row["client_id"]), set()).add(str(row["tool_schema"]))

    assert by_client
    assert all(len(schemas) == 1 for schemas in by_client.values())


def test_generated_rows_match_the_run_matrix_workload_schema(
    calibration, clients
) -> None:
    rows = generate_requests(
        calibration=calibration, clients=clients, request_count=20, seed=5
    )

    for row in rows:
        assert set(WORKLOAD_FIELDS) <= set(row)
        assert isinstance(row["true_length"], int)
        assert row["source"] == SOURCE_SYNTHETIC


def test_zero_tool_share_tracks_the_trace(calibration, clients) -> None:
    rows = generate_requests(
        calibration=calibration, clients=clients, request_count=2000, seed=5
    )
    observed = sum(1 for row in rows if not row["tool_schema"]) / len(rows)

    # Sanity band, not an exact match - this is a sampled marginal.
    assert observed == pytest.approx(calibration.zero_tool_fraction, abs=0.05)


def test_completion_length_marginals_track_the_trace(calibration, clients) -> None:
    rows = generate_requests(
        calibration=calibration, clients=clients, request_count=2000, seed=5
    )
    lengths = [int(row["true_length"]) for row in rows]

    assert percentile(lengths, 0.50) == pytest.approx(calibration.completion_p50, rel=0.2)
    assert percentile(lengths, 0.99) == pytest.approx(calibration.completion_p99, rel=0.2)


def test_turn_depth_marginal_tracks_the_trace(calibration, clients) -> None:
    rows = generate_requests(
        calibration=calibration, clients=clients, request_count=2000, seed=5
    )
    depths = [int(row["turn_depth"]) for row in rows]

    assert percentile(depths, 0.50) == pytest.approx(
        calibration.turn_depth_p50, abs=1
    )


def test_generation_is_deterministic_under_a_seed(calibration, clients) -> None:
    first = generate_requests(
        calibration=calibration, clients=clients, request_count=100, seed=11
    )
    second = generate_requests(
        calibration=calibration, clients=clients, request_count=100, seed=11
    )
    third = generate_requests(
        calibration=calibration, clients=clients, request_count=100, seed=12
    )

    assert first == second
    assert first != third


def test_client_construction_is_deterministic_under_a_seed() -> None:
    left = build_clients(
        train_examples=train_pool(), per_stratum=3, seed=7, tool_names_of=tool_names
    )
    right = build_clients(
        train_examples=train_pool(), per_stratum=3, seed=7, tool_names_of=tool_names
    )

    assert left == right


def test_real_traces_are_emitted_and_marked_as_not_synthetic(calibration) -> None:
    rows = trace_rows(calibration)

    assert rows
    assert all(row["source"] == SOURCE_TRACE for row in rows)
    assert all(row["synthetic"] is False for row in rows)
    assert all(set(WORKLOAD_FIELDS) <= set(row) for row in rows)


def test_manifest_records_every_parameter_and_its_source(calibration, clients) -> None:
    synthetic = generate_requests(
        calibration=calibration, clients=clients, request_count=200, seed=5
    )
    traces = trace_rows(calibration)
    manifest = build_manifest(
        calibration=calibration,
        clients=clients,
        synthetic=synthetic,
        traces=traces,
        seed=5,
    )

    assert manifest["calibration"]["source_measurement"].endswith(".jsonl.gz")
    assert manifest["calibration"]["trace_sha256"]
    for key in (
        "zero_tool_fraction",
        "completion_tokens_p50",
        "completion_tokens_p99",
        "turn_depth_p50",
    ):
        assert key in manifest["calibration"]
    # The synthetic multi-tenancy must not masquerade as measured.
    assert "NOT trace-derived" in manifest["multi_tenancy"]["note"]
    assert manifest["rows"]["total"] == len(synthetic) + len(traces)
    assert set(manifest["multi_tenancy"]["clients_per_stratum"]) == set(STRATA)


def test_stratum_assignment_survives_the_real_vocabulary_parser(clients) -> None:
    """A client's rendered schema must parse back to the tool set it was built from."""
    for client in clients:
        assert tool_names(client.tool_schema) == tuple(sorted(client.tool_names))


def test_every_row_has_a_positive_baseline_service_ms(calibration, clients) -> None:
    """runner.py rejects the whole workload on the first non-positive value."""
    rows = generate_requests(
        calibration=calibration, clients=clients, request_count=500, seed=5
    ) + trace_rows(calibration)

    assert rows
    for row in rows:
        assert isinstance(row["baseline_service_ms"], float)
        assert row["baseline_service_ms"] > 0.0


def test_baseline_service_ms_uses_the_legacy_proxy_formula(calibration, clients) -> None:
    """Same formula and constant as workload_builder, so slowdown is comparable."""
    rows = generate_requests(
        calibration=calibration, clients=clients, request_count=50, seed=5
    )

    for row in rows:
        assert row["baseline_service_ms"] == pytest.approx(
            round(int(row["true_length"]) * DEFAULT_PER_TOKEN_MS, 6)
        )


def test_trace_rows_use_the_proxy_not_measured_wall_clock(calibration) -> None:
    """e2e_ms is full-chain wall clock; using it would put the two subsets on
    different scales and silently corrupt slowdown comparisons."""
    rows = trace_rows(calibration)

    assert rows
    for row in rows:
        assert row["baseline_service_ms"] == pytest.approx(
            round(int(row["true_length"]) * DEFAULT_PER_TOKEN_MS, 6)
        )
        # The measured value is preserved, just not used as the baseline.
        assert "trace_e2e_ms" in row


def test_per_token_ms_is_configurable(calibration, clients) -> None:
    rows = generate_requests(
        calibration=calibration,
        clients=clients,
        request_count=20,
        seed=5,
        per_token_ms=4.0,
    )

    for row in rows:
        assert row["baseline_service_ms"] == pytest.approx(
            round(int(row["true_length"]) * 4.0, 6)
        )


def test_baseline_helper_rejects_non_positive_inputs() -> None:
    with pytest.raises(ValueError, match="true_length"):
        baseline_service_ms(0, DEFAULT_PER_TOKEN_MS)
    with pytest.raises(ValueError, match="per_token_ms"):
        baseline_service_ms(10, 0.0)


def test_manifest_documents_the_baseline_formula(calibration, clients) -> None:
    synthetic = generate_requests(
        calibration=calibration, clients=clients, request_count=50, seed=5
    )
    manifest = build_manifest(
        calibration=calibration,
        clients=clients,
        synthetic=synthetic,
        traces=trace_rows(calibration),
        seed=5,
    )

    baseline = manifest["baseline_service_ms"]
    assert baseline["formula"] == "true_length * per_token_ms"
    assert baseline["per_token_ms"] == DEFAULT_PER_TOKEN_MS
    assert "proxy" in baseline["claim"]
    assert "NOT trace-derived" in baseline["note"]
