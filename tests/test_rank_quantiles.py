import hashlib
import importlib
import json


def write_training_labels(path, count=6_000):
    with path.open("w", encoding="utf-8") as handle:
        for index in range(1, count + 1):
            handle.write(
                json.dumps(
                    {
                        "sample_id": f"toolace-{index:06d}:0000",
                        "prompt": f"prompt {index}",
                        "tool_schema": f"schema {index}",
                        "output_length": index,
                    }
                )
                + "\n"
            )


def test_builder_writes_nearest_rank_manifest_and_sidecar(tmp_path):
    rank_quantiles = importlib.import_module("scheduler_benchmark.rank_quantiles")
    labels = tmp_path / "labels.jsonl"
    sidecar = tmp_path / "sidecar.jsonl"
    manifest = tmp_path / "manifest.json"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"checkpoint")
    write_training_labels(labels)

    built = rank_quantiles.build_rank_quantile_artifacts(
        labels_path=labels,
        checkpoint=checkpoint,
        sidecar_path=sidecar,
        manifest_path=manifest,
        model_version="bert-prompt_schema-tier2-seed17",
        expected_count=6_000,
    )

    assert built["sample_count"] == 6_000
    assert built["percentiles"]["10"] == 600
    assert built["percentiles"]["99"] == 5_940
    assert built["global_quantiles"] == {"50": 3_000, "70": 4_200, "90": 5_400}
    assert built["source_sha256"] == hashlib.sha256(labels.read_bytes()).hexdigest()
    assert built["checkpoint_sha256"] == hashlib.sha256(b"checkpoint").hexdigest()
    assert built["approximation_notice"] == rank_quantiles.APPROXIMATION_NOTICE
    rows = [json.loads(line) for line in sidecar.read_text().splitlines()]
    assert rows[0] == {
        "request_id": "toolace-000001:0000",
        "prompt_text": "prompt 1",
        "tool_schema_text": "schema 1",
        "output_length": 1,
        "split": "train",
    }
    stored = rank_quantiles.ReplayStore.from_path(sidecar).get(
        "toolace-000001:0000"
    )
    assert stored.prompt_text == "prompt 1"
    assert stored.tool_schema_text == "schema 1"


def test_mapper_clamps_interpolates_ratios_and_signals(tmp_path):
    rank_quantiles = importlib.import_module("scheduler_benchmark.rank_quantiles")
    labels = tmp_path / "labels.jsonl"
    sidecar = tmp_path / "sidecar.jsonl"
    manifest = tmp_path / "manifest.json"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"checkpoint")
    write_training_labels(labels)
    rank_quantiles.build_rank_quantile_artifacts(
        labels_path=labels,
        checkpoint=checkpoint,
        sidecar_path=sidecar,
        manifest_path=manifest,
        model_version="bert-prompt_schema-tier2-seed17",
        expected_count=6_000,
    )

    mapper = rank_quantiles.RankQuantileMapper.from_path(manifest)

    prediction = mapper.map_score(0.105)
    assert prediction.quantiles == {50: 630.0, 70: 882.0, 90: 1_134.0}
    assert prediction.signals == {
        "quantile_spread": 2_400.0,
        "ood_distance": 0.0,
        "feature_coverage": 1.0,
        "rank_score": 0.105,
    }
    assert mapper.map_score(0.0).quantiles[50] == 600.0
    assert mapper.map_score(1.0).quantiles[50] == 5_940.0
