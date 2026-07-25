from __future__ import annotations

import json
import shutil
from pathlib import Path

from .train_ranker import TrainingExample, load_tier1_examples, split_by_session, train


MATRIX_VARIANTS = ("prompt_only", "prompt_schema", "full_context")
MATRIX_SEEDS = (17, 42, 73)


def build_bert_run_configs(base: dict[str, object]) -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    for variant in MATRIX_VARIANTS:
        for seed in MATRIX_SEEDS:
            configs.append(
                {
                    **base,
                    "run_name": f"bert-{variant}-tier1-seed{seed}",
                    "variant": variant,
                    "seed": seed,
                    "save_steps": 200,
                }
            )
    return configs


def structural_features(example: TrainingExample) -> list[float]:
    return [
        float(len(example.prompt)),
        float(len(example.prompt.split())),
        float(len(example.tool_schema)),
        float(example.tool_schema.count('"name"')),
        float(len(example.history)),
    ]


def train_lightgbm_baseline(
    *, labels_path: Path, output_dir: Path, seed: int = 42
) -> dict[str, object]:
    from lightgbm import LGBMRegressor
    from scipy.stats import kendalltau

    examples = load_tier1_examples(
        labels_path, sources={"toolace"}, include_context=True
    )
    train_examples, validation_examples = split_by_session(examples)
    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=seed,
        verbosity=-1,
    )
    model.fit(
        [structural_features(item) for item in train_examples],
        [item.output_length for item in train_examples],
    )
    predictions = model.predict(
        [structural_features(item) for item in validation_examples]
    )
    tau = kendalltau(predictions, [item.output_length for item in validation_examples])
    output_dir.mkdir(parents=True, exist_ok=False)
    model.booster_.save_model(str(output_dir / "model.txt"))
    metrics = {
        "run_name": "lightgbm-structural-tier1-seed42",
        "variant": "structural_features",
        "seed": seed,
        "validation_tau": float(tau.statistic),
        "validation_pvalue": float(tau.pvalue),
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
    }
    (output_dir / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    return metrics


def run_matrix(
    *,
    base_config: dict[str, object],
    labels_path: Path,
    config_dir: Path,
    work_dir: Path,
    results_dir: Path,
) -> dict[str, object]:
    import torch

    config_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    matrix_results = results_dir / "tier1-matrix"
    result_configs = results_dir / "configs"
    matrix_results.mkdir(parents=True, exist_ok=True)
    result_configs.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "tier1-matrix-summary.json"
    runs: list[dict[str, object]] = []

    for config in build_bert_run_configs(base_config):
        run_name = str(config["run_name"])
        config_path = config_dir / f"{run_name}.json"
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        shutil.copy2(config_path, result_configs / config_path.name)
        result_run = matrix_results / run_name
        metrics_path = result_run / "final" / "validation_metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
        else:
            run_work = work_dir / run_name
            metrics = train(
                config=config,
                labels_path=labels_path,
                output_dir=run_work,
            )
            result_run.mkdir(parents=True, exist_ok=True)
            shutil.copytree(run_work / "final", result_run / "final")
        runs.append({"run_name": run_name, **metrics})
        summary_path.write_text(
            json.dumps({"expected_runs": 10, "completed_runs": len(runs), "runs": runs}, indent=2, sort_keys=True) + "\n"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    lightgbm_name = "lightgbm-structural-tier1-seed42"
    lightgbm_dir = matrix_results / lightgbm_name / "final"
    lightgbm_metrics = lightgbm_dir / "validation_metrics.json"
    if lightgbm_metrics.exists():
        metrics = json.loads(lightgbm_metrics.read_text())
    else:
        metrics = train_lightgbm_baseline(
            labels_path=labels_path,
            output_dir=lightgbm_dir,
        )
    runs.append(metrics)
    summary = {"expected_runs": 10, "completed_runs": len(runs), "runs": runs}
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
