from __future__ import annotations

import json
import shutil
from pathlib import Path

from .tier2 import latest_results, read_jsonl
from .train_ranker import TrainingExample, train
from .training_matrix import MATRIX_SEEDS, MATRIX_VARIANTS, structural_features


def load_tier2_split_examples(
    *,
    sample_path: Path,
    ledger_path: Path,
    train_pool_limit: int | None = None,
) -> tuple[dict[str, list[TrainingExample]], dict[str, dict[str, int]]]:
    results = {
        str(row["sample_id"]): row for row in latest_results(read_jsonl(ledger_path))
    }
    splits: dict[str, list[TrainingExample]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    counts = {
        split: {"pool": 0, "eligible": 0, "censored": 0, "failed_or_missing": 0}
        for split in splits
    }
    for source in read_jsonl(sample_path):
        split = str(source["tier2_split"])
        if split not in splits:
            raise ValueError(f"unsupported Tier-2 split: {split}")
        if split == "train" and train_pool_limit is not None and counts[split]["pool"] >= train_pool_limit:
            continue
        counts[split]["pool"] += 1
        label = results.get(str(source["sample_id"]))
        if not label or label.get("status") != "ok":
            counts[split]["failed_or_missing"] += 1
            continue
        if label.get("censored"):
            counts[split]["censored"] += 1
            continue
        splits[split].append(
            TrainingExample(
                sample_id=str(source["sample_id"]),
                session_id=str(source["session_id"]),
                prompt=str(source["prompt"]),
                output_length=int(label["output_length"]),
                generator_id=str(label["generator_id"]),
                tool_schema=str(source.get("tool_schema", "")),
                history=tuple(tuple(map(str, item)) for item in source.get("history", [])),
            )
        )
        counts[split]["eligible"] += 1
    return splits, counts


def build_tier2_run_specs(tier1_results_dir: Path) -> list[dict[str, object]]:
    return [
        {
            "run_name": f"bert-{variant}-tier2-seed{seed}",
            "variant": variant,
            "seed": seed,
            "initial_model_path": (
                tier1_results_dir / f"bert-{variant}-tier1-seed{seed}" / "final"
            ),
        }
        for variant in MATRIX_VARIANTS
        for seed in MATRIX_SEEDS
    ]


def train_lightgbm_tier2(
    *,
    splits: dict[str, list[TrainingExample]],
    output_dir: Path,
    seed: int = 42,
) -> dict[str, object]:
    from lightgbm import LGBMRegressor
    from scipy.stats import kendalltau

    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=seed,
        verbosity=-1,
    )
    model.fit(
        [structural_features(item) for item in splits["train"]],
        [item.output_length for item in splits["train"]],
    )
    metrics: dict[str, object] = {
        "run_name": "lightgbm-structural-tier2-seed42",
        "variant": "structural_features",
        "seed": seed,
        "train_examples": len(splits["train"]),
    }
    for split in ("validation", "test"):
        predictions = model.predict([structural_features(item) for item in splits[split]])
        tau = kendalltau(predictions, [item.output_length for item in splits[split]])
        metrics[f"{split}_tau"] = float(tau.statistic)
        metrics[f"{split}_pvalue"] = float(tau.pvalue)
        metrics[f"{split}_examples"] = len(splits[split])
    output_dir.mkdir(parents=True, exist_ok=False)
    model.booster_.save_model(str(output_dir / "model.txt"))
    (output_dir / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    return metrics


def _load_tier1_config(config_dir: Path, *, variant: str, seed: int) -> dict[str, object]:
    path = config_dir / f"bert-{variant}-tier1-seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing Tier-1 config: {path}")
    return json.loads(path.read_text())


def run_tier2_matrix(
    *,
    sample_path: Path,
    ledger_path: Path,
    tier1_results_dir: Path,
    tier1_config_dir: Path,
    work_dir: Path,
    results_dir: Path,
) -> dict[str, object]:
    import torch

    splits, exclusion_counts = load_tier2_split_examples(
        sample_path=sample_path, ledger_path=ledger_path
    )
    matrix_dir = results_dir / "tier2-matrix"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "tier2-matrix-summary.json"
    runs: list[dict[str, object]] = []
    for spec in build_tier2_run_specs(tier1_results_dir):
        run_name = str(spec["run_name"])
        result_final = matrix_dir / run_name / "final"
        metrics_path = result_final / "validation_metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
        else:
            initial_model_path = Path(spec["initial_model_path"])
            if not initial_model_path.exists():
                raise FileNotFoundError(f"missing corresponding Tier-1 checkpoint: {initial_model_path}")
            config = _load_tier1_config(
                tier1_config_dir,
                variant=str(spec["variant"]),
                seed=int(spec["seed"]),
            )
            config.update(
                {
                    "run_name": run_name,
                    "label_tier": "tier2-qwen3.5-9b",
                    "save_steps": 200,
                }
            )
            run_work = work_dir / run_name
            metrics = train(
                config=config,
                labels_path=sample_path,
                output_dir=run_work,
                split_examples=splits,
                initial_model_path=initial_model_path,
            )
            result_final.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(run_work / "final", result_final)
        run_record = {"run_name": run_name, **metrics, "censor_exclusion_counts": exclusion_counts}
        runs.append(run_record)
        summary_path.write_text(json.dumps({
            "expected_runs": 10,
            "completed_runs": len(runs),
            "censor_exclusion_counts": exclusion_counts,
            "runs": runs,
        }, indent=2, sort_keys=True) + "\n")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    lightgbm_name = "lightgbm-structural-tier2-seed42"
    lightgbm_final = matrix_dir / lightgbm_name / "final"
    lightgbm_metrics_path = lightgbm_final / "validation_metrics.json"
    if lightgbm_metrics_path.exists():
        metrics = json.loads(lightgbm_metrics_path.read_text())
    else:
        metrics = train_lightgbm_tier2(splits=splits, output_dir=lightgbm_final)
    runs.append({**metrics, "censor_exclusion_counts": exclusion_counts})
    summary = {
        "expected_runs": 10,
        "completed_runs": len(runs),
        "censor_exclusion_counts": exclusion_counts,
        "runs": runs,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def run_nested_learning_curve(
    *,
    sample_path: Path,
    ledger_path: Path,
    tier1_checkpoint: Path,
    tier1_config_path: Path,
    work_dir: Path,
    results_dir: Path,
    summary_path: Path,
) -> dict[str, object]:
    import torch

    points: list[dict[str, object]] = []
    for pool_size in (500, 1000, 2000, 4000):
        splits, exclusion_counts = load_tier2_split_examples(
            sample_path=sample_path,
            ledger_path=ledger_path,
            train_pool_limit=pool_size,
        )
        run_name = f"bert-full_context-tier2-seed42-pool{pool_size}"
        result_final = results_dir / run_name / "final"
        metrics_path = result_final / "validation_metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
        else:
            config = json.loads(tier1_config_path.read_text())
            config.update(
                {
                    "run_name": run_name,
                    "variant": "full_context",
                    "seed": 42,
                    "label_tier": "tier2-qwen3.5-9b",
                    "save_steps": 200,
                }
            )
            run_work = work_dir / run_name
            metrics = train(
                config=config,
                labels_path=sample_path,
                output_dir=run_work,
                split_examples=splits,
                initial_model_path=tier1_checkpoint,
            )
            result_final.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(run_work / "final", result_final)
        points.append(
            {
                "train_pool_size": pool_size,
                "effective_train_examples": exclusion_counts["train"]["eligible"],
                "censor_exclusion_counts": exclusion_counts,
                **metrics,
            }
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps({
            "variant": "full_context",
            "seed": 42,
            "nested_train_pool_sizes": [500, 1000, 2000, 4000],
            "no_new_labels": True,
            "completed_points": len(points),
            "points": points,
        }, indent=2, sort_keys=True) + "\n")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return json.loads(summary_path.read_text())
