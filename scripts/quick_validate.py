#!/usr/bin/env python
"""Tiny unified validation suite for the full U-CaGNN experiment pipeline.

This script is the single post-change validation entry point for the repository.
It runs tiny-scale experiments that cover the canonical recipe matrix, ablation
variants, observability paths, and evaluation modes so code changes are checked
against the full experiment surface before longer formal runs.
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

import torch

from experiments.ablation_configs import ABLATION_VARIANTS, make_ablation_config
from experiments.recipes import get_recipe, load_experiment_catalog
from experiments.run_experiment import MLFLOW_DB_PATH, build_config, run_experiment
from src.data.feature_policy import supports_feature_utility
from src.utils.cli_parsers import build_quick_validate_parser
from src.utils.config import DEFAULT_SEED, UCaGNNConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
THESIS_DB_PATH = REPO_ROOT / "results" / "thesis_experiments.db"
RUNTIME_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_EVAL_MODES = [
    "default",
    "interest_only",
    "conformity_suppressed",
]
_TINY_DATASET_LIMITS = {
    "amazonbook": 100,
    "movielens1m": 100,
    "movielens20m": 100,
    "kuairec_v2": 100,
    "taobao": 100,
    "kuairand1k": 100,
}


def _build_runtime_config(
    args: argparse.Namespace,
    dataset: str,
    *,
    recipe: str | None = None,
    preset: str | None = None,
    eval_scoring_mode: str | None = None,
    scoring_weight_mode: str | None = None,
    use_features: bool | None = None,
    feature_policy: str | None = None,
    graph_method: str | None = None,
    num_neighbors: list[int] | None = None,
    sample_interactions: int | None = None,
    loader_max_rows: int | None = None,
    patience: int | None = None,
    enable_profiling: bool | None = None,
    profiling_cadence: int | None = None,
) -> UCaGNNConfig:
    """Build a quick-validate config without emulating a full CLI namespace.

    Args:
        args: Parsed quick-validate CLI arguments.
        dataset: Dataset name for the run.
        recipe: Optional named recipe.
        preset: Optional preset override.
        eval_scoring_mode: Optional evaluation-time scoring mode.
        scoring_weight_mode: Optional score-mixture mode.
        use_features: Optional feature toggle override.
        feature_policy: Optional feature policy override.
        graph_method: Optional graph method override.
        num_neighbors: Optional fan-out override.
        sample_interactions: Optional tiny-run interaction budget.
        loader_max_rows: Optional loader row cap.
        patience: Optional early-stopping patience override.
        enable_profiling: Optional profiling toggle.
        profiling_cadence: Optional profiling cadence override.

    Returns:
        A validated runtime config tailored for quick validation.
    """
    config_inputs: dict[str, object] = {
        "dataset": dataset,
        "recipe": recipe,
        "preset": preset,
        "seed": DEFAULT_SEED,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_scoring_mode": eval_scoring_mode,
        "scoring_weight_mode": scoring_weight_mode,
        "use_features": use_features,
        "feature_policy": feature_policy,
        "graph_method": graph_method,
        "num_neighbors": num_neighbors,
        "sample_interactions": sample_interactions,
        "loader_max_rows": loader_max_rows,
        "device": RUNTIME_DEVICE,
        "data_dir": args.data_dir,
    }
    config = build_config(config_inputs)
    # Disable torch.compile for smoke tests: 1-epoch runs don't benefit
    # and running many configs in one process hits the recompile limit.
    config.use_torch_compile = False
    if patience is not None:
        config.patience = patience
    if enable_profiling is not None:
        config.enable_profiling = enable_profiling
    if profiling_cadence is not None:
        config.profiling_cadence = profiling_cadence
    return config


def _build_tiny_recipe_config(
    args: argparse.Namespace,
    dataset: str,
    *,
    recipe: str,
    eval_scoring_mode: str | None = None,
    use_features: bool | None = None,
    enable_profiling: bool = False,
) -> UCaGNNConfig:
    """Build a tiny validation config for a catalog recipe on one dataset."""
    dataset_limit = int(_TINY_DATASET_LIMITS.get(dataset, 100))
    return _build_runtime_config(
        args,
        dataset,
        recipe=recipe,
        eval_scoring_mode=eval_scoring_mode,
        use_features=use_features,
        sample_interactions=dataset_limit,
        loader_max_rows=dataset_limit,
        patience=1,
        enable_profiling=enable_profiling,
        profiling_cadence=1,
    )


def _canonical_recipe_names() -> list[str]:
    recipes = load_experiment_catalog().get("recipes", {})
    return sorted(name for name, spec in recipes.items() if "alias_for" not in spec)


def _select_values(
    requested: list[str] | None, available: list[str], label: str
) -> list[str]:
    if requested is None:
        return available
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown {label}: {', '.join(unknown)}")
    return [value for value in available if value in requested]


def _representative_dataset(datasets: list[str], preferred: list[str]) -> str:
    for candidate in preferred:
        if candidate in datasets:
            return candidate
    return datasets[0]


def _sqlite_count(query: str, params: tuple[object, ...]) -> int:
    with sqlite3.connect(THESIS_DB_PATH) as conn:
        row = conn.execute(query, params).fetchone()
    return int(row[0] or 0) if row is not None else 0


def _assert_experiment_logging(exp_id: int) -> None:
    experiments = _sqlite_count(
        "SELECT COUNT(*) FROM experiments WHERE id = ?", (exp_id,)
    )
    metrics = _sqlite_count(
        "SELECT COUNT(*) FROM metrics WHERE experiment_id = ?", (exp_id,)
    )
    if experiments == 0:
        raise AssertionError(
            f"SQLite experiment row missing for experiment_id={exp_id}"
        )
    if metrics == 0:
        raise AssertionError(f"SQLite metric rows missing for experiment_id={exp_id}")


def _assert_profiling_logging(exp_id: int) -> None:
    profiling_rows = _sqlite_count(
        "SELECT COUNT(*) FROM profiling WHERE experiment_id = ?", (exp_id,)
    )
    if profiling_rows == 0:
        raise AssertionError(f"Profiling rows missing for experiment_id={exp_id}")


def _assert_ranking_metrics(test_metrics: dict[str, float]) -> None:
    required_metric_names = (
        "NDCG@20",
        "Recall@20",
        "AveragePopularity@20",
        "NDCG@40",
        "Recall@40",
        "AveragePopularity@40",
    )
    for metric_name in required_metric_names:
        if metric_name not in test_metrics:
            raise AssertionError(f"Ranking metrics missing {metric_name}")


def _print_case(
    status: str,
    category: str,
    dataset: str,
    label: str,
    elapsed: float,
    detail: str = "",
) -> None:
    suffix = f" | {detail}" if detail else ""
    print(
        f"{status:<4} {category:<13} {dataset:<12} {label:<40} {elapsed:>7.2f}s{suffix}"
    )


def _run_single_case(
    *,
    category: str,
    dataset: str,
    label: str,
    config: UCaGNNConfig,
    preset: str | None,
    intervention: str,
    recipe_name: str | None = None,
    save_checkpoint: bool = False,
    enable_mlflow: bool = False,
    auto_resume: bool = False,
    checkpoint_path: str | None = None,
    expect_logging: bool = False,
    expect_profiling: bool = False,
    expect_metrics: bool = False,
    expect_mlflow_db_touch: bool = False,
) -> tuple[dict, float]:
    mlflow_mtime_before = (
        MLFLOW_DB_PATH.stat().st_mtime_ns
        if expect_mlflow_db_touch and MLFLOW_DB_PATH.exists()
        else None
    )
    started = time.perf_counter()
    result = run_experiment(
        config,
        preset=preset,
        intervention=intervention,
        save_checkpoint=save_checkpoint,
        enable_mlflow=enable_mlflow,
        mlflow_experiment_name="ucagnn-quick-validate",
        recipe_name=recipe_name,
        checkpoint_path=checkpoint_path,
        checkpoint_every=1,
        auto_resume=auto_resume,
    )
    elapsed = time.perf_counter() - started

    exp_id = int(result["exp_id"])
    if expect_logging:
        _assert_experiment_logging(exp_id)
    if expect_profiling:
        _assert_profiling_logging(exp_id)
    if expect_metrics:
        _assert_ranking_metrics(result["test_metrics"])
    if expect_mlflow_db_touch:
        mlflow_mtime_after = (
            MLFLOW_DB_PATH.stat().st_mtime_ns if MLFLOW_DB_PATH.exists() else None
        )
        if mlflow_mtime_after is None or mlflow_mtime_after == mlflow_mtime_before:
            raise AssertionError("MLflow probe did not update results/mlflow.db")

    return result, elapsed


def _run_recipe_category(args: argparse.Namespace, results: list[dict]) -> None:
    recipe_names = _select_values(
        args.recipe_names, _canonical_recipe_names(), "recipe names"
    )
    print(
        f"Recipe coverage: {len(args.datasets)} datasets x {len(recipe_names)} canonical recipes"
    )
    for dataset in args.datasets:
        for recipe_name in recipe_names:
            recipe = get_recipe(recipe_name)
            config = _build_tiny_recipe_config(args, dataset, recipe=recipe_name)
            label = f"recipe:{recipe_name}"
            try:
                _, elapsed = _run_single_case(
                    category="recipes",
                    dataset=dataset,
                    label=label,
                    config=config,
                    preset=recipe.get("preset"),
                    intervention=f"quick_recipe_{recipe_name}",
                    recipe_name=recipe_name,
                )
                results.append(
                    {
                        "status": "pass",
                        "category": "recipes",
                        "dataset": dataset,
                        "label": label,
                        "elapsed": elapsed,
                    }
                )
                _print_case("OK", "recipes", dataset, label, elapsed)
            except Exception as exc:
                elapsed = 0.0
                results.append(
                    {
                        "status": "fail",
                        "category": "recipes",
                        "dataset": dataset,
                        "label": label,
                        "elapsed": elapsed,
                        "detail": str(exc),
                    }
                )
                _print_case("FAIL", "recipes", dataset, label, elapsed, str(exc))
                if args.fail_fast:
                    return


def _run_ablation_category(args: argparse.Namespace, results: list[dict]) -> None:
    variants = _select_values(
        args.ablation_variants, sorted(ABLATION_VARIANTS), "ablation variants"
    )
    print(
        f"Ablation coverage: {len(args.datasets)} datasets x {len(variants)} variants"
    )
    for dataset in args.datasets:
        dataset_limit = int(_TINY_DATASET_LIMITS.get(dataset, 100))
        for variant in variants:
            config = make_ablation_config(
                variant,
                dataset=dataset,
                data_dir=args.data_dir,
                device=RUNTIME_DEVICE,
                epochs=args.epochs,
                batch_size=args.batch_size,
                sample_interactions=dataset_limit,
                loader_max_rows=dataset_limit,
            )
            config.use_torch_compile = False
            config.patience = 1
            config.enable_profiling = False
            config.profiling_cadence = 1
            label = f"ablation:{variant}"
            try:
                _, elapsed = _run_single_case(
                    category="ablations",
                    dataset=dataset,
                    label=label,
                    config=config,
                    preset="ucagnn",
                    intervention=f"quick_ablation_{variant}",
                )
                results.append(
                    {
                        "status": "pass",
                        "category": "ablations",
                        "dataset": dataset,
                        "label": label,
                        "elapsed": elapsed,
                    }
                )
                _print_case("OK", "ablations", dataset, label, elapsed)
            except Exception as exc:
                elapsed = 0.0
                results.append(
                    {
                        "status": "fail",
                        "category": "ablations",
                        "dataset": dataset,
                        "label": label,
                        "elapsed": elapsed,
                        "detail": str(exc),
                    }
                )
                _print_case("FAIL", "ablations", dataset, label, elapsed, str(exc))
                if args.fail_fast:
                    return


def _run_observability_category(args: argparse.Namespace, results: list[dict]) -> None:
    profiling_recipes = [
        "ucagnn",
        "ucagnn_knn",
    ]
    feature_datasets = [
        dataset for dataset in args.datasets if supports_feature_utility(dataset)
    ]
    print(
        "Observability coverage: "
        f"{len(args.datasets)} datasets x {len(profiling_recipes)} profiling probes, "
        f"{len(feature_datasets)} feature probes, "
        f"{len(args.datasets)} resume probes"
    )
    if not torch.cuda.is_available():
        for dataset in args.datasets:
            for recipe_name in profiling_recipes:
                label = f"profiling:{recipe_name}"
                results.append(
                    {
                        "status": "skip",
                        "category": "observability",
                        "dataset": dataset,
                        "label": label,
                        "elapsed": 0.0,
                        "detail": "CUDA unavailable",
                    }
                )
                _print_case(
                    "SKIP", "observability", dataset, label, 0.0, "CUDA unavailable"
                )
    else:
        for dataset in args.datasets:
            for recipe_name in profiling_recipes:
                recipe = get_recipe(recipe_name)
                config = _build_tiny_recipe_config(
                    args,
                    dataset,
                    recipe=recipe_name,
                    enable_profiling=True,
                )
                label = f"profiling:{recipe_name}"
                try:
                    _, elapsed = _run_single_case(
                        category="observability",
                        dataset=dataset,
                        label=label,
                        config=config,
                        preset=recipe.get("preset"),
                        intervention=f"quick_profile_{recipe_name}",
                        recipe_name=recipe_name,
                        expect_logging=True,
                        expect_profiling=True,
                    )
                    results.append(
                        {
                            "status": "pass",
                            "category": "observability",
                            "dataset": dataset,
                            "label": label,
                            "elapsed": elapsed,
                        }
                    )
                    _print_case("OK", "observability", dataset, label, elapsed)
                except Exception as exc:
                    results.append(
                        {
                            "status": "fail",
                            "category": "observability",
                            "dataset": dataset,
                            "label": label,
                            "elapsed": 0.0,
                            "detail": str(exc),
                        }
                    )
                    _print_case("FAIL", "observability", dataset, label, 0.0, str(exc))
                    if args.fail_fast:
                        return

    feature_recipe = "ucagnn"
    feature_label = "features:ucagnn"
    for dataset in feature_datasets:
        feature_config = _build_tiny_recipe_config(
            args,
            dataset,
            recipe=feature_recipe,
            use_features=True,
        )
        try:
            _, elapsed = _run_single_case(
                category="observability",
                dataset=dataset,
                label=feature_label,
                config=feature_config,
                preset="ucagnn",
                intervention="quick_features",
                recipe_name=feature_recipe,
                expect_logging=True,
                expect_metrics=True,
            )
            results.append(
                {
                    "status": "pass",
                    "category": "observability",
                    "dataset": dataset,
                    "label": feature_label,
                    "elapsed": elapsed,
                }
            )
            _print_case("OK", "observability", dataset, feature_label, elapsed)
        except Exception as exc:
            results.append(
                {
                    "status": "fail",
                    "category": "observability",
                    "dataset": dataset,
                    "label": feature_label,
                    "elapsed": 0.0,
                    "detail": str(exc),
                }
            )
            _print_case("FAIL", "observability", dataset, feature_label, 0.0, str(exc))
            if args.fail_fast:
                return

    resume_label = "checkpoint-resume:ucagnn"
    for dataset in args.datasets:
        checkpoint_path = (
            REPO_ROOT
            / "results"
            / "checkpoints"
            / f"quick_validate_resume_probe_{dataset}.pt"
        )
        checkpoint_path.unlink(missing_ok=True)
        resume_config = _build_tiny_recipe_config(args, dataset, recipe="ucagnn")
        try:
            _, first_elapsed = _run_single_case(
                category="observability",
                dataset=dataset,
                label=resume_label,
                config=resume_config,
                preset="ucagnn",
                intervention="quick_resume_probe",
                recipe_name="ucagnn",
                save_checkpoint=True,
                enable_mlflow=args.mlflow,
                auto_resume=False,
                checkpoint_path=str(checkpoint_path),
                expect_logging=True,
                expect_metrics=True,
                expect_mlflow_db_touch=args.mlflow,
            )
            second_started = time.perf_counter()
            resumed_result = run_experiment(
                resume_config,
                preset="ucagnn",
                intervention="quick_resume_probe",
                save_checkpoint=True,
                enable_mlflow=False,
                mlflow_experiment_name="ucagnn-quick-validate",
                recipe_name="ucagnn",
                checkpoint_path=str(checkpoint_path),
                checkpoint_every=1,
                auto_resume=True,
            )
            second_elapsed = time.perf_counter() - second_started
            if not resumed_result.get("resumed"):
                raise AssertionError("Auto-resume probe did not report resumed=True")
            results.append(
                {
                    "status": "pass",
                    "category": "observability",
                    "dataset": dataset,
                    "label": resume_label,
                    "elapsed": first_elapsed + second_elapsed,
                }
            )
            _print_case(
                "OK",
                "observability",
                dataset,
                resume_label,
                first_elapsed + second_elapsed,
            )
        except Exception as exc:
            results.append(
                {
                    "status": "fail",
                    "category": "observability",
                    "dataset": dataset,
                    "label": resume_label,
                    "elapsed": 0.0,
                    "detail": str(exc),
                }
            )
            _print_case("FAIL", "observability", dataset, resume_label, 0.0, str(exc))
            if args.fail_fast:
                return
        finally:
            checkpoint_path.unlink(missing_ok=True)


def _run_evaluation_category(args: argparse.Namespace, results: list[dict]) -> None:
    eval_dataset = _representative_dataset(
        args.datasets, ["movielens1m", "kuairec_v2", "taobao"]
    )
    for mode in DEFAULT_EVAL_MODES:
        config = _build_tiny_recipe_config(
            args,
            eval_dataset,
            recipe="ucagnn_knn",
            eval_scoring_mode=mode,
        )
        label = f"eval:{mode}"
        try:
            _, elapsed = _run_single_case(
                category="evaluation",
                dataset=eval_dataset,
                label=label,
                config=config,
                preset="ucagnn",
                intervention=f"quick_eval_{mode}",
                recipe_name="ucagnn_knn",
                expect_metrics=True,
            )
            results.append(
                {
                    "status": "pass",
                    "category": "evaluation",
                    "dataset": eval_dataset,
                    "label": label,
                    "elapsed": elapsed,
                }
            )
            _print_case("OK", "evaluation", eval_dataset, label, elapsed)
        except Exception as exc:
            results.append(
                {
                    "status": "fail",
                    "category": "evaluation",
                    "dataset": eval_dataset,
                    "label": label,
                    "elapsed": 0.0,
                    "detail": str(exc),
                }
            )
            _print_case("FAIL", "evaluation", eval_dataset, label, 0.0, str(exc))
            if args.fail_fast:
                return


def _print_summary(results: list[dict], total_elapsed: float) -> None:
    categories = sorted({row["category"] for row in results})
    failures = [row for row in results if row["status"] == "fail"]
    skips = [row for row in results if row["status"] == "skip"]

    print("=" * 78)
    print("VALIDATION SUMMARY")
    print("=" * 78)
    for category in categories:
        subset = [row for row in results if row["category"] == category]
        passed = sum(1 for row in subset if row["status"] == "pass")
        failed = sum(1 for row in subset if row["status"] == "fail")
        skipped = sum(1 for row in subset if row["status"] == "skip")
        print(
            f"{category:<13} pass={passed:<4} fail={failed:<4} skip={skipped:<4} total={len(subset)}"
        )

    if failures:
        print("-" * 78)
        print("FAILURES")
        print("-" * 78)
        for row in failures:
            print(
                f"{row['category']}: {row['dataset']} :: {row['label']} :: {row.get('detail', '')}"
            )

    if skips:
        print("-" * 78)
        print("SKIPS")
        print("-" * 78)
        for row in skips:
            print(
                f"{row['category']}: {row['dataset']} :: {row['label']} :: {row.get('detail', '')}"
            )

    print("-" * 78)
    print(
        f"TOTAL: {total_elapsed:.2f}s | FAILURES: {len(failures)} | SKIPS: {len(skips)}"
    )
    print("=" * 78)


def main() -> int:
    args = build_quick_validate_parser().parse_args()
    start_time = time.perf_counter()
    results: list[dict] = []
    category_runners = {
        "recipes": _run_recipe_category,
        "ablations": _run_ablation_category,
        "observability": _run_observability_category,
        "evaluation": _run_evaluation_category,
    }

    print("=" * 78)
    print("QUICK VALIDATION")
    print("=" * 78)
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Categories: {', '.join(args.categories)}")
    print(f"Epochs: {args.epochs} | Batch size: {args.batch_size}")
    print(f"MLflow probe: {'enabled' if args.mlflow else 'disabled'}")

    for category in args.categories:
        category_runners[category](args, results)
        if args.fail_fast and any(row["status"] == "fail" for row in results):
            _print_summary(results, time.perf_counter() - start_time)
            return 1

    total_elapsed = time.perf_counter() - start_time
    _print_summary(results, total_elapsed)
    return 0 if not any(row["status"] == "fail" for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
