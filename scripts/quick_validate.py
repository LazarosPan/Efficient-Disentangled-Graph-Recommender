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

import torch
from experiments.ablation_configs import (
    ABLATION_VARIANTS,
    build_ablation_base_kwargs,
    make_ablation_config,
)
from experiments.recipes import get_recipe, recipe_names
from experiments.run_experiment import (
    build_config,
    build_runtime_config_inputs,
    run_experiment,
)
from src.data.feature_policy import supports_feature_utility
from src.training import THESIS_PRIMARY_METRICS
from src.utils.cli_parsers import (
    build_quick_validate_parser,
)
from src.utils.config import DEFAULT_SEED, UCaGNNConfig
from src.utils.project_paths import CHECKPOINT_DIR, MLFLOW_DB_PATH, THESIS_DB_PATH

RUNTIME_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QUICK_VALIDATE_EPOCHS = 1
QUICK_VALIDATE_BATCH_SIZE = 128
_TINY_DATASET_LIMITS = {
    "amazonbook": 100,
    "movielens1m": 100,
    "movielens20m": 100,
    "kuairec_v2": 100,
    "taobao": 100,
    "kuairand1k": 100,
}


def _build_tiny_runtime_overrides(dataset: str) -> dict[str, int | bool]:
    """Build the shared tiny-run overrides for a dataset.

    This single-owner helper keeps every smoke-run default in one place:
    one epoch, the tiny batch size, the dataset-specific interaction and row
    caps, patience=1, and ``use_torch_compile=False``. Compile is disabled for
    smoke validation because the tiny runs are meant to be fast, stable
    pipeline checks rather than a benchmark for compile startup or backend
    coverage.

    Args:
        dataset: Dataset name for the validation run.

    Returns:
        A mapping containing the tiny-run epochs, batch size, interaction cap,
        loader row cap, patience, and compile toggle.

    """
    dataset_limit = int(_TINY_DATASET_LIMITS.get(dataset, 100))
    return {
        "epochs": QUICK_VALIDATE_EPOCHS,
        "batch_size": QUICK_VALIDATE_BATCH_SIZE,
        "sample_interactions": dataset_limit,
        "loader_max_rows": dataset_limit,
        "patience": 1,
        "use_torch_compile": False,
    }


def _apply_tiny_runtime_overrides(
    config: UCaGNNConfig,
    tiny_overrides: dict[str, int | bool],
) -> UCaGNNConfig:
    """Apply the shared tiny-run runtime knobs to a config.

    Args:
        config: Config instance to mutate.
        tiny_overrides: Shared tiny-run runtime values.

    Returns:
        The updated config.

    """
    config.use_torch_compile = bool(tiny_overrides["use_torch_compile"])
    config.patience = int(tiny_overrides["patience"])
    return config


def _build_runtime_config(
    args: argparse.Namespace,
    dataset: str,
    *,
    recipe: str | None = None,
    preset: str | None = None,
    use_features: bool | None = None,
    feature_policy: str | None = None,
    num_neighbors: list[int] | None = None,
    epochs: int = QUICK_VALIDATE_EPOCHS,
    batch_size: int = QUICK_VALIDATE_BATCH_SIZE,
    sample_interactions: int | None = None,
    loader_max_rows: int | None = None,
) -> UCaGNNConfig:
    """Build a quick-validate config without emulating a full CLI namespace.

    Args:
        args: Parsed quick-validate CLI arguments.
        dataset: Dataset name for the run.
        recipe: Optional named recipe.
        preset: Optional preset override.
        use_features: Optional feature toggle override.
        feature_policy: Optional feature policy override.
        num_neighbors: Optional fan-out override.
        epochs: Optional tiny-run epoch override.
        batch_size: Optional tiny-run batch-size override.
        sample_interactions: Optional tiny-run interaction budget.
        loader_max_rows: Optional loader row cap.

    Returns:
        A validated runtime config tailored for quick validation.

    """
    config_inputs = build_runtime_config_inputs(
        dataset=dataset,
        recipe=recipe,
        preset=preset,
        seed=DEFAULT_SEED,
        data_dir=args.data_dir,
        device=RUNTIME_DEVICE,
        epochs=epochs,
        batch_size=batch_size,
        auto_batch_size=False,
        use_features=use_features,
        feature_policy=feature_policy,
        num_neighbors=num_neighbors,
        sample_interactions=sample_interactions,
        loader_max_rows=loader_max_rows,
    )
    config = build_config(config_inputs)
    return config


def _build_tiny_recipe_config(
    args: argparse.Namespace,
    dataset: str,
    *,
    recipe: str,
    use_features: bool | None = None,
) -> UCaGNNConfig:
    """Build a tiny validation config for a catalog recipe on one dataset."""
    tiny_overrides = _build_tiny_runtime_overrides(dataset)
    config = _build_runtime_config(
        args,
        dataset,
        recipe=recipe,
        use_features=use_features,
        epochs=int(tiny_overrides["epochs"]),
        batch_size=int(tiny_overrides["batch_size"]),
        sample_interactions=tiny_overrides["sample_interactions"],
        loader_max_rows=tiny_overrides["loader_max_rows"],
    )
    return _apply_tiny_runtime_overrides(config, tiny_overrides)


def _build_tiny_ablation_config(
    args: argparse.Namespace,
    dataset: str,
    *,
    variant: str,
) -> UCaGNNConfig:
    """Build a tiny validation config for an ablation variant on one dataset."""
    tiny_overrides = _build_tiny_runtime_overrides(dataset)
    config = make_ablation_config(
        variant=variant,
        **build_ablation_base_kwargs(
            dataset=dataset,
            data_dir=args.data_dir,
            device=RUNTIME_DEVICE,
            epochs=int(tiny_overrides["epochs"]),
            batch_size=int(tiny_overrides["batch_size"]),
            sample_interactions=int(tiny_overrides["sample_interactions"]),
            loader_max_rows=int(tiny_overrides["loader_max_rows"]),
        ),
    )
    return _apply_tiny_runtime_overrides(config, tiny_overrides)


def _select_values(
    requested: list[str] | None,
    available: list[str],
    label: str,
) -> list[str]:
    if requested is None:
        return available
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown {label}: {', '.join(unknown)}")
    return [value for value in available if value in requested]


def _sqlite_count(query: str, params: tuple[object, ...]) -> int:
    with sqlite3.connect(THESIS_DB_PATH) as conn:
        row = conn.execute(query, params).fetchone()
    return int(row[0] or 0) if row is not None else 0


def _assert_experiment_logging(exp_id: int) -> None:
    experiments = _sqlite_count(
        "SELECT COUNT(*) FROM experiments WHERE id = ?",
        (exp_id,),
    )
    metrics = _sqlite_count(
        "SELECT COUNT(*) FROM metrics WHERE experiment_id = ?",
        (exp_id,),
    )
    if experiments == 0:
        raise AssertionError(
            f"SQLite experiment row missing for experiment_id={exp_id}",
        )
    if metrics == 0:
        raise AssertionError(f"SQLite metric rows missing for experiment_id={exp_id}")


def _assert_ranking_metrics(test_metrics: dict[str, float]) -> None:
    for metric_name in THESIS_PRIMARY_METRICS:
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
        f"{status:<4} {category:<13} {dataset:<12} {label:<40} {elapsed:>7.2f}s{suffix}",
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
    if expect_metrics:
        _assert_ranking_metrics(result["test_metrics"])
    if expect_mlflow_db_touch:
        mlflow_mtime_after = MLFLOW_DB_PATH.stat().st_mtime_ns if MLFLOW_DB_PATH.exists() else None
        if mlflow_mtime_after is None or mlflow_mtime_after == mlflow_mtime_before:
            raise AssertionError("MLflow probe did not update results/mlflow.db")

    return result, elapsed


def _run_recipe_category(args: argparse.Namespace, results: list[dict]) -> None:
    selected_recipe_names = _select_values(
        args.recipe_names,
        recipe_names(include_aliases=False),
        "recipe names",
    )
    print(
        ""
        f"Recipe coverage: {len(args.datasets)} datasets x "
        f"{len(selected_recipe_names)} canonical recipes",
    )
    for dataset in args.datasets:
        for recipe_name in selected_recipe_names:
            recipe = get_recipe(recipe_name)
            config = _build_tiny_recipe_config(
                args,
                dataset,
                recipe=recipe_name,
            )
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
                    },
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
                    },
                )
                _print_case("FAIL", "recipes", dataset, label, elapsed, str(exc))
                if args.fail_fast:
                    return


def _run_ablation_category(args: argparse.Namespace, results: list[dict]) -> None:
    variants = _select_values(
        args.ablation_variants,
        sorted(ABLATION_VARIANTS),
        "ablation variants",
    )
    print(
        f"Ablation coverage: {len(args.datasets)} datasets x {len(variants)} variants",
    )
    for dataset in args.datasets:
        for variant in variants:
            config = _build_tiny_ablation_config(
                args,
                dataset,
                variant=variant,
            )
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
                    },
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
                    },
                )
                _print_case("FAIL", "ablations", dataset, label, elapsed, str(exc))
                if args.fail_fast:
                    return


def _run_observability_category(args: argparse.Namespace, results: list[dict]) -> None:
    feature_datasets = [dataset for dataset in args.datasets if supports_feature_utility(dataset)]
    print(
        ""
        f"Observability coverage: {len(feature_datasets)} feature probes, "
        f"{len(args.datasets)} resume probes",
    )

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
                },
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
                },
            )
            _print_case("FAIL", "observability", dataset, feature_label, 0.0, str(exc))
            if args.fail_fast:
                return

    resume_label = "checkpoint-resume:ucagnn"
    for dataset in args.datasets:
        checkpoint_path = CHECKPOINT_DIR / f"quick_validate_resume_probe_{dataset}.pt"
        checkpoint_path.unlink(missing_ok=True)
        resume_config = _build_tiny_recipe_config(
            args,
            dataset,
            recipe="ucagnn",
        )
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
                },
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
                },
            )
            _print_case("FAIL", "observability", dataset, resume_label, 0.0, str(exc))
            if args.fail_fast:
                return
        finally:
            checkpoint_path.unlink(missing_ok=True)


def _run_evaluation_category(args: argparse.Namespace, results: list[dict]) -> None:
    preferred = ["movielens1m", "kuairec_v2", "taobao"]
    eval_dataset = next((c for c in preferred if c in args.datasets), args.datasets[0])
    config = _build_tiny_recipe_config(args, eval_dataset, recipe="ucagnn")
    label = "eval:refined"
    try:
        _, elapsed = _run_single_case(
            category="evaluation",
            dataset=eval_dataset,
            label=label,
            config=config,
            preset="ucagnn",
            intervention="quick_eval_refined",
            recipe_name="ucagnn",
            expect_metrics=True,
        )
        results.append(
            {
                "status": "pass",
                "category": "evaluation",
                "dataset": eval_dataset,
                "label": label,
                "elapsed": elapsed,
            },
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
            },
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
            ""
            f"{category:<13} pass={passed:<4} fail={failed:<4} "
            f"skip={skipped:<4} total={len(subset)}",
        )

    if failures:
        print("-" * 78)
        print("FAILURES")
        print("-" * 78)
        for row in failures:
            print(
                f"{row['category']}: {row['dataset']} :: {row['label']} :: {row.get('detail', '')}",
            )

    if skips:
        print("-" * 78)
        print("SKIPS")
        print("-" * 78)
        for row in skips:
            print(
                f"{row['category']}: {row['dataset']} :: {row['label']} :: {row.get('detail', '')}",
            )

    print("-" * 78)
    print(
        f"TOTAL: {total_elapsed:.2f}s | FAILURES: {len(failures)} | SKIPS: {len(skips)}",
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
    print(f"Epochs: {QUICK_VALIDATE_EPOCHS} | Batch size: {QUICK_VALIDATE_BATCH_SIZE}")
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
