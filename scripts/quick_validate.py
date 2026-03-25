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
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from experiments.ablation_configs import ABLATION_VARIANTS, make_ablation_config
from experiments.recipes import get_recipe, load_experiment_catalog
from experiments.run_experiment import MLFLOW_DB_PATH, build_config, run_experiment
from scripts._workflow_helpers import dataset_limit, timed_run_experiment


PROJECT_ROOT = Path(__file__).parent.parent
THESIS_DB_PATH = PROJECT_ROOT / "results" / "thesis_experiments.db"

DEFAULT_DATASETS = [
    "amazonbook",
    "movielens1m",
    "movielens20m",
    "kuairec_v2",
    "taobao",
    "kuairand1k",
]
DEFAULT_CATEGORIES = ["recipes", "ablations", "observability", "evaluation"]
DEFAULT_EVAL_MODES = [
    "default",
    "interest_only",
    "conformity_only",
    "counterfactual_only",
    "conformity_suppressed",
]

TINY_LOADER_MAX_ROWS = {
    "amazonbook": 100,
    "movielens1m": 100,
    "movielens20m": 100,
    "kuairec_v2": 100,
    "taobao": 100,
    "kuairand1k": 100,
}

TINY_SAMPLE_INTERACTIONS = {
    "amazonbook": 100,
    "movielens1m": 100,
    "movielens20m": 100,
    "kuairec_v2": 100,
    "taobao": 100,
    "kuairand1k": 100,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run unified tiny-scale validation across the full experiment surface"
    )
    parser.add_argument(
        "--datasets", nargs="*", default=DEFAULT_DATASETS, help="Datasets to validate"
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        choices=DEFAULT_CATEGORIES,
        default=DEFAULT_CATEGORIES,
        help="Validation categories to run",
    )
    parser.add_argument(
        "--recipe-names",
        nargs="*",
        default=None,
        help="Optional canonical recipe filter",
    )
    parser.add_argument(
        "--ablation-variants",
        nargs="*",
        default=None,
        help="Optional ablation-variant filter",
    )
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--device", default="cuda", help="Execution device")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    parser.add_argument(
        "--epochs", type=int, default=1, help="Epochs for each tiny validation run"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for tiny validation runs",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable the optional MLflow observability probe",
    )
    parser.add_argument(
        "--fail-fast", action="store_true", help="Stop after the first failure"
    )
    return parser.parse_args()


def _build_run_namespace(
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
    training_mode: str | None = None,
    num_neighbors: list[int] | None = None,
    sample_interactions: int | None = None,
    loader_max_rows: int | None = None,
    intervention: str | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=dataset,
        recipe=recipe,
        preset=preset,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        embed_dim=None,
        n_gnn_layers=None,
        interest_gnn_layers=None,
        conformity_gnn_layers=None,
        lr=None,
        eval_scoring_mode=eval_scoring_mode,
        scoring_weight_mode=scoring_weight_mode,
        use_features=use_features,
        feature_policy=feature_policy,
        graph_method=graph_method,
        training_mode=training_mode,
        num_neighbors=num_neighbors,
        sample_interactions=sample_interactions,
        loader_max_rows=loader_max_rows,
        device=args.device,
        data_dir=args.data_dir,
        intervention=intervention,
    )


def _build_runtime_config(
    namespace: argparse.Namespace,
    *,
    patience: int | None = None,
    enable_profiling: bool | None = None,
    profiling_cadence: int | None = None,
):
    config = build_config(namespace)
    if patience is not None:
        config.patience = patience
    if enable_profiling is not None:
        config.enable_profiling = enable_profiling
    if profiling_cadence is not None:
        config.profiling_cadence = profiling_cadence
    return config


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
        "NDCG@50",
        "Recall@50",
        "AveragePopularity@50",
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
    config,
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
    result, elapsed = timed_run_experiment(
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
            namespace = _build_run_namespace(
                args,
                dataset,
                recipe=recipe_name,
                sample_interactions=dataset_limit(
                    dataset, TINY_SAMPLE_INTERACTIONS, default=100
                ),
                loader_max_rows=dataset_limit(
                    dataset, TINY_LOADER_MAX_ROWS, default=100
                ),
            )
            config = _build_runtime_config(
                namespace,
                patience=1,
                enable_profiling=False,
                profiling_cadence=1,
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
        for variant in variants:
            config = make_ablation_config(
                variant,
                dataset=dataset,
                data_dir=args.data_dir,
                seed=args.seed,
                device=args.device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                sample_interactions=dataset_limit(
                    dataset, TINY_SAMPLE_INTERACTIONS, default=100
                ),
                loader_max_rows=dataset_limit(
                    dataset, TINY_LOADER_MAX_ROWS, default=100
                ),
            )
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
                    preset="full",
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
    probe_dataset = _representative_dataset(
        args.datasets, ["movielens1m", "kuairec_v2", "amazonbook"]
    )
    feature_dataset = _representative_dataset(
        args.datasets, ["kuairec_v2", "kuairand1k", "movielens1m"]
    )

    profiling_recipes = [
        "full_full_graph_dense",
        "full_cached_propagation_cagra",
        "full_mini_batch_knn",
    ]
    if not torch.cuda.is_available():
        for recipe_name in profiling_recipes:
            label = f"profiling:{recipe_name}"
            results.append(
                {
                    "status": "skip",
                    "category": "observability",
                    "dataset": probe_dataset,
                    "label": label,
                    "elapsed": 0.0,
                    "detail": "CUDA unavailable",
                }
            )
            _print_case(
                "SKIP", "observability", probe_dataset, label, 0.0, "CUDA unavailable"
            )
    else:
        for recipe_name in profiling_recipes:
            recipe = get_recipe(recipe_name)
            namespace = _build_run_namespace(
                args,
                probe_dataset,
                recipe=recipe_name,
                sample_interactions=dataset_limit(
                    probe_dataset, TINY_SAMPLE_INTERACTIONS, default=100
                ),
                loader_max_rows=dataset_limit(
                    probe_dataset, TINY_LOADER_MAX_ROWS, default=100
                ),
            )
            config = _build_runtime_config(
                namespace,
                patience=1,
                enable_profiling=True,
                profiling_cadence=1,
            )
            label = f"profiling:{recipe_name}"
            try:
                _, elapsed = _run_single_case(
                    category="observability",
                    dataset=probe_dataset,
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
                        "dataset": probe_dataset,
                        "label": label,
                        "elapsed": elapsed,
                    }
                )
                _print_case("OK", "observability", probe_dataset, label, elapsed)
            except Exception as exc:
                results.append(
                    {
                        "status": "fail",
                        "category": "observability",
                        "dataset": probe_dataset,
                        "label": label,
                        "elapsed": 0.0,
                        "detail": str(exc),
                    }
                )
                _print_case(
                    "FAIL", "observability", probe_dataset, label, 0.0, str(exc)
                )
                if args.fail_fast:
                    return

    feature_recipe = "full_full_graph_knn"
    feature_namespace = _build_run_namespace(
        args,
        feature_dataset,
        recipe=feature_recipe,
        use_features=True,
        sample_interactions=dataset_limit(
            feature_dataset, TINY_SAMPLE_INTERACTIONS, default=100
        ),
        loader_max_rows=dataset_limit(
            feature_dataset, TINY_LOADER_MAX_ROWS, default=100
        ),
    )
    feature_config = _build_runtime_config(
        feature_namespace,
        patience=1,
        enable_profiling=False,
        profiling_cadence=1,
    )
    feature_label = "features:full_full_graph_knn"
    try:
        _, elapsed = _run_single_case(
            category="observability",
            dataset=feature_dataset,
            label=feature_label,
            config=feature_config,
            preset="full",
            intervention="quick_features",
            recipe_name=feature_recipe,
            expect_logging=True,
            expect_metrics=True,
        )
        results.append(
            {
                "status": "pass",
                "category": "observability",
                "dataset": feature_dataset,
                "label": feature_label,
                "elapsed": elapsed,
            }
        )
        _print_case("OK", "observability", feature_dataset, feature_label, elapsed)
    except Exception as exc:
        results.append(
            {
                "status": "fail",
                "category": "observability",
                "dataset": feature_dataset,
                "label": feature_label,
                "elapsed": 0.0,
                "detail": str(exc),
            }
        )
        _print_case(
            "FAIL", "observability", feature_dataset, feature_label, 0.0, str(exc)
        )
        if args.fail_fast:
            return

    checkpoint_path = (
        PROJECT_ROOT / "results" / "checkpoints" / "quick_validate_resume_probe.pt"
    )
    checkpoint_path.unlink(missing_ok=True)
    resume_namespace = _build_run_namespace(
        args,
        probe_dataset,
        recipe="full_full_graph_dense",
        sample_interactions=dataset_limit(
            probe_dataset, TINY_SAMPLE_INTERACTIONS, default=100
        ),
        loader_max_rows=dataset_limit(probe_dataset, TINY_LOADER_MAX_ROWS, default=100),
    )
    resume_config = _build_runtime_config(
        resume_namespace,
        patience=1,
        enable_profiling=False,
        profiling_cadence=1,
    )
    resume_label = "checkpoint-resume:full_full_graph_dense"
    try:
        _, first_elapsed = _run_single_case(
            category="observability",
            dataset=probe_dataset,
            label=resume_label,
            config=resume_config,
            preset="full",
            intervention="quick_resume_probe",
            recipe_name="full_full_graph_dense",
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
            preset="full",
            intervention="quick_resume_probe",
            save_checkpoint=True,
            enable_mlflow=False,
            mlflow_experiment_name="ucagnn-quick-validate",
            recipe_name="full_full_graph_dense",
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
                "dataset": probe_dataset,
                "label": resume_label,
                "elapsed": first_elapsed + second_elapsed,
            }
        )
        _print_case(
            "OK",
            "observability",
            probe_dataset,
            resume_label,
            first_elapsed + second_elapsed,
        )
    except Exception as exc:
        results.append(
            {
                "status": "fail",
                "category": "observability",
                "dataset": probe_dataset,
                "label": resume_label,
                "elapsed": 0.0,
                "detail": str(exc),
            }
        )
        _print_case("FAIL", "observability", probe_dataset, resume_label, 0.0, str(exc))
        if args.fail_fast:
            return
    finally:
        checkpoint_path.unlink(missing_ok=True)


def _run_evaluation_category(args: argparse.Namespace, results: list[dict]) -> None:
    eval_dataset = _representative_dataset(
        args.datasets, ["movielens1m", "kuairec_v2", "taobao"]
    )
    for mode in DEFAULT_EVAL_MODES:
        namespace = _build_run_namespace(
            args,
            eval_dataset,
            recipe="full_full_graph_knn",
            eval_scoring_mode=mode,
            sample_interactions=dataset_limit(
                eval_dataset, TINY_SAMPLE_INTERACTIONS, default=100
            ),
            loader_max_rows=dataset_limit(
                eval_dataset, TINY_LOADER_MAX_ROWS, default=100
            ),
        )
        config = _build_runtime_config(
            namespace,
            patience=1,
            enable_profiling=False,
            profiling_cadence=1,
        )
        label = f"eval:{mode}"
        try:
            _, elapsed = _run_single_case(
                category="evaluation",
                dataset=eval_dataset,
                label=label,
                config=config,
                preset="full",
                intervention=f"quick_eval_{mode}",
                recipe_name="full_full_graph_knn",
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
    args = parse_args()
    start_time = time.perf_counter()
    results: list[dict] = []

    print("=" * 78)
    print("QUICK VALIDATION")
    print("=" * 78)
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Categories: {', '.join(args.categories)}")
    print(f"Epochs: {args.epochs} | Batch size: {args.batch_size} | Seed: {args.seed}")
    print(f"MLflow probe: {'enabled' if args.mlflow else 'disabled'}")

    if "recipes" in args.categories:
        _run_recipe_category(args, results)
        if args.fail_fast and any(row["status"] == "fail" for row in results):
            _print_summary(results, time.perf_counter() - start_time)
            return 1
    if "ablations" in args.categories:
        _run_ablation_category(args, results)
        if args.fail_fast and any(row["status"] == "fail" for row in results):
            _print_summary(results, time.perf_counter() - start_time)
            return 1
    if "observability" in args.categories:
        _run_observability_category(args, results)
        if args.fail_fast and any(row["status"] == "fail" for row in results):
            _print_summary(results, time.perf_counter() - start_time)
            return 1
    if "evaluation" in args.categories:
        _run_evaluation_category(args, results)

    total_elapsed = time.perf_counter() - start_time
    _print_summary(results, total_elapsed)
    return 0 if not any(row["status"] == "fail" for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
