"""CLI parser builders for utility scripts and data exploration entry points."""

from __future__ import annotations

import argparse
from pathlib import Path


_BENCHMARK_DATASETS = [
    "amazonbook",
    "movielens1m",
    "movielens20m",
    "kuairec_v2",
    "taobao",
    "kuairand1k",
]
_VALIDATION_CATEGORIES = ["recipes", "ablations", "observability", "evaluation"]


def build_quick_validate_parser() -> argparse.ArgumentParser:
    """Build the tiny validation CLI parser.

    Returns:
        Configured parser for ``scripts/quick_validate.py``.
    """
    parser = argparse.ArgumentParser(
        description="Run unified tiny-scale validation across the full experiment surface"
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=_BENCHMARK_DATASETS,
        help="Datasets to validate",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        choices=_VALIDATION_CATEGORIES,
        default=_VALIDATION_CATEGORIES,
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
    return parser


def build_evaluate_scoring_modes_parser() -> argparse.ArgumentParser:
    """Build the same-checkpoint scoring-mode evaluation CLI parser.

    Returns:
        Configured parser for ``scripts/evaluate_scoring_modes.py``.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a single checkpoint under multiple scoring modes"
    )
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to a completed training checkpoint produced by run_experiment.py",
    )
    parser.add_argument(
        "--modes",
        nargs="*",
        default=["default", "interest_only", "conformity_suppressed"],
        choices=[
            "default",
            "interest_only",
            "conformity_only",
            "counterfactual_only",
            "conformity_suppressed",
        ],
        help="Evaluation-time scoring modes to compare",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test", "both"],
        default="test",
        help="Which split to evaluate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Evaluation batch size for full-catalog scoring",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override; defaults to the checkpoint config device",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional JSON output path for the collected metric table",
    )
    return parser


def build_query_results_parser() -> argparse.ArgumentParser:
    """Build the experiment-results query CLI parser.

    Returns:
        Configured parser for ``scripts/query_results.py``.
    """
    parser = argparse.ArgumentParser(description="Query experiment results")
    parser.add_argument(
        "--view",
        choices=["all", "completed", "attention", "errors"],
        default="all",
        help="Select a convenience exploration view before applying any extra filters",
    )
    parser.add_argument(
        "--batch-id", default=None, help="Filter experiment list to one batch id"
    )
    parser.add_argument(
        "--status",
        choices=["running", "completed", "oom", "failed", "unknown"],
        help="Filter experiment list by status",
    )
    parser.add_argument("--exp", type=int, help="Show experiment details")
    parser.add_argument("--metrics", type=int, help="Show metrics for experiment")
    parser.add_argument("--profiling", type=int, help="Show profiling for experiment")
    parser.add_argument("--alpha", type=int, help="Show alpha drift for experiment")
    parser.add_argument(
        "--bottleneck", type=int, help="Show bottleneck analysis for experiment"
    )
    return parser


def build_explore_all_datasets_parser() -> argparse.ArgumentParser:
    """Build the dataset-visualization CLI parser.

    Returns:
        Configured parser for ``src/data_exploration/explore_all_datasets.py``.
    """
    parser = argparse.ArgumentParser(
        description="Visualize all six benchmark datasets through canonical loaders."
    )
    parser.add_argument("--data-dir", default="data", help="Root data directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "dataset_visualizations",
        help="Directory where figures will be written",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=_BENCHMARK_DATASETS,
        choices=_BENCHMARK_DATASETS,
        help="Datasets to visualize",
    )
    parser.add_argument(
        "--dpi", type=int, default=180, help="Figure DPI for saved images"
    )
    return parser


def build_data_information_parser() -> argparse.ArgumentParser:
    """Build the dataset-information report CLI parser.

    Returns:
        Configured parser for ``src/data_exploration/data_information.py``.
    """
    parser = argparse.ArgumentParser(description="Generate dataset information report.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for report (default: data/datasets_information.md)",
    )
    parser.add_argument(
        "--audit-json",
        type=str,
        default=None,
        help="Optional path for machine-readable feature audit export",
    )
    return parser


__all__ = [
    "build_data_information_parser",
    "build_evaluate_scoring_modes_parser",
    "build_explore_all_datasets_parser",
    "build_query_results_parser",
    "build_quick_validate_parser",
]
