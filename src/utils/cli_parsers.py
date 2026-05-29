"""CLI parser builders for utility scripts and data exploration entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

BENCHMARK_DATASETS = [
    "amazonbook",
    "movielens1m",
    "movielens20m",
    "kuairec_v2",
    "taobao",
    "kuairand1k",
]
BENCHMARK_DATASET_TIERS: dict[str, list[str]] = {
    "small": ["amazonbook", "movielens1m"],
    "medium": ["kuairec_v2", "kuairand1k"],
    "large": ["movielens20m", "taobao"],
}
BENCHMARK_DATASET_TIERS["all"] = BENCHMARK_DATASETS
BENCHMARK_TIER_CHOICES = list(BENCHMARK_DATASET_TIERS)
PRESET_CHOICES = ["ucagnn", "lightgcn", "dice_like"]
SCORING_WEIGHT_MODE_CHOICES = ["fixed", "learned"]
EVALUATE_SCORING_MODE_CHOICES = [
    "default",
    "interest_only",
    "conformity_only",
    "conformity_suppressed",
]
DEFAULT_EVALUATE_SCORING_MODES = [
    "default",
    "interest_only",
    "conformity_suppressed",
]
_VALIDATION_CATEGORIES = ["recipes", "ablations", "observability", "evaluation"]


def normalize_benchmark_datasets_arg(raw: object) -> list[str]:
    """Normalize the benchmark ``datasets`` field to a list of selectors."""
    if isinstance(raw, (list, tuple)):
        return list(raw)
    if isinstance(raw, str):
        return [part.strip() for part in raw.split(",") if part.strip()]
    return ["all"]


def resolve_benchmark_datasets(tiers: list[str] | str) -> list[str]:
    """Expand tier selectors or explicit datasets to a deduplicated dataset list."""
    if isinstance(tiers, str):
        tiers = [tiers]
    if "all" in tiers:
        return list(BENCHMARK_DATASET_TIERS["all"])

    known_datasets = set(BENCHMARK_DATASET_TIERS["all"])
    seen: dict[str, None] = {}
    for tier in tiers:
        if tier in BENCHMARK_DATASET_TIERS:
            for dataset_name in BENCHMARK_DATASET_TIERS[tier]:
                seen[dataset_name] = None
        elif tier in known_datasets:
            seen[tier] = None
        else:
            raise ValueError(
                (
                    "Unknown dataset or tier '{tier}'. Expected one of "
                    f"{sorted(list(BENCHMARK_DATASET_TIERS) + list(known_datasets))}."
                ),
            )
    return list(seen)


def add_device_and_data_dir_args(
    container: argparse._ActionsContainer,
    *,
    device_default: str | None,
    data_dir_default: str | None,
    device_help: str,
    data_dir_help: str,
) -> None:
    """Add the shared device and data-directory options.

    Args:
        container: Parser or argument group receiving the options.
        device_default: Default value for ``--device``.
        data_dir_default: Default value for ``--data-dir``.
        device_help: Help text for ``--device``.
        data_dir_help: Help text for ``--data-dir``.

    Returns:
        None.

    """
    container.add_argument("--device", default=device_default, help=device_help)
    container.add_argument("--data-dir", default=data_dir_default, help=data_dir_help)


def add_mlflow_destination_args(
    container: argparse._ActionsContainer,
    *,
    experiment_name_default: str | None,
    tracking_uri_help: str,
    experiment_name_help: str,
) -> None:
    """Add the shared MLflow destination arguments.

    Args:
        container: Parser or argument group receiving the options.
        experiment_name_default: Default value for ``--mlflow-experiment-name``.
        tracking_uri_help: Help text for ``--mlflow-tracking-uri``.
        experiment_name_help: Help text for ``--mlflow-experiment-name``.

    Returns:
        None.

    """
    container.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help=tracking_uri_help,
    )
    container.add_argument(
        "--mlflow-experiment-name",
        default=experiment_name_default,
        help=experiment_name_help,
    )


def add_batch_execution_args(
    container: argparse._ActionsContainer,
    *,
    experiment_name_default: str | None,
    no_mlflow_help: str,
    tracking_uri_help: str,
    experiment_name_help: str,
    batch_id_help: str,
    resume_batch_help: str,
    dry_run_help: str,
) -> None:
    """Add the shared execution/tracking options for batch-oriented commands.

    Args:
        container: Parser or argument group receiving the options.
        experiment_name_default: Default MLflow experiment name.
        no_mlflow_help: Help text for ``--no-mlflow``.
        tracking_uri_help: Help text for ``--mlflow-tracking-uri``.
        experiment_name_help: Help text for ``--mlflow-experiment-name``.
        batch_id_help: Help text for ``--batch-id``.
        resume_batch_help: Help text for ``--resume-batch``.
        dry_run_help: Help text for ``--dry-run``.

    Returns:
        None.

    """
    container.add_argument(
        "--no-mlflow",
        action="store_true",
        help=no_mlflow_help,
    )
    add_mlflow_destination_args(
        container,
        experiment_name_default=experiment_name_default,
        tracking_uri_help=tracking_uri_help,
        experiment_name_help=experiment_name_help,
    )
    container.add_argument(
        "--batch-id",
        default=None,
        help=batch_id_help,
    )
    container.add_argument(
        "--resume-batch",
        action="store_true",
        help=resume_batch_help,
    )
    container.add_argument(
        "--dry-run",
        action="store_true",
        help=dry_run_help,
    )


def add_change_note_arg(
    container: argparse._ActionsContainer,
    *,
    help_text: str,
) -> None:
    """Add the shared optional change-note argument.

    Args:
        container: Parser or argument group receiving the option.
        help_text: Help text for ``--change-note``.

    Returns:
        None.

    """
    container.add_argument(
        "--change-note",
        default=None,
        help=help_text,
    )


def add_overwrite_checkpoint_arg(
    container: argparse._ActionsContainer,
    *,
    help_text: str,
) -> None:
    """Add the shared explicit checkpoint replacement flag.

    Args:
        container: Parser or argument group receiving the option.
        help_text: Help text for ``--overwrite-checkpoint``.

    Returns:
        None.

    """
    container.add_argument(
        "--overwrite-checkpoint",
        action="store_true",
        help=help_text,
    )


def add_execution_tracking_group(
    parser: argparse.ArgumentParser,
    *,
    experiment_name_default: str | None,
    no_mlflow_help: str,
    tracking_uri_help: str,
    experiment_name_help: str,
    batch_id_help: str,
    resume_batch_help: str,
    dry_run_help: str,
    group_title: str = "execution and tracking",
    device_default: str = "cuda",
    data_dir_default: str = "data",
    device_help: str = "Device",
    data_dir_help: str = "Data directory",
) -> argparse._ArgumentGroup:
    """Add the standard execution and tracking argument group.

    Args:
        parser: Parser receiving the new argument group.
        experiment_name_default: Default MLflow experiment name.
        no_mlflow_help: Help text for ``--no-mlflow``.
        tracking_uri_help: Help text for ``--mlflow-tracking-uri``.
        experiment_name_help: Help text for ``--mlflow-experiment-name``.
        batch_id_help: Help text for ``--batch-id``.
        resume_batch_help: Help text for ``--resume-batch``.
        dry_run_help: Help text for ``--dry-run``.
        group_title: Title for the created argument group.
        device_default: Default ``--device`` value.
        data_dir_default: Default ``--data-dir`` value.
        device_help: Help text for ``--device``.
        data_dir_help: Help text for ``--data-dir``.

    Returns:
        argparse._ArgumentGroup: The created argument group so callers can extend it.

    """
    group = parser.add_argument_group(group_title)
    add_device_and_data_dir_args(
        group,
        device_default=device_default,
        data_dir_default=data_dir_default,
        device_help=device_help,
        data_dir_help=data_dir_help,
    )
    add_batch_execution_args(
        group,
        experiment_name_default=experiment_name_default,
        no_mlflow_help=no_mlflow_help,
        tracking_uri_help=tracking_uri_help,
        experiment_name_help=experiment_name_help,
        batch_id_help=batch_id_help,
        resume_batch_help=resume_batch_help,
        dry_run_help=dry_run_help,
    )
    return group


def build_quick_validate_parser() -> argparse.ArgumentParser:
    """Build the tiny validation CLI parser.

    Returns:
        Configured parser for ``scripts/quick_validate.py``.

    """
    parser = argparse.ArgumentParser(
        description="Run unified tiny-scale validation across the full experiment surface",
    )
    parser.set_defaults(
        datasets=list(BENCHMARK_DATASETS),
        categories=list(_VALIDATION_CATEGORIES),
        recipe_names=None,
        ablation_variants=None,
        data_dir="data",
        mlflow=False,
        fail_fast=False,
    )
    return parser


def build_evaluate_scoring_modes_parser() -> argparse.ArgumentParser:
    """Build the same-checkpoint scoring-mode evaluation CLI parser.

    Returns:
        Configured parser for ``scripts/evaluate_scoring_modes.py``.

    """
    parser = argparse.ArgumentParser(
        description="Evaluate a single checkpoint under multiple scoring modes",
    )
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to a completed training checkpoint produced by run_experiment.py",
    )
    parser.add_argument(
        "--modes",
        nargs="*",
        default=DEFAULT_EVALUATE_SCORING_MODES,
        choices=EVALUATE_SCORING_MODE_CHOICES,
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
        choices=["all", "completed", "attention", "errors", "comparison"],
        default=None,
        help=(
            "Select a convenience exploration view before applying any extra "
            "filters. Omit all flags to show the top-20 completed runs by "
            "NDCG@20."
        ),
    )
    return parser


def build_explore_all_datasets_parser() -> argparse.ArgumentParser:
    """Build the dataset-visualization CLI parser.

    Returns:
        Configured parser for ``src/data_exploration/explore_all_datasets.py``.

    """
    parser = argparse.ArgumentParser(
        description="Visualize all six benchmark datasets through canonical loaders.",
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
        default=list(BENCHMARK_DATASETS),
        choices=BENCHMARK_DATASETS,
        help="Datasets to visualize",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Figure DPI for saved images",
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
    "BENCHMARK_DATASETS",
    "BENCHMARK_DATASET_TIERS",
    "BENCHMARK_TIER_CHOICES",
    "PRESET_CHOICES",
    "SCORING_WEIGHT_MODE_CHOICES",
    "add_batch_execution_args",
    "add_change_note_arg",
    "add_device_and_data_dir_args",
    "add_mlflow_destination_args",
    "add_overwrite_checkpoint_arg",
    "build_data_information_parser",
    "build_evaluate_scoring_modes_parser",
    "build_explore_all_datasets_parser",
    "build_query_results_parser",
    "build_quick_validate_parser",
    "normalize_benchmark_datasets_arg",
    "resolve_benchmark_datasets",
]
