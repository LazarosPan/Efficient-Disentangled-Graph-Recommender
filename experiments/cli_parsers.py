"""CLI parser builders for U-CaGNN experiment entry points."""

from __future__ import annotations

import argparse
import typing

from src.utils.cli_parsers import (
    BENCHMARK_TIER_CHOICES,
    PRESET_CHOICES,
    add_change_note_arg,
    add_device_and_data_dir_args,
    add_execution_tracking_group,
    add_mlflow_destination_args,
    add_overwrite_checkpoint_arg,
)

from experiments.ablation_configs import ABLATION_VARIANTS
from experiments.recipes import formal_profile_names, recipe_names, search_space_names


def _formal_run_profile_hint(message: str) -> str:
    """Return a formal-run hint when a profile slug is passed to the wrong CLI."""
    marker = "unrecognized arguments:"
    if marker not in message:
        return ""

    unrecognized = message.split(marker, maxsplit=1)[1].strip().split()
    if not unrecognized:
        return ""

    profiles = set(formal_profile_names())
    matching_profile = next((arg for arg in unrecognized if arg in profiles), None)
    if matching_profile is None:
        return ""

    return f"\n\nDid you mean to run:\n  uv run formal-run --profile {matching_profile} ?"


def build_run_experiment_parser() -> argparse.ArgumentParser:
    """Build the single-run experiment CLI parser.

    Returns:
        Configured parser for ``experiments/run_experiment.py``.

    """

    class ExperimentArgumentParser(argparse.ArgumentParser):
        """Custom ArgumentParser for single-run experiments to handle profile typos."""

        def error(self, message: str) -> typing.NoReturn:
            """Override error to suggest formal-run command if a profile name is passed."""
            message += _formal_run_profile_hint(message)
            super().error(message)

    parser = ExperimentArgumentParser(description="Run a U-CaGNN experiment")

    sel = parser.add_argument_group("experiment selection")
    sel.add_argument("--dataset", default="movielens1m", help="Dataset name")
    sel.add_argument("--recipe", choices=recipe_names(), help="Named experiment recipe")
    sel.add_argument("--preset", choices=PRESET_CHOICES, help="Config preset")
    sel.add_argument("--intervention", default=None, help="Ablation intervention name")
    sel.add_argument(
        "--list-recipes",
        action="store_true",
        help="Print available named recipes and exit",
    )

    cp = parser.add_argument_group("checkpointing")
    cp.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional explicit checkpoint path",
    )
    cp.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Save checkpoint every N epochs",
    )
    add_overwrite_checkpoint_arg(
        cp,
        help_text="Delete any existing checkpoint at the resolved path and force a fresh run",
    )

    tr = parser.add_argument_group("tracking")
    tr.add_argument(
        "--mlflow-run-name",
        default=None,
        help="Optional explicit MLflow run name",
    )
    tr.add_argument(
        "--experiment-id",
        default=None,
        help="Optional thesis experiment identifier tag, e.g. E1",
    )
    add_change_note_arg(
        tr,
        help_text="Optional short note describing the current code change or run intent",
    )

    parser.set_defaults(
        enable_mlflow=True,
        auto_resume=True,
        data_dir="data",
        device="cuda",
        mlflow_tracking_uri=None,
        mlflow_experiment_name="ucagnn-thesis",
    )
    return parser


def build_benchmark_parser() -> argparse.ArgumentParser:
    """Build the benchmark matrix CLI parser.

    Returns:
        Configured parser for ``experiments/run_benchmark.py``.

    """
    parser = argparse.ArgumentParser(description="Run U-CaGNN benchmark matrix")

    mx = parser.add_argument_group("benchmark matrix")
    mx.add_argument(
        "--datasets",
        default="small,medium",
        help=(
            "Comma-separated dataset tiers to run (choices: "
            f"{', '.join(BENCHMARK_TIER_CHOICES)}). Use 'all' as a shorthand "
            "for all tiers."
        ),
    )
    mx.add_argument(
        "--presets",
        nargs="*",
        default=["ucagnn", "lightgcn_paper", "dice_paper"],
        choices=PRESET_CHOICES,
        help="Presets to run",
    )
    ex = add_execution_tracking_group(
        parser,
        experiment_name_default="ucagnn-benchmark",
        no_mlflow_help="Disable MLflow tracking for all benchmark runs",
        tracking_uri_help="Override MLflow tracking URI for all benchmark runs",
        experiment_name_help="MLflow experiment name for benchmark runs",
        batch_id_help=("Optional batch identifier for grouping and resuming benchmark runs"),
        resume_batch_help=(
            "Skip benchmark items already recorded with a terminal status for this batch id"
        ),
        dry_run_help="Print plan without running",
        device_default="cuda",
        data_dir_default="data",
        device_help="Device",
        data_dir_help="Data directory",
    )
    ex.add_argument(
        "--profile-name",
        default=None,
        help="Optional semantic formal profile label to persist alongside batch metadata",
    )
    add_change_note_arg(
        ex,
        help_text="Optional short note describing the current code change or run intent",
    )
    add_overwrite_checkpoint_arg(
        ex,
        help_text="Delete any existing checkpoint for each run and force fresh training",
    )

    return parser


def build_formal_run_parser() -> argparse.ArgumentParser:
    """Build the formal-run CLI parser.

    Returns:
        Configured parser for the ``formal-run`` command.

    """
    profiles = formal_profile_names()
    parser = argparse.ArgumentParser(
        description=(
            "Run the formal U-CaGNN experiment matrix with semantic profile-based resume."
        ),
    )
    parser.add_argument(
        "--profile",
        "--version",
        dest="profile",
        default=None,
        help=(
            "Optional semantic formal profile slug, or a comma-separated queue "
            "of slugs to run sequentially. Supported profiles: " + ", ".join(profiles) + "."
        ),
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help=(
            "Print the predefined formal profiles from "
            "experiments/experiment_catalog.json and exit."
        ),
    )
    add_overwrite_checkpoint_arg(
        parser,
        help_text="Delete any existing checkpoint for each resolved run and force fresh training.",
    )
    return parser


def build_search_parser() -> argparse.ArgumentParser:
    """Build the Optuna search CLI parser.

    Returns:
        Configured parser for ``experiments/run_search.py``.

    """
    spaces = search_space_names()
    parser = argparse.ArgumentParser(
        description="Run Optuna searches over configured U-CaGNN search spaces.",
    )

    sel = parser.add_argument_group("search selection")
    sel.add_argument(
        "--space",
        choices=spaces,
        help="Search-space id from experiments/search_spaces.json.",
    )
    sel.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset name to narrow the selected search space.",
    )
    sel.add_argument(
        "--trials",
        type=int,
        default=None,
        help=(
            "Target fresh informative Optuna trials per dataset: COMPLETE plus real "
            "PRUNED, excluding FAIL/RUNNING, historically imported rows, and "
            "duplicate-skip prunes. "
            "Defaults to the search-space value."
        ),
    )
    sel.add_argument(
        "--study-name",
        default=None,
        help="Optuna study name. Defaults to <space>-<dataset-or-all>.",
    )
    sel.add_argument(
        "--list-spaces",
        action="store_true",
        help="Print available Optuna search spaces and exit.",
    )

    ex = parser.add_argument_group("execution")
    ex.add_argument(
        "--storage",
        default="sqlite:///results/optuna_studies.db",
        help="Optuna storage URI.",
    )
    ex.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved bounds and base configs without training.",
    )
    add_device_and_data_dir_args(
        ex,
        device_default="cuda",
        data_dir_default="data",
        device_help="Device for training trials.",
        data_dir_help="Dataset root directory.",
    )
    ex.add_argument(
        "--mlflow",
        dest="no_mlflow",
        action="store_false",
        help="Enable MLflow tracking for trial runs. Disabled by default for search.",
    )
    ex.add_argument(
        "--no-mlflow",
        dest="no_mlflow",
        action="store_true",
        help="Disable MLflow tracking for trial runs. This is the search default.",
    )
    add_mlflow_destination_args(
        ex,
        experiment_name_default="ucagnn-optuna",
        tracking_uri_help="Override MLflow tracking URI for trial runs.",
        experiment_name_help="MLflow experiment name for trial runs.",
    )
    add_overwrite_checkpoint_arg(
        ex,
        help_text=("Accepted for compatibility; search runs do not save or resume checkpoints."),
    )
    parser.set_defaults(no_mlflow=True)

    return parser


def build_ablation_parser() -> argparse.ArgumentParser:
    """Build the ablation study CLI parser.

    Returns:
        Configured parser for ``experiments/run_ablation.py``.

    """
    variant_names = list(ABLATION_VARIANTS.keys())
    parser = argparse.ArgumentParser(description="Run U-CaGNN ablation study")

    sel = parser.add_argument_group("ablation selection")
    sel.add_argument(
        "--datasets",
        nargs="*",
        required=True,
        help=(
            "Dataset names or benchmark tiers to run, e.g. amazonbook kuairec_v2 "
            f"or small medium (choices: {', '.join(BENCHMARK_TIER_CHOICES)})"
        ),
    )
    sel.add_argument(
        "--variants",
        nargs="*",
        default=variant_names,
        choices=variant_names,
        help="Ablation variants to run",
    )

    add_overwrite_checkpoint_arg(
        parser,
        help_text="Delete any existing checkpoint for a resolved ablation run and retrain it.",
    )
    return parser


__all__ = [
    "build_ablation_parser",
    "build_benchmark_parser",
    "build_formal_run_parser",
    "build_run_experiment_parser",
    "build_search_parser",
]
