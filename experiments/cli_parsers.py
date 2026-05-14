"""CLI parser builders for U-CaGNN experiment entry points."""

from __future__ import annotations

import argparse

from src.utils.cli_parsers import (
    BENCHMARK_TIER_CHOICES,
    PRESET_CHOICES,
    SCORING_WEIGHT_MODE_CHOICES,
    add_batch_size_probe_args,
    add_device_and_data_dir_args,
    add_execution_tracking_group,
    add_mlflow_destination_args,
    add_sample_limit_args,
)
from src.utils.config import SUPPORTED_LR_SCHEDULERS

from experiments.ablation_configs import ABLATION_VARIANTS
from experiments.recipes import formal_profile_names, recipe_names


def build_run_experiment_parser() -> argparse.ArgumentParser:
    """Build the single-run experiment CLI parser.

    Returns:
        Configured parser for ``experiments/run_experiment.py``.

    """
    parser = argparse.ArgumentParser(description="Run a U-CaGNN experiment")

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

    ov = parser.add_argument_group("runtime overrides")
    ov.add_argument("--epochs", type=int, default=None, help="Override epochs")
    ov.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    add_batch_size_probe_args(
        ov,
        auto_help="Probe the largest feasible batch size for the current dataset on CUDA",
        fixed_help="Disable runtime batch-size probing and use --batch-size directly",
        candidates_help="Descending candidate batch sizes for the auto-batch probe",
    )
    ov.add_argument("--embed-dim", type=int, default=None, help="Override embed dim")
    ov.add_argument(
        "--single-branch-gnn-layers",
        type=int,
        default=None,
        help="Override the LightGCN single-branch depth",
    )
    ov.add_argument(
        "--interest-gnn-layers",
        type=int,
        default=None,
        help="Override the interest-branch GNN depth",
    )
    ov.add_argument(
        "--conformity-gnn-layers",
        type=int,
        default=None,
        help="Override the conformity-branch GNN depth",
    )
    ov.add_argument("--lr", type=float, default=None, help="Override learning rate")
    ov.add_argument(
        "--lr-scheduler",
        choices=SUPPORTED_LR_SCHEDULERS,
        default=None,
        help="Override the learning rate scheduler type.",
    )
    ov.add_argument(
        "--lr-scheduler-factor",
        type=float,
        default=None,
        help="Override the scheduler factor where applicable.",
    )
    ov.add_argument(
        "--lr-scheduler-patience",
        type=int,
        default=None,
        help="Override scheduler patience for plateau scheduler.",
    )

    sc = parser.add_argument_group("scoring and features")
    sc.add_argument(
        "--eval-scoring-mode",
        choices=["default", "interest_only", "conformity_suppressed"],
        default=None,
        help="Evaluation-time scoring mode for Recall/NDCG",
    )
    sc.add_argument(
        "--scoring-weight-mode",
        choices=SCORING_WEIGHT_MODE_CHOICES,
        default=None,
        help="Scoring mixture mode: fixed config weights or learned simplex weights",
    )
    sc.add_argument(
        "--use-features",
        dest="use_features",
        action="store_true",
        help="Enable dataset side features when available",
    )
    sc.add_argument(
        "--no-features",
        dest="use_features",
        action="store_false",
        help="Disable dataset side features even when available",
    )
    sc.add_argument(
        "--feature-policy",
        choices=["thesis_default", "all_optional"],
        default=None,
        help=(
            "Feature-loading policy: thesis_default enforces the safe thesis "
            "allowlist; all_optional restores the full optional side-feature "
            "scans."
        ),
    )

    dg = parser.add_argument_group("data and graph")
    dg.add_argument(
        "--num-neighbors",
        type=int,
        nargs="+",
        default=None,
        help="Fan-out per GNN layer for mini_batch mode (e.g., 10 10)",
    )
    add_sample_limit_args(
        dg,
        sample_help="Optional interaction budget for sampled runs such as preflight",
        loader_help="Optional early row cap for dataset loading during fast smoke/preflight runs",
    )
    add_device_and_data_dir_args(
        dg,
        device_default="cuda",
        data_dir_default="data",
        device_help="Device (cuda/cpu)",
        data_dir_help="Data directory",
    )

    cp = parser.add_argument_group("checkpointing")
    cp.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Skip saving checkpoint",
    )
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
    cp.add_argument(
        "--auto-resume",
        dest="auto_resume",
        action="store_true",
        help="Resume automatically from a matching checkpoint",
    )
    cp.add_argument(
        "--no-auto-resume",
        dest="auto_resume",
        action="store_false",
        help="Disable automatic checkpoint resume for this run",
    )
    cp.add_argument(
        "--overwrite-checkpoint",
        action="store_true",
        help="Delete any existing checkpoint at the resolved path and force a fresh run",
    )

    tr = parser.add_argument_group("tracking")
    tr.add_argument(
        "--enable-mlflow",
        dest="enable_mlflow",
        action="store_true",
        help="Explicitly enable MLflow tracking",
    )
    tr.add_argument(
        "--no-mlflow",
        dest="enable_mlflow",
        action="store_false",
        help="Disable MLflow tracking for this run",
    )
    add_mlflow_destination_args(
        tr,
        experiment_name_default="ucagnn-thesis",
        tracking_uri_help="Override MLflow tracking URI",
        experiment_name_help="MLflow experiment name",
    )
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
    tr.add_argument(
        "--change-note",
        default=None,
        help="Optional short note describing the current code change or run intent",
    )

    parser.set_defaults(
        enable_mlflow=True,
        auto_resume=True,
        use_features=None,
        auto_batch_size=None,
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
        default=["ucagnn", "lightgcn", "dice_like"],
        choices=PRESET_CHOICES,
        help="Presets to run",
    )
    mx.add_argument(
        "--scoring-weight-modes",
        nargs="*",
        default=["learned"],
        choices=SCORING_WEIGHT_MODE_CHOICES,
        help=(
            "Score-mixture modes to run. LightGCN stays fixed-only because "
            "learned weights are inapplicable without dual branches."
        ),
    )

    ov = parser.add_argument_group("runtime overrides")
    ov.add_argument("--epochs", type=int, default=None, help="Override epochs for all")
    ov.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size for all runs",
    )
    add_batch_size_probe_args(
        ov,
        auto_help="Probe the largest feasible dataset-specific batch size for CUDA runs",
        fixed_help="Disable runtime batch-size probing for the benchmark matrix",
        candidates_help="Descending candidate batch sizes for the benchmark auto-batch probe",
    )
    ov.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate for all runs",
    )
    ov.add_argument(
        "--lr-scheduler",
        choices=[*SUPPORTED_LR_SCHEDULERS, "all"],
        default=None,
        help="Override the learning rate scheduler type for all runs or use 'all' to sweep.",
    )
    ov.add_argument(
        "--lr-scheduler-factor",
        type=float,
        default=None,
        help="Override the scheduler factor where applicable.",
    )
    ov.add_argument(
        "--lr-scheduler-patience",
        type=int,
        default=None,
        help="Override scheduler patience for plateau scheduler.",
    )
    ov.add_argument(
        "--num-neighbors",
        nargs="*",
        type=int,
        default=None,
        help="Optional mini-batch fan-out override applied to all matrix items",
    )
    add_sample_limit_args(
        ov,
        sample_help="Optional interaction budget for sampled benchmark passes",
        loader_help="Optional dataset loader row cap for all runs",
    )
    es = ov.add_mutually_exclusive_group()
    es.add_argument(
        "--early-stopping",
        dest="use_early_stopping",
        action="store_true",
        help="Enable early stopping for all benchmark runs",
    )
    es.add_argument(
        "--no-early-stopping",
        dest="use_early_stopping",
        action="store_false",
        help="Disable early stopping for all benchmark runs",
    )

    ex = add_execution_tracking_group(
        parser,
        experiment_name_default="ucagnn-benchmark",
        no_mlflow_help="Disable MLflow tracking for all benchmark runs",
        tracking_uri_help="Override MLflow tracking URI for all benchmark runs",
        experiment_name_help="MLflow experiment name for benchmark runs",
        batch_id_help=(
            "Optional batch identifier for grouping and resuming benchmark runs"
        ),
        resume_batch_help=(
            "Skip benchmark items already recorded with a terminal status "
            "for this batch id"
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
    ex.add_argument(
        "--change-note",
        default=None,
        help="Optional short note describing the current code change or run intent",
    )
    ex.add_argument(
        "--overwrite-checkpoint",
        action="store_true",
        help="Delete any existing checkpoint for each run and force fresh training",
    )

    parser.set_defaults(use_early_stopping=None, auto_batch_size=None)
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
            "Optional semantic formal profile slug. Supported profiles: "
        + ", ".join(profiles)
        + "."
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
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume the latest saved formal run state instead of creating a new one.",
    )
    parser.add_argument(
        "--new-run",
        action="store_true",
        help="Force a fresh run even if a saved formal-run state exists.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Start the selected profile from the beginning under a fresh batch id.",
    )
    add_device_and_data_dir_args(
        parser,
        device_default=None,
        data_dir_default=None,
        device_help="Device override. Defaults to cuda for new runs.",
        data_dir_help="Data directory override.",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging for this formal run.",
    )
    add_mlflow_destination_args(
        parser,
        experiment_name_default=None,
        tracking_uri_help="Optional MLflow tracking URI override.",
        experiment_name_help="Optional MLflow experiment name override. Defaults to ucagnn-formal.",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help=(
            "Override the profile's dataset tiers. Comma-separated tier names "
            f"(choices: {', '.join(BENCHMARK_TIER_CHOICES)}) or explicit dataset "
            "names like movielens1m. Use 'all' for all tiers. Example: "
            "--datasets movielens1m,taobao"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the formal run plan without executing it.",
    )
    parser.add_argument(
        "--change-note",
        default=None,
        help="Optional short note describing the current code change or run intent.",
    )
    parser.add_argument(
        "--overwrite-checkpoint",
        action="store_true",
        help=(
            "Delete any existing checkpoint for each resolved run and "
            "force fresh training."
        ),
    )
    return parser


def build_ablation_parser() -> argparse.ArgumentParser:
    """Build the ablation study CLI parser.

    Returns:
        Configured parser for ``experiments/run_ablation.py``.

    """
    variant_names = list(ABLATION_VARIANTS.keys())
    parser = argparse.ArgumentParser(description="Run U-CaGNN ablation study")

    sel = parser.add_argument_group("ablation selection")
    sel.add_argument("--dataset", required=True, help="Dataset name")
    sel.add_argument(
        "--variants",
        nargs="*",
        default=variant_names,
        choices=variant_names,
        help="Ablation variants to run",
    )

    ov = parser.add_argument_group("runtime overrides")
    ov.add_argument("--epochs", type=int, default=None, help="Override epochs")
    ov.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    add_sample_limit_args(
        ov,
        sample_help="Optional interaction budget for sampled ablation smoke runs",
        loader_help="Optional early row cap for dataset loading during fast ablation smoke runs",
    )

    add_execution_tracking_group(
        parser,
        experiment_name_default="ucagnn-ablation",
        no_mlflow_help="Disable MLflow tracking for all ablation runs",
        tracking_uri_help="Override MLflow tracking URI for all ablation runs",
        experiment_name_help="MLflow experiment name for ablation runs",
        batch_id_help="Optional batch identifier for grouping and resuming ablation runs",
        resume_batch_help=(
            "Skip ablation variants already recorded with a terminal status "
            "for this batch id"
        ),
        dry_run_help="Print plan without running",
        device_default="cuda",
        data_dir_default="data",
        device_help="Device",
        data_dir_help="Data directory",
    )
    return parser


__all__ = [
    "build_ablation_parser",
    "build_benchmark_parser",
    "build_formal_run_parser",
    "build_run_experiment_parser",
]
