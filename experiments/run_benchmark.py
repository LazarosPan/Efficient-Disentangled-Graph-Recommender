#!/usr/bin/env python
"""Formal matrix benchmark runner for dataset x preset.

Usage:
    uv run formal-run --profile default
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
import traceback
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from scripts._workflow_helpers import (
    configure_cli_logging,
    metric_value,
    print_batch_summary_counts,
    resolve_batch_id,
    thesis_metric_values,
)
from src.training import THESIS_PRIMARY_METRICS
from src.utils.benchmark_datasets import (
    normalize_benchmark_datasets_arg,
    resolve_benchmark_datasets,
)
from src.utils.config import (
    BENCHMARK_CONFIG_FIELDS,
    DEFAULT_SEED,
    PAPER_BASELINE_PRESETS,
    UCaGNNConfig,
)
from src.utils.experiment_logger import RUNTIME_PROBE_METRIC_NAMES, ExperimentLogger
from src.utils.experiment_naming import format_num_neighbors_payload
from src.utils.project_paths import FORMAL_RUN_STATE_PATH, THESIS_DB_PATH

from experiments.benchmark_resolvers import (
    resolve_benchmark_graph_policy_values,
    resolve_benchmark_lr_scheduler_values,
    resolve_benchmark_num_neighbor_values,
    resolve_benchmark_preprocessing_preset_values,
)
from experiments.cli_parsers import build_benchmark_parser, build_formal_run_parser
from experiments.recipes import (
    default_formal_profile_name,
    formal_profile_names,
    get_formal_profile,
    resolve_profile_num_neighbors,
)
from experiments.run_experiment import (
    build_benchmark_config_inputs,
    build_config,
    normalize_benchmark_config_overrides,
    normalize_config_inputs,
    recoverable_checkpoint_for_config,
    run_experiment,
)

logger = logging.getLogger("ucagnn.benchmark")
STATE_PATH = FORMAL_RUN_STATE_PATH
DEFAULT_PROFILE_NAME = default_formal_profile_name()


FORMAL_RUN_STATE_FIELDS = frozenset(
    {
        "profile_name",
        "profile_slug",
        "batch_id",
        "resumed",
        "last_started_at_utc",
        "last_finished_at_utc",
        "last_exit_code",
        "benchmark_args",
    },
)
BENCHMARK_MATRIX_FIELDS = (
    "datasets",
    "presets",
    "profile_name",
    "profile_slug",
    "runtime_probe_target_epochs",
)
RUNTIME_ONLY_BENCHMARK_FIELDS = (
    "device",
    "data_dir",
    "no_mlflow",
    "mlflow_tracking_uri",
    "mlflow_experiment_name",
    "overwrite_checkpoint",
    "change_note",
    "batch_id",
    "resume_batch",
    "dry_run",
)

NORMALIZED_BENCHMARK_FIELDS = (
    *BENCHMARK_MATRIX_FIELDS,
    *BENCHMARK_CONFIG_FIELDS,
    "graph_policy_options",
    "preprocessing_preset_options",
    "change_note",
    "no_mlflow",
    "mlflow_tracking_uri",
    "mlflow_experiment_name",
    "overwrite_checkpoint",
    "batch_id",
    "resume_batch",
    "dry_run",
)
PLAN_COMPARISON_FIELDS = tuple(
    field_name
    for field_name in NORMALIZED_BENCHMARK_FIELDS
    if field_name not in RUNTIME_ONLY_BENCHMARK_FIELDS
)
BenchmarkPlanItem = tuple[str, str, str, str | None, str, tuple[int, ...]]


def _write_state(payload: dict[str, object]) -> None:
    """Persist the formal-run state file."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_saved_formal_state() -> dict[str, object] | None:
    """Load and validate the saved formal-run state."""
    if not STATE_PATH.exists():
        return None

    raw_state = json.loads(STATE_PATH.read_text())
    if not isinstance(raw_state, dict):
        raise ValueError(
            "Saved formal-run state is malformed. Delete it and start a fresh formal run.",
        )

    unexpected_fields = sorted(set(raw_state) - FORMAL_RUN_STATE_FIELDS)
    if unexpected_fields:
        raise ValueError(
            (
                "Saved formal-run state contains unexpected fields "
                f"{unexpected_fields}. Delete it and start a fresh formal run."
            ),
        )

    raw_profile_name = raw_state.get("profile_name")
    if not isinstance(raw_profile_name, str) or not raw_profile_name.strip():
        raise ValueError(
            (
                "Saved formal-run state is missing profile_name. Delete it and start "
                "a fresh formal run."
            ),
        )

    try:
        profile_bundle = get_formal_profile(raw_profile_name)
    except KeyError as exc:
        raise ValueError(
            (
                "The saved formal-run state references a profile that is no longer "
                "defined. Delete it and start a fresh formal run."
            ),
        ) from exc

    raw_benchmark_args = raw_state.get("benchmark_args")
    if not isinstance(raw_benchmark_args, Mapping):
        raise ValueError(
            (
                "Saved formal-run state is missing benchmark_args. Delete it and "
                "start a fresh formal run."
            ),
        )

    benchmark_args = _coerce_benchmark_args(
        raw_benchmark_args,
        fallback_profile_name=str(profile_bundle["id"]),
    )
    benchmark_args["profile_name"] = str(profile_bundle["id"])
    benchmark_args["profile_slug"] = str(profile_bundle["name"])

    saved_state = dict(raw_state)
    saved_state["profile_name"] = str(profile_bundle["id"])
    saved_state["profile_slug"] = str(profile_bundle["name"])
    saved_state["benchmark_args"] = benchmark_args
    return saved_state


def _normalize_benchmark_args(
    raw_args: argparse.Namespace | Mapping[str, object] | object,
    *,
    fallback_profile_name: str | None = None,
) -> dict[str, object]:
    """Return a JSON-safe benchmark payload."""
    benchmark_args = normalize_config_inputs(raw_args)
    unexpected_fields = sorted(set(benchmark_args) - set(NORMALIZED_BENCHMARK_FIELDS))
    if unexpected_fields:
        raise ValueError(
            (
                f"Formal-run state contains unexpected fields {unexpected_fields}. "
                "Start a fresh formal run."
            ),
        )

    normalized_args = {
        field_name: benchmark_args.get(field_name) for field_name in BENCHMARK_MATRIX_FIELDS
    }
    normalized_args.update(normalize_benchmark_config_overrides(benchmark_args))
    normalized_args["change_note"] = benchmark_args.get("change_note")
    normalized_args["no_mlflow"] = bool(benchmark_args.get("no_mlflow", False))
    normalized_args["mlflow_tracking_uri"] = benchmark_args.get("mlflow_tracking_uri")
    normalized_args["mlflow_experiment_name"] = benchmark_args.get(
        "mlflow_experiment_name",
    )
    normalized_args["overwrite_checkpoint"] = bool(
        benchmark_args.get("overwrite_checkpoint", False),
    )
    normalized_args["batch_id"] = benchmark_args.get("batch_id")
    normalized_args["resume_batch"] = bool(benchmark_args.get("resume_batch", False))
    normalized_args["dry_run"] = bool(benchmark_args.get("dry_run", False))
    normalized_args["datasets"] = normalize_benchmark_datasets_arg(
        benchmark_args.get("datasets"),
    )
    normalized_args["presets"] = list(benchmark_args["presets"])
    normalized_args["profile_name"] = benchmark_args.get("profile_name") or fallback_profile_name
    normalized_args["profile_slug"] = benchmark_args.get("profile_slug")
    return normalized_args


def _is_normalized_benchmark_args(
    benchmark_args: Mapping[str, object],
) -> bool:
    """Return whether a mapping already matches the normalized benchmark shape."""
    return (
        set(benchmark_args) == set(NORMALIZED_BENCHMARK_FIELDS)
        and isinstance(
            benchmark_args.get("datasets"),
            list,
        )
        and isinstance(benchmark_args.get("presets"), list)
        and (
            benchmark_args.get("profile_name") is None
            or isinstance(benchmark_args.get("profile_name"), str)
        )
    )


def _coerce_benchmark_args(
    raw_args: argparse.Namespace | Mapping[str, object] | object,
    *,
    fallback_profile_name: str | None = None,
) -> dict[str, object]:
    """Return benchmark args, reusing an already-normalized payload when possible."""
    if isinstance(raw_args, Mapping) and _is_normalized_benchmark_args(raw_args):
        return dict(raw_args)
    return _normalize_benchmark_args(
        raw_args,
        fallback_profile_name=fallback_profile_name,
    )


def _benchmark_plan_signature(
    args: argparse.Namespace | Mapping[str, object] | object,
) -> tuple[object, ...]:
    """Return the semantic plan signature used to compare saved formal runs."""
    normalized_args = _coerce_benchmark_args(args)
    return tuple(normalized_args[field_name] for field_name in PLAN_COMPARISON_FIELDS)


def _build_new_run_args(
    cli_args: argparse.Namespace,
    profile_name: str,
) -> dict[str, object]:
    """Build benchmark arguments for a fresh formal run."""
    profile_bundle = get_formal_profile(profile_name)
    matrix = profile_bundle["matrix"]
    assert isinstance(matrix, dict)
    config_overrides = profile_bundle["config_overrides"]
    assert isinstance(config_overrides, dict)
    datasets_value = matrix.get("datasets", "all")
    benchmark_args = normalize_benchmark_config_overrides(config_overrides)
    benchmark_args["sample_interactions"] = None
    benchmark_args["loader_max_rows"] = None
    benchmark_args.update(
        {
            "datasets": normalize_benchmark_datasets_arg(datasets_value),
            "presets": list(matrix["presets"]),
            "profile_name": str(profile_bundle["id"]),
            "profile_slug": str(profile_bundle["name"]),
            "runtime_probe_target_epochs": (
                profile_bundle["runtime_probe"]["target_epochs"]
                if profile_bundle.get("runtime_probe") is not None
                else None
            ),
            "change_note": None,
            "device": "cuda",
            "data_dir": "data",
            "no_mlflow": False,
            "mlflow_tracking_uri": None,
            "mlflow_experiment_name": "ucagnn-formal",
            "overwrite_checkpoint": bool(
                getattr(cli_args, "overwrite_checkpoint", False),
            ),
            "batch_id": resolve_batch_id(None, prefix=f"formal-{profile_name}"),
            "resume_batch": True,
            "dry_run": False,
        },
    )
    return benchmark_args


def _parse_formal_profile_sequence(raw_profile: str | None) -> list[str]:
    """Return one or more resolved formal profile identifiers from CLI input."""
    if raw_profile is None:
        return [DEFAULT_PROFILE_NAME]

    profile_names = [part.strip() for part in raw_profile.split(",") if part.strip()]
    if not profile_names:
        raise ValueError("--profile must name at least one formal profile.")
    return [str(get_formal_profile(profile_name)["id"]) for profile_name in profile_names]


def _build_runtime_probe_estimate(
    *,
    target_epochs: int,
    observed_training_time_s: float,
    observed_epochs: int,
    train_batches_per_epoch: int,
) -> dict[str, float]:
    """Scale one runtime probe to the target epoch budget."""
    if target_epochs < 1:
        raise ValueError("target_epochs must be >= 1.")
    if observed_epochs < 1:
        raise ValueError("observed_epochs must be >= 1.")
    if train_batches_per_epoch < 1:
        raise ValueError("train_batches_per_epoch must be >= 1.")
    if observed_training_time_s <= 0 or not math.isfinite(observed_training_time_s):
        raise ValueError("observed_training_time_s must be finite and > 0.")

    observed_batches = float(observed_epochs * train_batches_per_epoch)
    observed_batches_per_second = observed_batches / float(observed_training_time_s)
    seconds_per_epoch = float(observed_training_time_s) / float(observed_epochs)
    estimated_train_time_s = seconds_per_epoch * float(target_epochs)
    return {
        "runtime_probe_target_epochs": float(target_epochs),
        "runtime_probe_observed_epochs": float(observed_epochs),
        "runtime_probe_train_batches_per_epoch": float(train_batches_per_epoch),
        "runtime_probe_observed_batches_per_second": observed_batches_per_second,
        "runtime_probe_seconds_per_epoch": seconds_per_epoch,
        "runtime_probe_estimated_train_time_s": estimated_train_time_s,
        "runtime_probe_estimated_remaining_train_time_s": max(
            0.0,
            estimated_train_time_s - float(observed_training_time_s),
        ),
    }


def _runtime_probe_estimate_from_result(
    benchmark_args: Mapping[str, object],
    result: Mapping[str, object],
) -> dict[str, float] | None:
    """Build a runtime-probe estimate for a completed benchmark result when configured."""
    target_epochs = benchmark_args.get("runtime_probe_target_epochs")
    if target_epochs is None:
        return None

    observed_training_time_s = result.get("training_time_s")
    observed_epochs = result.get("epochs_stopped_at")
    train_batches_per_epoch = result.get("train_batches_per_epoch")
    if (
        observed_training_time_s is None
        or observed_epochs is None
        or train_batches_per_epoch is None
    ):
        return None

    try:
        return _build_runtime_probe_estimate(
            target_epochs=int(target_epochs),
            observed_training_time_s=float(observed_training_time_s),
            observed_epochs=int(observed_epochs),
            train_batches_per_epoch=int(train_batches_per_epoch),
        )
    except ValueError:
        logger.warning(
            "Skipping runtime-probe estimate because observed timing data is incomplete.",
        )
        return None


def _log_runtime_probe_estimate(
    tracker: ExperimentLogger,
    exp_id: int,
    estimate: Mapping[str, float],
) -> None:
    """Persist runtime-probe approximation metrics under an explicit split label."""
    for metric_name in RUNTIME_PROBE_METRIC_NAMES:
        tracker.log_metric(
            exp_id,
            metric_name,
            estimate[metric_name],
            split="approximation",
        )


def _resolve_saved_benchmark_args(
    saved_benchmark_args: dict[str, object],
    expected_args: dict[str, object],
    cli_args: argparse.Namespace,
    *,
    profile_name: str,
    profile_slug: str,
) -> tuple[dict[str, object], str, bool]:
    """Resolve whether a saved plan can resume or must restart fresh."""
    if _benchmark_plan_signature(saved_benchmark_args) != _benchmark_plan_signature(
        expected_args,
    ):
        return expected_args, profile_name, False
    benchmark_args = dict(saved_benchmark_args)
    if getattr(cli_args, "overwrite_checkpoint", False):
        benchmark_args["overwrite_checkpoint"] = True
    benchmark_args["dry_run"] = False
    benchmark_args["resume_batch"] = True
    benchmark_args["sample_interactions"] = None
    benchmark_args["loader_max_rows"] = None
    benchmark_args["profile_name"] = profile_name
    benchmark_args["profile_slug"] = profile_slug
    return benchmark_args, profile_name, True


def _resolve_benchmark_args(
    cli_args: argparse.Namespace,
) -> tuple[dict[str, object], str, bool]:
    """Resolve whether to create a new formal run or resume the saved one."""
    current_profile_bundle = None
    requested_profile = None
    if cli_args.profile is not None:
        current_profile_bundle = get_formal_profile(cli_args.profile)
        requested_profile = str(current_profile_bundle["id"])

    try:
        saved_state = _load_saved_formal_state()
    except ValueError as exc:
        if "no longer defined" in str(exc):
            if requested_profile is None:
                raise
            logger.warning(
                (
                    "Ignoring stale formal-run state because it references a profile "
                    "that is no longer defined; starting requested profile '%s' fresh."
                ),
                requested_profile,
            )
            saved_state = None
        else:
            logger.warning(
                (
                    "Deleting legacy or incompatible formal-run state file because: "
                    "%s. Starting fresh."
                ),
                exc,
            )
            try:
                STATE_PATH.unlink(missing_ok=True)
            except Exception as unlink_exc:
                logger.debug("Failed to delete legacy state file: %s", unlink_exc)
            saved_state = None
    saved_benchmark_args: dict[str, object] | None = None

    saved_profile = None
    if saved_state is not None:
        saved_profile = str(saved_state["profile_name"])
        raw_saved_benchmark_args = saved_state["benchmark_args"]
        assert isinstance(raw_saved_benchmark_args, dict)
        saved_benchmark_args = dict(raw_saved_benchmark_args)

    if (
        requested_profile is None
        and saved_state is not None
        and saved_profile != DEFAULT_PROFILE_NAME
    ):
        logger.warning(
            (
                "Ignoring saved formal-run state for profile '%s'. Pass "
                "--profile %s to resume it; starting default profile '%s' fresh."
            ),
            saved_profile,
            saved_profile,
            DEFAULT_PROFILE_NAME,
        )

    should_resume_latest = (
        requested_profile is None
        and saved_state is not None
        and saved_profile == DEFAULT_PROFILE_NAME
    )
    if should_resume_latest:
        assert saved_benchmark_args is not None
        profile_name = saved_profile or DEFAULT_PROFILE_NAME
        expected_args = _build_new_run_args(cli_args, profile_name)
        benchmark_args, profile_name, resumed = _resolve_saved_benchmark_args(
            saved_benchmark_args,
            expected_args,
            cli_args,
            profile_name=profile_name,
            profile_slug=str(get_formal_profile(profile_name)["name"]),
        )
        return benchmark_args, profile_name, resumed

    if (
        requested_profile is not None
        and saved_state is not None
        and requested_profile == saved_profile
    ):
        assert saved_benchmark_args is not None
        expected_args = _build_new_run_args(cli_args, requested_profile)
        benchmark_args, profile_name, resumed = _resolve_saved_benchmark_args(
            saved_benchmark_args,
            expected_args,
            cli_args,
            profile_name=requested_profile,
            profile_slug=str(current_profile_bundle["name"]),
        )
        return benchmark_args, profile_name, resumed

    profile_name = requested_profile or DEFAULT_PROFILE_NAME
    benchmark_args = _build_new_run_args(cli_args, profile_name)
    return benchmark_args, profile_name, False


def _resolve_benchmark_num_neighbors_for_preset(
    benchmark_args: Mapping[str, object],
    preset: str,
    num_neighbors: list[int] | tuple[int, ...],
) -> list[int]:
    """Return the fan-out prefix needed by the preset's active depth."""
    default_config = UCaGNNConfig()
    if preset in {"lightgcn", "lightgcn_paper"}:
        required_hops = int(
            benchmark_args.get("single_branch_gnn_layers")
            or default_config.single_branch_gnn_layers,
        )
    else:
        required_hops = max(
            int(
                benchmark_args.get("interest_gnn_layers") or default_config.interest_gnn_layers,
            ),
            int(
                benchmark_args.get("conformity_gnn_layers") or default_config.conformity_gnn_layers,
            ),
        )
    if len(num_neighbors) < required_hops:
        raise ValueError(
            (
                f"num_neighbors length ({len(num_neighbors)}) must cover the active "
                f"{preset} depth ({required_hops})."
            ),
        )
    return list(num_neighbors[:required_hops])


def _benchmark_num_neighbors_summary(
    benchmark_args: Mapping[str, object],
) -> str:
    """Return a compact summary of the resolved num_neighbors payload."""
    raw_num_neighbors = benchmark_args.get("num_neighbors")
    if isinstance(raw_num_neighbors, Mapping):
        parts: list[str] = []
        for key, selected_neighbors in raw_num_neighbors.items():
            resolved = resolve_profile_num_neighbors(
                {"num_neighbors": selected_neighbors},
            )
            if resolved is None:
                continue
            parts.append(f"{key}: {format_num_neighbors_payload(resolved)}")
        return ", ".join(parts)

    resolved = resolve_profile_num_neighbors({"num_neighbors": raw_num_neighbors})
    if resolved is None:
        resolved = [list(UCaGNNConfig().num_neighbors)]
    return format_num_neighbors_payload(resolved) or ""


def build_benchmark_plan(
    args: argparse.Namespace | Mapping[str, object] | object,
) -> list[BenchmarkPlanItem]:
    """Build the ordered benchmark execution plan.

    The semantic thesis matrix remains dataset * preset, but the
    execution order intentionally keeps datasets as the innermost loop so a
    dataset-specific failure surfaces during the first method sweep instead of
    after one dataset has already exhausted every method combination.
    """
    benchmark_args = _coerce_benchmark_args(args)
    datasets = resolve_benchmark_datasets(benchmark_args["datasets"])
    presets = list(dict.fromkeys(benchmark_args["presets"]))
    num_neighbor_values_by_dataset = {
        dataset: resolve_benchmark_num_neighbor_values(benchmark_args, dataset=dataset)
        for dataset in datasets
    }
    lr_scheduler_values = resolve_benchmark_lr_scheduler_values(benchmark_args)
    lr_scheduler_values_by_preset = {
        preset: ["none"] if preset in PAPER_BASELINE_PRESETS else lr_scheduler_values
        for preset in presets
    }
    graph_policy_values = resolve_benchmark_graph_policy_values(benchmark_args)
    preprocessing_preset_values = resolve_benchmark_preprocessing_preset_values(
        benchmark_args,
    )
    return [
        (
            dataset,
            preset,
            lr_scheduler,
            None if preprocessing_preset is None else str(preprocessing_preset),
            str(graph_policy),
            tuple(num_neighbors),
        )
        for preset in presets
        for dataset in datasets
        for lr_scheduler in lr_scheduler_values_by_preset[preset]
        for preprocessing_preset in preprocessing_preset_values
        for graph_policy in graph_policy_values
        for num_neighbors in num_neighbor_values_by_dataset[dataset]
    ]


def _benchmark_item_label(
    dataset: str,
    preset: str,
    lr_scheduler: str,
    preprocessing_preset: str | None,
    graph_policy: str,
    neighbor_label: str,
) -> str:
    """Return the shared human-readable label for one benchmark item."""
    return (
        f"{dataset} / {preset} / {lr_scheduler} "
        f"/ {preprocessing_preset or 'default'} / {graph_policy} / nbr{neighbor_label}"
    )


def _record_benchmark_failure(
    failure_notes: list[str],
    *,
    dataset: str,
    preset: str,
    lr_scheduler: str,
    preprocessing_preset: str | None,
    graph_policy: str,
    neighbor_label: str,
    exc: Exception,
) -> None:
    """Log and store one benchmark failure through the shared format."""
    item_label = _benchmark_item_label(
        dataset,
        preset,
        lr_scheduler,
        preprocessing_preset,
        graph_policy,
        neighbor_label,
    )
    failure_notes.append(
        f"{item_label}: {type(exc).__name__}: {exc}",
    )
    logger.error("FAILED: %s", exc)
    traceback.print_exception(type(exc), exc, exc.__traceback__)


def run_benchmark(args: argparse.Namespace | Mapping[str, object] | object) -> int:
    """Execute the formal benchmark matrix from parsed arguments."""
    benchmark_args = _coerce_benchmark_args(args)

    configure_cli_logging()

    datasets = resolve_benchmark_datasets(benchmark_args["datasets"])
    presets = list(dict.fromkeys(benchmark_args["presets"]))
    batch_id = resolve_batch_id(benchmark_args["batch_id"], prefix="benchmark")
    profile_name = benchmark_args.get("profile_name")
    profile_slug = benchmark_args.get("profile_slug")
    experiments = build_benchmark_plan(benchmark_args)

    if not experiments:
        logger.error(
            "No benchmark experiments remain after applying the matrix filters.",
        )
        return 1

    tiers_label = (
        ",".join(benchmark_args["datasets"])
        if isinstance(benchmark_args["datasets"], list)
        else str(benchmark_args["datasets"])
    )
    print("=" * 70)
    print(f"BENCHMARK PLAN: {len(experiments)} experiments")
    print(
        f"  Datasets: {tiers_label} ({len(datasets)} datasets: {', '.join(datasets)})",
    )
    print(f"  Presets: {', '.join(presets)}")
    if benchmark_args["lr_scheduler"]:
        schedulers = benchmark_args["lr_scheduler"]
        if isinstance(schedulers, str):
            schedulers = [schedulers]
        print(f"  LR schedulers: {', '.join(schedulers)}")
    graph_policies = resolve_benchmark_graph_policy_values(benchmark_args)
    print(f"  Graph policies: {', '.join(str(policy) for policy in graph_policies)}")
    preprocessing_presets = resolve_benchmark_preprocessing_preset_values(benchmark_args)
    resolved_preprocessing_presets = [
        preset for preset in preprocessing_presets if preset is not None
    ]
    if resolved_preprocessing_presets:
        print(
            "  Preprocessing presets: "
            + ", ".join(
                str(preprocessing_preset) for preprocessing_preset in resolved_preprocessing_presets
            ),
        )
    neighbor_shapes = _benchmark_num_neighbors_summary(benchmark_args)
    if neighbor_shapes:
        print(f"  Neighbor shapes: {neighbor_shapes}")
    print(f"  Batch ID: {batch_id}")
    if profile_name:
        if profile_slug:
            print(f"  Profile: {profile_name} ({profile_slug})")
        else:
            print(f"  Profile: {profile_name}")
    if benchmark_args.get("change_note"):
        print(f"  Change note: {benchmark_args['change_note']}")
    print(f"  Resume batch: {benchmark_args['resume_batch']}")
    print("=" * 70)

    if benchmark_args["dry_run"]:
        print(
            "\n"
            f"{'#':>4} | {'Dataset':<15} | {'Preset':<12} | {'Scheduler':<12} | "
            f"{'Preproc':<24} | {'Graph':<16} | {'Neighbors':<10}",
        )
        print("-" * 140)
        for i, (ds, pr, scheduler, preprocessing_preset, graph_policy, neighbors) in enumerate(
            experiments,
            1,
        ):
            neighbor_label = format_num_neighbors_payload(neighbors) or ""
            print(
                f"{i:>4} | {ds:<15} | {pr:<12} | {scheduler:<12} | "
                f"{preprocessing_preset or '-'!s: <24} | "
                f"{graph_policy:<16} | {neighbor_label:<10}",
            )
        print(f"\nTotal: {len(experiments)} experiments (dry run, nothing executed)")
        return 0

    # Run experiments
    tracker = ExperimentLogger(db_path=str(THESIS_DB_PATH))
    completed = 0
    failed = 0
    skipped = 0
    results = []
    failure_notes: list[str] = []

    for i, (
        dataset,
        preset,
        lr_scheduler,
        preprocessing_preset,
        graph_policy,
        num_neighbors,
    ) in enumerate(experiments, 1):
        neighbor_list = list(num_neighbors)
        effective_neighbor_list = list(neighbor_list)
        raw_neighbor_label = format_num_neighbors_payload(neighbor_list) or ""
        neighbor_label = raw_neighbor_label

        try:
            effective_neighbor_list = _resolve_benchmark_num_neighbors_for_preset(
                benchmark_args,
                preset,
                neighbor_list,
            )
            neighbor_label = format_num_neighbors_payload(effective_neighbor_list) or ""
        except Exception as e:
            failed += 1
            _record_benchmark_failure(
                failure_notes,
                dataset=dataset,
                preset=preset,
                lr_scheduler=lr_scheduler,
                preprocessing_preset=preprocessing_preset,
                graph_policy=graph_policy,
                neighbor_label=neighbor_label,
                exc=e,
            )
            continue

        print(f"\n{'=' * 70}")
        print(
            f"[{i}/{len(experiments)}] "
            + _benchmark_item_label(
                dataset,
                preset,
                lr_scheduler,
                preprocessing_preset,
                graph_policy,
                neighbor_label,
            ),
        )
        print("=" * 70)

        try:
            config = build_config(
                build_benchmark_config_inputs(
                    benchmark_args,
                    dataset=dataset,
                    preset=preset,
                    lr_scheduler=lr_scheduler,
                    num_neighbors=effective_neighbor_list,
                    preprocessing_preset=preprocessing_preset,
                    graph_policy=graph_policy,
                ),
            )
        except Exception as e:
            failed += 1
            _record_benchmark_failure(
                failure_notes,
                dataset=dataset,
                preset=preset,
                lr_scheduler=lr_scheduler,
                preprocessing_preset=preprocessing_preset,
                graph_policy=graph_policy,
                neighbor_label=neighbor_label,
                exc=e,
            )
            continue

        existing = tracker.find_latest_batch_experiment(
            batch_id=batch_id,
            dataset=dataset,
            preset=preset,
            intervention=None,
            seed=DEFAULT_SEED,
            training_mode="mini_batch",
            config_filters={
                "graph_policy": graph_policy,
                "num_neighbors": effective_neighbor_list,
                **(
                    {"preprocessing_preset": preprocessing_preset}
                    if preprocessing_preset is not None
                    else {}
                ),
            },
        )
        recovered_checkpoint = None
        if not bool(benchmark_args.get("overwrite_checkpoint")):
            recovered_checkpoint = recoverable_checkpoint_for_config(
                config,
                preset=preset,
            )

        should_skip_existing = (
            benchmark_args["resume_batch"]
            and existing is not None
            and existing["status"] in ExperimentLogger.TERMINAL_STATUSES
        )
        if (
            should_skip_existing
            and existing["status"] in {"failed", "oom"}
            and not bool(benchmark_args.get("overwrite_checkpoint"))
            and recovered_checkpoint is not None
        ):
            should_skip_existing = False
            logger.info(
                "Retrying failed batch item (exp_id=%s, status=%s) from recoverable checkpoint.",
                existing["id"],
                existing["status"],
            )
        if should_skip_existing:
            skipped += 1
            logger.info(
                "Skipping existing batch item (exp_id=%s, status=%s)",
                existing["id"],
                existing["status"],
            )
            if existing["status"] == "completed":
                results.append(
                    {
                        "dataset": dataset,
                        "preset": preset,
                        "preprocessing_preset": preprocessing_preset,
                        "graph_policy": graph_policy,
                        "num_neighbors": neighbor_list,
                        "exp_id": existing["id"],
                        "metrics": tracker.get_metrics_for_split(
                            int(existing["id"]),
                            split="test",
                        ),
                        "elapsed_s": 0.0,
                        "run_mode": "existing",
                        "peak_vram_mb": None,
                        "epochs_stopped_at": None,
                        "checkpoint_path": None,
                    },
                )
            continue

        try:
            run_config = config
            run_checkpoint_path = None
            if recovered_checkpoint is not None:
                run_config, recovered_checkpoint_path = recovered_checkpoint
                run_checkpoint_path = str(recovered_checkpoint_path)
            t0 = time.time()
            result = run_experiment(
                run_config,
                preset=preset,
                enable_mlflow=not bool(benchmark_args["no_mlflow"]),
                mlflow_tracking_uri=benchmark_args["mlflow_tracking_uri"],
                mlflow_experiment_name=str(benchmark_args["mlflow_experiment_name"]),
                batch_id=batch_id,
                profile_name=profile_name,
                overwrite_checkpoint=bool(benchmark_args.get("overwrite_checkpoint")),
                change_note=benchmark_args.get("change_note"),
                checkpoint_path=run_checkpoint_path,
            )
            elapsed = time.time() - t0
            runtime_probe_estimate = _runtime_probe_estimate_from_result(
                benchmark_args,
                result,
            )
            if runtime_probe_estimate is not None:
                _log_runtime_probe_estimate(
                    tracker,
                    int(result["exp_id"]),
                    runtime_probe_estimate,
                )
                logger.info(
                    (
                        "Runtime probe approximation: %.1fs/epoch, %.2f batch/s, "
                        "estimated %.1fs for %.0f epochs"
                    ),
                    runtime_probe_estimate["runtime_probe_seconds_per_epoch"],
                    runtime_probe_estimate["runtime_probe_observed_batches_per_second"],
                    runtime_probe_estimate["runtime_probe_estimated_train_time_s"],
                    runtime_probe_estimate["runtime_probe_target_epochs"],
                )

            results.append(
                {
                    "dataset": dataset,
                    "preset": preset,
                    "preprocessing_preset": preprocessing_preset,
                    "graph_policy": graph_policy,
                    "num_neighbors": effective_neighbor_list,
                    "exp_id": result["exp_id"],
                    "metrics": result["test_metrics"],
                    "elapsed_s": elapsed,
                    "run_mode": "resumed" if result.get("resumed") else "new",
                    "peak_vram_mb": result.get("peak_vram_mb"),
                    "epochs_stopped_at": result.get("epochs_stopped_at"),
                    "checkpoint_path": result.get("checkpoint_path"),
                    "runtime_probe_estimate": runtime_probe_estimate,
                },
            )
            completed += 1
            logger.info(f"Completed in {elapsed:.1f}s (exp_id={result['exp_id']})")

        except Exception as e:
            failed += 1
            _record_benchmark_failure(
                failure_notes,
                dataset=dataset,
                preset=preset,
                lr_scheduler=lr_scheduler,
                preprocessing_preset=preprocessing_preset,
                graph_policy=graph_policy,
                neighbor_label=neighbor_label,
                exc=e,
            )
            continue

    sorted_results = sorted(
        results,
        key=lambda r: (
            str(r["dataset"]).lower(),
            -metric_value(r["metrics"], "NDCG@20"),
            -metric_value(r["metrics"], "NDCG@40"),
            -metric_value(r["metrics"], "Recall@20"),
            -metric_value(r["metrics"], "Recall@40"),
            metric_value(r["metrics"], "AveragePopularity@20"),
            metric_value(r["metrics"], "AveragePopularity@40"),
            str(r["preset"]).lower(),
            str(r.get("preprocessing_preset") or "").lower(),
            str(r["graph_policy"]).lower(),
            tuple(r["num_neighbors"]),
        ),
    )
    print_batch_summary_counts(
        title="BENCHMARK SUMMARY",
        completed=completed,
        failed=failed,
        skipped=skipped,
        total=len(experiments),
    )
    if sorted_results:
        print("Note: AvgPop@20 and AvgPop@40 are lower-is-better.")
        print(
            (
                "\n"
                + (
                    f"{'Dataset':<15} | {'Preset':<12} | "
                    f"{'Preproc':<24} | {'Graph':<16} | {'Neighbors':<10} | {'NDCG@20':>8} | "
                    f"{'Recall@20':>10} | {'AvgPop@20':>10} | {'NDCG@40':>8} | "
                    f"{'Recall@40':>10} | {'AvgPop@40':>10} | {'Epochs':>6} | "
                    f"{'PeakVRAM':>8} | Time | {'Mode':<8} | {'Experiment':<40}"
                )
            ),
        )
        print("-" * 253)
        for r in sorted_results:
            metric_values = thesis_metric_values(
                r["metrics"],
                THESIS_PRIMARY_METRICS,
            )
            neighbor_label = format_num_neighbors_payload(r["num_neighbors"]) or ""
            epochs = (
                str(r.get("epochs_stopped_at")) if r.get("epochs_stopped_at") is not None else "-"
            )
            peak_vram = f"{r['peak_vram_mb']:.0f}MB" if r.get("peak_vram_mb") is not None else "-"
            checkpoint_label = (
                Path(r["checkpoint_path"]).name if r.get("checkpoint_path") else f"exp{r['exp_id']}"
            )
            print(
                (
                    f"{r['dataset']:<15} | {r['preset']:<12} | "
                    f"{r.get('preprocessing_preset') or '-'!s: <24} | "
                    f"{r['graph_policy']:<16} | "
                    f"{neighbor_label:<10} | {metric_values['NDCG@20']:>8.4f} | "
                    f"{metric_values['Recall@20']:>10.4f} | "
                    f"{metric_values['AveragePopularity@20']:>10.4f} | "
                    f"{metric_values['NDCG@40']:>8.4f} | "
                    f"{metric_values['Recall@40']:>10.4f} | "
                    f"{metric_values['AveragePopularity@40']:>10.4f} | "
                    f"{epochs:>6} | {peak_vram:>8} | {r['elapsed_s']:.0f}s | "
                    f"{r['run_mode']:<8} | {checkpoint_label:<40}"
                ),
            )

    if failure_notes:
        print("\nFAILURE NOTES")
        for note in failure_notes:
            print(f"- {note}")

    tracker.close()
    return 0 if failed == 0 else 1


def main() -> int:
    """Parse CLI arguments and run the benchmark matrix."""
    args = build_benchmark_parser().parse_args()
    return run_benchmark(args)


def _run_single_formal_profile(
    profile_name: str | None,
    cli_args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    """Run one formal profile using the existing saved-state resolution rules."""
    profile_args = argparse.Namespace(**vars(cli_args))
    profile_args.profile = profile_name
    try:
        benchmark_args, resolved_profile_name, resumed = _resolve_benchmark_args(profile_args)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc.args[0] if exc.args else exc))

    state = {
        "profile_name": resolved_profile_name,
        "profile_slug": benchmark_args.get("profile_slug"),
        "batch_id": benchmark_args["batch_id"],
        "resumed": resumed,
        "last_started_at_utc": datetime.now(UTC).isoformat(),
        "last_finished_at_utc": None,
        "last_exit_code": None,
        "benchmark_args": dict(benchmark_args),
    }
    _write_state(state)

    print("=" * 70)
    print("FORMAL RUN")
    if benchmark_args.get("profile_slug"):
        print(f"  Profile: {resolved_profile_name} ({benchmark_args['profile_slug']})")
    else:
        print(f"  Profile: {resolved_profile_name}")
    if benchmark_args.get("change_note"):
        print(f"  Change note: {benchmark_args['change_note']}")
    print(f"  Batch ID: {benchmark_args['batch_id']}")
    print(f"  Resuming: {benchmark_args['resume_batch']}")
    print("  Full datasets: True")
    print("  OOM fallback: log-and-continue")
    print("=" * 70)

    exit_code = run_benchmark(benchmark_args)
    state["last_finished_at_utc"] = datetime.now(UTC).isoformat()
    state["last_exit_code"] = exit_code
    _write_state(state)
    return exit_code


def formal_main() -> int:
    """Run the formal experiment workflow through one simple entry point."""
    parser = build_formal_run_parser()
    cli_args = parser.parse_args()

    if cli_args.list_profiles:
        print("Available formal profiles:")
        for profile_name in formal_profile_names():
            profile = get_formal_profile(profile_name)
            print(f"  {profile['id']}: {profile['description']} [{profile['name']}]")
        return 0

    try:
        profile_names = _parse_formal_profile_sequence(cli_args.profile)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc.args[0] if exc.args else exc))

    if cli_args.profile is None:
        return _run_single_formal_profile(None, cli_args, parser)

    if len(profile_names) > 1:
        print("=" * 70)
        print("FORMAL RUN PROFILE QUEUE")
        for index, profile_name in enumerate(profile_names, 1):
            print(f"  {index}. {profile_name}")
        print("=" * 70)

    exit_codes: list[int] = []
    for profile_name in profile_names:
        exit_codes.append(_run_single_formal_profile(profile_name, cli_args, parser))

    return 0 if all(exit_code == 0 for exit_code in exit_codes) else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
