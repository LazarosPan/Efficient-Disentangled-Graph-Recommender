#!/usr/bin/env python
"""Formal matrix benchmark runner for dataset x preset.

Usage:
    uv run formal-run --dry-run
    uv run formal-run --profile default
    uv run formal-run --resume-latest
"""

from __future__ import annotations

import argparse
import json
import logging
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
from src.utils.cli_parsers import (
    normalize_benchmark_datasets_arg,
    resolve_benchmark_datasets,
)
from src.utils.config import DEFAULT_SEED, SUPPORTED_LR_SCHEDULERS, UCaGNNConfig
from src.utils.experiment_logger import ExperimentLogger
from src.utils.project_paths import FORMAL_RUN_STATE_PATH, THESIS_DB_PATH

from experiments.cli_parsers import build_benchmark_parser, build_formal_run_parser
from experiments.recipes import (
    default_formal_profile_name,
    formal_profile_names,
    get_formal_profile,
)
from experiments.run_experiment import (
    BENCHMARK_CONFIG_FIELDS,
    build_benchmark_config_inputs,
    build_config,
    normalize_benchmark_config_overrides,
    normalize_config_inputs,
    run_experiment,
)

logger = logging.getLogger("ucagnn.benchmark")
STATE_PATH = FORMAL_RUN_STATE_PATH
DEFAULT_PROFILE_NAME = default_formal_profile_name()


DEFAULT_SCORING_WEIGHT_MODES = ["learned"]
BENCHMARK_MATRIX_FIELDS = (
    "datasets",
    "presets",
    "scoring_weight_modes",
    "profile_name",
    "profile_slug",
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
    field_name for field_name in NORMALIZED_BENCHMARK_FIELDS if field_name not in RUNTIME_ONLY_BENCHMARK_FIELDS
)


def _write_state(payload: dict[str, object]) -> None:
    """Persist the formal-run state file."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


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
            f"Formal-run state contains unexpected fields {unexpected_fields}. Start a fresh formal run.",
        )

    normalized_args = {field_name: benchmark_args.get(field_name) for field_name in BENCHMARK_MATRIX_FIELDS}
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
    scoring_weight_modes = normalized_args["scoring_weight_modes"]
    if not isinstance(scoring_weight_modes, (list, tuple)) or not scoring_weight_modes:
        scoring_weight_modes = DEFAULT_SCORING_WEIGHT_MODES
    normalized_args["datasets"] = normalize_benchmark_datasets_arg(
        benchmark_args.get("datasets"),
    )
    normalized_args["presets"] = list(benchmark_args["presets"])
    normalized_args["scoring_weight_modes"] = list(scoring_weight_modes)
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
        and isinstance(
            benchmark_args.get("scoring_weight_modes"),
            list,
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
    # CLI --datasets overrides the profile's dataset selection.
    cli_datasets = getattr(cli_args, "datasets", None)
    datasets_value = cli_datasets if cli_datasets is not None else matrix.get("datasets", "all")
    benchmark_args = normalize_benchmark_config_overrides(config_overrides)
    benchmark_args.update(
        {
            "datasets": normalize_benchmark_datasets_arg(datasets_value),
            "presets": list(matrix["presets"]),
            "scoring_weight_modes": list(
                matrix.get("scoring_weight_modes", DEFAULT_SCORING_WEIGHT_MODES),
            ),
            "profile_name": str(profile_bundle["id"]),
            "profile_slug": str(profile_bundle["name"]),
            "change_note": getattr(cli_args, "change_note", None),
            "device": cli_args.device or "cuda",
            "data_dir": cli_args.data_dir or "data",
            "no_mlflow": bool(cli_args.no_mlflow),
            "mlflow_tracking_uri": cli_args.mlflow_tracking_uri,
            "mlflow_experiment_name": cli_args.mlflow_experiment_name or "ucagnn-formal",
            "overwrite_checkpoint": bool(
                getattr(cli_args, "overwrite_checkpoint", False),
            ),
            "batch_id": resolve_batch_id(None, prefix=f"formal-{profile_name}"),
            "resume_batch": True,
            "dry_run": cli_args.dry_run,
        },
    )
    return benchmark_args


def _override_resumed_args(
    benchmark_args: dict[str, object],
    cli_args: argparse.Namespace,
    profile_name: str,
) -> dict[str, object]:
    """Apply a small set of runtime overrides when resuming a saved run."""
    overridden_args = dict(benchmark_args)
    for attribute in (
        "device",
        "data_dir",
        "mlflow_tracking_uri",
        "mlflow_experiment_name",
    ):
        value = getattr(cli_args, attribute)
        if value is not None:
            overridden_args[attribute] = value
    if cli_args.no_mlflow:
        overridden_args["no_mlflow"] = True
    if getattr(cli_args, "overwrite_checkpoint", False):
        overridden_args["overwrite_checkpoint"] = True
    overridden_args["dry_run"] = cli_args.dry_run
    overridden_args["resume_batch"] = True
    overridden_args["sample_interactions"] = None
    overridden_args["loader_max_rows"] = None
    if cli_args.restart:
        overridden_args["batch_id"] = resolve_batch_id(
            None,
            prefix=f"formal-{profile_name}",
        )
    return overridden_args


def _resolve_benchmark_args(
    cli_args: argparse.Namespace,
) -> tuple[dict[str, object], str, bool]:
    """Resolve whether to create a new formal run or resume the saved one."""
    saved_state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else None
    current_profile_bundle = None
    saved_benchmark_args: dict[str, object] | None = None

    requested_profile = None
    if cli_args.profile is not None:
        current_profile_bundle = get_formal_profile(cli_args.profile)
        requested_profile = str(current_profile_bundle["id"])
    saved_profile = None
    saved_profile_bundle = None
    if saved_state is not None:
        raw_saved_profile = str(
            saved_state.get("profile_name") or saved_state.get("version") or DEFAULT_PROFILE_NAME,
        )
        try:
            saved_profile_bundle = get_formal_profile(raw_saved_profile)
            saved_profile = str(saved_profile_bundle["id"])
        except KeyError:
            saved_profile = raw_saved_profile
            saved_profile_bundle = saved_state.get("profile_config")
        raw_saved_benchmark_args = saved_state.get("benchmark_args")
        if isinstance(raw_saved_benchmark_args, dict):
            try:
                saved_benchmark_args = _coerce_benchmark_args(
                    raw_saved_benchmark_args,
                    fallback_profile_name=raw_saved_profile,
                )
            except ValueError:
                # Stale state with an unrecognized schema; treat as missing.
                saved_benchmark_args = None
                saved_profile = None
                saved_profile_bundle = None

    should_resume_latest = cli_args.resume_latest or (
        requested_profile is None and not cli_args.new_run and saved_state is not None
    )
    if should_resume_latest:
        if saved_state is None:
            raise ValueError("No saved formal-run state exists to resume.")
        if saved_profile_bundle is None:
            if cli_args.resume_latest:
                raise ValueError(
                    (
                        "The saved formal-run state references a profile that is no longer "
                        "defined. Use --new-run or --restart to start a fresh batch."
                    ),
                )
            profile_name = requested_profile or DEFAULT_PROFILE_NAME
            benchmark_args = _build_new_run_args(cli_args, profile_name)
            return benchmark_args, profile_name, False
        assert saved_benchmark_args is not None
        benchmark_args = saved_benchmark_args
        profile_name = saved_profile or DEFAULT_PROFILE_NAME
        expected_args = _build_new_run_args(cli_args, profile_name)
        if _benchmark_plan_signature(benchmark_args) != _benchmark_plan_signature(
            expected_args,
        ):
            if cli_args.resume_latest:
                raise ValueError(
                    (
                        "The saved formal-run state no longer matches the current profile "
                        "bundle. Use --new-run or --restart to start a fresh "
                        "mini-batch-only batch."
                    ),
                )
            benchmark_args = expected_args
            return benchmark_args, profile_name, False
        benchmark_args = _override_resumed_args(benchmark_args, cli_args, profile_name)
        benchmark_args["profile_name"] = profile_name
        benchmark_args["profile_slug"] = str(get_formal_profile(profile_name)["name"])
        return benchmark_args, profile_name, True

    if (
        requested_profile is not None
        and saved_state is not None
        and not cli_args.new_run
        and not cli_args.restart
        and requested_profile == saved_profile
        and saved_profile_bundle == current_profile_bundle
    ):
        assert saved_benchmark_args is not None
        benchmark_args = saved_benchmark_args
        expected_args = _build_new_run_args(cli_args, requested_profile)
        if _benchmark_plan_signature(benchmark_args) != _benchmark_plan_signature(
            expected_args,
        ):
            profile_name = requested_profile
            benchmark_args = _build_new_run_args(cli_args, profile_name)
            return benchmark_args, profile_name, False
        benchmark_args = _override_resumed_args(
            benchmark_args,
            cli_args,
            requested_profile,
        )
        benchmark_args["profile_name"] = requested_profile
        benchmark_args["profile_slug"] = str(current_profile_bundle["name"])
        return benchmark_args, requested_profile, True

    profile_name = requested_profile or DEFAULT_PROFILE_NAME
    benchmark_args = _build_new_run_args(cli_args, profile_name)
    return benchmark_args, profile_name, False


def _scoring_weight_modes_for_preset(
    preset: str,
    requested_modes: list[str],
) -> list[str]:
    """Return the applicable scoring-weight sweep values for one preset."""
    if preset in {"lightgcn", "dice_like"}:
        return ["fixed"] if "fixed" in requested_modes else []
    return list(requested_modes)


def _resolve_benchmark_num_neighbors_for_preset(
    benchmark_args: Mapping[str, object],
    preset: str,
    num_neighbors: list[int] | tuple[int, ...],
) -> list[int]:
    """Return the fan-out prefix needed by the preset's active depth."""
    default_config = UCaGNNConfig()
    if preset == "lightgcn":
        required_hops = int(
            benchmark_args.get("single_branch_gnn_layers") or default_config.single_branch_gnn_layers,
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
            f"num_neighbors length ({len(num_neighbors)}) must cover the active {preset} depth ({required_hops}).",
        )
    return list(num_neighbors[:required_hops])


def build_benchmark_plan(
    args: argparse.Namespace | Mapping[str, object] | object,
) -> list[tuple[str, str, str, str, tuple[int, ...]]]:
    """Build the ordered benchmark execution plan.

    The semantic thesis matrix remains dataset * preset, but the
    execution order intentionally keeps datasets as the innermost loop so a
    dataset-specific failure surfaces during the first method sweep instead of
    after one dataset has already exhausted every method combination.
    """
    benchmark_args = _coerce_benchmark_args(args)
    datasets = resolve_benchmark_datasets(benchmark_args["datasets"])
    presets = list(dict.fromkeys(benchmark_args["presets"]))
    num_neighbors_options = benchmark_args["num_neighbors_options"] or (
        [benchmark_args["num_neighbors"]]
        if benchmark_args["num_neighbors"] is not None
        else [list(UCaGNNConfig().num_neighbors)]
    )
    lr_scheduler_values = benchmark_args["lr_scheduler"]
    if isinstance(lr_scheduler_values, str):
        lr_scheduler_values = [lr_scheduler_values]
    if lr_scheduler_values == ["all"]:
        lr_scheduler_values = list(SUPPORTED_LR_SCHEDULERS)
    return [
        (dataset, preset, scoring_weight_mode, lr_scheduler, tuple(num_neighbors))
        for preset in presets
        for scoring_weight_mode in _scoring_weight_modes_for_preset(
            preset,
            benchmark_args["scoring_weight_modes"],
        )
        for dataset in datasets
        for lr_scheduler in lr_scheduler_values
        for num_neighbors in num_neighbors_options
    ]


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
    print(f"  Score-mix modes: {', '.join(benchmark_args['scoring_weight_modes'])}")
    if benchmark_args["lr_scheduler"]:
        schedulers = benchmark_args["lr_scheduler"]
        if isinstance(schedulers, str):
            schedulers = [schedulers]
        print(f"  LR schedulers: {', '.join(schedulers)}")
    if benchmark_args["num_neighbors_options"]:
        neighbor_shapes = ", ".join(
            "[" + ", ".join(str(value) for value in neighbors) + "]"
            for neighbors in benchmark_args["num_neighbors_options"]
        )
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
            f"\n{'#':>4} | {'Dataset':<15} | {'Preset':<12} | {'ScoreMix':<8} | {'Scheduler':<12} | {'Neighbors':<10}",
        )
        print("-" * 106)
        for i, (ds, pr, swm, scheduler, neighbors) in enumerate(experiments, 1):
            neighbor_label = "-".join(str(value) for value in neighbors)
            print(
                f"{i:>4} | {ds:<15} | {pr:<12} | {swm:<8} | {scheduler:<12} | {neighbor_label:<10}",
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

    for i, (dataset, preset, scoring_weight_mode, lr_scheduler, num_neighbors) in enumerate(experiments, 1):
        neighbor_list = list(num_neighbors)
        effective_neighbor_list = list(neighbor_list)
        raw_neighbor_label = "-".join(str(value) for value in neighbor_list)
        neighbor_label = raw_neighbor_label

        try:
            effective_neighbor_list = _resolve_benchmark_num_neighbors_for_preset(
                benchmark_args,
                preset,
                neighbor_list,
            )
            neighbor_label = "-".join(str(value) for value in effective_neighbor_list)
        except Exception as e:
            failed += 1
            failure_notes.append(
                (
                    f"{dataset} / {preset} / {scoring_weight_mode} / {lr_scheduler} / "
                    f"nbr{neighbor_label}: {type(e).__name__}: {e}"
                ),
            )
            logger.error(f"FAILED: {e}")
            traceback.print_exc()
            continue

        print(f"\n{'=' * 70}")
        print(
            f"[{i}/{len(experiments)}] {dataset} / {preset} / {scoring_weight_mode} / nbr{neighbor_label}",
        )
        print("=" * 70)

        existing = tracker.find_latest_batch_experiment(
            batch_id=batch_id,
            dataset=dataset,
            preset=preset,
            intervention=None,
            seed=DEFAULT_SEED,
            training_mode="mini_batch",
            config_filters={
                "scoring_weight_mode": scoring_weight_mode,
                "num_neighbors": effective_neighbor_list,
            },
        )
        if (
            benchmark_args["resume_batch"]
            and existing is not None
            and existing["status"] in ExperimentLogger.TERMINAL_STATUSES
        ):
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
                        "scoring_weight_mode": scoring_weight_mode,
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
            config = build_config(
                build_benchmark_config_inputs(
                    benchmark_args,
                    dataset=dataset,
                    preset=preset,
                    lr_scheduler=lr_scheduler,
                    scoring_weight_mode=scoring_weight_mode,
                    num_neighbors=effective_neighbor_list,
                ),
            )
            t0 = time.time()
            result = run_experiment(
                config,
                preset=preset,
                enable_mlflow=not bool(benchmark_args["no_mlflow"]),
                mlflow_tracking_uri=benchmark_args["mlflow_tracking_uri"],
                mlflow_experiment_name=str(benchmark_args["mlflow_experiment_name"]),
                batch_id=batch_id,
                profile_name=profile_name,
                overwrite_checkpoint=bool(benchmark_args.get("overwrite_checkpoint")),
                change_note=benchmark_args.get("change_note"),
            )
            elapsed = time.time() - t0

            results.append(
                {
                    "dataset": dataset,
                    "preset": preset,
                    "scoring_weight_mode": scoring_weight_mode,
                    "num_neighbors": effective_neighbor_list,
                    "exp_id": result["exp_id"],
                    "metrics": result["test_metrics"],
                    "elapsed_s": elapsed,
                    "run_mode": "resumed" if result.get("resumed") else "new",
                    "peak_vram_mb": result.get("peak_vram_mb"),
                    "epochs_stopped_at": result.get("epochs_stopped_at"),
                    "checkpoint_path": result.get("checkpoint_path"),
                },
            )
            completed += 1
            logger.info(f"Completed in {elapsed:.1f}s (exp_id={result['exp_id']})")

        except Exception as e:
            failed += 1
            failure_notes.append(
                f"{dataset} / {preset} / {scoring_weight_mode} / nbr{neighbor_label}: {type(e).__name__}: {e}",
            )
            logger.error(f"FAILED: {e}")
            traceback.print_exc()
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
            str(r["scoring_weight_mode"]).lower(),
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
                    f"{'Dataset':<15} | {'Preset':<12} | {'ScoreMix':<8} | "
                    f"{'Neighbors':<10} | {'NDCG@20':>8} | {'Recall@20':>10} | "
                    f"{'AvgPop@20':>10} | {'NDCG@40':>8} | {'Recall@40':>10} | "
                    f"{'AvgPop@40':>10} | {'Epochs':>6} | {'PeakVRAM':>8} | "
                    f"Time | {'Mode':<8} | {'Experiment':<40}"
                )
            ),
        )
        print("-" * 219)
        for r in sorted_results:
            metric_values = thesis_metric_values(
                r["metrics"],
                (
                    "NDCG@20",
                    "Recall@20",
                    "AveragePopularity@20",
                    "NDCG@40",
                    "Recall@40",
                    "AveragePopularity@40",
                ),
            )
            neighbor_label = "-".join(str(value) for value in r["num_neighbors"])
            epochs = str(r.get("epochs_stopped_at")) if r.get("epochs_stopped_at") is not None else "-"
            peak_vram = f"{r['peak_vram_mb']:.0f}MB" if r.get("peak_vram_mb") is not None else "-"
            checkpoint_label = Path(r["checkpoint_path"]).name if r.get("checkpoint_path") else f"exp{r['exp_id']}"
            print(
                (
                    f"{r['dataset']:<15} | {r['preset']:<12} | "
                    f"{r['scoring_weight_mode']:<8} | {neighbor_label:<10} | "
                    f"{metric_values['NDCG@20']:>8.4f} | "
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
        benchmark_args, profile_name, resumed = _resolve_benchmark_args(cli_args)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc.args[0] if exc.args else exc))

    state = {
        "profile_name": profile_name,
        "profile_slug": benchmark_args.get("profile_slug"),
        "batch_id": benchmark_args["batch_id"],
        "resumed": resumed,
        "last_started_at_utc": datetime.now(UTC).isoformat(),
        "last_finished_at_utc": None,
        "last_exit_code": None,
        "benchmark_args": dict(benchmark_args),
    }
    if not cli_args.dry_run:
        _write_state(state)

    print("=" * 70)
    print("FORMAL RUN")
    if benchmark_args.get("profile_slug"):
        print(f"  Profile: {profile_name} ({benchmark_args['profile_slug']})")
    else:
        print(f"  Profile: {profile_name}")
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
    if not cli_args.dry_run:
        _write_state(state)
    return exit_code


if __name__ == "__main__":
    import sys

    sys.exit(main())
