#!/usr/bin/env python
"""Formal matrix benchmark runner for dataset x preset x graph_method.

Usage:
    python experiments/run_benchmark.py --tier small --dry-run
    python experiments/run_benchmark.py --tier small --presets ucagnn
    python experiments/run_benchmark.py --tier all
"""

from __future__ import annotations

import argparse
import json
from itertools import product
import logging
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.recipes import (
    default_formal_profile_name,
    formal_profile_names,
    get_formal_profile,
)
from experiments.run_experiment import (
    DB_PATH,
    DEFAULT_PRESET_ORDER,
    PRESETS,
    build_config,
    canonicalize_preset_names,
    run_experiment,
)
from scripts._workflow_helpers import metric_value, resolve_batch_id
from src.utils.config import DEFAULT_SEED
from src.utils.experiment_logger import ExperimentLogger

logger = logging.getLogger("ucagnn.benchmark")
STATE_PATH = Path(__file__).parent.parent / "results" / "formal_run_state.json"
DEFAULT_PROFILE_NAME = default_formal_profile_name()

TIERS = {
    "small": ["amazonbook", "movielens1m"],
    "medium": ["movielens20m", "kuairec_v2"],
    "large": ["taobao", "kuairand1k"],
}
TIERS["all"] = TIERS["small"] + TIERS["medium"] + TIERS["large"]

DEFAULT_GRAPH_METHODS = ["cagra", "knn"]
DEFAULT_SCORING_WEIGHT_MODES = ["learned"]
SCORING_WEIGHT_MODE_CHOICES = ["learned", "fixed"]


def _normalize_profile_name(raw_profile: str) -> str:
    """Normalize a user-facing formal profile label into a filesystem-safe slug."""
    normalized = "".join(
        character.lower() if character.isalnum() else "-"
        for character in raw_profile.strip()
    )
    collapsed = "-".join(part for part in normalized.split("-") if part)
    if not collapsed:
        raise ValueError("Profile label must contain at least one letter or number.")
    return collapsed


def _resolve_profile_bundle(profile_name: str) -> dict[str, object]:
    """Return the predefined support-parameter bundle for a formal profile."""
    try:
        return get_formal_profile(profile_name)
    except KeyError as exc:
        supported = ", ".join(formal_profile_names())
        raise ValueError(
            f"Unknown formal profile {profile_name!r}. Supported profiles: {supported}."
        ) from exc


def _build_batch_id(profile_name: str) -> str:
    """Build a fresh execution batch identifier for a formal profile run."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"formal-{profile_name}-{timestamp}"


def _load_state() -> dict[str, object] | None:
    """Load the last formal-run state file if it exists."""
    if not STATE_PATH.exists():
        return None
    return json.loads(STATE_PATH.read_text())


def _write_state(payload: dict[str, object]) -> None:
    """Persist the formal-run state file."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _args_to_state(args: argparse.Namespace) -> dict[str, object]:
    """Convert benchmark arguments into a JSON-safe state payload."""
    return {
        "tier": args.tier,
        "presets": list(args.presets),
        "graph_methods": list(args.graph_methods),
        "scoring_weight_modes": list(args.scoring_weight_modes),
        "profile_name": args.profile_name,
        "epochs": args.epochs,
        "use_early_stopping": args.use_early_stopping,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "single_branch_gnn_layers": args.single_branch_gnn_layers,
        "interest_gnn_layers": args.interest_gnn_layers,
        "conformity_gnn_layers": args.conformity_gnn_layers,
        "dropout": args.dropout,
        "num_neighbors": list(args.num_neighbors)
        if args.num_neighbors is not None
        else None,
        "hard_negative_ratio": args.hard_negative_ratio,
        "curriculum_phase1_end": args.curriculum_phase1_end,
        "curriculum_phase2_end": args.curriculum_phase2_end,
        "loss_schedule": getattr(args, "loss_schedule", None),
        "loader_max_rows": args.loader_max_rows,
        "device": args.device,
        "data_dir": args.data_dir,
        "no_mlflow": args.no_mlflow,
        "mlflow_tracking_uri": args.mlflow_tracking_uri,
        "mlflow_experiment_name": args.mlflow_experiment_name,
        "batch_id": args.batch_id,
        "resume_batch": args.resume_batch,
        "dry_run": args.dry_run,
    }


def _state_to_args(state: dict[str, object]) -> argparse.Namespace:
    """Restore benchmark arguments from saved state."""
    benchmark_args = state["benchmark_args"]
    assert isinstance(benchmark_args, dict)
    scoring_weight_modes = benchmark_args.get("scoring_weight_modes")
    if not isinstance(scoring_weight_modes, list) or not scoring_weight_modes:
        scoring_weight_modes = list(DEFAULT_SCORING_WEIGHT_MODES)
    legacy_depth = benchmark_args.get("n_gnn_layers")
    normalized_args = {
        "batch_size": None,
        "lr": None,
        "num_neighbors": None,
        "hard_negative_ratio": None,
        "curriculum_phase1_end": None,
        "curriculum_phase2_end": None,
        "loss_schedule": None,
        "loader_max_rows": None,
        "sample_interactions": None,
        "scoring_weight_modes": list(scoring_weight_modes),
        "use_early_stopping": benchmark_args.get("use_early_stopping", True),
        "profile_name": state.get("profile_name") or state.get("version"),
        "single_branch_gnn_layers": benchmark_args.get(
            "single_branch_gnn_layers", legacy_depth
        ),
        "interest_gnn_layers": benchmark_args.get("interest_gnn_layers", legacy_depth),
        "conformity_gnn_layers": benchmark_args.get(
            "conformity_gnn_layers", legacy_depth
        ),
        "dropout": benchmark_args.get("dropout"),
        **benchmark_args,
    }
    return argparse.Namespace(**normalized_args)


def _plans_match(
    saved_args: argparse.Namespace,
    expected_args: argparse.Namespace,
) -> bool:
    """Return whether a saved formal benchmark plan matches the current profile plan."""
    comparable_fields = (
        "tier",
        "presets",
        "graph_methods",
        "scoring_weight_modes",
        "epochs",
        "use_early_stopping",
        "batch_size",
        "lr",
        "single_branch_gnn_layers",
        "interest_gnn_layers",
        "conformity_gnn_layers",
        "dropout",
        "num_neighbors",
        "hard_negative_ratio",
        "curriculum_phase1_end",
        "curriculum_phase2_end",
        "loss_schedule",
        "loader_max_rows",
        "sample_interactions",
        "profile_name",
    )
    for field_name in comparable_fields:
        if getattr(saved_args, field_name, None) != getattr(
            expected_args, field_name, None
        ):
            return False
    return True


def _build_new_run_args(
    cli_args: argparse.Namespace,
    profile_name: str,
) -> argparse.Namespace:
    """Build benchmark arguments for a fresh formal run."""
    profile_bundle = _resolve_profile_bundle(profile_name)
    matrix = profile_bundle["matrix"]
    assert isinstance(matrix, dict)
    config_overrides = profile_bundle["config_overrides"]
    assert isinstance(config_overrides, dict)
    legacy_depth = config_overrides.get("n_gnn_layers")
    return argparse.Namespace(
        tier=str(matrix["tier"]),
        presets=list(matrix["presets"]),
        graph_methods=list(matrix["graph_methods"]),
        scoring_weight_modes=list(
            matrix.get("scoring_weight_modes", DEFAULT_SCORING_WEIGHT_MODES)
        ),
        profile_name=profile_name,
        epochs=int(config_overrides["epochs"]),
        use_early_stopping=bool(config_overrides.get("use_early_stopping", True)),
        batch_size=int(config_overrides["batch_size"]),
        lr=float(config_overrides["lr"]),
        single_branch_gnn_layers=int(config_overrides["single_branch_gnn_layers"])
        if config_overrides.get("single_branch_gnn_layers") is not None
        else int(legacy_depth)
        if legacy_depth is not None
        else None,
        interest_gnn_layers=int(config_overrides["interest_gnn_layers"])
        if config_overrides.get("interest_gnn_layers") is not None
        else int(legacy_depth)
        if legacy_depth is not None
        else None,
        conformity_gnn_layers=int(config_overrides["conformity_gnn_layers"])
        if config_overrides.get("conformity_gnn_layers") is not None
        else int(legacy_depth)
        if legacy_depth is not None
        else None,
        dropout=float(config_overrides["dropout"])
        if config_overrides.get("dropout") is not None
        else None,
        num_neighbors=list(config_overrides["num_neighbors"]),
        hard_negative_ratio=float(config_overrides.get("hard_negative_ratio", 0.0)),
        curriculum_phase1_end=int(config_overrides.get("curriculum_phase1_end", 15)),
        curriculum_phase2_end=int(config_overrides.get("curriculum_phase2_end", 30)),
        loss_schedule=str(config_overrides["loss_schedule"])
        if config_overrides.get("loss_schedule") is not None
        else None,
        loader_max_rows=None,
        sample_interactions=None,
        device=cli_args.device or "cuda",
        data_dir=cli_args.data_dir or "data",
        no_mlflow=bool(cli_args.no_mlflow),
        mlflow_tracking_uri=cli_args.mlflow_tracking_uri,
        mlflow_experiment_name=cli_args.mlflow_experiment_name or "ucagnn-formal",
        batch_id=_build_batch_id(profile_name),
        resume_batch=True,
        dry_run=cli_args.dry_run,
    )


def _override_resumed_args(
    benchmark_args: argparse.Namespace,
    cli_args: argparse.Namespace,
    profile_name: str,
) -> argparse.Namespace:
    """Apply a small set of runtime overrides when resuming a saved run."""
    for attribute in (
        "device",
        "data_dir",
        "mlflow_tracking_uri",
        "mlflow_experiment_name",
    ):
        value = getattr(cli_args, attribute)
        if value is not None:
            setattr(benchmark_args, attribute, value)
    if cli_args.no_mlflow:
        benchmark_args.no_mlflow = True
    benchmark_args.dry_run = cli_args.dry_run
    benchmark_args.resume_batch = True
    benchmark_args.sample_interactions = None
    benchmark_args.loader_max_rows = None
    if cli_args.restart:
        benchmark_args.batch_id = _build_batch_id(profile_name)
    return benchmark_args


def _resolve_benchmark_args(
    cli_args: argparse.Namespace,
) -> tuple[argparse.Namespace, str, bool]:
    """Resolve whether to create a new formal run or resume the saved one."""
    saved_state = _load_state()
    current_profile_bundle = None

    requested_profile = None
    if cli_args.profile is not None:
        current_profile_bundle = _resolve_profile_bundle(
            _normalize_profile_name(cli_args.profile)
        )
        requested_profile = str(current_profile_bundle["name"])
    saved_profile = None
    saved_profile_bundle = None
    if saved_state is not None:
        raw_saved_profile = str(
            saved_state.get("profile_name")
            or saved_state.get("version")
            or DEFAULT_PROFILE_NAME
        )
        try:
            saved_profile_bundle = _resolve_profile_bundle(raw_saved_profile)
            saved_profile = str(saved_profile_bundle["name"])
        except ValueError:
            saved_profile = raw_saved_profile
            saved_profile_bundle = saved_state.get("profile_config")

    should_resume_latest = cli_args.resume_latest or (
        requested_profile is None and not cli_args.new_run and saved_state is not None
    )
    if should_resume_latest:
        if saved_state is None:
            raise ValueError("No saved formal-run state exists to resume.")
        if saved_profile_bundle is None:
            if cli_args.resume_latest:
                raise ValueError(
                    "The saved formal-run state references a profile that is no longer defined. "
                    "Use --new-run or --restart to start a fresh batch."
                )
            profile_name = requested_profile or DEFAULT_PROFILE_NAME
            benchmark_args = _build_new_run_args(cli_args, profile_name)
            return benchmark_args, profile_name, False
        benchmark_args = _state_to_args(saved_state)
        profile_name = saved_profile or DEFAULT_PROFILE_NAME
        expected_args = _build_new_run_args(cli_args, profile_name)
        if not _plans_match(benchmark_args, expected_args):
            if cli_args.resume_latest:
                raise ValueError(
                    "The saved formal-run state no longer matches the current profile bundle. "
                    "Use --new-run or --restart to start a fresh mini-batch-only batch."
                )
            benchmark_args = expected_args
            return benchmark_args, profile_name, False
        benchmark_args = _override_resumed_args(benchmark_args, cli_args, profile_name)
        benchmark_args.profile_name = profile_name
        return benchmark_args, profile_name, True

    if (
        requested_profile is not None
        and saved_state is not None
        and not cli_args.new_run
        and not cli_args.restart
        and requested_profile == saved_profile
        and saved_profile_bundle == current_profile_bundle
    ):
        benchmark_args = _state_to_args(saved_state)
        expected_args = _build_new_run_args(cli_args, requested_profile)
        if not _plans_match(benchmark_args, expected_args):
            profile_name = requested_profile
            benchmark_args = _build_new_run_args(cli_args, profile_name)
            return benchmark_args, profile_name, False
        benchmark_args = _override_resumed_args(
            benchmark_args,
            cli_args,
            requested_profile,
        )
        benchmark_args.profile_name = requested_profile
        return benchmark_args, requested_profile, True

    profile_name = requested_profile or DEFAULT_PROFILE_NAME
    benchmark_args = _build_new_run_args(cli_args, profile_name)
    return benchmark_args, profile_name, False


def _scoring_weight_modes_for_preset(
    preset: str,
    requested_modes: list[str],
) -> list[str]:
    """Return the applicable scoring-weight sweep values for one preset."""
    if preset == "lightgcn":
        return ["fixed"] if "fixed" in requested_modes else []
    return list(requested_modes)


def build_benchmark_plan(
    args: argparse.Namespace,
) -> list[tuple[str, str, str, str]]:
    """Build the ordered benchmark execution plan.

    The semantic thesis matrix remains dataset × preset × graph_method, but the
    execution order intentionally keeps datasets as the innermost loop so a
    dataset-specific failure surfaces during the first method sweep instead of
    after one dataset has already exhausted every method combination.
    """
    datasets = TIERS[args.tier]
    experiments: list[tuple[str, str, str, str]] = []
    presets = canonicalize_preset_names(args.presets)
    for preset, graph_method in product(presets, args.graph_methods):
        for scoring_weight_mode in _scoring_weight_modes_for_preset(
            preset,
            args.scoring_weight_modes,
        ):
            for dataset in datasets:
                experiments.append(
                    (
                        dataset,
                        preset,
                        graph_method,
                        scoring_weight_mode,
                    )
                )
    return experiments


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark CLI parser."""
    parser = argparse.ArgumentParser(description="Run U-CaGNN benchmark matrix")
    parser.add_argument(
        "--tier", choices=list(TIERS.keys()), default="small", help="Dataset tier"
    )
    parser.add_argument(
        "--presets",
        nargs="*",
        default=list(DEFAULT_PRESET_ORDER),
        choices=list(PRESETS.keys()),
        help="Presets to run (canonical defaults: ucagnn, lightgcn, dice_like).",
    )
    parser.add_argument(
        "--graph-methods",
        nargs="*",
        default=DEFAULT_GRAPH_METHODS,
        choices=DEFAULT_GRAPH_METHODS,
        help="Graph construction methods to run across the matrix",
    )
    parser.add_argument(
        "--scoring-weight-modes",
        nargs="*",
        default=DEFAULT_SCORING_WEIGHT_MODES,
        choices=SCORING_WEIGHT_MODE_CHOICES,
        help="Score-mixture modes to run. LightGCN stays fixed-only because learned weights are inapplicable without dual branches.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override epochs for all"
    )
    early_stopping_group = parser.add_mutually_exclusive_group()
    early_stopping_group.add_argument(
        "--early-stopping",
        dest="use_early_stopping",
        action="store_true",
        help="Enable early stopping for all benchmark runs.",
    )
    early_stopping_group.add_argument(
        "--no-early-stopping",
        dest="use_early_stopping",
        action="store_false",
        help="Disable early stopping for all benchmark runs.",
    )
    parser.set_defaults(use_early_stopping=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size for all runs in the matrix.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate for all runs in the matrix.",
    )
    parser.add_argument(
        "--num-neighbors",
        nargs="*",
        type=int,
        default=None,
        help="Optional mini-batch fan-out override applied to all matrix items.",
    )
    parser.add_argument(
        "--loader-max-rows",
        type=int,
        default=None,
        help="Optional dataset loader row cap for all runs in the matrix.",
    )
    parser.add_argument(
        "--sample-interactions",
        type=int,
        default=None,
        help="Optional interaction budget for sampled benchmark passes.",
    )
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking for all benchmark runs",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="Override MLflow tracking URI for all benchmark runs",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        default="ucagnn-benchmark",
        help="MLflow experiment name for benchmark runs",
    )
    parser.add_argument(
        "--batch-id",
        default=None,
        help="Optional batch identifier for grouping and resuming benchmark runs.",
    )
    parser.add_argument(
        "--profile-name",
        default=None,
        help="Optional semantic formal profile label to persist alongside batch metadata.",
    )
    parser.add_argument(
        "--resume-batch",
        action="store_true",
        help="Skip benchmark items already recorded with a terminal status for this batch id.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print plan without running"
    )
    return parser


def run_benchmark(args: argparse.Namespace) -> int:
    """Execute the formal benchmark matrix from parsed arguments."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    datasets = TIERS[args.tier]
    args.presets = canonicalize_preset_names(args.presets)
    batch_id = resolve_batch_id(args.batch_id, prefix="benchmark")
    profile_name = getattr(args, "profile_name", None)
    experiments = build_benchmark_plan(args)

    if not experiments:
        logger.error(
            "No benchmark experiments remain after applying the matrix filters."
        )
        return 1

    print("=" * 70)
    print(f"BENCHMARK PLAN: {len(experiments)} experiments")
    print(f"  Tier: {args.tier} ({len(datasets)} datasets: {', '.join(datasets)})")
    print(f"  Presets: {', '.join(args.presets)}")
    print(f"  Graph methods: {', '.join(args.graph_methods)}")
    print(f"  Score-mix modes: {', '.join(args.scoring_weight_modes)}")
    print(f"  Batch ID: {batch_id}")
    if profile_name:
        print(f"  Profile: {profile_name}")
    print(f"  Resume batch: {args.resume_batch}")
    print("=" * 70)

    if args.dry_run:
        print(
            f"\n{'#':>4} | {'Dataset':<15} | {'Preset':<12} | {'Graph':<5} | {'ScoreMix':<8}"
        )
        print("-" * 81)
        for i, (ds, pr, gm, swm) in enumerate(experiments, 1):
            print(f"{i:>4} | {ds:<15} | {pr:<12} | {gm:<5} | {swm:<8}")
        print(f"\nTotal: {len(experiments)} experiments (dry run, nothing executed)")
        return 0

    # Run experiments
    tracker = ExperimentLogger(db_path=str(DB_PATH))
    completed = 0
    failed = 0
    skipped = 0
    results = []

    for i, (
        dataset,
        preset,
        graph_method,
        scoring_weight_mode,
    ) in enumerate(experiments, 1):
        print(f"\n{'=' * 70}")
        print(
            f"[{i}/{len(experiments)}] {dataset} / {preset} / {graph_method} / {scoring_weight_mode}"
        )
        print("=" * 70)

        existing = tracker.find_latest_batch_experiment(
            batch_id=batch_id,
            dataset=dataset,
            preset=preset,
            intervention=None,
            seed=DEFAULT_SEED,
            training_mode="mini_batch",
            graph_method=graph_method,
            config_filters={"scoring_weight_mode": scoring_weight_mode},
        )
        if (
            args.resume_batch
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
                        "graph_method": graph_method,
                        "exp_id": existing["id"],
                        "metrics": tracker.get_metrics_for_split(
                            int(existing["id"]),
                            split="test",
                        ),
                        "elapsed_s": 0.0,
                    }
                )
            continue

        try:
            config_inputs: dict[str, object] = {
                "dataset": dataset,
                "recipe": None,
                "preset": preset,
                "seed": DEFAULT_SEED,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "single_branch_gnn_layers": getattr(
                    args,
                    "single_branch_gnn_layers",
                    None,
                ),
                "interest_gnn_layers": getattr(args, "interest_gnn_layers", None),
                "conformity_gnn_layers": getattr(args, "conformity_gnn_layers", None),
                "dropout": getattr(args, "dropout", None),
                "lr": args.lr,
                "use_early_stopping": args.use_early_stopping,
                "scoring_weight_mode": scoring_weight_mode,
                "graph_method": graph_method,
                "num_neighbors": args.num_neighbors,
                "hard_negative_ratio": getattr(args, "hard_negative_ratio", None),
                "curriculum_phase1_end": getattr(
                    args,
                    "curriculum_phase1_end",
                    None,
                ),
                "curriculum_phase2_end": getattr(
                    args,
                    "curriculum_phase2_end",
                    None,
                ),
                "sample_interactions": args.sample_interactions,
                "loader_max_rows": args.loader_max_rows,
                "loss_schedule": getattr(args, "loss_schedule", None),
                "device": args.device,
                "data_dir": args.data_dir,
            }

            config = build_config(config_inputs)
            t0 = time.time()
            result = run_experiment(
                config,
                preset=preset,
                enable_mlflow=not args.no_mlflow,
                mlflow_tracking_uri=args.mlflow_tracking_uri,
                mlflow_experiment_name=args.mlflow_experiment_name,
                batch_id=batch_id,
                profile_name=profile_name,
            )
            elapsed = time.time() - t0

            results.append(
                {
                    "dataset": dataset,
                    "preset": preset,
                    "graph_method": graph_method,
                    "scoring_weight_mode": scoring_weight_mode,
                    "exp_id": result["exp_id"],
                    "metrics": result["test_metrics"],
                    "elapsed_s": elapsed,
                }
            )
            completed += 1
            logger.info(f"Completed in {elapsed:.1f}s (exp_id={result['exp_id']})")

        except Exception as e:
            failed += 1
            logger.error(f"FAILED: {e}")
            traceback.print_exc()
            continue

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Completed: {completed}/{len(experiments)}")
    print(f"Failed: {failed}/{len(experiments)}")
    print(f"Skipped via resume: {skipped}")

    if results:
        print("Note: AvgPop@20 and AvgPop@40 are lower-is-better.")
        print(
            f"\n{'Dataset':<15} | {'Preset':<12} | {'Graph':<5} | {'ScoreMix':<8} | {'NDCG@20':>8} | {'Recall@20':>10} | {'AvgPop@20':>10} | {'NDCG@40':>8} | {'Recall@40':>10} | {'AvgPop@40':>10} | Time"
        )
        print("-" * 162)
        for r in results:
            ndcg_20 = metric_value(r["metrics"], "NDCG@20")
            recall_20 = metric_value(r["metrics"], "Recall@20")
            avg_pop_20 = metric_value(r["metrics"], "AveragePopularity@20")
            ndcg_40 = metric_value(r["metrics"], "NDCG@40")
            recall_40 = metric_value(r["metrics"], "Recall@40")
            avg_pop_40 = metric_value(r["metrics"], "AveragePopularity@40")
            print(
                f"{r['dataset']:<15} | {r['preset']:<12} | {r['graph_method']:<5} | {r['scoring_weight_mode']:<8} | "
                f"{ndcg_20:>8.4f} | {recall_20:>10.4f} | {avg_pop_20:>10.4f} | {ndcg_40:>8.4f} | {recall_40:>10.4f} | {avg_pop_40:>10.4f} | {r['elapsed_s']:.0f}s"
            )

    tracker.close()
    return 0 if failed == 0 else 1


def main() -> int:
    """Parse CLI arguments and run the benchmark matrix."""
    parser = build_parser()
    args = parser.parse_args()
    return run_benchmark(args)


def _build_formal_run_parser() -> argparse.ArgumentParser:
    """Build the simple formal-run CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run the formal U-CaGNN experiment matrix with semantic profile-based resume."
    )
    parser.add_argument(
        "--profile",
        "--version",
        dest="profile",
        default=None,
        help=(
            "Optional semantic formal profile slug. "
            f"Supported profiles: {', '.join(formal_profile_names())}."
        ),
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="Print the predefined formal profiles from experiments/experiment_catalog.json and exit.",
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
    parser.add_argument(
        "--device",
        default=None,
        help="Device override. Defaults to cuda for new runs.",
    )
    parser.add_argument("--data-dir", default=None, help="Data directory override.")
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging for this formal run.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="Optional MLflow tracking URI override.",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        default=None,
        help="Optional MLflow experiment name override. Defaults to ucagnn-formal.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the formal run plan without executing it.",
    )
    return parser


def formal_main() -> int:
    """Run the formal experiment workflow through one simple entry point."""
    parser = _build_formal_run_parser()
    cli_args = parser.parse_args()

    if cli_args.list_profiles:
        print("Available formal profiles:")
        for profile_name in formal_profile_names():
            profile = _resolve_profile_bundle(profile_name)
            print(f"  {profile_name}: {profile['description']}")
        return 0

    try:
        benchmark_args, profile_name, resumed = _resolve_benchmark_args(cli_args)
    except ValueError as exc:
        parser.error(str(exc))

    state = {
        "profile_name": profile_name,
        "batch_id": benchmark_args.batch_id,
        "resumed": resumed,
        "last_started_at_utc": datetime.now(timezone.utc).isoformat(),
        "last_finished_at_utc": None,
        "last_exit_code": None,
        "benchmark_args": _args_to_state(benchmark_args),
    }
    if not cli_args.dry_run:
        _write_state(state)

    print("=" * 70)
    print("FORMAL RUN")
    print(f"  Profile: {profile_name}")
    print(f"  Batch ID: {benchmark_args.batch_id}")
    print(f"  Resuming: {benchmark_args.resume_batch}")
    print("  Full datasets: True")
    print("  OOM fallback: log-and-continue")
    print("=" * 70)

    exit_code = run_benchmark(benchmark_args)
    state["last_finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    state["last_exit_code"] = exit_code
    if not cli_args.dry_run:
        _write_state(state)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
