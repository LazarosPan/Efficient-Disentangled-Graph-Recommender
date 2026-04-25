#!/usr/bin/env python
"""Formal matrix benchmark runner for dataset x preset x graph_method.

Usage:
    uv run formal-run --dry-run
    uv run formal-run --profile default
    uv run formal-run --resume-latest
"""

from __future__ import annotations

from collections.abc import Mapping
import argparse
import json
from itertools import product
import logging
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from experiments.cli_parsers import build_benchmark_parser, build_formal_run_parser
from experiments.recipes import (
    default_formal_profile_name,
    formal_profile_names,
    get_formal_profile,
)
from experiments.run_experiment import (
    DB_PATH,
    build_config,
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

DEFAULT_SCORING_WEIGHT_MODES = ["learned"]
PLAN_COMPARISON_FIELDS = (
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


def _normalize_benchmark_args(
    raw_args: argparse.Namespace | Mapping[str, object] | object,
    *,
    fallback_profile_name: str | None = None,
) -> dict[str, object]:
    """Return a JSON-safe benchmark payload."""
    benchmark_args = raw_args if isinstance(raw_args, Mapping) else vars(raw_args)
    allowed_fields = {
        "tier",
        "presets",
        "graph_methods",
        "scoring_weight_modes",
        "profile_name",
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
        "device",
        "data_dir",
        "no_mlflow",
        "mlflow_tracking_uri",
        "mlflow_experiment_name",
        "batch_id",
        "resume_batch",
        "dry_run",
    }
    unexpected_fields = sorted(set(benchmark_args) - allowed_fields)
    if unexpected_fields:
        raise ValueError(
            "Formal-run state contains unexpected fields "
            f"{unexpected_fields}. Start a fresh formal run."
        )
    scoring_weight_modes = benchmark_args.get("scoring_weight_modes")
    if not isinstance(scoring_weight_modes, (list, tuple)) or not scoring_weight_modes:
        scoring_weight_modes = list(DEFAULT_SCORING_WEIGHT_MODES)
    num_neighbors = benchmark_args.get("num_neighbors")
    return {
        "tier": str(benchmark_args["tier"]),
        "presets": list(benchmark_args["presets"]),
        "graph_methods": list(benchmark_args["graph_methods"]),
        "scoring_weight_modes": list(scoring_weight_modes),
        "profile_name": benchmark_args.get("profile_name") or fallback_profile_name,
        "epochs": benchmark_args.get("epochs"),
        "use_early_stopping": benchmark_args.get("use_early_stopping", True),
        "batch_size": benchmark_args.get("batch_size"),
        "lr": benchmark_args.get("lr"),
        "single_branch_gnn_layers": benchmark_args.get("single_branch_gnn_layers"),
        "interest_gnn_layers": benchmark_args.get("interest_gnn_layers"),
        "conformity_gnn_layers": benchmark_args.get("conformity_gnn_layers"),
        "dropout": benchmark_args.get("dropout"),
        "num_neighbors": list(num_neighbors) if num_neighbors is not None else None,
        "hard_negative_ratio": benchmark_args.get("hard_negative_ratio"),
        "curriculum_phase1_end": benchmark_args.get("curriculum_phase1_end"),
        "curriculum_phase2_end": benchmark_args.get("curriculum_phase2_end"),
        "loss_schedule": benchmark_args.get("loss_schedule"),
        "loader_max_rows": benchmark_args.get("loader_max_rows"),
        "sample_interactions": benchmark_args.get("sample_interactions"),
        "device": benchmark_args.get("device"),
        "data_dir": benchmark_args.get("data_dir"),
        "no_mlflow": benchmark_args.get("no_mlflow"),
        "mlflow_tracking_uri": benchmark_args.get("mlflow_tracking_uri"),
        "mlflow_experiment_name": benchmark_args.get("mlflow_experiment_name"),
        "batch_id": benchmark_args.get("batch_id"),
        "resume_batch": benchmark_args.get("resume_batch"),
        "dry_run": benchmark_args.get("dry_run"),
    }


def _benchmark_plan_signature(
    args: argparse.Namespace | Mapping[str, object] | object,
) -> tuple[object, ...]:
    """Return the semantic plan signature used to compare saved formal runs."""
    normalized_args = _normalize_benchmark_args(args)
    return tuple(normalized_args[field_name] for field_name in PLAN_COMPARISON_FIELDS)


def _build_new_run_args(
    cli_args: argparse.Namespace,
    profile_name: str,
) -> dict[str, object]:
    """Build benchmark arguments for a fresh formal run."""
    profile_bundle = _resolve_profile_bundle(profile_name)
    matrix = profile_bundle["matrix"]
    assert isinstance(matrix, dict)
    config_overrides = profile_bundle["config_overrides"]
    assert isinstance(config_overrides, dict)
    return {
        "tier": str(matrix["tier"]),
        "presets": list(matrix["presets"]),
        "graph_methods": list(matrix["graph_methods"]),
        "scoring_weight_modes": list(
            matrix.get("scoring_weight_modes", DEFAULT_SCORING_WEIGHT_MODES)
        ),
        "profile_name": profile_name,
        "epochs": int(config_overrides["epochs"]),
        "use_early_stopping": bool(config_overrides.get("use_early_stopping", True)),
        "batch_size": int(config_overrides["batch_size"]),
        "lr": float(config_overrides["lr"]),
        "single_branch_gnn_layers": int(config_overrides["single_branch_gnn_layers"])
        if config_overrides.get("single_branch_gnn_layers") is not None
        else None,
        "interest_gnn_layers": int(config_overrides["interest_gnn_layers"])
        if config_overrides.get("interest_gnn_layers") is not None
        else None,
        "conformity_gnn_layers": int(config_overrides["conformity_gnn_layers"])
        if config_overrides.get("conformity_gnn_layers") is not None
        else None,
        "dropout": float(config_overrides["dropout"])
        if config_overrides.get("dropout") is not None
        else None,
        "num_neighbors": list(config_overrides["num_neighbors"]),
        "hard_negative_ratio": float(config_overrides.get("hard_negative_ratio", 0.0)),
        "curriculum_phase1_end": int(config_overrides.get("curriculum_phase1_end", 15)),
        "curriculum_phase2_end": int(config_overrides.get("curriculum_phase2_end", 30)),
        "loss_schedule": str(config_overrides["loss_schedule"])
        if config_overrides.get("loss_schedule") is not None
        else None,
        "loader_max_rows": None,
        "sample_interactions": None,
        "device": cli_args.device or "cuda",
        "data_dir": cli_args.data_dir or "data",
        "no_mlflow": bool(cli_args.no_mlflow),
        "mlflow_tracking_uri": cli_args.mlflow_tracking_uri,
        "mlflow_experiment_name": cli_args.mlflow_experiment_name or "ucagnn-formal",
        "batch_id": _build_batch_id(profile_name),
        "resume_batch": True,
        "dry_run": cli_args.dry_run,
    }


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
    overridden_args["dry_run"] = cli_args.dry_run
    overridden_args["resume_batch"] = True
    overridden_args["sample_interactions"] = None
    overridden_args["loader_max_rows"] = None
    if cli_args.restart:
        overridden_args["batch_id"] = _build_batch_id(profile_name)
    return overridden_args


def _resolve_benchmark_args(
    cli_args: argparse.Namespace,
) -> tuple[dict[str, object], str, bool]:
    """Resolve whether to create a new formal run or resume the saved one."""
    saved_state = _load_state()
    current_profile_bundle = None
    saved_benchmark_args: dict[str, object] | None = None

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
        raw_saved_benchmark_args = saved_state.get("benchmark_args")
        if isinstance(raw_saved_benchmark_args, dict):
            saved_benchmark_args = _normalize_benchmark_args(
                raw_saved_benchmark_args,
                fallback_profile_name=raw_saved_profile,
            )

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
        assert saved_benchmark_args is not None
        benchmark_args = saved_benchmark_args
        profile_name = saved_profile or DEFAULT_PROFILE_NAME
        expected_args = _build_new_run_args(cli_args, profile_name)
        if _benchmark_plan_signature(benchmark_args) != _benchmark_plan_signature(
            expected_args
        ):
            if cli_args.resume_latest:
                raise ValueError(
                    "The saved formal-run state no longer matches the current profile bundle. "
                    "Use --new-run or --restart to start a fresh mini-batch-only batch."
                )
            benchmark_args = expected_args
            return benchmark_args, profile_name, False
        benchmark_args = _override_resumed_args(benchmark_args, cli_args, profile_name)
        benchmark_args["profile_name"] = profile_name
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
            expected_args
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
    args: argparse.Namespace | Mapping[str, object] | object,
) -> list[tuple[str, str, str, str]]:
    """Build the ordered benchmark execution plan.

    The semantic thesis matrix remains dataset × preset × graph_method, but the
    execution order intentionally keeps datasets as the innermost loop so a
    dataset-specific failure surfaces during the first method sweep instead of
    after one dataset has already exhausted every method combination.
    """
    benchmark_args = _normalize_benchmark_args(args)
    datasets = TIERS[str(benchmark_args["tier"])]
    experiments: list[tuple[str, str, str, str]] = []
    presets = list(dict.fromkeys(benchmark_args["presets"]))
    for preset, graph_method in product(presets, benchmark_args["graph_methods"]):
        for scoring_weight_mode in _scoring_weight_modes_for_preset(
            preset,
            benchmark_args["scoring_weight_modes"],
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


def run_benchmark(args: argparse.Namespace | Mapping[str, object] | object) -> int:
    """Execute the formal benchmark matrix from parsed arguments."""
    benchmark_args = _normalize_benchmark_args(args)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    datasets = TIERS[str(benchmark_args["tier"])]
    presets = list(dict.fromkeys(benchmark_args["presets"]))
    batch_id = resolve_batch_id(benchmark_args["batch_id"], prefix="benchmark")
    profile_name = benchmark_args.get("profile_name")
    experiments = build_benchmark_plan(benchmark_args)

    if not experiments:
        logger.error(
            "No benchmark experiments remain after applying the matrix filters."
        )
        return 1

    print("=" * 70)
    print(f"BENCHMARK PLAN: {len(experiments)} experiments")
    print(
        f"  Tier: {benchmark_args['tier']} ({len(datasets)} datasets: {', '.join(datasets)})"
    )
    print(f"  Presets: {', '.join(presets)}")
    print(f"  Graph methods: {', '.join(benchmark_args['graph_methods'])}")
    print(f"  Score-mix modes: {', '.join(benchmark_args['scoring_weight_modes'])}")
    print(f"  Batch ID: {batch_id}")
    if profile_name:
        print(f"  Profile: {profile_name}")
    print(f"  Resume batch: {benchmark_args['resume_batch']}")
    print("=" * 70)

    if benchmark_args["dry_run"]:
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
                "epochs": benchmark_args["epochs"],
                "batch_size": benchmark_args["batch_size"],
                "single_branch_gnn_layers": benchmark_args["single_branch_gnn_layers"],
                "interest_gnn_layers": benchmark_args["interest_gnn_layers"],
                "conformity_gnn_layers": benchmark_args["conformity_gnn_layers"],
                "dropout": benchmark_args["dropout"],
                "lr": benchmark_args["lr"],
                "use_early_stopping": benchmark_args["use_early_stopping"],
                "scoring_weight_mode": scoring_weight_mode,
                "graph_method": graph_method,
                "num_neighbors": benchmark_args["num_neighbors"],
                "hard_negative_ratio": benchmark_args["hard_negative_ratio"],
                "curriculum_phase1_end": benchmark_args["curriculum_phase1_end"],
                "curriculum_phase2_end": benchmark_args["curriculum_phase2_end"],
                "sample_interactions": benchmark_args["sample_interactions"],
                "loader_max_rows": benchmark_args["loader_max_rows"],
                "loss_schedule": benchmark_args["loss_schedule"],
                "device": benchmark_args["device"],
                "data_dir": benchmark_args["data_dir"],
            }

            config = build_config(config_inputs)
            t0 = time.time()
            result = run_experiment(
                config,
                preset=preset,
                enable_mlflow=not bool(benchmark_args["no_mlflow"]),
                mlflow_tracking_uri=benchmark_args["mlflow_tracking_uri"],
                mlflow_experiment_name=str(benchmark_args["mlflow_experiment_name"]),
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
    args = build_benchmark_parser().parse_args()
    return run_benchmark(args)


def formal_main() -> int:
    """Run the formal experiment workflow through one simple entry point."""
    parser = build_formal_run_parser()
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
        "batch_id": benchmark_args["batch_id"],
        "resumed": resumed,
        "last_started_at_utc": datetime.now(timezone.utc).isoformat(),
        "last_finished_at_utc": None,
        "last_exit_code": None,
        "benchmark_args": _normalize_benchmark_args(benchmark_args),
    }
    if not cli_args.dry_run:
        _write_state(state)

    print("=" * 70)
    print("FORMAL RUN")
    print(f"  Profile: {profile_name}")
    print(f"  Batch ID: {benchmark_args['batch_id']}")
    print(f"  Resuming: {benchmark_args['resume_batch']}")
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
    import sys

    sys.exit(main())
