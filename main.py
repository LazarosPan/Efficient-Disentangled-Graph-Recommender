"""Simple formal experiment entry point with persisted resume state."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from experiments.recipes import formal_profile_names, get_formal_profile
from experiments.run_benchmark import (
    run_benchmark,
)

STATE_PATH = Path(__file__).parent / "results" / "formal_run_state.json"
DEFAULT_PROFILE_NAME = "v1"


def _timestamp_slug() -> str:
    """Return a compact UTC timestamp slug."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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
    return f"formal-{profile_name}-{_timestamp_slug()}"


def _load_state() -> dict[str, object] | None:
    """Load the last formal-run state file if it exists."""
    if not STATE_PATH.exists():
        return None
    return json.loads(STATE_PATH.read_text())


def _write_state(payload: dict[str, object]) -> None:
    """Persist the formal-run state file."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _build_parser() -> argparse.ArgumentParser:
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
            "Optional semantic formal profile such as v1. "
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


def _args_to_state(args: argparse.Namespace) -> dict[str, object]:
    """Convert benchmark arguments into a JSON-safe state payload."""
    return {
        "tier": args.tier,
        "presets": list(args.presets),
        "training_modes": list(args.training_modes),
        "graph_methods": list(args.graph_methods),
        "seeds": list(args.seeds),
        "profile_name": args.profile_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "num_neighbors": list(args.num_neighbors) if args.num_neighbors is not None else None,
        "loader_max_rows": args.loader_max_rows,
        "device": args.device,
        "data_dir": args.data_dir,
        "no_mlflow": args.no_mlflow,
        "mlflow_tracking_uri": args.mlflow_tracking_uri,
        "mlflow_experiment_name": args.mlflow_experiment_name,
        "batch_id": args.batch_id,
        "resume_batch": args.resume_batch,
        "fallback_on_oom": args.fallback_on_oom,
        "dry_run": args.dry_run,
    }


def _state_to_args(state: dict[str, object]) -> argparse.Namespace:
    """Restore benchmark arguments from saved state."""
    benchmark_args = state["benchmark_args"]
    assert isinstance(benchmark_args, dict)
    normalized_args = {
        "batch_size": None,
        "lr": None,
        "num_neighbors": None,
        "loader_max_rows": None,
        "sample_interactions": None,
        "profile_name": state.get("profile_name") or state.get("version"),
        "fallback_on_oom": "none",
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
        "training_modes",
        "graph_methods",
        "seeds",
        "epochs",
        "batch_size",
        "lr",
        "num_neighbors",
        "loader_max_rows",
        "sample_interactions",
        "fallback_on_oom",
        "profile_name",
    )
    for field_name in comparable_fields:
        if getattr(saved_args, field_name, None) != getattr(expected_args, field_name, None):
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
    return argparse.Namespace(
        tier=str(matrix["tier"]),
        presets=list(matrix["presets"]),
        training_modes=list(matrix["training_modes"]),
        graph_methods=list(matrix["graph_methods"]),
        seeds=list(matrix["seeds"]),
        profile_name=profile_name,
        epochs=int(config_overrides["epochs"]),
        batch_size=int(config_overrides["batch_size"]),
        lr=float(config_overrides["lr"]),
        num_neighbors=list(config_overrides["num_neighbors"]),
        loader_max_rows=None,
        sample_interactions=None,
        device=cli_args.device or "cuda",
        data_dir=cli_args.data_dir or "data",
        no_mlflow=bool(cli_args.no_mlflow),
        mlflow_tracking_uri=cli_args.mlflow_tracking_uri,
        mlflow_experiment_name=cli_args.mlflow_experiment_name or "ucagnn-formal",
        batch_id=_build_batch_id(profile_name),
        resume_batch=True,
        fallback_on_oom="none",
        dry_run=cli_args.dry_run,
    )


def _override_resumed_args(
    benchmark_args: argparse.Namespace,
    cli_args: argparse.Namespace,
    profile_name: str,
) -> argparse.Namespace:
    """Apply a small set of runtime overrides when resuming a saved run."""
    if cli_args.device is not None:
        benchmark_args.device = cli_args.device
    if cli_args.data_dir is not None:
        benchmark_args.data_dir = cli_args.data_dir
    if cli_args.mlflow_tracking_uri is not None:
        benchmark_args.mlflow_tracking_uri = cli_args.mlflow_tracking_uri
    if cli_args.mlflow_experiment_name is not None:
        benchmark_args.mlflow_experiment_name = cli_args.mlflow_experiment_name
    if cli_args.no_mlflow:
        benchmark_args.no_mlflow = True
    benchmark_args.dry_run = cli_args.dry_run
    benchmark_args.resume_batch = True
    benchmark_args.sample_interactions = None
    benchmark_args.loader_max_rows = None
    benchmark_args.fallback_on_oom = "none"
    if cli_args.restart:
        benchmark_args.batch_id = _build_batch_id(profile_name)
    return benchmark_args


def _resolve_benchmark_args(
    cli_args: argparse.Namespace,
) -> tuple[argparse.Namespace, str, bool]:
    """Resolve whether to create a new formal run or resume the saved one."""
    saved_state = _load_state()
    current_profile_bundle = None

    requested_profile = (
        _normalize_profile_name(cli_args.profile) if cli_args.profile is not None else None
    )
    if requested_profile is not None:
        current_profile_bundle = _resolve_profile_bundle(requested_profile)
    saved_profile = None
    saved_profile_bundle = None
    if saved_state is not None:
        saved_profile = str(
            saved_state.get("profile_name")
            or saved_state.get("version")
            or DEFAULT_PROFILE_NAME
        )
        saved_profile_bundle = saved_state.get("profile_config")

    should_resume_latest = cli_args.resume_latest or (
        requested_profile is None and not cli_args.new_run and saved_state is not None
    )
    if should_resume_latest:
        if saved_state is None:
            raise ValueError("No saved formal-run state exists to resume.")
        benchmark_args = _state_to_args(saved_state)
        profile_name = saved_profile or DEFAULT_PROFILE_NAME
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


def main() -> int:
    """Run the formal experiment workflow through one simple entry point."""
    parser = _build_parser()
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
        "version": profile_name,
        "profile_name": profile_name,
        "profile_config": _resolve_profile_bundle(profile_name),
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
