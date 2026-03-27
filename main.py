"""Simple formal experiment entry point with persisted resume state."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from experiments.run_benchmark import (
    DEFAULT_GRAPH_METHODS,
    DEFAULT_SEEDS,
    DEFAULT_TRAINING_MODES,
    PRESETS,
    TIERS,
    run_benchmark,
)

STATE_PATH = Path(__file__).parent / "results" / "formal_run_state.json"


def _timestamp_slug() -> str:
    """Return a compact UTC timestamp slug."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _normalize_version(raw_version: str) -> str:
    """Normalize a user-facing version label into a filesystem-safe slug."""
    normalized = "".join(
        character.lower() if character.isalnum() else "-"
        for character in raw_version.strip()
    )
    collapsed = "-".join(part for part in normalized.split("-") if part)
    if not collapsed:
        raise ValueError("Version label must contain at least one letter or number.")
    return collapsed


def _build_batch_id(version: str, restart: bool) -> str:
    """Build the batch identifier used by the formal benchmark runner."""
    if restart:
        return f"formal-{version}-restart-{_timestamp_slug()}"
    return f"formal-{version}"


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
        description="Run the formal U-CaGNN experiment matrix with simple versioned resume."
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Optional version label such as v1 or thesis-main. Re-run the same version to resume it.",
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
        help="Start the selected version from the beginning under a fresh batch id.",
    )
    parser.add_argument(
        "--tier",
        choices=list(TIERS.keys()),
        default=None,
        help="Dataset tier for a new formal run. Defaults to all datasets.",
    )
    parser.add_argument(
        "--presets",
        nargs="*",
        default=None,
        help="Presets for a new formal run. Defaults to the full benchmark preset list.",
    )
    parser.add_argument(
        "--training-modes",
        nargs="*",
        choices=DEFAULT_TRAINING_MODES,
        default=None,
        help="Training modes for a new formal run. Defaults to the full benchmark list.",
    )
    parser.add_argument(
        "--graph-methods",
        nargs="*",
        choices=DEFAULT_GRAPH_METHODS,
        default=None,
        help="Graph methods for a new formal run. Defaults to the full benchmark list.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Seeds for a new formal run. Defaults to the benchmark seed set.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch override for the whole formal run.",
    )
    parser.add_argument(
        "--sample-interactions",
        type=int,
        default=None,
        help="Optional sampled interaction budget. Leave unset for real formal runs.",
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
        "--fallback-on-oom",
        choices=["none", "cached_propagation", "mini_batch"],
        default=None,
        help="Optional explicit fallback mode for OOM runs.",
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
        "epochs": args.epochs,
        "sample_interactions": args.sample_interactions,
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
    return argparse.Namespace(**benchmark_args)


def _build_new_run_args(cli_args: argparse.Namespace, version: str) -> argparse.Namespace:
    """Build benchmark arguments for a fresh formal run."""
    return argparse.Namespace(
        tier=cli_args.tier or "all",
        presets=cli_args.presets or list(PRESETS.keys()),
        training_modes=cli_args.training_modes or list(DEFAULT_TRAINING_MODES),
        graph_methods=cli_args.graph_methods or list(DEFAULT_GRAPH_METHODS),
        seeds=cli_args.seeds or list(DEFAULT_SEEDS),
        epochs=cli_args.epochs,
        sample_interactions=cli_args.sample_interactions,
        device=cli_args.device or "cuda",
        data_dir=cli_args.data_dir or "data",
        no_mlflow=bool(cli_args.no_mlflow),
        mlflow_tracking_uri=cli_args.mlflow_tracking_uri,
        mlflow_experiment_name=cli_args.mlflow_experiment_name or "ucagnn-formal",
        batch_id=_build_batch_id(version, restart=cli_args.restart),
        resume_batch=not cli_args.restart,
        fallback_on_oom=cli_args.fallback_on_oom or "none",
        dry_run=cli_args.dry_run,
    )


def _override_resumed_args(
    benchmark_args: argparse.Namespace,
    cli_args: argparse.Namespace,
    version: str,
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
    if cli_args.fallback_on_oom is not None:
        benchmark_args.fallback_on_oom = cli_args.fallback_on_oom
    if cli_args.epochs is not None:
        benchmark_args.epochs = cli_args.epochs
    if cli_args.no_mlflow:
        benchmark_args.no_mlflow = True
    benchmark_args.dry_run = cli_args.dry_run
    benchmark_args.resume_batch = not cli_args.restart
    if cli_args.restart:
        benchmark_args.batch_id = _build_batch_id(version, restart=True)
    return benchmark_args


def _resolve_benchmark_args(
    cli_args: argparse.Namespace,
) -> tuple[argparse.Namespace, str, bool]:
    """Resolve whether to create a new formal run or resume the saved one."""
    saved_state = _load_state()

    should_resume_latest = cli_args.resume_latest or (
        cli_args.version is None and not cli_args.new_run and saved_state is not None
    )
    if should_resume_latest:
        if saved_state is None:
            raise ValueError("No saved formal-run state exists to resume.")
        benchmark_args = _state_to_args(saved_state)
        version = str(saved_state["version"])
        benchmark_args = _override_resumed_args(benchmark_args, cli_args, version)
        return benchmark_args, version, True

    version = _normalize_version(cli_args.version or _timestamp_slug())
    benchmark_args = _build_new_run_args(cli_args, version)
    return benchmark_args, version, False


def main() -> int:
    """Run the formal experiment workflow through one simple entry point."""
    parser = _build_parser()
    cli_args = parser.parse_args()

    try:
        benchmark_args, version, resumed = _resolve_benchmark_args(cli_args)
    except ValueError as exc:
        parser.error(str(exc))

    state = {
        "version": version,
        "batch_id": benchmark_args.batch_id,
        "resumed": resumed,
        "last_started_at_utc": datetime.now(timezone.utc).isoformat(),
        "last_finished_at_utc": None,
        "last_exit_code": None,
        "benchmark_args": _args_to_state(benchmark_args),
    }
    _write_state(state)

    print("=" * 70)
    print("FORMAL RUN")
    print(f"  Version: {version}")
    print(f"  Batch ID: {benchmark_args.batch_id}")
    print(f"  Resuming: {benchmark_args.resume_batch}")
    print("=" * 70)

    exit_code = run_benchmark(benchmark_args)
    state["last_finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    state["last_exit_code"] = exit_code
    _write_state(state)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
