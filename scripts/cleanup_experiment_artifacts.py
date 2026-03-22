#!/usr/bin/env python
"""Delete repository-local experiment tracking files and artifacts.

Dry-run is the default. Use ``--yes`` to execute.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
SQLITE_DB_PATH = RESULTS_DIR / "thesis_experiments.db"
MLFLOW_DB_PATH = RESULTS_DIR / "mlflow.db"
MLFLOW_ARTIFACTS_DIR = REPO_ROOT / "mlruns"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"


@dataclass(frozen=True)
class CleanupTarget:
    label: str
    path: Path
    kind: str
    description: str


TARGETS = {
    "sqlite": CleanupTarget(
        label="sqlite",
        path=SQLITE_DB_PATH,
        kind="file",
        description="Delete the thesis SQLite database file under results/.",
    ),
    "mlflow-db": CleanupTarget(
        label="mlflow-db",
        path=MLFLOW_DB_PATH,
        kind="file",
        description="Delete the MLflow backend SQLite database file under results/.",
    ),
    "mlflow-artifacts": CleanupTarget(
        label="mlflow-artifacts",
        path=MLFLOW_ARTIFACTS_DIR,
        kind="directory",
        description="Delete the repository-local MLflow artifact directory mlruns/.",
    ),
    "checkpoints": CleanupTarget(
        label="checkpoints",
        path=CHECKPOINT_DIR,
        kind="directory",
        description="Delete all locally saved checkpoints under results/checkpoints/.",
    ),
}

SQLITE_SIDE_SUFFIXES = ("-wal", "-shm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete repository-local experiment tracking files and artifacts",
    )
    parser.add_argument(
        "--sqlite",
        action="store_true",
        help="Delete results/thesis_experiments.db and its SQLite sidecar files.",
    )
    parser.add_argument(
        "--mlflow-db",
        action="store_true",
        help="Delete results/mlflow.db and its SQLite sidecar files.",
    )
    parser.add_argument(
        "--mlflow-artifacts",
        action="store_true",
        help="Delete the repository-local mlruns/ artifact directory.",
    )
    parser.add_argument(
        "--checkpoints",
        action="store_true",
        help="Delete the results/checkpoints/ directory.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Select all cleanup targets.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Execute deletions. Without this flag the script performs a dry run.",
    )
    return parser.parse_args()


def selected_target_keys(args: argparse.Namespace) -> list[str]:
    selected: list[str] = []
    if args.all or args.sqlite:
        selected.append("sqlite")
    if args.all or args.mlflow_db:
        selected.append("mlflow-db")
    if args.all or args.mlflow_artifacts:
        selected.append("mlflow-artifacts")
    if args.all or args.checkpoints:
        selected.append("checkpoints")
    return selected


def iter_sidecar_paths(path: Path) -> list[Path]:
    return [Path(f"{path}{suffix}") for suffix in SQLITE_SIDE_SUFFIXES]


def print_plan(selected: list[CleanupTarget], dry_run: bool) -> None:
    mode = "DRY RUN" if dry_run else "EXECUTE"
    print("=" * 72)
    print(f"EXPERIMENT ARTIFACT CLEANUP ({mode})")
    print("=" * 72)
    for target in selected:
        exists = target.path.exists()
        print(f"- {target.label}: {target.path}")
        print(f"  kind: {target.kind}")
        print(f"  exists: {'yes' if exists else 'no'}")
        print(f"  note: {target.description}")
        if target.kind == "file":
            sidecars = iter_sidecar_paths(target.path)
            present_sidecars = [str(sidecar) for sidecar in sidecars if sidecar.exists()]
            if present_sidecars:
                print(f"  sidecars: {', '.join(present_sidecars)}")
    print("=" * 72)


def delete_file_target(path: Path) -> None:
    path.unlink(missing_ok=True)
    for sidecar in iter_sidecar_paths(path):
        sidecar.unlink(missing_ok=True)


def delete_directory_target(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def main() -> int:
    args = parse_args()
    keys = selected_target_keys(args)
    dry_run = not args.yes

    if not keys:
        print("No cleanup targets selected. Choose one or more flags such as --sqlite, --mlflow-db, --mlflow-artifacts, --checkpoints, or --all.")
        return 1

    selected = [TARGETS[key] for key in keys]
    print_plan(selected, dry_run)

    if dry_run:
        print("No changes applied. Re-run with --yes to execute.")
        return 0

    for target in selected:
        if target.kind == "file":
            delete_file_target(target.path)
        else:
            delete_directory_target(target.path)

    print("Cleanup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())