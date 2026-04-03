#!/usr/bin/env python
"""Delete repository-local experiment artifacts and generated run-state files."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
MLFLOW_DB_PATH = RESULTS_DIR / "mlflow.db"
FORMAL_RUN_STATE_PATH = RESULTS_DIR / "formal_run_state.json"
MLFLOW_ARTIFACTS_DIR = REPO_ROOT / "mlruns"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"


@dataclass(frozen=True)
class CleanupTarget:
    label: str
    path: Path
    kind: str
    description: str


TARGETS = (
    CleanupTarget(
        label="mlflow-db",
        path=MLFLOW_DB_PATH,
        kind="file",
        description="Delete the MLflow backend SQLite database file under results/.",
    ),
    CleanupTarget(
        label="formal-run-state",
        path=FORMAL_RUN_STATE_PATH,
        kind="file",
        description="Delete the generated formal-run resume state JSON under results/.",
    ),
    CleanupTarget(
        label="mlflow-artifacts",
        path=MLFLOW_ARTIFACTS_DIR,
        kind="directory",
        description="Delete the repository-local MLflow artifact directory mlruns/.",
    ),
    CleanupTarget(
        label="checkpoints",
        path=CHECKPOINT_DIR,
        kind="directory",
        description="Delete all locally saved checkpoints under results/checkpoints/.",
    ),
)

SQLITE_SIDE_SUFFIXES = ("-wal", "-shm")


def iter_sidecar_paths(path: Path) -> list[Path]:
    return [Path(f"{path}{suffix}") for suffix in SQLITE_SIDE_SUFFIXES]


def print_plan(selected: tuple[CleanupTarget, ...]) -> None:
    print("=" * 72)
    print("RESET EXPERIMENT ARTIFACTS")
    print("=" * 72)
    for target in selected:
        exists = target.path.exists()
        print(f"- {target.label}: {target.path}")
        print(f"  kind: {target.kind}")
        print(f"  exists: {'yes' if exists else 'no'}")
        print(f"  note: {target.description}")
        if target.kind == "file":
            sidecars = iter_sidecar_paths(target.path)
            present_sidecars = [
                str(sidecar) for sidecar in sidecars if sidecar.exists()
            ]
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
    selected = TARGETS
    print_plan(selected)

    for target in selected:
        if target.kind == "file":
            delete_file_target(target.path)
        else:
            delete_directory_target(target.path)

    print("Experiment artifact cleanup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
