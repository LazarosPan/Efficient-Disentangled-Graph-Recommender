"""Repository-local shared paths for experiment orchestration and cleanup tools."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
THESIS_DB_PATH = RESULTS_DIR / "thesis_experiments.db"
MLFLOW_DB_PATH = RESULTS_DIR / "mlflow.db"
FORMAL_RUN_STATE_PATH = RESULTS_DIR / "formal_run_state.json"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
MLFLOW_ARTIFACTS_DIR = REPO_ROOT / "mlruns"
SQLITE_SIDE_SUFFIXES = ("-wal", "-shm")


def iter_sqlite_sidecar_paths(path: Path) -> list[Path]:
    """Return SQLite sidecar paths for one database file.

    Args:
        path: Base SQLite database path.

    Returns:
        List of the ``-wal`` and ``-shm`` sidecar paths for ``path``.

    """
    return [Path(f"{path}{suffix}") for suffix in SQLITE_SIDE_SUFFIXES]


__all__ = [
    "CHECKPOINT_DIR",
    "FORMAL_RUN_STATE_PATH",
    "MLFLOW_ARTIFACTS_DIR",
    "MLFLOW_DB_PATH",
    "REPO_ROOT",
    "RESULTS_DIR",
    "SQLITE_SIDE_SUFFIXES",
    "THESIS_DB_PATH",
    "iter_sqlite_sidecar_paths",
]
