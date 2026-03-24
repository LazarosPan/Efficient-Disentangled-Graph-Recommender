#!/usr/bin/env python
"""Delete the repository-local thesis SQLite database and its sidecar files."""

from __future__ import annotations

from pathlib import Path


DEFAULT_DB_PATH = Path(__file__).parent.parent / "results" / "thesis_experiments.db"
SQLITE_SIDE_SUFFIXES = ("-wal", "-shm")


def _iter_db_paths(db_path: Path) -> list[Path]:
    return [db_path, *[Path(f"{db_path}{suffix}") for suffix in SQLITE_SIDE_SUFFIXES]]


def print_plan(db_path: Path) -> None:
    print("=" * 72)
    print("RESET THESIS SQLITE")
    print("=" * 72)
    print(f"Database: {db_path}")
    for path in _iter_db_paths(db_path):
        print(f"- {path} | exists={'yes' if path.exists() else 'no'}")
    print("=" * 72)


def delete_database(db_path: Path) -> None:
    for path in _iter_db_paths(db_path):
        path.unlink(missing_ok=True)


def main() -> int:
    db_path = DEFAULT_DB_PATH.resolve()
    print_plan(db_path)
    delete_database(db_path)
    print("SQLite reset complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
