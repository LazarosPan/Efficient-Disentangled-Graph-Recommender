#!/usr/bin/env python
"""Reset experiment-tracking tables in the SQLite database.

Dry-run is the default. Use ``--yes`` to execute.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path


DEFAULT_DB_PATH = Path(__file__).parent.parent / "results" / "thesis_experiments.db"
TABLES = ("profiling", "metrics", "experiments")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reset thesis experiment SQLite tables")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to the SQLite database.",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        choices=TABLES,
        help="Subset of tables to clear. Default clears all tables.",
    )
    parser.add_argument(
        "--drop-and-recreate",
        action="store_true",
        help="Drop selected tables and recreate the schema via ExperimentLogger.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Execute changes. Without this flag the script performs a dry run.",
    )
    return parser.parse_args()


def ordered_tables(selected_tables: list[str]) -> list[str]:
    order = {name: idx for idx, name in enumerate(TABLES)}
    return sorted(selected_tables, key=lambda name: order[name])


def validate_selection(selected_tables: list[str]) -> str | None:
    selected = set(selected_tables)
    if "experiments" in selected and ({"profiling", "metrics"} - selected):
        return (
            "Clearing 'experiments' requires also clearing 'profiling' and 'metrics' "
            "because SQLite foreign keys are enabled."
        )
    return None


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def print_plan(
    db_path: Path,
    selected_tables: list[str],
    drop_and_recreate: bool,
    dry_run: bool,
) -> None:
    mode = "DRY RUN" if dry_run else "EXECUTE"
    print("=" * 72)
    print(f"EXPERIMENT DB RESET ({mode})")
    print("=" * 72)
    print(f"Database: {db_path}")
    print(f"Tables:   {', '.join(selected_tables)}")
    print(f"Mode:     {'drop + recreate' if drop_and_recreate else 'delete rows'}")
    print("=" * 72)


def delete_rows(conn: sqlite3.Connection, selected_tables: list[str]) -> None:
    for table_name in selected_tables:
        conn.execute(f"DELETE FROM {table_name}")


def drop_tables(conn: sqlite3.Connection, selected_tables: list[str]) -> None:
    conn.execute("DROP VIEW IF EXISTS experiment_summary")
    for table_name in reversed(selected_tables):
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")


def main() -> int:
    args = parse_args()
    db_path = args.db_path.resolve()
    selected_tables = ordered_tables(args.tables or list(TABLES))
    dry_run = not args.yes

    selection_error = validate_selection(selected_tables)
    if selection_error is not None:
        print(selection_error)
        return 1

    print_plan(db_path, selected_tables, args.drop_and_recreate, dry_run)

    if not db_path.exists():
        print(f"Database does not exist: {db_path}")
        return 1

    if dry_run:
        print("No changes applied. Re-run with --yes to execute.")
        return 0

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON")

    try:
        missing = [table_name for table_name in selected_tables if not table_exists(conn, table_name)]
        if missing:
            print(f"Missing table(s): {', '.join(missing)}")
            return 1

        if args.drop_and_recreate:
            drop_tables(conn, selected_tables)
            conn.commit()
            conn.close()

            sys.path.insert(0, str(Path(__file__).parent.parent))
            from src.utils.experiment_logger import ExperimentLogger

            logger = ExperimentLogger(db_path=str(db_path))
            logger.close()
        else:
            delete_rows(conn, selected_tables)
            conn.commit()
            conn.close()

        print("Database reset complete.")
        return 0
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())