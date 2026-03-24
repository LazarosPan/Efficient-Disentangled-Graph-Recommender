#!/usr/bin/env python
"""Audit metric names across source files, SQLite, and MLflow.

The allowed metric families are derived from `.github/skills/pytorch-geometric/metrics.md`.
This script is intentionally read-only unless `--strict` is used for CI-style failure.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
PYG_METRICS_DOC = (
    PROJECT_ROOT / ".github" / "skills" / "pytorch-geometric" / "metrics.md"
)
THESIS_DB_PATH = PROJECT_ROOT / "results" / "thesis_experiments.db"
MLFLOW_DB_PATH = PROJECT_ROOT / "results" / "mlflow.db"
SCAN_ROOTS = ("src", "scripts", "experiments")
DOC_ROOTS = (".github/skills/ucagnn-implementation", "docs/ucagnn_implementation")
METRIC_LITERAL_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)@(\d+)\b")
MLFLOW_TEST_PREFIX = "test_"


@dataclass(frozen=True)
class MetricOccurrence:
    family: str
    literal: str
    location: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit repo/database metric names against the PyG metrics doc"
    )
    parser.add_argument(
        "--include-docs",
        action="store_true",
        help="Also scan implementation docs for metric literals",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when non-PyG metric names are found",
    )
    return parser.parse_args()


def _allowed_metric_families() -> set[str]:
    content = PYG_METRICS_DOC.read_text(encoding="utf-8")
    families: set[str] = set()
    for match in re.finditer(r"`(LinkPred[A-Za-z0-9]+)`", content):
        name = match.group(1)
        if name in {"LinkPredMetric", "LinkPredMetricCollection"}:
            continue
        families.add(name.removeprefix("LinkPred"))
    return families


def _iter_files(include_docs: bool) -> list[Path]:
    roots = list(SCAN_ROOTS)
    if include_docs:
        roots.extend(DOC_ROOTS)
    files: list[Path] = []
    for root in roots:
        root_path = PROJECT_ROOT / root
        if not root_path.exists():
            continue
        for path in root_path.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".md", ".toml", ".json"}:
                files.append(path)
    return sorted(files)


def _scan_source_literals(
    allowed: set[str], include_docs: bool
) -> tuple[list[MetricOccurrence], list[MetricOccurrence]]:
    allowed_hits: list[MetricOccurrence] = []
    disallowed_hits: list[MetricOccurrence] = []
    for path in _iter_files(include_docs):
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for family, k_value in METRIC_LITERAL_RE.findall(text):
            occurrence = MetricOccurrence(
                family=family, literal=f"{family}@{k_value}", location=relative
            )
            if family in allowed:
                allowed_hits.append(occurrence)
            else:
                disallowed_hits.append(occurrence)
    return allowed_hits, disallowed_hits


def _query_sqlite_metric_names(db_path: Path) -> list[str]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT DISTINCT metric_name FROM metrics WHERE metric_name LIKE '%@%' ORDER BY metric_name"
        ).fetchall()
    finally:
        conn.close()
    return [row[0] for row in rows]


def _query_mlflow_metric_names(db_path: Path) -> list[str]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT DISTINCT key FROM metrics WHERE key LIKE 'test_%' ORDER BY key"
        ).fetchall()
    finally:
        conn.close()
    names: list[str] = []
    for (key,) in rows:
        if not key.startswith(MLFLOW_TEST_PREFIX):
            continue
        unsuffixed = key[len(MLFLOW_TEST_PREFIX) :]
        normalized = unsuffixed.replace("_at_", "@")
        names.append(normalized)
    return sorted(set(names))


def _partition_metric_names(
    names: list[str], allowed: set[str]
) -> tuple[list[str], list[str]]:
    allowed_names: list[str] = []
    disallowed_names: list[str] = []
    for name in names:
        match = METRIC_LITERAL_RE.fullmatch(name)
        if match is None:
            disallowed_names.append(name)
            continue
        family = match.group(1)
        if family in allowed:
            allowed_names.append(name)
        else:
            disallowed_names.append(name)
    return allowed_names, disallowed_names


def _print_section(title: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)


def _print_occurrences(title: str, occurrences: list[MetricOccurrence]) -> None:
    print(title)
    if not occurrences:
        print("  none")
        return
    for occurrence in occurrences:
        print(f"  {occurrence.literal:<24} {occurrence.location}")


def _print_names(title: str, names: list[str]) -> None:
    print(title)
    if not names:
        print("  none")
        return
    for name in names:
        print(f"  {name}")


def main() -> int:
    args = parse_args()
    allowed = _allowed_metric_families()
    source_allowed, source_disallowed = _scan_source_literals(
        allowed, include_docs=args.include_docs
    )
    sqlite_names = _query_sqlite_metric_names(THESIS_DB_PATH)
    sqlite_allowed, sqlite_disallowed = _partition_metric_names(sqlite_names, allowed)
    mlflow_names = _query_mlflow_metric_names(MLFLOW_DB_PATH)
    mlflow_allowed, mlflow_disallowed = _partition_metric_names(mlflow_names, allowed)

    _print_section("PYG METRIC AUDIT")
    print("Allowed metric families from metrics.md:")
    print("  " + ", ".join(sorted(allowed)))
    print(
        "  note: Diversity and Personalization are allowed by PyG, but not currently logged by the evaluator."
    )
    print(
        "  note: database findings report distinct metric names across stored history, so stale rows from older runs remain visible until cleaned."
    )

    print("-" * 78)
    _print_occurrences("Implementation-source metric literals", source_allowed)
    _print_occurrences("Non-PyG source metric literals", source_disallowed)

    print("-" * 78)
    _print_names("SQLite metric names", sqlite_allowed)
    _print_names("Non-PyG SQLite metric names", sqlite_disallowed)

    print("-" * 78)
    _print_names("MLflow metric names", mlflow_allowed)
    _print_names("Non-PyG MLflow metric names", mlflow_disallowed)

    has_findings = bool(source_disallowed or sqlite_disallowed or mlflow_disallowed)
    print("-" * 78)
    print(f"Findings: {'present' if has_findings else 'none'}")
    return 1 if args.strict and has_findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
