#!/usr/bin/env python
"""Provide a concise command reference for the repository's CLI surface."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CommandEntry:
    """Structured help entry for one command.

    Args:
        name: Stable command key used for filtering.
        command: Full command invocation.
        summary: One-line purpose statement.
        details: Short operational note about when to use it.
        group: Display group name.
        examples: Representative invocations.
    """

    name: str
    command: str
    summary: str
    details: str
    group: str
    examples: tuple[str, ...]


COMMANDS = (
    CommandEntry(
        name="quick-validate",
        command="uv run quick-validate",
        summary="Default tiny end-to-end validator across recipes, ablations, observability, and evaluation.",
        details="Use this after code changes. Add --mlflow only when you need the optional MLflow probe.",
        group="Canonical Workflow",
        examples=("uv run quick-validate", "uv run quick-validate --mlflow"),
    ),
    CommandEntry(
        name="experiment",
        command="uv run experiment",
        summary="Run one concrete experiment recipe or preset configuration.",
        details="Use this for a single tracked training run, not for matrix orchestration.",
        group="Canonical Workflow",
        examples=(
            "uv run experiment --list-recipes",
            "uv run experiment --dataset movielens1m --recipe full_full_graph_knn",
        ),
    ),
    CommandEntry(
        name="benchmark",
        command="uv run benchmark",
        summary="Execute the formal dataset × preset × training_mode × graph_method matrix.",
        details="Start with --dry-run before launching a wider benchmark tier.",
        group="Canonical Workflow",
        examples=(
            "uv run benchmark --tier small --dry-run",
            "uv run benchmark --tier small",
        ),
    ),
    CommandEntry(
        name="ablation",
        command="uv run ablation",
        summary="Run component-removal studies for one dataset.",
        details="Use --dry-run to inspect the planned variants before execution.",
        group="Canonical Workflow",
        examples=(
            "uv run ablation --dataset movielens1m --dry-run",
            "uv run ablation --dataset movielens1m",
        ),
    ),
    CommandEntry(
        name="reset-experiment-db",
        command="uv run reset-experiment-db",
        summary="Delete only the thesis SQLite database and its sidecars.",
        details="Use this when you want a fresh thesis DB without touching MLflow artifacts or checkpoints.",
        group="Analysis And Maintenance",
        examples=("uv run reset-experiment-db",),
    ),
    CommandEntry(
        name="cleanup-experiment-artifacts",
        command="uv run cleanup-experiment-artifacts",
        summary="Delete the local MLflow DB, MLflow artifacts, and checkpoints.",
        details="Use this when you want to clear repository-local run artifacts beyond the thesis SQLite record.",
        group="Analysis And Maintenance",
        examples=("uv run cleanup-experiment-artifacts",),
    ),
    CommandEntry(
        name="query-results",
        command="uv run query-results",
        summary="Inspect the thesis SQLite database from the command line.",
        details="This is the supported result-inspection path until a smaller plotting/reporting path exists.",
        group="Analysis And Maintenance",
        examples=(
            "uv run query-results",
            "uv run query-results --metrics 12",
            "uv run query-results --profiling 12",
        ),
    ),
    CommandEntry(
        name="audit-metrics",
        command="uv run audit-metrics",
        summary="Check source code and stored metrics against the allowed PyG metric families.",
        details="Use this when you touch evaluator logging or result-reporting semantics.",
        group="Analysis And Maintenance",
        examples=("uv run audit-metrics", "uv run audit-metrics --strict"),
    ),
    CommandEntry(
        name="verify-setup",
        command="uv run verify-setup",
        summary="Check environment imports and basic project readiness.",
        details="Use this for setup debugging, not as the normal post-change validator.",
        group="Specialized Diagnostics",
        examples=("uv run verify-setup", "uv run verify-setup --all"),
    ),
    CommandEntry(
        name="verify-sqlite",
        command="uv run verify-sqlite",
        summary="Exercise ExperimentLogger and SQLite schema behavior in isolation.",
        details="Use this when database setup or tracking behavior is suspicious.",
        group="Specialized Diagnostics",
        examples=("uv run verify-sqlite", "uv run verify-sqlite --keep-db"),
    ),
    CommandEntry(
        name="preflight",
        command="uv run preflight",
        summary="Run a representative smoke harness with checkpoint and resume checks.",
        details="Use this before longer runs when you want a realistic but still limited orchestration probe.",
        group="Specialized Diagnostics",
        examples=(
            "uv run preflight --dry-run",
            "uv run preflight --profile fast",
        ),
    ),
    CommandEntry(
        name="feature-probes",
        command="uv run feature-probes",
        summary="Run tiny feature-utility and feature-policy thesis probes.",
        details="Use this when validating whether side features help or whether broader optional scans are worth keeping.",
        group="Specialized Diagnostics",
        examples=("uv run feature-probes",),
    ),
    CommandEntry(
        name="verify-pipeline",
        command="uv run verify-pipeline",
        summary="Legacy alias for quick-validate.",
        details="Prefer quick-validate for new usage. Keep this only for compatibility with older commands or notes.",
        group="Compatibility",
        examples=("uv run verify-pipeline",),
    ),
    CommandEntry(
        name="download-datasets",
        command="uv run download-datasets",
        summary="Fetch the small subset of PyG-managed datasets the repo can bootstrap automatically.",
        details="This is a setup helper, not a routine development command.",
        group="Data",
        examples=("uv run download-datasets",),
    ),
)

GROUPS = tuple(dict.fromkeys(entry.group for entry in COMMANDS))
TERMS = (
    ("ablation", "Run variants with parts removed to measure impact."),
    ("recipe", "Named experiment setup from the catalog."),
    ("preset", "Model configuration family."),
    (
        "sample-interactions",
        "Run on a smaller canonical interaction sample for smoke testing.",
    ),
    ("--keep-db", "Keep the temporary verification database instead of deleting it."),
    ("--mlflow", "Enable MLflow logging for commands that keep it off by default."),
    ("--no-auto-resume", "Start a fresh run even if a matching checkpoint exists."),
)


def parse_args() -> argparse.Namespace:
    """Parse list-commands filters and display controls.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Show a concise command reference for the repository"
    )
    parser.add_argument(
        "--group",
        choices=GROUPS,
        help="Show only one command group",
    )
    parser.add_argument(
        "--command",
        choices=[entry.name for entry in COMMANDS],
        help="Show detailed help for one command",
    )
    parser.add_argument(
        "--no-terms",
        action="store_true",
        help="Hide the glossary section in summary views",
    )
    return parser.parse_args()


def _print_command(entry: CommandEntry) -> None:
    """Print one detailed command help entry.

    Args:
        entry: Command entry to display.
    """
    print(entry.name)
    print(f"  {entry.command}")
    print(f"  {entry.summary}")
    print(f"  {entry.details}")
    print("  Examples:")
    for example in entry.examples:
        print(f"    {example}")


def _print_group(group: str) -> None:
    """Print all commands for one group.

    Args:
        group: Group label to display.
    """
    print(group)
    for entry in COMMANDS:
        if entry.group != group:
            continue
        print(f"  {entry.command}")
        print(f"    {entry.summary}")
    print()


def _print_terms() -> None:
    """Print the short workflow glossary."""
    print("Terms")
    for term, description in TERMS:
        print(f"  {term}: {description}")


def main() -> int:
    args = parse_args()
    print("U-CaGNN command reference")
    print()

    if args.command is not None:
        entry = next(command for command in COMMANDS if command.name == args.command)
        _print_command(entry)
        return 0

    if args.group is not None:
        _print_group(args.group)
        if not args.no_terms:
            _print_terms()
        return 0

    for group in GROUPS:
        _print_group(group)
    if not args.no_terms:
        _print_terms()
    print()
    print(
        "Tip: use --command <name> for one command or --group <group> to narrow the output."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
