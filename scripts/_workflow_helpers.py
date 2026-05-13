"""Shared helpers for experiment-oriented scripts."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC, datetime


def configure_cli_logging(level: int = logging.INFO) -> None:
    """Configure the repository's standard CLI logging format.

    Args:
        level: Logging level passed to ``logging.basicConfig``.

    Returns:
        None.

    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def resolve_batch_id(provided: str | None, prefix: str) -> str:
    """Return an explicit or generated batch id for grouped runs."""
    if provided:
        return provided
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}"


def metric_value(metrics: Mapping[str, float], metric_name: str) -> float:
    """Return a metric value, falling back from @40 to @20 when needed."""
    if metric_name in metrics:
        return metrics[metric_name]
    if metric_name.endswith("@40"):
        fallback = metric_name.replace("@40", "@20")
        return metrics.get(fallback, 0.0)
    return 0.0


def thesis_metric_values(
    metrics: Mapping[str, float],
    metric_names: tuple[str, ...],
) -> dict[str, float]:
    """Return the requested thesis metrics with fallback handling applied.

    Args:
        metrics: Metric mapping from an experiment result.
        metric_names: Ordered thesis metric names to extract.

    Returns:
        dict[str, float]: Metric-name to scalar-value mapping.

    """
    return {metric_name: metric_value(metrics, metric_name) for metric_name in metric_names}


def print_batch_summary_counts(
    *,
    title: str,
    completed: int,
    failed: int,
    skipped: int,
    total: int,
) -> None:
    """Print the shared batch-run summary header and counters.

    Args:
        title: Human-readable summary title.
        completed: Number of completed runs.
        failed: Number of failed runs.
        skipped: Number of resumed/skipped runs.
        total: Total number of planned runs.

    Returns:
        None.

    """
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"Completed: {completed}/{total}")
    print(f"Failed: {failed}/{total}")
    print(f"Skipped via resume: {skipped}")
