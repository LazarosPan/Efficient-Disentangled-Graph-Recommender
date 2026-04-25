"""Shared helpers for experiment-oriented scripts."""

from __future__ import annotations

from datetime import datetime, timezone


def resolve_batch_id(provided: str | None, prefix: str) -> str:
    """Return an explicit or generated batch id for grouped runs."""
    if provided:
        return provided
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}"


def metric_value(metrics: dict[str, float], metric_name: str) -> float:
    """Return a metric value, falling back from @40 to @20 when needed."""
    if metric_name in metrics:
        return metrics[metric_name]
    if metric_name.endswith("@40"):
        fallback = metric_name.replace("@40", "@20")
        return metrics.get(fallback, 0.0)
    return 0.0
