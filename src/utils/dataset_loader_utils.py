"""Shared dataset-loader utilities for local-path resolution and safe field parsing."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def resolve_local_dataset_dir(
    candidates: Sequence[Path],
    required_files: Sequence[str],
    missing_message: str,
) -> Path:
    """Return the first local dataset directory that contains all required files."""
    for raw_dir in candidates:
        if all((raw_dir / name).exists() for name in required_files):
            return raw_dir
    raise FileNotFoundError(missing_message)


def get_optional_csv_field(parts: Sequence[str], idx: int) -> str | None:
    """Return a CSV field by index, or None when the field is absent."""
    if idx < 0 or idx >= len(parts):
        return None
    return parts[idx]


def try_parse_int(value: str | None) -> int | None:
    """Parse an integer field, returning None when parsing fails."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def try_parse_float(value: str | None) -> float | None:
    """Parse a floating-point field, returning None when parsing fails."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def try_parse_timestamp_seconds(value: str | None) -> int | None:
    """Parse a timestamp field that may arrive as an integer or float string."""
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
