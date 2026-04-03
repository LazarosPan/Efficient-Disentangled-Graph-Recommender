"""Shared dataset-loader utilities for local-path resolution and safe field parsing."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np


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


def downcast_numeric_array(
    values: np.ndarray,
    *,
    allow_float16: bool = False,
) -> np.ndarray:
    """Downcast a numeric NumPy array to the smallest safe storage dtype.

    Args:
        values: Numeric array to narrow.
        allow_float16: Whether floating arrays may be narrowed to ``float16``.

    Returns:
        Array with the narrowest practical dtype in the same numeric family.
    """
    if values.size == 0 or values.dtype == np.bool_:
        return values
    if np.issubdtype(values.dtype, np.integer):
        target_dtype = np.result_type(
            np.min_scalar_type(values.min()),
            np.min_scalar_type(values.max()),
        )
        return values.astype(target_dtype, copy=False)
    if not np.issubdtype(values.dtype, np.floating):
        return values

    if not allow_float16:
        return values.astype(np.float32, copy=False)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return values.astype(np.float16, copy=False)
    if (
        finite_values.min() >= np.finfo(np.float16).min
        and finite_values.max() <= np.finfo(np.float16).max
    ):
        return values.astype(np.float16, copy=False)
    return values.astype(np.float32, copy=False)
