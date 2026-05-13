"""Shared dataset-loader utilities for local-path resolution and numeric downcasting."""

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


def downcast_numeric_array(
    values: np.ndarray,
    *,
    allow_float16: bool = False,
) -> np.ndarray:
    """Return a numeric array narrowed to the smallest safe NumPy storage dtype.

    Args:
        values: Numeric array to narrow.
        allow_float16: Whether floating arrays may be narrowed to ``float16``.

    Returns:
        Array narrowed by actual value range rather than by the original dtype
        alone. Integer arrays are checked against every standard NumPy signed or
        unsigned width with ``np.iinfo``. Floating arrays are checked against
        ``np.finfo`` and may narrow to ``float16`` only when explicitly allowed.
        Boolean, empty, and non-numeric arrays are returned unchanged.

    """
    if values.size == 0 or values.dtype == np.bool_:
        return values
    if np.issubdtype(values.dtype, np.integer):
        min_value = int(values.min())
        max_value = int(values.max())
        if min_value >= 0:
            for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
                if max_value <= np.iinfo(dtype).max:
                    return values.astype(dtype, copy=False)
            return values.astype(np.uint64, copy=False)
        for dtype in (np.int8, np.int16, np.int32, np.int64):
            info = np.iinfo(dtype)
            if info.min <= min_value and max_value <= info.max:
                return values.astype(dtype, copy=False)
        return values.astype(np.int64, copy=False)
    if not np.issubdtype(values.dtype, np.floating):
        return values

    finite_values = values[np.isfinite(values)]
    candidate_dtypes: tuple[type[np.floating], ...] = (np.float16, np.float32, np.float64) if allow_float16 else (np.float32, np.float64)
    if finite_values.size == 0:
        return values.astype(candidate_dtypes[0], copy=False)

    min_value = float(finite_values.min())
    max_value = float(finite_values.max())
    for dtype in candidate_dtypes:
        info = np.finfo(dtype)
        min_supported = float(info.min)
        max_supported = float(info.max)
        if min_supported <= min_value and max_value <= max_supported:
            return values.astype(dtype, copy=False)
    return values.astype(np.float64, copy=False)


def downcast_numeric_arrays(
    *values: np.ndarray,
    allow_float16: bool = False,
) -> tuple[np.ndarray, ...]:
    """Return multiple arrays narrowed with the shared numeric downcast policy.

    Args:
        *values: Numeric arrays to narrow independently.
        allow_float16: Whether floating arrays may narrow to ``float16``.

    Returns:
        tuple[np.ndarray, ...]: Arrays downcast in the same order they were passed.

    """
    return tuple(downcast_numeric_array(value, allow_float16=allow_float16) for value in values)
