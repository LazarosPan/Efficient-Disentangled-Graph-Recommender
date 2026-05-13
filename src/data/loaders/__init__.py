"""Loader registry — every loader returns a CanonicalInteractions."""

from __future__ import annotations

from ._registry import (
    LOADERS,
    load_dataset,
    resolve_preprocessing_preset,
)

__all__ = [
    "LOADERS",
    "load_dataset",
    "resolve_preprocessing_preset",
]
