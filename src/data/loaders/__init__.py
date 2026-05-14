"""Loader registry — every loader returns a CanonicalInteractions."""

from __future__ import annotations

from ._registry import (
    LOADERS,
    default_preprocessing_preset,
    load_dataset,
    resolve_preprocessing_preset,
)

__all__ = [
    "LOADERS",
    "default_preprocessing_preset",
    "load_dataset",
    "resolve_preprocessing_preset",
]
