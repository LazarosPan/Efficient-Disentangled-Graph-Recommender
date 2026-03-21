"""Loader registry — every loader returns a CanonicalInteractions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from ..canonical import CanonicalInteractions

from .movielens1m import load_movielens1m
from .movielens20m import load_movielens20m
from .taobao import load_taobao
from .kuairec_v2 import load_kuairec_v2
from .amazonbook import load_amazonbook
from .kuairand1k import load_kuairand1k

LOADERS: dict[str, Callable[..., CanonicalInteractions]] = {
    "movielens1m": load_movielens1m,
    "movielens20m": load_movielens20m,
    "taobao": load_taobao,
    "kuairec_v2": load_kuairec_v2,
    "amazonbook": load_amazonbook,
    "kuairand1k": load_kuairand1k,
}


def load_dataset(name: str, data_dir: str = "data") -> CanonicalInteractions:
    """Load a dataset by name."""
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(LOADERS.keys())}")
    return LOADERS[name](data_dir)
