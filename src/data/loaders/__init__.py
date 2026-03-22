"""Loader registry — every loader returns a CanonicalInteractions."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from ...feature_policy import DEFAULT_FEATURE_POLICY, FeaturePolicyName

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


@lru_cache(maxsize=32)
def _load_dataset_cached(
    name: str,
    data_dir: str,
    max_rows: int,
    include_optional_features: bool,
    feature_policy: FeaturePolicyName,
) -> CanonicalInteractions:
    return LOADERS[name](
        data_dir,
        max_rows=max_rows,
        include_optional_features=include_optional_features,
        feature_policy=feature_policy,
    )


def load_dataset(
    name: str,
    data_dir: str = "data",
    max_rows: int | None = None,
    include_optional_features: bool = True,
    feature_policy: FeaturePolicyName = DEFAULT_FEATURE_POLICY,
) -> CanonicalInteractions:
    """Load a dataset by name."""
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(LOADERS.keys())}")
    if max_rows is None:
        return LOADERS[name](
            data_dir,
            max_rows=max_rows,
            include_optional_features=include_optional_features,
            feature_policy=feature_policy,
        )
    return _load_dataset_cached(
        name,
        data_dir,
        max_rows,
        include_optional_features,
        feature_policy,
    )
